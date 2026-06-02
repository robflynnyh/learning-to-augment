from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Rank1Perturbation:
    a: Tensor
    b: Tensor
    antithetic: bool = False

    @property
    def num_candidates(self) -> int:
        return int(self.a.shape[0])


@dataclass(frozen=True)
class EggrollPerturbations:
    layers: Dict[str, Rank1Perturbation]

    @property
    def num_candidates(self) -> int:
        if not self.layers:
            return 1
        return next(iter(self.layers.values())).num_candidates

    def get(self, name: str) -> Optional[Rank1Perturbation]:
        return self.layers.get(name)

    def items(self) -> Iterator[Tuple[str, Rank1Perturbation]]:
        return iter(self.layers.items())

    def to(self, device: torch.device | str) -> "EggrollPerturbations":
        return EggrollPerturbations(
            {
                name: Rank1Perturbation(
                    a=pert.a.to(device),
                    b=pert.b.to(device),
                    antithetic=pert.antithetic,
                )
                for name, pert in self.layers.items()
            }
        )


class EggrollLinear(nn.Module):
    """Linear layer with optional candidate-specific rank-1 EGGROLL noise."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Tensor,
        perturbation: Optional[Rank1Perturbation] = None,
        sigma: float = 0.0,
    ) -> Tensor:
        if perturbation is None or sigma == 0.0:
            return F.linear(x, self.weight, self.bias)

        if x.shape[1] != perturbation.a.shape[0]:
            raise ValueError(
                "EGGROLL candidate axis mismatch: "
                f"x has N={x.shape[1]}, perturbation has N={perturbation.a.shape[0]}"
            )
        if x.shape[-1] != perturbation.b.shape[-1]:
            raise ValueError(
                "EGGROLL feature mismatch: "
                f"x has D={x.shape[-1]}, perturbation.b has D={perturbation.b.shape[-1]}"
            )

        if x.dim() == 3:
            base = torch.einsum("bnd,od->bno", x, self.weight)
            scale = torch.einsum("bnd,nd->bn", x, perturbation.b)
            delta = scale[..., None] * perturbation.a[None, :, :]
        elif x.dim() == 4:
            base = torch.einsum("bntd,od->bnto", x, self.weight)
            scale = torch.einsum("bntd,nd->bnt", x, perturbation.b)
            delta = scale[..., None] * perturbation.a[None, :, None, :]
        else:
            raise ValueError("EggrollLinear expects [B,N,D] or [B,N,T,D] when perturbed")

        y = base + sigma * delta
        if self.bias is not None:
            y = y + self.bias
        return y


def iter_eggroll_linears(module: nn.Module) -> Iterator[Tuple[str, EggrollLinear]]:
    for name, child in module.named_modules():
        if isinstance(child, EggrollLinear):
            yield name, child


def sample_rank1_perturbations(
    module: nn.Module,
    num_candidates: int,
    *,
    antithetic: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> EggrollPerturbations:
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    if antithetic and num_candidates % 2 != 0:
        raise ValueError("antithetic EGGROLL requires an even num_candidates")

    layers: Dict[str, Rank1Perturbation] = {}
    for name, layer in iter_eggroll_linears(module):
        cur_device = device if device is not None else layer.weight.device
        cur_dtype = dtype if dtype is not None else layer.weight.dtype
        sample_count = num_candidates // 2 if antithetic else num_candidates
        a = torch.randn(
            sample_count,
            layer.out_features,
            generator=generator,
            device=cur_device,
            dtype=cur_dtype,
        )
        b = torch.randn(
            sample_count,
            layer.in_features,
            generator=generator,
            device=cur_device,
            dtype=cur_dtype,
        )
        if antithetic:
            a = torch.cat([a, -a], dim=0)
            b = torch.cat([b, b], dim=0)
        layers[name] = Rank1Perturbation(a=a, b=b, antithetic=antithetic)
    return EggrollPerturbations(layers)


def group_normalise_rewards(quality: Tensor, eps: float = 1e-8) -> Tensor:
    if quality.dim() != 2:
        raise ValueError("quality must have shape [B, N]")
    mean = quality.mean(dim=1, keepdim=True)
    std = quality.std(dim=1, keepdim=True, unbiased=False)
    return (quality - mean) / (std + eps)


def eggroll_delta_matrix(rewards: Tensor, perturbation: Rank1Perturbation, sigma: float) -> Tensor:
    if rewards.dim() != 1:
        raise ValueError("rewards must have shape [N]")
    if rewards.shape[0] != perturbation.num_candidates:
        raise ValueError("rewards and perturbation candidate counts differ")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if perturbation.antithetic:
        pair_count = perturbation.num_candidates // 2
        rewards_pos = rewards[:pair_count]
        rewards_neg = rewards[pair_count:]
        a = perturbation.a[:pair_count]
        b = perturbation.b[:pair_count]
        reward_diff = rewards_pos - rewards_neg
        delta = torch.einsum("p,po,pi->oi", reward_diff, a, b)
        return delta / (2 * pair_count * sigma)

    delta = torch.einsum("n,no,ni->oi", rewards, perturbation.a, perturbation.b)
    return delta / (perturbation.num_candidates * sigma)


def eggroll_update_shared_params(
    module: nn.Module,
    perturbations: EggrollPerturbations,
    rewards: Tensor,
    sigma: float,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Tensor]:
    """Apply the ascent EGGROLL estimate through a descent optimizer."""

    deltas: Dict[str, Tensor] = {}
    optimizer.zero_grad(set_to_none=True)
    layer_map = dict(iter_eggroll_linears(module))
    for name, perturbation in perturbations.items():
        if name not in layer_map:
            raise KeyError(f"No EggrollLinear named {name!r} in module")
        layer = layer_map[name]
        delta = eggroll_delta_matrix(rewards.to(layer.weight.device), perturbation, sigma)
        layer.weight.grad = -delta.to(dtype=layer.weight.dtype, device=layer.weight.device)
        deltas[name] = delta.detach()
    optimizer.step()
    return deltas
