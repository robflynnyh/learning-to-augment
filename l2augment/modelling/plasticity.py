from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from l2augment.utils.eggroll import EggrollLinear, EggrollPerturbations


@dataclass
class FastWeightFactors:
    F: Tensor
    base_weight_norm: Optional[Tensor] = None


@dataclass
class FastWeightUpdate:
    P: Tensor
    Q: Tensor
    eta: Tensor
    rho: Tensor


class FastWeightState:
    def __init__(self, factors: Mapping[str, FastWeightFactors]) -> None:
        self.factors = dict(factors)

    def __getitem__(self, name: str) -> FastWeightFactors:
        return self.factors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.factors

    def items(self) -> Iterator[Tuple[str, FastWeightFactors]]:
        return iter(self.factors.items())

    def keys(self) -> Iterable[str]:
        return self.factors.keys()

    def to(self, device: torch.device | str) -> "FastWeightState":
        return FastWeightState(
            {
                name: FastWeightFactors(
                    F=factors.F.to(device),
                    base_weight_norm=(
                        None if factors.base_weight_norm is None else factors.base_weight_norm.to(device)
                    ),
                )
                for name, factors in self.factors.items()
            }
        )


@dataclass
class AdaptedLinearSpec:
    name: str
    in_features: int
    out_features: int
    base_weight_norm: Tensor


@dataclass
class FastWeightForwardContext:
    fast_state: Optional[FastWeightState]
    batch_size: int
    num_candidates: int
    capture_activations: bool = False
    activations: Optional[Dict[str, Tensor]] = None


_FAST_WEIGHT_CONTEXT: ContextVar[Optional[FastWeightForwardContext]] = ContextVar(
    "fast_weight_context",
    default=None,
)


@contextmanager
def fast_weight_context(
    fast_state: Optional[FastWeightState],
    batch_size: int,
    num_candidates: int,
    *,
    capture_activations: bool = False,
):
    ctx = FastWeightForwardContext(
        fast_state=fast_state,
        batch_size=batch_size,
        num_candidates=num_candidates,
        capture_activations=capture_activations,
        activations={} if capture_activations else None,
    )
    token = _FAST_WEIGHT_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _FAST_WEIGHT_CONTEXT.reset(token)


class FastWeightLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, module_name: str) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("FastWeightLinear only supports nn.Linear modules")
        self.base_linear = base_linear
        self.module_name = module_name

    @property
    def in_features(self) -> int:
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        return self.base_linear.out_features

    def forward(
        self,
        x: Tensor,
        fast_state: Optional[FastWeightState] = None,
        batch_size: Optional[int] = None,
        num_candidates: Optional[int] = None,
    ) -> Tensor:
        ctx = _FAST_WEIGHT_CONTEXT.get()
        if fast_state is None and ctx is not None:
            fast_state = ctx.fast_state
        if batch_size is None and ctx is not None:
            batch_size = ctx.batch_size
        if num_candidates is None and ctx is not None:
            num_candidates = ctx.num_candidates

        if fast_state is None or self.module_name not in fast_state:
            y = self.base_linear(x)
            if ctx is not None and ctx.capture_activations and ctx.activations is not None:
                ctx.activations[self.module_name] = _reshape_linear_output(y, ctx.batch_size, ctx.num_candidates).detach()
            return y

        if batch_size is None or num_candidates is None:
            raise ValueError("batch_size and num_candidates are required with fast_state")

        original_shape = x.shape
        if x.shape[0] != batch_size * num_candidates:
            raise ValueError(
                f"Expected flattened batch {batch_size * num_candidates}, got {x.shape[0]}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input feature dim {self.in_features}, got {x.shape[-1]}")

        x_seq = x.reshape(batch_size, num_candidates, -1, self.in_features)
        y_seq = fast_weight_linear(
            x_seq,
            self.base_linear.weight,
            self.base_linear.bias,
            fast_state[self.module_name].F,
        )
        y = y_seq.reshape(*original_shape[:-1], self.out_features)

        if ctx is not None and ctx.capture_activations and ctx.activations is not None:
            ctx.activations[self.module_name] = y_seq.detach()
        return y


def _reshape_linear_output(y: Tensor, batch_size: int, num_candidates: int) -> Tensor:
    return y.reshape(batch_size, num_candidates, -1, y.shape[-1])


def fast_weight_linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    F_fast: Tensor,
) -> Tensor:
    base = torch.einsum("bntd,od->bnto", x, weight)
    delta = torch.einsum("bntd,bnod->bnto", x, F_fast)
    y = base + delta
    if bias is not None:
        y = y + bias
    return y


def init_fast_state(
    *,
    batch_size: int,
    num_candidates: int,
    module_specs: Mapping[str, Tuple[int, int] | AdaptedLinearSpec],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> FastWeightState:
    factors: Dict[str, FastWeightFactors] = {}
    for name, spec in module_specs.items():
        if isinstance(spec, AdaptedLinearSpec):
            out_features, in_features = spec.out_features, spec.in_features
            base_weight_norm = spec.base_weight_norm.to(device=device, dtype=dtype)
        else:
            out_features, in_features = spec
            base_weight_norm = None
        factors[name] = FastWeightFactors(
            F=torch.zeros(batch_size, num_candidates, out_features, in_features, device=device, dtype=dtype),
            base_weight_norm=base_weight_norm,
        )
    return FastWeightState(factors)


def dense_fast_weight_update(update: FastWeightUpdate) -> Tensor:
    eta = update.eta
    if eta.dim() != 3:
        raise ValueError("eta must have shape [B, N, R] or [B, N, 1]")
    if eta.shape[-1] not in (1, update.P.shape[-1]):
        raise ValueError("eta rank dimension must be 1 or match P/Q update rank")
    scaled_P = update.P * eta.unsqueeze(-2)
    return torch.einsum("bnor,bnir->bnoi", scaled_P, update.Q)


def apply_fast_updates(
    fast_state: FastWeightState,
    updates: Mapping[str, FastWeightUpdate],
    *,
    max_fast_rank: Optional[int] = None,
    max_fast_norm_ratio: Optional[float] = None,
    eps: float = 1e-8,
    return_metrics: bool = False,
) -> FastWeightState | Tuple[FastWeightState, Dict[str, Tensor]]:
    del max_fast_rank
    next_factors: Dict[str, FastWeightFactors] = {}
    state_ratios: List[Tensor] = []
    update_ratios: List[Tensor] = []
    clipped_flags: List[Tensor] = []
    for name, factors in fast_state.items():
        if name not in updates:
            next_factors[name] = factors
            continue

        update = updates[name]
        rho = update.rho.clamp(0.0, 1.0)
        rho_view = rho.reshape(rho.shape[0], rho.shape[1], 1, 1)
        dense_update = dense_fast_weight_update(update).to(dtype=factors.F.dtype, device=factors.F.device)
        F_next = rho_view * factors.F + dense_update

        if factors.base_weight_norm is not None:
            base_norm = factors.base_weight_norm.to(device=F_next.device, dtype=torch.float32).reshape(1, 1)
            update_norm = dense_update.float().norm(dim=(-2, -1))
            update_ratios.append(update_norm / (base_norm + eps))

        if max_fast_norm_ratio is not None and factors.base_weight_norm is not None:
            fast_norm = F_next.float().norm(dim=(-2, -1), keepdim=True)
            max_norm = (
                factors.base_weight_norm.to(device=F_next.device, dtype=torch.float32).reshape(1, 1, 1, 1)
                * max_fast_norm_ratio
            )
            scale = (max_norm / (fast_norm + eps)).clamp(max=1.0).to(dtype=F_next.dtype)
            clipped_flags.append((scale < 1.0).reshape(-1).float())
            F_next = F_next * scale

        if factors.base_weight_norm is not None:
            base_norm = factors.base_weight_norm.to(device=F_next.device, dtype=torch.float32).reshape(1, 1)
            state_norm = F_next.float().norm(dim=(-2, -1))
            state_ratios.append(state_norm / (base_norm + eps))

        next_factors[name] = FastWeightFactors(
            F=F_next,
            base_weight_norm=factors.base_weight_norm,
        )
    next_state = FastWeightState(next_factors)
    metrics = _fast_update_metrics(state_ratios, update_ratios, clipped_flags, device=next_state_metrics_device(next_state))
    if return_metrics:
        return next_state, metrics
    return next_state


def next_state_metrics_device(fast_state: FastWeightState) -> torch.device:
    first = next(iter(fast_state.factors.values()), None)
    if first is None:
        return torch.device("cpu")
    return first.F.device


def _fast_update_metrics(
    state_ratios: List[Tensor],
    update_ratios: List[Tensor],
    clipped_flags: List[Tensor],
    *,
    device: torch.device,
) -> Dict[str, Tensor]:
    if state_ratios:
        state_flat = torch.cat([item.reshape(-1).float() for item in state_ratios])
    else:
        state_flat = torch.zeros(1, device=device)
    if update_ratios:
        update_flat = torch.cat([item.reshape(-1).float() for item in update_ratios])
    else:
        update_flat = torch.zeros(1, device=device)
    if clipped_flags:
        clipped_flat = torch.cat([item.reshape(-1).float() for item in clipped_flags])
    else:
        clipped_flat = torch.zeros(1, device=device)
    return {
        "fast_state_norm_ratio_mean": state_flat.mean(),
        "fast_state_norm_ratio_max": state_flat.max(),
        "fast_update_norm_ratio_mean": update_flat.mean(),
        "fast_update_norm_ratio_max": update_flat.max(),
        "fast_weight_clipped_fraction": clipped_flat.mean(),
    }


def _unwrap_fast_weight_linear(module: nn.Module) -> nn.Module:
    if isinstance(module, FastWeightLinear):
        return module.base_linear
    return module


def discover_attention_out_proj_modules(model: nn.Module) -> List[str]:
    return [
        name
        for name, module in model.named_modules()
        if name.endswith("attend.fn.out_proj")
        and isinstance(_unwrap_fast_weight_linear(module), nn.Linear)
    ]


def collect_linear_specs(model: nn.Module, target_modules: Iterable[str]) -> Dict[str, AdaptedLinearSpec]:
    specs: Dict[str, AdaptedLinearSpec] = {}
    module_map = dict(model.named_modules())
    for name in target_modules:
        module = module_map.get(name)
        linear = _unwrap_fast_weight_linear(module) if module is not None else None
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"Target module {name!r} is not an nn.Linear")
        specs[name] = AdaptedLinearSpec(
            name=name,
            in_features=linear.in_features,
            out_features=linear.out_features,
            base_weight_norm=linear.weight.detach().norm(),
        )
    return specs


def wrap_linear_modules(model: nn.Module, target_modules: Iterable[str]) -> Dict[str, AdaptedLinearSpec]:
    target_list = list(target_modules)
    specs = collect_linear_specs(model, target_list)
    for name in target_list:
        parent, attr = _resolve_parent_module(model, name)
        child = getattr(parent, attr)
        if isinstance(child, FastWeightLinear):
            continue
        setattr(parent, attr, FastWeightLinear(child, name))
    return specs


def _resolve_parent_module(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def asr_forward_with_fast_state(
    *,
    asr_model: nn.Module,
    audio: Tensor,
    lengths: Optional[Tensor],
    fast_state: Optional[FastWeightState],
    batch_size: int,
    num_candidates: int,
    return_selected_activations: bool = True,
):
    flat_audio = audio.reshape(batch_size * num_candidates, *audio.shape[2:])
    flat_lengths = None if lengths is None else lengths.reshape(batch_size * num_candidates)
    kwargs = {"audio_signal": flat_audio}
    if flat_lengths is not None:
        kwargs["length"] = flat_lengths
    with torch.no_grad(), fast_weight_context(
        fast_state,
        batch_size,
        num_candidates,
        capture_activations=return_selected_activations,
    ) as ctx:
        output = asr_model(**kwargs)
    return output, (ctx.activations or {})


class LayerTokenProjector(nn.Module):
    def __init__(self, input_dim: int, token_dim: int) -> None:
        super().__init__()
        self.proj = EggrollLinear(input_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        activation: Tensor,
        perturbation=None,
        sigma: float = 0.0,
    ) -> Tensor:
        if activation.dim() < 3:
            raise ValueError("activation must include [B, N, ...] axes")
        x = activation.reshape(*activation.shape[:2], -1, activation.shape[-1]).mean(dim=-2)
        return self.norm(torch.tanh(self.proj(x, perturbation, sigma)))


class LayerCommunicationModule(nn.Module):
    def __init__(self, token_dim: int, comm_dim: int) -> None:
        super().__init__()
        self.in_proj = EggrollLinear(token_dim, comm_dim)
        self.out_proj = EggrollLinear(comm_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens: Tensor,
        perturbations: Optional[EggrollPerturbations],
        sigma: float,
        name_prefix: str,
    ) -> Tensor:
        h = F.gelu(
            self.in_proj(
                tokens,
                None if perturbations is None else perturbations.get(f"{name_prefix}.in_proj"),
                sigma,
            )
        )
        h = self.out_proj(
            h,
            None if perturbations is None else perturbations.get(f"{name_prefix}.out_proj"),
            sigma,
        )
        pooled = h.mean(dim=2, keepdim=True)
        return self.norm(tokens + h + pooled)


class LowRankUpdateHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        out_features: int,
        in_features: int,
        update_rank: int,
        max_eta: float,
        default_rho: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.update_rank = update_rank
        self.max_eta = max_eta
        self.default_rho = default_rho
        self.eps = eps
        self.p_head = EggrollLinear(token_dim, out_features * update_rank)
        self.q_head = EggrollLinear(token_dim, in_features * update_rank)
        self.eta_head = EggrollLinear(token_dim, update_rank)
        self.rho_head = EggrollLinear(token_dim, 1)

    def forward(
        self,
        token: Tensor,
        perturbations: Optional[EggrollPerturbations],
        sigma: float,
        name_prefix: str,
    ) -> FastWeightUpdate:
        B, N, _ = token.shape
        P = self.p_head(
            token,
            None if perturbations is None else perturbations.get(f"{name_prefix}.p_head"),
            sigma,
        ).reshape(B, N, self.out_features, self.update_rank)
        Q = self.q_head(
            token,
            None if perturbations is None else perturbations.get(f"{name_prefix}.q_head"),
            sigma,
        ).reshape(B, N, self.in_features, self.update_rank)
        P = P / (P.norm(dim=-2, keepdim=True) + self.eps)
        Q = Q / (Q.norm(dim=-2, keepdim=True) + self.eps)

        eta = self.max_eta * torch.tanh(
            self.eta_head(
                token,
                None if perturbations is None else perturbations.get(f"{name_prefix}.eta_head"),
                sigma,
            )
        )
        rho_raw = self.rho_head(
            token,
            None if perturbations is None else perturbations.get(f"{name_prefix}.rho_head"),
            sigma,
        )
        default_logit = torch.logit(torch.tensor(self.default_rho, device=token.device, dtype=token.dtype))
        rho = torch.sigmoid(rho_raw + default_logit).clamp(0.0, 1.0)
        return FastWeightUpdate(P=P, Q=Q, eta=eta, rho=rho)


class PlasticityPolicy(nn.Module):
    def __init__(
        self,
        module_specs: Mapping[str, Tuple[int, int] | AdaptedLinearSpec],
        *,
        activation_dims: Optional[Mapping[str, int]] = None,
        token_dim: int = 128,
        comm_dim: int = 128,
        update_rank: int = 1,
        max_eta: float = 1e-4,
        default_rho: float = 0.95,
    ) -> None:
        super().__init__()
        self.module_names = list(module_specs.keys())
        self.token_dim = token_dim
        self.projectors = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for name, spec in module_specs.items():
            if isinstance(spec, AdaptedLinearSpec):
                in_features, out_features = spec.in_features, spec.out_features
            else:
                out_features, in_features = spec
            act_dim = activation_dims[name] if activation_dims is not None else out_features
            safe_name = self._safe_name(name)
            self.projectors[safe_name] = LayerTokenProjector(act_dim, token_dim)
            self.heads[safe_name] = LowRankUpdateHead(
                token_dim=token_dim,
                out_features=out_features,
                in_features=in_features,
                update_rank=update_rank,
                max_eta=max_eta,
                default_rho=default_rho,
            )
        self.communication = LayerCommunicationModule(token_dim, comm_dim)

    def forward(
        self,
        activations: Mapping[str, Tensor],
        fast_state: Optional[FastWeightState] = None,
        perturbations: Optional[EggrollPerturbations] = None,
        sigma: float = 0.0,
        config=None,
    ) -> Dict[str, FastWeightUpdate]:
        del fast_state, config
        token_stack = self.project_layer_tokens(activations, perturbations, sigma)
        token_stack = self.communication(token_stack, perturbations, sigma, "communication")
        updates = {}
        for idx, name in enumerate(self.module_names):
            safe_name = self._safe_name(name)
            updates[name] = self.heads[safe_name](
                token_stack[:, :, idx],
                perturbations,
                sigma,
                f"heads.{safe_name}",
            )
        return updates

    def project_layer_tokens(
        self,
        activations: Mapping[str, Tensor],
        perturbations: Optional[EggrollPerturbations] = None,
        sigma: float = 0.0,
    ) -> Tensor:
        tokens = []
        for name in self.module_names:
            if name not in activations:
                raise KeyError(f"Missing activation for target module {name!r}")
            safe_name = self._safe_name(name)
            tokens.append(
                self.projectors[safe_name](
                    activations[name],
                    None if perturbations is None else perturbations.get(f"projectors.{safe_name}.proj"),
                    sigma,
                )
            )
        return torch.stack(tokens, dim=2)

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace(".", "__")


def infer_with_plasticity(
    *,
    asr_model: nn.Module,
    learned_updater: PlasticityPolicy,
    recording_audio: Tensor,
    tokenizer,
    config,
    module_specs: Mapping[str, Tuple[int, int] | AdaptedLinearSpec],
):
    from l2augment.rollout.gpu_plasticity import decode_output, segment_recording

    chunks, lengths = segment_recording(
        recording_audio,
        chunk_size=config.rollout.chunk_size_frames,
        overlap=config.rollout.get("chunk_overlap_frames", 0),
    )
    fast_state = init_fast_state(
        batch_size=1,
        num_candidates=1,
        module_specs=module_specs,
        device=recording_audio.device,
        dtype=recording_audio.dtype,
    )
    transcript_parts = []
    for t in range(chunks.shape[0]):
        audio = chunks[t : t + 1, None]
        lens = lengths[t : t + 1, None]
        output, activations = asr_forward_with_fast_state(
            asr_model=asr_model,
            audio=audio,
            lengths=lens,
            fast_state=fast_state,
            batch_size=1,
            num_candidates=1,
            return_selected_activations=True,
        )
        updates = learned_updater(activations, fast_state=fast_state, perturbations=None)
        fast_state = apply_fast_updates(
            fast_state,
            updates,
            max_fast_norm_ratio=config.plasticity.max_fast_norm_ratio,
        )
        transcript_parts.append(decode_output(output, tokenizer, 1, 1)[0][0])
    return " ".join(part for part in transcript_parts if part)
