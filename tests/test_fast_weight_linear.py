import torch
from torch import nn

from l2augment.modelling.plasticity import (
    FastWeightFactors,
    FastWeightLinear,
    FastWeightState,
    FastWeightUpdate,
    apply_fast_updates,
    dense_fast_weight_update,
    fast_weight_linear,
)


def test_fast_weight_linear_matches_explicit_dense_weight():
    torch.manual_seed(0)
    B, N, T, D_in, D_out = 2, 3, 4, 5, 7
    x = torch.randn(B, N, T, D_in)
    weight = torch.randn(D_out, D_in)
    bias = torch.randn(D_out)
    F = torch.randn(B, N, D_out, D_in)

    implicit = fast_weight_linear(x, weight, bias, F)
    dense_weight = weight[None, None] + F
    explicit = torch.einsum("bntd,bnod->bnto", x, dense_weight) + bias

    torch.testing.assert_close(implicit, explicit, rtol=1e-5, atol=1e-6)


def test_dense_fast_state_update_matches_explicit_formula():
    torch.manual_seed(3)
    B, N, D_in, D_out, R = 2, 3, 4, 5, 2
    F_old = torch.randn(B, N, D_out, D_in)
    P = torch.randn(B, N, D_out, R)
    Q = torch.randn(B, N, D_in, R)
    eta = torch.randn(B, N, R) * 0.01
    rho = torch.rand(B, N, 1)
    update = FastWeightUpdate(P=P, Q=Q, eta=eta, rho=rho)
    state = FastWeightState({"adapt": FastWeightFactors(F=F_old.clone())})

    next_state = apply_fast_updates(state, {"adapt": update})

    dense_update = torch.einsum("bnor,bnir->bnoi", P * eta[..., None, :], Q)
    expected = rho[..., None] * F_old + dense_update
    torch.testing.assert_close(dense_fast_weight_update(update), dense_update)
    torch.testing.assert_close(next_state["adapt"].F, expected)


def test_fast_weight_candidates_are_isolated():
    torch.manual_seed(1)
    B, N, T, D_in, D_out = 1, 2, 3, 4, 5
    base_linear = nn.Linear(D_in, D_out)
    wrapped = FastWeightLinear(base_linear, "adapt")
    x = torch.randn(B * N, T, D_in)
    F = torch.zeros(B, N, D_out, D_in)
    F[:, 0] = torch.randn(B, D_out, D_in)
    fast_state = FastWeightState({"adapt": FastWeightFactors(F=F)})

    out = wrapped(x, fast_state=fast_state, batch_size=B, num_candidates=N)
    out_bn = out.reshape(B, N, T, D_out)
    base = base_linear(x).reshape(B, N, T, D_out)

    torch.testing.assert_close(out_bn[:, 1], base[:, 1], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(out_bn[:, 0], base[:, 0])


def test_dense_fast_state_norm_clipping_enforces_ratio():
    torch.manual_seed(4)
    B, N, D_in, D_out, R = 2, 2, 3, 4, 1
    P = torch.randn(B, N, D_out, R)
    Q = torch.randn(B, N, D_in, R)
    update = FastWeightUpdate(
        P=P,
        Q=Q,
        eta=torch.ones(B, N, R),
        rho=torch.zeros(B, N, 1),
    )
    state = FastWeightState(
        {
            "adapt": FastWeightFactors(
                F=torch.zeros(B, N, D_out, D_in),
                base_weight_norm=torch.tensor(2.0),
            )
        }
    )

    next_state, metrics = apply_fast_updates(
        state,
        {"adapt": update},
        max_fast_norm_ratio=0.05,
        return_metrics=True,
    )

    ratios = next_state["adapt"].F.float().norm(dim=(-2, -1)) / 2.0
    assert ratios.max() <= 0.050001
    assert metrics["fast_weight_clipped_fraction"].item() > 0.0
