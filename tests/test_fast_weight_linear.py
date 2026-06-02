import torch
from torch import nn

from l2augment.modelling.plasticity import (
    FastWeightFactors,
    FastWeightLinear,
    FastWeightState,
    fast_weight_linear,
)


def test_fast_weight_linear_matches_explicit_dense_weight():
    torch.manual_seed(0)
    B, N, T, D_in, D_out, R = 2, 3, 4, 5, 7, 2
    x = torch.randn(B, N, T, D_in)
    weight = torch.randn(D_out, D_in)
    bias = torch.randn(D_out)
    A = torch.randn(B, N, D_out, R)
    Bf = torch.randn(B, N, D_in, R)

    implicit = fast_weight_linear(x, weight, bias, A, Bf)
    dense_delta = torch.einsum("bnor,bnir->bnoi", A, Bf)
    dense_weight = weight[None, None] + dense_delta
    explicit = torch.einsum("bntd,bnod->bnto", x, dense_weight) + bias

    torch.testing.assert_close(implicit, explicit, rtol=1e-5, atol=1e-6)


def test_fast_weight_candidates_are_isolated():
    torch.manual_seed(1)
    B, N, T, D_in, D_out, R = 1, 2, 3, 4, 5, 1
    base_linear = nn.Linear(D_in, D_out)
    wrapped = FastWeightLinear(base_linear, "adapt")
    x = torch.randn(B * N, T, D_in)
    A = torch.zeros(B, N, D_out, R)
    Bf = torch.zeros(B, N, D_in, R)
    A[:, 0] = torch.randn(B, D_out, R)
    Bf[:, 0] = torch.randn(B, D_in, R)
    fast_state = FastWeightState({"adapt": FastWeightFactors(A=A, B=Bf)})

    out = wrapped(x, fast_state=fast_state, batch_size=B, num_candidates=N)
    out_bn = out.reshape(B, N, T, D_out)
    base = base_linear(x).reshape(B, N, T, D_out)

    torch.testing.assert_close(out_bn[:, 1], base[:, 1], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(out_bn[:, 0], base[:, 0])
