import torch

from l2augment.utils.eggroll import (
    EggrollLinear,
    EggrollPerturbations,
    Rank1Perturbation,
    eggroll_update_shared_params,
    group_normalise_rewards,
)


def test_eggroll_linear_matches_explicit_rank1_matrix_3d():
    torch.manual_seed(0)
    B, N, D_in, D_out = 3, 4, 5, 7
    sigma = 0.25
    layer = EggrollLinear(D_in, D_out, bias=True)
    x = torch.randn(B, N, D_in)
    pert = Rank1Perturbation(a=torch.randn(N, D_out), b=torch.randn(N, D_in))

    implicit = layer(x, pert, sigma)
    explicit = torch.empty_like(implicit)
    for bidx in range(B):
        for nidx in range(N):
            matrix = layer.weight + sigma * torch.outer(pert.a[nidx], pert.b[nidx])
            explicit[bidx, nidx] = x[bidx, nidx] @ matrix.T + layer.bias

    torch.testing.assert_close(implicit, explicit, rtol=1e-5, atol=1e-6)


def test_eggroll_linear_matches_explicit_rank1_matrix_4d():
    torch.manual_seed(1)
    B, N, T, D_in, D_out = 2, 3, 4, 5, 6
    sigma = 0.1
    layer = EggrollLinear(D_in, D_out, bias=False)
    x = torch.randn(B, N, T, D_in)
    pert = Rank1Perturbation(a=torch.randn(N, D_out), b=torch.randn(N, D_in))

    implicit = layer(x, pert, sigma)
    explicit = torch.empty_like(implicit)
    for bidx in range(B):
        for nidx in range(N):
            matrix = layer.weight + sigma * torch.outer(pert.a[nidx], pert.b[nidx])
            explicit[bidx, nidx] = x[bidx, nidx] @ matrix.T

    torch.testing.assert_close(implicit, explicit, rtol=1e-5, atol=1e-6)


def test_reward_clamp_uses_max_not_min():
    wer = torch.tensor([0.0, 0.5, 1.0, 1.5])
    quality = 1.0 - wer.clamp(max=1.0)
    torch.testing.assert_close(quality, torch.tensor([1.0, 0.5, 0.0, 0.0]))


def test_group_normalise_rewards_over_candidate_dimension():
    quality = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    rewards = group_normalise_rewards(quality)
    torch.testing.assert_close(rewards.mean(dim=1), torch.zeros(2), atol=1e-7, rtol=0.0)


def test_eggroll_optimizer_update_moves_in_reward_increasing_direction():
    layer = EggrollLinear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    pert = EggrollPerturbations(
        {"": Rank1Perturbation(a=torch.ones(1, 1), b=torch.ones(1, 1), antithetic=False)}
    )
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    eggroll_update_shared_params(layer, pert, rewards=torch.ones(1), sigma=1.0, optimizer=optimizer)

    assert layer.weight.item() > 0.0
