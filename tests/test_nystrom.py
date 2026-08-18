import torch

from models.dual2fair.sinkhorn import (
    build_adaptive_nystrom_state, compute_dense_barycenter,
    compute_lowrank_barycenter, lowrank_matvec, solve_item_sinkhorn_dense,
    solve_item_sinkhorn_lowrank)


def explicit_kernel(state):
    return state.F @ state.A_pinv @ state.F.T + state.gamma * torch.ones(
        len(state.F), len(state.F), dtype=state.F.dtype)


def test_nystrom_certificate_and_implicit_products():
    items = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    frequencies = torch.arange(1, 9, dtype=torch.float32)
    state = build_adaptive_nystrom_state(
        items, frequencies, epsilon_v=0.5, initial_rank=4,
        max_rank=8, tolerance=2.0, seed=3)
    kernel = explicit_kernel(state)
    vector = torch.randn(8)
    assert torch.allclose(lowrank_matvec(state, vector), kernel @ vector, atol=1e-5)
    diagonal = (state.F @ state.A_pinv * state.F).sum(1)
    error = max(0.0, 1.0 - diagonal.min().item())
    k_min = torch.exp(torch.tensor(-4.0)).item()
    expected_gamma = max(0.0, error - k_min / 2.0)
    assert abs(state.gamma.item() - expected_gamma) < 1e-5


def test_lowrank_sinkhorn_and_barycenter_match_explicit_state():
    items = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
    frequencies = torch.arange(1, 7, dtype=torch.float32)
    state = build_adaptive_nystrom_state(
        items, frequencies, epsilon_v=0.7, initial_rank=6,
        max_rank=6, tolerance=2.0, seed=2)
    source = torch.full((6,), 1 / 6.)
    target = torch.tensor([.1, .1, .2, .2, .2, .2])
    solve_item_sinkhorn_lowrank(state, source, target, max_iter=1000,
                                tolerance=1e-5)
    kernel = explicit_kernel(state)
    coupling = state.a[:, None] * kernel * state.b[None, :]
    assert torch.allclose(coupling.sum(1), source, atol=2e-3)
    assert torch.allclose(coupling.sum(0), target, atol=2e-3)
    expected = coupling @ items / coupling.sum(1, keepdim=True)
    actual = compute_lowrank_barycenter(state, items)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_dense_sinkhorn_reference_path_marginals_and_barycenter():
    items = torch.nn.functional.normalize(torch.randn(5, 3), dim=1)
    source = torch.full((5,), 1 / 5.)
    target = torch.tensor([.1, .2, .2, .2, .3])
    state = solve_item_sinkhorn_dense(items, source, target, epsilon_v=0.6,
                                      max_iter=1000, tolerance=1e-6)
    assert torch.allclose(state.plan.sum(1), source, atol=2e-3)
    assert torch.allclose(state.plan.sum(0), target, atol=2e-3)
    expected = state.plan @ items / state.plan.sum(1, keepdim=True)
    actual = compute_dense_barycenter(state, items)
    assert torch.allclose(actual, expected, atol=1e-6)
