import torch

from models.dual2fair.alignment import linear_mmd_loss, linear_mmd_state, weighted_mean


def test_weighted_linear_mmd_matches_first_moment_formula():
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
    y = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    p = torch.tensor([0.25, 0.75])
    q = torch.tensor([0.6, 0.4])
    mu_p = weighted_mean(x, p)
    mu_q = weighted_mean(y, q)
    loss = linear_mmd_loss(x, p, y, q)
    assert torch.allclose(mu_p, p[:, None].mul(x).sum(0))
    assert torch.allclose(mu_q, q[:, None].mul(y).sum(0))
    assert torch.allclose(loss, (mu_p - mu_q).pow(2).sum())


def test_linear_mmd_mean_shift_exactly_matches_target_mean():
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
    y = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    p = torch.tensor([0.25, 0.75])
    q = torch.tensor([0.6, 0.4])
    state = linear_mmd_state(x, p, y, q)
    z = x + state.delta
    shifted_mean = weighted_mean(z, p)
    assert torch.allclose(shifted_mean, state.target_mean)


def test_item_mmd_nonzero_when_same_support_has_different_weights():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source = torch.tensor([0.9, 0.1])
    target = torch.tensor([0.1, 0.9])
    loss = linear_mmd_loss(x, source, x, target)
    assert loss > 0


def test_linear_mmd_has_no_correspondence_state():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    p = torch.tensor([0.5, 0.5])
    state = linear_mmd_state(x, p, x, p)
    assert hasattr(state, 'delta')
    assert not hasattr(state, 'indices')
    assert not hasattr(state, 'transport')
    assert torch.allclose(state.delta, torch.zeros_like(state.delta))
