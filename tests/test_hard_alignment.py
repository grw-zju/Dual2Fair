import torch
import torch.nn.functional as F

from models.dual2fair.alignment import hard_item_match_blockwise, hard_user_match


def test_user_hard_matching_uses_geometry_and_target_prior():
    x = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q = torch.tensor([0.1, 0.9])
    epsilon = 0.2
    assignments = hard_user_match(x, prototypes, q, epsilon, 1e-8)
    expected_scores = F.normalize(x, dim=1) @ F.normalize(prototypes, dim=1).T
    expected_scores = expected_scores + epsilon * torch.log(q.clamp_min(1e-8))[None, :]
    assert torch.equal(assignments, torch.argmax(expected_scores, dim=1))
    z = prototypes[assignments]
    assert torch.allclose(z, prototypes[assignments])


def test_hard_matching_has_no_global_column_capacity_constraint():
    x = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q = torch.tensor([0.5, 0.5])
    assignments = hard_user_match(x, prototypes, q, epsilon=0.05, eps0=1e-8)
    assert torch.equal(assignments, torch.zeros(3, dtype=torch.long))


def test_item_hard_matching_uses_target_prior_and_can_override_close_geometry():
    x = torch.tensor([[1.0, 0.0]])
    y = F.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.1]]), dim=1)
    nu = torch.tensor([0.01, 0.99])
    assignments = hard_item_match_blockwise(x, y, nu, epsilon=0.1,
                                            source_chunk_size=1,
                                            target_chunk_size=1)
    dense_scores = x @ y.T + 0.1 * torch.log(nu.clamp_min(1e-8))[None, :]
    assert torch.equal(assignments, torch.argmax(dense_scores, dim=1))
    assert assignments.item() == 1


def test_blockwise_hard_item_matching_matches_dense_exactly():
    generator = torch.Generator().manual_seed(7)
    x = F.normalize(torch.randn(5, 4, generator=generator), dim=1)
    y = F.normalize(torch.randn(6, 4, generator=generator), dim=1)
    nu = torch.tensor([0.4, 0.1, 0.2, 0.05, 0.15, 0.1])
    blockwise = hard_item_match_blockwise(x, y, nu, epsilon=0.2,
                                          source_chunk_size=2,
                                          target_chunk_size=3)
    dense = torch.argmax(x @ y.T + 0.2 * torch.log(nu.clamp_min(1e-8))[None, :], dim=1)
    assert torch.equal(blockwise, dense)
