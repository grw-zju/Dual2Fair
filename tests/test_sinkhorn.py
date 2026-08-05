import torch

from models.dual2fair.transport import dense_log_sinkhorn
from models.dual2fair.user_calibration import UserRepresentationCalibration


def test_sinkhorn_marginals():
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    source = torch.tensor([0.4, 0.6])
    target = torch.tensor([0.5, 0.5])
    plan = dense_log_sinkhorn(cost, source, target, epsilon=0.2,
                              max_iter=500, convergence_tol=1e-7)
    assert torch.allclose(plan.sum(1), source, atol=1e-4)
    assert torch.allclose(plan.sum(0), target, atol=1e-4)


def test_row_normalized_barycentric_projection():
    plan = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    projected = UserRepresentationCalibration.barycentric_projection(plan, targets)
    expected = torch.tensor([[1 / 3, 4 / 3], [3 / 7, 8 / 7]])
    assert torch.allclose(projected, expected)


def test_projection_invariant_to_row_scaling():
    plan = torch.rand(5, 3)
    targets = torch.rand(3, 4)
    scales = torch.rand(5, 1) + 0.1
    first = UserRepresentationCalibration.barycentric_projection(plan, targets)
    second = UserRepresentationCalibration.barycentric_projection(plan * scales, targets)
    assert torch.allclose(first, second, atol=1e-6)


def test_projection_does_not_collapse_norm():
    targets = torch.ones(4, 6)
    plan = torch.full((100, 4), 1.0 / 400.0)
    projected = UserRepresentationCalibration.barycentric_projection(plan, targets)
    assert projected.norm(dim=1).mean() > 2.0
