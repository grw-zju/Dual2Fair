import torch

from models.dual2fair.user_calibration import UserRepresentationCalibration


def make_module():
    module = UserRepresentationCalibration(
        representation_dim=3, gmm_clusters=2, epsilon_u=0.2,
        sinkhorn_max_iter=500, sinkhorn_tol=1e-5, random_state=7)
    module.set_user_groups([0, 1], [2, 3], {0: {0, 1}, 1: {1, 2}, 2: {0}, 3: {2}})
    return module


def test_global_user_ot_and_residual_formula():
    module = make_module()
    users = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                          [0.5, 0.5, 0.], [0., 0.5, 0.5]])
    items = torch.eye(3, requires_grad=True)
    module.refresh(users, items)
    plan = module.user_transport
    source = torch.full((2,), 0.5)
    assert torch.allclose(plan.sum(1), source, atol=2e-3)
    assert torch.allclose(plan.sum(0), module.gmm_weights, atol=2e-3)
    calibrated = module.calibrate(users)
    assert torch.allclose(calibrated[:2], users[:2])
    assert 0.0 < module.rho_u.item() < 1.0
    assert not torch.allclose(calibrated[2:], users[2:])


def test_history_stop_gradient():
    module = make_module()
    users = torch.randn(4, 3, requires_grad=True)
    items = torch.randn(3, 3, requires_grad=True)
    x, _ = module.calibration_representations([2, 3], users, items)
    x.sum().backward()
    assert items.grad is None
    assert users.grad is not None
