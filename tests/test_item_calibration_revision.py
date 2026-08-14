import torch

from models.dual2fair.item_calibration import ItemRepresentationCalibration


def test_relevance_target_endpoints_and_source():
    module = ItemRepresentationCalibration(4, omega=.2)
    merit = torch.tensor([1., 2., 4., 8.])
    uniform = module.relevance_target(merit, omega=1)
    merit_only = module.relevance_target(merit, omega=0)
    assert torch.allclose(uniform, torch.full((4,), .25))
    assert torch.all(merit_only >= 0)
    assert torch.allclose(merit_only.sum(), torch.tensor(1.))
    source = module.source_marginal(torch.tensor([1., 4., 9., 16.]))
    assert torch.allclose(source.sum(), torch.tensor(1.))
    assert source[0] > source[-1]


def test_item_confidence_residual_and_lowrank_state():
    torch.manual_seed(4)
    module = ItemRepresentationCalibration(
        4, epsilon_v=.8, nystrom_initial_rank=4,
        nystrom_max_rank=8, nystrom_tol=2.0)
    items = torch.randn(8, 4)
    frequencies = torch.arange(1, 9, dtype=torch.float32)
    merit = torch.arange(8, 0, -1, dtype=torch.float32)
    module.refresh(items, frequencies, merit)
    calibrated = module.calibrate(items)
    assert calibrated.shape == items.shape
    assert not torch.allclose(calibrated, items)
    assert 0 < module.rho_v.item() < 1
    assert module.transport_state.F.shape[0] == len(items)
