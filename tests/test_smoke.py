import pytest
import torch

from models.dual2fair import Dual2Fair


@pytest.mark.parametrize('backbone_name', ['lightgcn', 'neumf', 'vaecf'])
def test_standard_backbone_smoke(dataset, config, backbone_factory, backbone_name):
    backbone = backbone_factory(backbone_name)
    backbone.eval()
    scores = backbone.compute_all_scores()
    assert scores.shape == (dataset.n_users, dataset.n_items)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize('backbone_name', ['lightgcn', 'neumf', 'vaecf'])
def test_dual2fair_smoke(dataset, config, backbone_factory, backbone_name):
    model = Dual2Fair(backbone_factory(backbone_name), dataset, config,
                      'cpu', backbone_name)
    output = model.update_calibration_state()
    scores = model.compute_all_scores()
    assert scores.shape == (dataset.n_users, dataset.n_items)
    assert torch.isfinite(scores).all()
    assert output.user_ot_objective is not None
    assert output.item_ot_objective is not None
