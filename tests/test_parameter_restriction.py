import copy

import torch
import yaml

from data.dataset_utils import load_dataset
from models.dual2fair import Dual2Fair
from run import init_backbone


def config():
    with open('config/default.yaml') as handle:
        settings = yaml.safe_load(handle)
    settings['model']['embedding_dim'] = 8
    settings['dual2fair'].update({
        'gmm_clusters': 2, 'training_candidate_size': 6,
        'nystrom_initial_rank': 4, 'nystrom_max_rank': 8,
        'nystrom_tol': 2.0, 'sinkhorn_max_iter': 30})
    return settings


def test_fairness_parameters_are_only_calibration_spaces():
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    settings = config()
    backbone = init_backbone('lightgcn', dataset, settings, torch.device('cpu'))
    model = Dual2Fair(backbone, dataset, settings, 'cpu', 'lightgcn')
    model.refresh_calibration_state()
    allowed = {id(parameter) for parameter in model.fairness_correction_parameters()}
    expected = {id(parameter) for parameter in model.user_calibration.W_u_c.parameters()}
    expected |= {id(parameter) for parameter in model.user_calibration.W_u_h.parameters()}
    expected |= {id(parameter) for parameter in model.item_calibration.W_v.parameters()}
    assert allowed == expected
    forbidden = list(backbone.parameters()) + list(model.user_calibration.P_u.parameters())
    forbidden += list(model.item_calibration.P_v.parameters())
    assert not any(id(parameter) in allowed for parameter in forbidden)


def test_checkpoint_state_restores_parameters_ema_and_calibration_state():
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    settings = config()
    backbone = init_backbone('lightgcn', dataset, settings, torch.device('cpu'))
    model = Dual2Fair(backbone, dataset, settings, 'cpu', 'lightgcn')
    model.refresh_calibration_state()
    checkpoint = model.checkpoint_state(epoch=3, iteration=7,
                                        selection_metadata={'validation': {'NDCG': .5}})
    expected_state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.123)
        for name in model.ema_state:
            model.ema_state[name].add_(1.0)
    model.calibration_state = None
    model.load_checkpoint_state(checkpoint)
    for name, value in model.state_dict().items():
        assert torch.allclose(value, expected_state[name])
    for name, value in model.ema_state.items():
        assert torch.allclose(value, checkpoint['ema_state'][name])
    assert model.calibration_state.refresh_index == checkpoint['calibration_state'].refresh_index
    assert model.user_calibration.user_transport is not None
    assert checkpoint['epoch'] == 3
    assert checkpoint['iteration'] == 7
