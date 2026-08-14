import copy

import pytest
import torch
import yaml

from data.dataset_utils import load_dataset
from evaluation.evaluator import Evaluator
from models.dual2fair import Dual2Fair
from models.dual2fair.hierarchical_opt import HierarchicalAlternatingOptimizer
from run import init_backbone, sample_negatives


def tiny_config(mode):
    with open('config/default.yaml') as handle:
        settings = yaml.safe_load(handle)
    settings['model'].update({'embedding_dim': 8, 'batch_size': 8})
    settings['dual2fair'].update({
        'alignment_mode': mode,
        'gmm_clusters': 2,
        'training_candidate_size': 6,
        'nystrom_initial_rank': 4,
        'nystrom_max_rank': 8,
        'nystrom_tol': 2.0,
        'sinkhorn_max_iter': 20,
        'mirror_interval': 1,
    })
    settings['evaluation']['require_uif_reference'] = False
    return settings


@pytest.mark.parametrize('mode', ['ot', 'hard', 'mmd'])
def test_alignment_modes_complete_lightgcn_smoke(mode):
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    settings = tiny_config(mode)
    backbone = init_backbone('lightgcn', dataset, settings, torch.device('cpu'))
    model = Dual2Fair(backbone, dataset, settings, 'cpu', 'lightgcn')
    optimizer = HierarchicalAlternatingOptimizer(
        model,
        accuracy_learning_rate=settings['model']['learning_rate'],
        fairness_learning_rate=settings['dual2fair']['fairness_learning_rate'],
        mirror_interval=settings['dual2fair']['mirror_interval'])
    model.refresh_calibration_state()
    users, positive, negative = sample_negatives(dataset, 1)
    users, positive, negative = users[:8], positive[:8], negative[:8]

    def accuracy_loss():
        model.build_calibrated_embeddings()
        return model.bpr_loss(users, positive, negative)

    optimizer.step_iteration(accuracy_loss)
    model.refresh_calibration_state()
    optimizer.refresh_correction(0.1, 0.1)
    model.build_calibrated_embeddings()
    results = Evaluator(dataset, split='val', require_uif_reference=False).evaluate(model=model)
    assert results['evaluation_protocol'] == 'full_warm_start_val'
    assert torch.isfinite(model.compute_user_fixed_coupling_loss())
    assert torch.isfinite(model.compute_item_fixed_coupling_loss())
    assert model.calibration_state.alignment_mode == mode


@pytest.mark.parametrize('mode', ['hard', 'mmd'])
def test_fairness_update_restricts_parameters_for_alternative_alignment_modes(mode):
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    settings = tiny_config(mode)
    backbone = init_backbone('lightgcn', dataset, settings, torch.device('cpu'))
    model = Dual2Fair(backbone, dataset, settings, 'cpu', 'lightgcn')
    model.refresh_calibration_state()
    optimizer = HierarchicalAlternatingOptimizer(
        model,
        accuracy_learning_rate=settings['model']['learning_rate'],
        fairness_learning_rate=settings['dual2fair']['fairness_learning_rate'],
        mirror_interval=99)
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    optimizer.refresh_correction(0.1, 0.1, strategy='standard_alternating')
    allowed_prefixes = (
        'user_calibration.W_u_c.',
        'user_calibration.W_u_h.',
        'item_calibration.W_v.',
    )
    for name, parameter in model.named_parameters():
        changed = not torch.allclose(before[name], parameter.detach())
        if changed:
            assert name.startswith(allowed_prefixes)
    for name in before:
        if name.startswith(('backbone.', 'user_calibration.P_u.', 'item_calibration.P_v.')):
            assert torch.allclose(before[name], dict(model.named_parameters())[name].detach())
    assert torch.allclose(before['user_calibration.rho_u_tilde'],
                          model.user_calibration.rho_u_tilde.detach())
    assert torch.allclose(before['item_calibration.rho_v_tilde'],
                          model.item_calibration.rho_v_tilde.detach())
