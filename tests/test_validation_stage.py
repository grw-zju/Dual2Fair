import torch

from data.dataset_utils import load_dataset
from run import get_device, init_backbone, load_config, train_dual2fair


def test_validation_only_stage_returns_validation_metrics_without_test_evaluation(tmp_path):
    config = load_config()
    config['model']['embedding_dim'] = 8
    config['model']['max_epochs'] = 1
    config['model']['early_stop_patience'] = 1
    config['model']['batch_size'] = 8
    config['evaluation']['require_uif_reference'] = False
    config['dual2fair'].update({
        'gmm_clusters': 2,
        'training_candidate_size': 6,
        'nystrom_initial_rank': 4,
        'nystrom_max_rank': 8,
        'nystrom_tol': 2.0,
        'sinkhorn_max_iter': 20,
    })
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    model, _, _, _, results = train_dual2fair(
        dataset, 'lightgcn', config, torch.device('cpu'),
        eval_mode='full', loss_type='bpr', save_path=None,
        evaluation_stage='validation')
    assert results['evaluation_stage'] == 'validation'
    assert results['evaluation_protocol'] == 'full_warm_start_val'
    assert model._best_validation_results['evaluation_protocol'] == 'full_warm_start_val'
