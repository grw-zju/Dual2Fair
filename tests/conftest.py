import copy

import pytest
import torch

from data.dataset_utils import load_dataset
from run import init_backbone, load_config


@pytest.fixture
def dataset():
    return load_dataset('demo', min_ui=3, min_ii=1,
                        data_split_seed=2026, negative_sampling_seed=42)


@pytest.fixture
def config():
    settings = copy.deepcopy(load_config('config/default.yaml'))
    settings['model'].update({'embedding_dim': 8, 'max_epochs': 1,
                              'early_stop_patience': 1, 'batch_size': 256})
    settings['dual2fair'].update({'n_clusters': 2, 'item_anchor_count': 4,
                                  'sinkhorn_max_iter': 30})
    return settings


@pytest.fixture
def backbone_factory(dataset, config):
    def factory(name):
        model = init_backbone(name, dataset, config, torch.device('cpu'))
        model._backbone_name_hint = name
        return model
    return factory
