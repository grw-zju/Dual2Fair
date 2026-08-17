import inspect

import run
from models.dual2fair.hierarchical_opt import HierarchicalAlternatingOptimizer


def test_config_default_yaml_is_formal_default_and_activates_complete_strategy():
    config = run.load_config()
    assert config['model']['embedding_dim'] == 64
    assert config['dual2fair']['optimization_strategy'] == 'hierarchical_mirror'
    assert config['dual2fair']['mirror_interval'] == 3
    assert not hasattr(run, 'DEFAULT_CONFIG')
    source = inspect.getsource(HierarchicalAlternatingOptimizer.refresh_correction)
    assert 'hierarchical_mirror' in source
