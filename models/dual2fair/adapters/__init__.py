from .base import BackboneAdapter
from .lightgcn_adapter import LightGCNAdapter
from .neumf_adapter import NeuMFAdapter
from .vaecf_adapter import VAECFAdapter


ADAPTERS = {
    'lightgcn': LightGCNAdapter,
    'neumf': NeuMFAdapter,
    'vaecf': VAECFAdapter,
}


def get_adapter(name, backbone):
    if name not in ADAPTERS:
        raise ValueError(f'Unsupported backbone adapter: {name}')
    return ADAPTERS[name](backbone)


__all__ = ['BackboneAdapter', 'LightGCNAdapter', 'NeuMFAdapter', 'VAECFAdapter',
           'get_adapter']
