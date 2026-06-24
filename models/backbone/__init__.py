from .base_backbone import BaseBackbone
from .neumf import NeuMF
from .vaecf import VAECF
from .lightgcn import LightGCN


BACKBONES = {
    'neumf': NeuMF,
    'vaecf': VAECF,
    'lightgcn': LightGCN,
}


def get_backbone(name, n_users, n_items, **kwargs):
    return BACKBONES[name](n_users, n_items, **kwargs)
