import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_utils import load_dataset
from models.dual2fair import Dual2Fair
from run import init_backbone, load_config


def gradient_norm(loss, tensor):
    gradient = torch.autograd.grad(loss, tensor, retain_graph=True, allow_unused=True)[0]
    return 0.0 if gradient is None else float(gradient.norm())


def analyze(config_path='', strategy='Dual2Fair'):
    config = load_config(config_path)
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    backbone = init_backbone('lightgcn', dataset, config, torch.device('cpu'))
    model = Dual2Fair(backbone, dataset, config, 'cpu', 'lightgcn')
    model.refresh_calibration_state()
    users, items = model.get_native_embeddings()
    users.retain_grad(); items.retain_grad()
    user_loss = model.compute_user_fixed_coupling_loss()
    item_loss = model.compute_item_fixed_coupling_loss()
    user_on_users = gradient_norm(user_loss, users)
    user_on_items = gradient_norm(user_loss, items)
    item_on_users = gradient_norm(item_loss, users)
    item_on_items = gradient_norm(item_loss, items)
    return {
        'strategy': strategy,
        'Leak_u_to_v': user_on_items / (user_on_users + config['dual2fair']['eps0']),
        'Leak_v_to_u': item_on_users / (item_on_items + config['dual2fair']['eps0'])}


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))
