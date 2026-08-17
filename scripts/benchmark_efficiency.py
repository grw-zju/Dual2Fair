import argparse
import json
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_utils import load_dataset
from run import init_backbone, load_config
from models.dual2fair import Dual2Fair


def benchmark(dataset_name='demo', backbone_name='lightgcn', config_path='',
              method='dual2fair_lowrank', gpu=-1, data_dir=None):
    config = load_config(config_path)
    device = torch.device('cpu' if gpu < 0 or not torch.cuda.is_available() else f'cuda:{gpu}')
    dataset_config = config['dataset'][dataset_name]
    dataset = load_dataset(dataset_name,
                           data_dir=data_dir,
                           min_ui=dataset_config['min_user_interactions'],
                           min_ii=dataset_config['min_item_interactions'])
    backbone = init_backbone(backbone_name, dataset, config, device)
    model = Dual2Fair(backbone, dataset, config, device, backbone_name).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if method == 'standard':
        refresh_time = 0.0
    elif method == 'dual2fair_lowrank':
        refresh_started = time.perf_counter()
        model.refresh_calibration_state()
        refresh_time = time.perf_counter() - refresh_started
    elif method == 'dual2fair_dense':
        raise NotImplementedError('Dense all-item benchmark is only permitted for explicit small-data studies')
    else:
        raise ValueError(method)
    total = time.perf_counter() - started
    peak = (torch.cuda.max_memory_allocated(device) if device.type == 'cuda'
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    inference_started = time.perf_counter()
    with torch.no_grad():
        user = torch.tensor([0], device=device)
        items = torch.arange(dataset.n_items, device=device)
        users = user.expand_as(items)
        (model(users, items) if method != 'standard' else backbone(users, items))
    inference = time.perf_counter() - inference_started
    result = {'dataset': dataset_name, 'backbone': backbone_name, 'method': method,
              'time_per_epoch_seconds': total, 'calibration_refresh_seconds': refresh_time,
              'peak_memory_bytes': int(peak), 'inference_seconds_per_user': inference}
    if dataset_name == 'demo':
        result['note'] = 'Demo/smoke-test mode; this output does not reproduce Tables IV-V.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='demo')
    parser.add_argument('--backbone', default='lightgcn')
    parser.add_argument('--config', default='')
    parser.add_argument('--method', default='dual2fair_lowrank',
                        choices=['standard', 'dual2fair_lowrank', 'dual2fair_dense'])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args()
    if args.dataset == 'demo':
        print('Demo/smoke-test mode; this output does not reproduce Tables IV-V.')
    print(json.dumps(benchmark(args.dataset, args.backbone, args.config,
                               args.method, args.gpu, args.data_dir), indent=2))
