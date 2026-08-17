import argparse
import json
import os
import resource
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_utils import load_dataset
from models.dual2fair import Dual2Fair
from run import init_backbone, load_config, sample_negatives, train_epoch_bpr


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reset_memory(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    if device.type == 'cuda':
        return int(torch.cuda.max_memory_allocated(device))
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def timed(device, fn):
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return time.perf_counter() - start, value


def dual2fair_epoch(model, dataset, optimizer, device, batch_size, n_neg):
    users, positive, negative = sample_negatives(dataset, n_neg)
    for start in range(0, len(users), batch_size):
        end = min(start + batch_size, len(users))
        u = users[start:end].to(device)
        p = positive[start:end].to(device)
        n = negative[start:end].to(device)
        optimizer.zero_grad()
        model.build_calibrated_embeddings()
        loss = model.bpr_loss(u, p, n)
        loss.backward()
        optimizer.step()
        model.clear_scoring_state()
    model.refresh_calibration_state()


def inference_latency(model, dataset, device, method, warmup_repeats, measure_repeats):
    users_to_score = list(range(min(dataset.n_users, max(1, measure_repeats))))
    items = torch.arange(dataset.n_items, device=device)

    def score_user(user_id):
        user = torch.full_like(items, int(user_id))
        with torch.no_grad():
            return model(user, items)

    for user_id in users_to_score[:max(1, warmup_repeats)]:
        score_user(user_id)
    sync(device)
    start = time.perf_counter()
    for repeat in range(measure_repeats):
        score_user(users_to_score[repeat % len(users_to_score)])
    sync(device)
    return (time.perf_counter() - start) / max(1, measure_repeats)


def benchmark(dataset_name='demo', backbone_name='lightgcn', config_path='',
              method='dual2fair_lowrank', gpu=-1, data_dir=None,
              warmup_repeats=3, measure_repeats=10, checkpoint=None):
    config = load_config(config_path)
    if method == 'dual2fair_dense':
        config['dual2fair']['item_solver_mode'] = 'dense'
    elif method == 'dual2fair_lowrank':
        config['dual2fair']['item_solver_mode'] = 'lowrank'
    device = torch.device('cpu' if gpu < 0 or not torch.cuda.is_available() else f'cuda:{gpu}')
    dataset_config = config['dataset'][dataset_name]
    dataset = load_dataset(dataset_name, data_dir=data_dir,
                           min_ui=dataset_config['min_user_interactions'],
                           min_ii=dataset_config['min_item_interactions'])
    backbone = init_backbone(backbone_name, dataset, config, device)
    model_config = config['model']
    batch_size = model_config.get('batch_size', 4096)
    n_neg = model_config.get('n_neg', 1)
    reset_memory(device)

    if method == 'standard':
        optimizer = torch.optim.Adam(backbone.parameters(),
                                     lr=model_config.get('learning_rate', 1e-3),
                                     weight_decay=model_config.get('weight_decay', 0.0))
        epoch_time, _ = timed(device, lambda: train_epoch_bpr(
            backbone, dataset, optimizer, device, batch_size, n_neg, scheduler=None,
            clip_norm=model_config.get('gradient_clip_val', 1.0)))
        refresh_time = 0.0
        serving_model = backbone
        if checkpoint:
            backbone.load_state_dict(torch.load(checkpoint, map_location=device))
        if hasattr(backbone, '_update_cache'):
            backbone._update_cache()
    else:
        model = Dual2Fair(backbone, dataset, config, device, backbone_name).to(device)
        if checkpoint:
            model.load_checkpoint_state(torch.load(checkpoint, map_location=device))
        optimizer = torch.optim.Adam(model.accuracy_parameters(),
                                     lr=model_config.get('learning_rate', 1e-3),
                                     weight_decay=model_config.get('weight_decay', 0.0))
        refresh_time, _ = timed(device, model.refresh_calibration_state)
        epoch_time, _ = timed(device, lambda: dual2fair_epoch(
            model, dataset, optimizer, device, batch_size, n_neg))
        model.build_calibrated_embeddings()
        serving_model = model

    memory = peak_memory(device)
    latency = inference_latency(serving_model, dataset, device, method,
                                warmup_repeats, measure_repeats)
    result = {
        'dataset': dataset_name,
        'backbone': backbone_name,
        'method': method,
        'time_per_epoch_seconds': epoch_time,
        'calibration_refresh_seconds': refresh_time,
        'peak_memory_bytes': memory,
        'inference_seconds_per_user': latency,
        'warmup_repeats': warmup_repeats,
        'measure_repeats': measure_repeats,
    }
    if dataset_name == 'demo':
        result['note'] = 'Demo/smoke-test mode; this output does not reproduce Tables IV-V.'
    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark training, refresh, memory, and serving latency.')
    parser.add_argument('--dataset', default='demo')
    parser.add_argument('--backbone', default='lightgcn')
    parser.add_argument('--config', default='')
    parser.add_argument('--method', default='dual2fair_lowrank',
                        choices=['standard', 'dual2fair_lowrank', 'dual2fair_dense'])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--warmup-repeats', type=int, default=3)
    parser.add_argument('--measure-repeats', type=int, default=10)
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()
    if args.dataset == 'demo':
        print('Demo/smoke-test mode; this output does not reproduce Tables IV-V.')
    result = benchmark(args.dataset, args.backbone, args.config, args.method,
                       args.gpu, args.data_dir, args.warmup_repeats,
                       args.measure_repeats, args.checkpoint)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
