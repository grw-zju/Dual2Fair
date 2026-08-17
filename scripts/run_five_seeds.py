import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np


SEEDS = [42, 43, 44, 45, 46]


def aggregate_runs(runs):
    if len(runs) != 5 or len({run['model_seed'] for run in runs}) != 5:
        raise ValueError('Aggregation requires five distinct model seeds')
    invariants = ('split_hash', 'dataset', 'backbone', 'method', 'eval_mode')
    for key in invariants:
        if len({str(run.get(key)) for run in runs}) != 1:
            raise ValueError(f'Run metadata mismatch: {key}')
    metrics = ('NDCG', 'Hit', 'DUF', 'DIF', 'UIF')
    aggregate = {'runs': runs, 'seeds': [run['model_seed'] for run in runs],
                 'run_level_metrics': {metric: [] for metric in metrics}}
    for metric in metrics:
        values = np.asarray([run[metric] for run in runs], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Run metric is missing or non-finite: {metric}')
        aggregate['run_level_metrics'][metric] = [float(value) for value in values]
        # UIF must be averaged across already-computed run-level UIF values.
        # Do not recompute UIF by plugging mean NDCG/DUF/DIF into the UIF formula.
        aggregate[metric] = {
            'mean': float(values.mean()),
            'std': float(values.std(ddof=1)),
            'ci95': float(1.96 * values.std(ddof=1) / math.sqrt(5))}
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--config', default='')
    parser.add_argument('--uif-reference-file')
    parser.add_argument('--evaluation-stage', default='both', choices=['validation', 'test', 'both'])
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output-root', default='results')
    args = parser.parse_args()
    directory = os.path.join(args.output_root, args.dataset, args.backbone, args.method)
    os.makedirs(directory, exist_ok=True)
    runs = []
    for seed in SEEDS:
        suffix = f'_seed_{seed}'
        command = [sys.executable, 'run.py', '--dataset', args.dataset,
                   '--backbone', args.backbone, '--method', args.method,
                   '--config', args.config, '--seed', str(seed),
                   '--split-seed', str(args.split_seed), '--eval_mode', 'full',
                   '--evaluation-stage', args.evaluation_stage,
                   '--gpu', str(args.gpu), '--results_dir', directory,
                   '--save_dir', os.path.join(directory, 'checkpoints'),
                   '--output-suffix', suffix]
        if args.uif_reference_file:
            command.extend(['--uif-reference-file', args.uif_reference_file])
        subprocess.run(command, check=True)
        path = os.path.join(directory,
                            f'{args.dataset}_{args.backbone}_{args.method}{suffix}.json')
        with open(path) as handle:
            run = json.load(handle)
        destination = os.path.join(directory, f'seed_{seed}.json')
        with open(destination, 'w') as handle:
            json.dump(run, handle, indent=2)
        runs.append(run)
    aggregate = aggregate_runs(runs)
    with open(os.path.join(directory, 'aggregate.json'), 'w') as handle:
        json.dump(aggregate, handle, indent=2)


if __name__ == '__main__':
    main()
