#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import subprocess
import sys

SEEDS = [42, 43, 44, 45, 46]
LAMBDA_GRID = [0.01, 0.05, 0.1, 0.5, 1, 10]


def run_command(command, dry_run):
    print(' '.join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description='Orchestrate the five-seed validation-selection-test workflow.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--results-root', default='results/grid')
    parser.add_argument('--uif-reference-file')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not args.uif_reference_file:
        raise SystemExit('Provide --uif-reference-file built from five Standard runs before grid search.')

    os.makedirs(args.results_root, exist_ok=True)
    plan = []
    for lambda1, lambda2 in itertools.product(LAMBDA_GRID, LAMBDA_GRID):
        for seed in SEEDS:
            output_suffix = f'_l1_{lambda1:g}_l2_{lambda2:g}_seed_{seed}'
            command = [
                sys.executable, 'run.py', '--dataset', args.dataset,
                '--backbone', args.backbone, '--method', 'dual2fair',
                '--lambda1', str(lambda1), '--lambda2', str(lambda2),
                '--seed', str(seed), '--config', args.config,
                '--uif-reference-file', args.uif_reference_file,
                '--evaluation-stage', 'validation',
                '--results_dir', args.results_root,
                '--output-suffix', output_suffix,
            ]
            plan.append(command)
            run_command(command, args.dry_run)
    with open(os.path.join(args.results_root, 'grid_plan.json'), 'w') as handle:
        json.dump({'dataset': args.dataset, 'backbone': args.backbone,
                   'seeds': SEEDS, 'lambda_grid': LAMBDA_GRID,
                   'commands': plan}, handle, indent=2)


if __name__ == '__main__':
    main()
