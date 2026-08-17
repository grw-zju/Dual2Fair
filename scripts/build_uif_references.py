#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

import yaml

SEEDS = [42, 43, 44, 45, 46]


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def metric_block(record, split):
    if split in {'validation', 'val'}:
        block = record.get('validation_results') or record.get('validation') or record.get('val')
        if block is None:
            raise ValueError(f"{record.get('method')} seed {record.get('model_seed')} lacks validation metrics")
        return block
    return record.get('test_results') or record.get('test') or record


def validate_and_build(paths, expected_seeds):
    if len(paths) != 5:
        raise ValueError('Exactly five Standard result JSON files are required')
    records = [load_json(path) for path in paths]
    seeds = [int(record.get('model_seed')) for record in records]
    if sorted(seeds) != sorted(expected_seeds) or len(set(seeds)) != 5:
        raise ValueError(f'Expected five distinct seeds {expected_seeds}, got {sorted(seeds)}')
    dataset = {record.get('dataset') for record in records}
    backbone = {record.get('backbone') for record in records}
    split_hash = {record.get('split_hash') for record in records}
    method = {record.get('method') for record in records}
    if len(dataset) != 1 or len(backbone) != 1 or len(split_hash) != 1:
        raise ValueError('dataset/backbone/split_hash must match across runs')
    if method != {'standard'}:
        raise ValueError(f'UIF references must be built from Standard runs, got {method}')
    output = {
        'dataset': dataset.pop(),
        'backbone': backbone.pop(),
        'split_hash': split_hash.pop(),
        'seeds': sorted(seeds),
    }
    for split in ('validation', 'test'):
        dufs, difs = [], []
        for record in records:
            block = metric_block(record, split)
            for metric in ('DUF', 'DIF'):
                if metric not in block:
                    raise ValueError(f'Missing {metric} in {split} block for seed {record.get("model_seed")}')
                value = float(block[metric])
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite {metric} in {split} block')
                (dufs if metric == 'DUF' else difs).append(value)
        output[split] = {
            'DUF_ref': float(sum(dufs) / len(dufs)),
            'DIF_ref': float(sum(difs) / len(difs)),
        }
    return output


def write_reference(reference, path):
    if not path:
        print(json.dumps(reference, indent=2))
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        if path.endswith(('.yaml', '.yml')):
            yaml.safe_dump(reference, handle, sort_keys=False)
        else:
            json.dump(reference, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Build split-specific UIF references from five Standard runs.')
    parser.add_argument('results', nargs='+')
    parser.add_argument('--expected-seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output')
    args = parser.parse_args()
    write_reference(validate_and_build(args.results, args.expected_seeds), args.output)


if __name__ == '__main__':
    main()
