#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy import stats


METADATA_KEYS = ('dataset', 'backbone', 'data_split_seed', 'split_hash', 'eval_mode')


def expand_inputs(values):
    paths = []
    for value in values:
        if os.path.isdir(value):
            paths.extend(sorted(glob.glob(os.path.join(value, '*.json'))))
        else:
            matches = sorted(glob.glob(value))
            paths.extend(matches if matches else [value])
    return paths


def infer_split(record):
    if record.get('split'):
        return str(record['split'])
    protocol = str(record.get('evaluation_protocol', ''))
    for split in ('validation', 'val', 'test'):
        if protocol.endswith('_' + split) or ('_' + split + '_') in protocol:
            return 'val' if split == 'validation' else split
    return record.get('selection_split')


def load_runs(inputs, metric):
    runs = OrderedDict()
    for path in expand_inputs(inputs):
        with open(path, 'r') as handle:
            record = json.load(handle)
        if metric not in record:
            raise ValueError(f'{path} is missing metric {metric}')
        if 'model_seed' not in record:
            raise ValueError(f'{path} is missing model_seed')
        seed = int(record['model_seed'])
        if seed in runs:
            raise ValueError(f'Duplicate seed {seed} in inputs')
        value = float(record[metric])
        if not math.isfinite(value):
            raise ValueError(f'{path} has non-finite {metric}')
        runs[seed] = {'path': path, 'record': record, 'value': value,
                      'split': infer_split(record)}
    if len(runs) < 2:
        raise ValueError('At least two matched seeds are required for a paired t-test')
    return runs


def check_metadata(left, right, metric):
    if set(left) != set(right):
        raise ValueError('Seed mismatch: left=' + repr(sorted(left))
                         + ', right=' + repr(sorted(right)))
    for seed in left:
        a = left[seed]['record']
        b = right[seed]['record']
        for key in METADATA_KEYS:
            if a.get(key) != b.get(key):
                raise ValueError(f'Seed {seed} metadata mismatch for {key}: {a.get(key)} vs {b.get(key)}')
        if left[seed]['split'] != right[seed]['split']:
            raise ValueError(f'Seed {seed} split mismatch: {left[seed]["split"]} vs {right[seed]["split"]}')
    split_values = {left[seed]['split'] for seed in left}
    if len(split_values) != 1:
        raise ValueError('Runs contain multiple splits: ' + repr(sorted(split_values)))
    if not split_values.pop():
        raise ValueError('Unable to infer split from result files')
    return metric


def paired_test(left, right, metric):
    check_metadata(left, right, metric)
    seeds = sorted(left)
    a = np.asarray([left[seed]['value'] for seed in seeds], dtype=float)
    b = np.asarray([right[seed]['value'] for seed in seeds], dtype=float)
    diff = a - b
    if np.allclose(diff, 0.0):
        statistic, p_value = 0.0, 1.0
    elif np.isclose(np.std(diff, ddof=1), 0.0):
        statistic = math.copysign(math.inf, float(np.mean(diff)))
        p_value = 0.0
    else:
        result = stats.ttest_rel(a, b, alternative='two-sided')
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    first = left[seeds[0]]['record']
    return {
        'dataset': first.get('dataset'),
        'backbone': first.get('backbone'),
        'split': left[seeds[0]]['split'],
        'metric': metric,
        'method_a': first.get('method'),
        'method_b': right[seeds[0]]['record'].get('method'),
        'seeds': ' '.join(str(seed) for seed in seeds),
        'n': len(seeds),
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
        'mean_diff_a_minus_b': float(np.mean(diff)),
        't_statistic': statistic,
        'p_value_two_sided': p_value,
        'p_lt_0_05': bool(p_value < 0.05),
    }


def write_rows(rows, output):
    fieldnames = list(rows[0].keys())
    handle = open(output, 'w', newline='') if output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description='Two-sided paired t-test over matched-seed result JSON files.')
    parser.add_argument('--method-a', nargs='+', required=True,
                        help='Directory, glob, or JSON files for method A')
    parser.add_argument('--method-b', nargs='+', required=True,
                        help='Directory, glob, or JSON files for method B')
    parser.add_argument('--metric', action='append', required=True,
                        help='Metric key to test; pass multiple times for multiple metrics')
    parser.add_argument('--output', help='Optional CSV output path')
    args = parser.parse_args()

    rows = []
    for metric in args.metric:
        left = load_runs(args.method_a, metric)
        right = load_runs(args.method_b, metric)
        rows.append(paired_test(left, right, metric))
    write_rows(rows, args.output)


if __name__ == '__main__':
    main()
