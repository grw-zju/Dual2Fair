#!/usr/bin/env python3
import argparse
import json
import math
import os
import re

import yaml


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def metric_mean(aggregate, metric):
    value = aggregate.get(metric)
    if isinstance(value, dict) and 'mean' in value:
        result = float(value['mean'])
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite aggregate metric {metric}')
    return result


def infer_lambda(aggregate, path, key):
    hp = aggregate.get('selected_hyperparameters') or aggregate.get('hyperparameters') or {}
    if key in hp:
        return float(hp[key])
    match = re.search(key + r'[_=]([0-9.]+)', os.path.basename(path))
    if match:
        return float(match.group(1))
    raise ValueError(f'Cannot infer {key} for {path}')


def candidate_from_path(path):
    aggregate = load_json(path)
    return {
        'path': path,
        'dataset': aggregate.get('dataset'),
        'backbone': aggregate.get('backbone'),
        'lambda1': infer_lambda(aggregate, path, 'lambda1'),
        'lambda2': infer_lambda(aggregate, path, 'lambda2'),
        'mean_validation_ndcg': metric_mean(aggregate, 'NDCG'),
        'mean_validation_uif': metric_mean(aggregate, 'UIF'),
    }


def select_config(standard_aggregate, candidates, retention_ratio=0.98):
    standard_ndcg = metric_mean(standard_aggregate, 'NDCG')
    threshold = retention_ratio * standard_ndcg
    normalized = []
    for item in candidates:
        candidate = dict(item)
        candidate['eligible'] = candidate['mean_validation_ndcg'] >= threshold
        normalized.append(candidate)
    eligible = [item for item in normalized if item['eligible']]
    if eligible:
        selected = min(eligible, key=lambda item: (
            item['mean_validation_uif'],
            -item['mean_validation_ndcg'],
            item['lambda1'],
            item['lambda2']))
        fallback = False
    else:
        selected = min(normalized, key=lambda item: (
            -item['mean_validation_ndcg'],
            item['mean_validation_uif'],
            item['lambda1'],
            item['lambda2']))
        fallback = True
    return {
        'dataset': standard_aggregate.get('dataset') or selected.get('dataset'),
        'backbone': standard_aggregate.get('backbone') or selected.get('backbone'),
        'standard_validation_ndcg': standard_ndcg,
        'retention_ratio': retention_ratio,
        'eligibility_threshold': threshold,
        'candidates': normalized,
        'selected': selected,
        'fallback_used': fallback,
    }


def main():
    parser = argparse.ArgumentParser(description='Select hyperparameters from five-seed validation aggregates.')
    parser.add_argument('--standard-aggregate', required=True)
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--retention-ratio', type=float, default=0.98)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    standard = load_json(args.standard_aggregate)
    candidates = [candidate_from_path(path) for path in args.candidate]
    result = select_config(standard, candidates, args.retention_ratio)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as handle:
        if args.output.endswith(('.yaml', '.yml')):
            yaml.safe_dump(result, handle, sort_keys=False)
        else:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
