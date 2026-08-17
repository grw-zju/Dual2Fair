#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_utils import DATASETS, load_dataset
from run import load_config

EXPECTED_STATS = {
    'movielens': {'users': 6034, 'items': 3125, 'interactions': 574376},
    'epinions': {'users': 20382, 'items': 30989, 'interactions': 542856},
    'gowalla': {'users': 29495, 'items': 40358, 'interactions': 2001700},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


def verify_dataset(root, name, config, manifest):
    data_dir = os.path.join(root, name)
    raw_path = os.path.join(data_dir, DATASETS[name]['file'])
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'Missing {name} file: {raw_path}')
    expected_hash = manifest.get(name, {}).get('sha256')
    if expected_hash and expected_hash != sha256(raw_path):
        raise ValueError(f'{name} SHA256 mismatch')
    dataset_config = config['dataset'][name]
    dataset = load_dataset(name, data_dir=data_dir,
                           min_ui=dataset_config['min_user_interactions'],
                           min_ii=dataset_config['min_item_interactions'])
    observed = {'users': dataset.n_users, 'items': dataset.n_items,
                'interactions': len(dataset.interactions)}
    if observed != EXPECTED_STATS[name]:
        raise ValueError(f'{name} statistics mismatch: expected {EXPECTED_STATS[name]}, got {observed}')
    return observed


def main():
    parser = argparse.ArgumentParser(description='Verify the external processed data package layout and statistics.')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--manifest')
    parser.add_argument('--dataset', action='append', choices=sorted(EXPECTED_STATS),
                        help='Dataset to verify; defaults to all datasets')
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    config = load_config('')
    datasets = args.dataset or list(EXPECTED_STATS)
    result = {name: verify_dataset(args.data_root, name, config, manifest)
              for name in datasets}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
