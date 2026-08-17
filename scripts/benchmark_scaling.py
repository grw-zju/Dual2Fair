import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_efficiency import benchmark


EXPECTED_ITEM_COUNTS = {20: 8072, 40: 16143, 60: 24215, 80: 32286, 100: 40358}
RAW_FILE = 'loc-gowalla_totalCheckins.txt'


def subset_file(root, fraction):
    manifest = os.path.join(root, f'{fraction}', 'manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as handle:
            data = json.load(handle)
        path = data.get('interactions') or data.get('path') or RAW_FILE
        return os.path.join(root, f'{fraction}', path), data
    return os.path.join(root, f'{fraction}', RAW_FILE), {}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def count_gowalla_items(path):
    items = set()
    with open(path, 'r', errors='ignore') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 5:
                items.add(parts[-1])
    return len(items)


def validate_subset(root, fraction):
    path, manifest = subset_file(root, fraction)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {fraction}% Gowalla subset input: {path}')
    observed = int(manifest.get('item_count') or count_gowalla_items(path))
    expected = EXPECTED_ITEM_COUNTS[fraction]
    if observed != expected:
        raise ValueError(f'{fraction}% subset item count mismatch: expected {expected}, got {observed}')
    actual_hash = sha256(path)
    expected_hash = manifest.get('sha256')
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(f'{fraction}% subset SHA256 mismatch')
    return {'fraction': fraction / 100.0, 'path': path, 'item_count': observed,
            'sha256': actual_hash, 'manifest': manifest or None}


def main():
    parser = argparse.ArgumentParser(description='Run Table-V scaling benchmark from external Gowalla subset inputs.')
    parser.add_argument('--dataset', default='gowalla', choices=['gowalla'])
    parser.add_argument('--subset-dir', required=True,
                        help='Directory containing 20/40/60/80/100 Gowalla subset folders')
    parser.add_argument('--backbone', default='lightgcn')
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--method', default='dual2fair_lowrank',
                        choices=['standard', 'dual2fair_lowrank', 'dual2fair_dense'])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--output-json')
    args = parser.parse_args()

    try:
        subsets = [validate_subset(args.subset_dir, fraction)
                   for fraction in EXPECTED_ITEM_COUNTS]
    except Exception as exc:
        print(str(exc) + '. Prepare exact external Gowalla subsets from the processed data package; this script does not invent subset manifests or reuse demo data.',
              file=sys.stderr)
        sys.exit(1)

    results = []
    for subset in subsets:
        data_dir = os.path.dirname(subset['path'])
        result = benchmark(args.dataset, args.backbone, args.config,
                           args.method, args.gpu, data_dir=data_dir)
        result.update(subset)
        results.append(result)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as handle:
            json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
