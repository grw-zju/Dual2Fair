import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_efficiency import benchmark


FRACTIONS = (20, 40, 60, 80, 100)
RAW_FILE = 'loc-gowalla_totalCheckins.txt'


def subset_path(root, fraction):
    return os.path.join(root, f'{fraction}', RAW_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gowalla', choices=['gowalla'])
    parser.add_argument('--subset-dir', required=True,
                        help='Directory containing 20/40/60/80/100 Gowalla subset folders')
    parser.add_argument('--backbone', default='lightgcn')
    parser.add_argument('--config', default='')
    parser.add_argument('--method', default='dual2fair_lowrank',
                        choices=['standard', 'dual2fair_lowrank', 'dual2fair_dense'])
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    missing = [fraction for fraction in FRACTIONS
               if not os.path.exists(subset_path(args.subset_dir, fraction))]
    if missing:
        print(
            'Missing Gowalla subset files for catalog fractions: '
            + ', '.join(f'{fraction}%' for fraction in missing)
            + '. Prepare full Gowalla data and explicit subset folders before running scaling benchmarks. '
            'This script does not reuse demo data or duplicate one catalog across fractions.',
            file=sys.stderr)
        sys.exit(1)

    results = []
    for fraction in FRACTIONS:
        data_dir = os.path.join(args.subset_dir, str(fraction))
        result = benchmark(args.dataset, args.backbone, args.config,
                           args.method, args.gpu, data_dir=data_dir)
        result['catalog_fraction'] = fraction / 100.0
        result['data_dir'] = data_dir
        results.append(result)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
