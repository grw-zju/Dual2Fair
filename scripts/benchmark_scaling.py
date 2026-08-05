import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_efficiency import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', type=int, default=10000)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--anchors', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    results = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        size = max(2, int(args.items * fraction))
        results.append({'fraction': fraction, **benchmark(
            size, args.dimension, min(args.anchors, size - 1),
            args.iterations, args.seed)})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
