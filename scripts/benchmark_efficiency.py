import argparse
import json
import time
import tracemalloc

import torch

from data.dataset_utils import load_dataset
from models.dual2fair.transport import (dense_log_sinkhorn,
                                        deterministic_landmark_indices,
                                        landmark_transport)


def benchmark(n_items, dimension, anchors, iterations, seed):
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(n_items, dimension, generator=generator)
    frequencies = torch.arange(1, n_items + 1, dtype=torch.float32)
    indices = deterministic_landmark_indices(frequencies, min(anchors, n_items - 1))
    source = torch.full((n_items,), 1.0 / n_items)
    target = torch.full((len(indices),), 1.0 / len(indices))
    tracemalloc.start()
    started = time.perf_counter()
    landmark_transport(embeddings, source, embeddings[indices], target,
                       max_iter=iterations)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'items': n_items, 'dimension': dimension, 'anchors': len(indices),
            'sinkhorn_iterations': iterations, 'seconds': elapsed,
            'peak_cpu_bytes': peak, 'backend': 'landmark'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', type=int, default=10000)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--anchors', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.items, args.dimension, args.anchors,
                               args.iterations, args.seed), indent=2))


if __name__ == '__main__':
    main()
