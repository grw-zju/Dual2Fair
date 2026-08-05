import argparse
import hashlib
import json
import os

from data.dataset_utils import DATASETS, load_dataset


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--min-user-interactions', type=int, default=1)
    parser.add_argument('--min-item-interactions', type=int, default=1)
    parser.add_argument('--split-seed', type=int, default=2026)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, min_ui=args.min_user_interactions,
                           min_ii=args.min_item_interactions,
                           data_split_seed=args.split_seed)
    source = os.path.join(dataset.data_dir, DATASETS[args.dataset]['file'])
    payload = {
        'dataset': args.dataset, 'source_url': DATASETS[args.dataset]['url'],
        'source_file': source, 'source_sha256': checksum(source),
        'users': dataset.n_users, 'items': dataset.n_items,
        'interactions': len(dataset.interactions), 'train': len(dataset.train_data),
        'validation': len(dataset.val_data), 'test': len(dataset.test_data),
        'sparsity': 1.0 - len(dataset.interactions) / (dataset.n_users * dataset.n_items),
        'split_hash': dataset.split_hash, 'timestamp_policy': 'chronological_leave_one_out',
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
