import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

DATASET_FILES = {
    'epinions': 'epinion_with_rating_timestamp_txt/rating_with_timestamp.txt',
    'movielens': 'ml-1m/ratings.dat',
    'gowalla': 'loc-gowalla_totalCheckins.txt',
}

EXPECTED = {
    'epinions': (20382, 30989, 542856),
    'movielens': (6034, 3125, 574376),
    'gowalla': (29495, 40358, 2001700),
}


def raw_file_path(name):
    return os.path.join(DATA_DIR, name, DATASET_FILES[name])


def audit(name):
    path = raw_file_path(name)
    if not os.path.exists(path):
        return {
            'dataset': name,
            'raw_file': path,
            'observed': None,
            'paper_reference': EXPECTED[name],
            'matches': False,
            'status': 'missing raw data; preprocessing audit not run',
        }
    sys.path.insert(0, ROOT)
    from data.dataset_utils import load_dataset
    dataset = load_dataset(name, min_ui=3, min_ii=1)
    observed = (dataset.n_users, dataset.n_items, len(dataset.interactions))
    return {
        'dataset': name,
        'raw_file': path,
        'observed': observed,
        'paper_reference': EXPECTED[name],
        'matches': observed == EXPECTED[name],
        'status': 'aligned' if observed == EXPECTED[name] else 'requires result reproduction check',
    }


if __name__ == '__main__':
    print(json.dumps([audit(name) for name in EXPECTED], indent=2))
