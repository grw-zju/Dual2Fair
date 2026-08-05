import hashlib
import json
import os
import urllib.request
import zipfile
import gzip
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'epinions': {
        'url': 'https://www.cse.msu.edu/~tangjili/datasetcode/epinions_with_rating_timestamp_txt.zip',
        'file': 'epinion_with_rating_timestamp_txt/rating_with_timestamp.txt',
    },
    'movielens': {
        'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'file': 'ml-1m/ratings.dat',
    },
    'gowalla': {
        'url': 'https://huggingface.co/datasets/habedi/gowalla-dataset/resolve/main/original_data/loc-gowalla_totalCheckins.txt.gz?download=true',
        'file': 'loc-gowalla_totalCheckins.txt',
    },
    'demo': {
        'url': None,
        'file': 'interactions.csv',
    },
}


class Dataset:
    def __init__(self, name, data_dir=None, min_ui=5, min_ii=5,
                 data_split_seed=2026, negative_sampling_seed=42,
                 split_path=None):
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        self.name = name
        self.data_dir = data_dir or os.path.join(DATA_DIR, name)
        self.min_ui = min_ui
        self.min_ii = min_ii
        self.data_split_seed = int(data_split_seed)
        self.negative_sampling_seed = int(negative_sampling_seed)
        self.split_path = split_path
        self.user_id_map = {}
        self.item_id_map = {}
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        raw_path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        if not os.path.exists(raw_path):
            self._download_data()

        loaders = {
            'epinions': self._load_epinions,
            'movielens': self._load_movielens,
            'gowalla': self._load_gowalla,
            'demo': self._load_demo,
        }
        df = self._filter(loaders[self.name]())
        if df.empty:
            raise ValueError(f"Dataset {self.name} is empty after filtering")

        self.n_users = int(df['user_id'].max() + 1)
        self.n_items = int(df['item_id'].max() + 1)
        self.interactions = df.reset_index(drop=True)

        if self.split_path and os.path.exists(self.split_path):
            self.train_data, self.val_data, self.test_data = self.load_split(self.split_path)
        else:
            self.train_data, self.val_data, self.test_data = self._split_loo(df)
            if self.split_path:
                self.save_split(self.split_path)

        self.train_user_items = defaultdict(set)
        for u, i in zip(self.train_data['user_id'].values, self.train_data['item_id'].values):
            self.train_user_items[int(u)].add(int(i))
        self.user_items = self.train_user_items

        self.item_freq = defaultdict(int)
        for i in self.train_data['item_id'].values:
            self.item_freq[int(i)] += 1
        self.user_freq = defaultdict(int)
        for u in self.train_data['user_id'].values:
            self.user_freq[int(u)] += 1
        for u in range(self.n_users):
            self.user_freq[u] += 0
        for i in range(self.n_items):
            self.item_freq[i] += 0

        self.adj_mat = self._build_adj_mat()
        self.interact_mat = self._build_interact_mat()
        self._val_neg_items = self._precompute_eval_negatives(
            self.get_val_dict(), self.negative_sampling_seed)
        self._test_neg_items = self._precompute_eval_negatives(
            self.get_test_dict(), self.negative_sampling_seed + 1)
        self.split_hash = self._compute_split_hash()

    def _download_data(self):
        info = DATASETS[self.name]
        if not info['url']:
            raise FileNotFoundError(
                f"Missing bundled dataset file: {os.path.join(self.data_dir, info['file'])}")
        os.makedirs(self.data_dir, exist_ok=True)
        url = info['url']
        print(f"Downloading {self.name}...")
        if self.name == 'gowalla':
            gz_path = os.path.join(self.data_dir, 'gowalla_checkins.txt.gz')
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, 'rb') as f_in:
                with open(os.path.join(self.data_dir, info['file']), 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_path)
        elif url.endswith('.zip'):
            zip_path = os.path.join(self.data_dir, f'{self.name}.zip')
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as archive:
                archive.extractall(self.data_dir)
            os.remove(zip_path)
        else:
            urllib.request.urlretrieve(url, os.path.join(self.data_dir, info['file']))

    def _load_epinions(self):
        path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
        if df.shape[1] < 2:
            raise ValueError('Epinions file must contain user and item columns')
        result = pd.DataFrame({
            'user_id_raw': pd.to_numeric(df.iloc[:, 0], errors='coerce'),
            'item_id_raw': pd.to_numeric(df.iloc[:, 1], errors='coerce'),
        })
        if df.shape[1] >= 4:
            result['timestamp'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        return self._remap_ids(result.dropna(subset=['user_id_raw', 'item_id_raw']))

    def _load_movielens(self):
        path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(path, sep='::', header=None,
                         names=['user_id_raw', 'item_id_raw', 'rating', 'timestamp'],
                         engine='python')
        df = df[df['rating'] >= 4].copy()
        return self._remap_ids(df[['user_id_raw', 'item_id_raw', 'timestamp']])

    def _load_gowalla(self):
        path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['user_id_raw', 'timestamp', 'latitude', 'longitude', 'item_id_raw'])
        parsed_timestamp = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[parsed_timestamp.notna()].copy()
        df['timestamp'] = parsed_timestamp[parsed_timestamp.notna()].astype('int64')
        return self._remap_ids(df[['user_id_raw', 'item_id_raw', 'timestamp']].dropna())

    def _load_demo(self):
        path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(path)
        required = {'user_id', 'item_id', 'timestamp'}
        if not required.issubset(df.columns):
            raise ValueError(f"Demo data requires columns: {sorted(required)}")
        df = df.rename(columns={'user_id': 'user_id_raw', 'item_id': 'item_id_raw'})
        return self._remap_ids(df[['user_id_raw', 'item_id_raw', 'timestamp']])

    def _remap_ids(self, df):
        df = df.copy()
        user_values = sorted(df['user_id_raw'].dropna().unique().tolist())
        item_values = sorted(df['item_id_raw'].dropna().unique().tolist())
        self.user_id_map = {old: new for new, old in enumerate(user_values)}
        self.item_id_map = {old: new for new, old in enumerate(item_values)}
        df['user_id'] = df['user_id_raw'].map(self.user_id_map)
        df['item_id'] = df['item_id_raw'].map(self.item_id_map)
        columns = ['user_id', 'item_id']
        if 'timestamp' in df.columns:
            columns.append('timestamp')
        result = df[columns].dropna(subset=['user_id', 'item_id']).astype(
            {'user_id': int, 'item_id': int})
        if 'timestamp' in result.columns:
            result = result.dropna(subset=['timestamp']).sort_values(
                ['timestamp', 'user_id', 'item_id'], kind='mergesort')
        return result.drop_duplicates(
            subset=['user_id', 'item_id'], keep='last').reset_index(drop=True)

    def _filter(self, df):
        filtered = df.copy()
        while not filtered.empty:
            user_counts = filtered['user_id'].value_counts()
            item_counts = filtered['item_id'].value_counts()
            next_df = filtered[
                filtered['user_id'].isin(user_counts[user_counts >= self.min_ui].index)
                & filtered['item_id'].isin(item_counts[item_counts >= self.min_ii].index)
            ]
            if len(next_df) == len(filtered):
                break
            filtered = next_df
        if filtered.empty:
            return filtered
        raw = filtered.rename(columns={'user_id': 'user_id_raw', 'item_id': 'item_id_raw'})
        return self._remap_ids(raw)

    def _split_loo(self, df):
        train_rows, val_rows, test_rows = [], [], []
        rng = np.random.RandomState(self.data_split_seed)
        has_timestamp = 'timestamp' in df.columns
        for uid, group in df.groupby('user_id', sort=True):
            if has_timestamp:
                ordered = group.sort_values(['timestamp', 'item_id'], kind='mergesort')
                items = ordered['item_id'].astype(int).tolist()
            else:
                items = group['item_id'].astype(int).tolist()
                rng.shuffle(items)
            if len(items) >= 3:
                train_rows.extend((int(uid), item) for item in items[:-2])
                val_rows.append((int(uid), items[-2]))
                test_rows.append((int(uid), items[-1]))
            elif len(items) == 2:
                train_rows.append((int(uid), items[0]))
                test_rows.append((int(uid), items[1]))
            elif items:
                train_rows.append((int(uid), items[0]))
        columns = ['user_id', 'item_id']
        return (pd.DataFrame(train_rows, columns=columns),
                pd.DataFrame(val_rows, columns=columns),
                pd.DataFrame(test_rows, columns=columns))

    def save_split(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            'dataset': self.name,
            'data_split_seed': self.data_split_seed,
            'min_ui': self.min_ui,
            'min_ii': self.min_ii,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'interaction_hash': self._compute_interaction_hash(),
            'train': self.train_data.values.tolist(),
            'validation': self.val_data.values.tolist(),
            'test': self.test_data.values.tolist(),
        }
        with open(path, 'w') as handle:
            json.dump(payload, handle, sort_keys=True)

    def load_split(self, path):
        with open(path, 'r') as handle:
            payload = json.load(handle)
        expected = {
            'dataset': self.name,
            'data_split_seed': self.data_split_seed,
            'min_ui': self.min_ui,
            'min_ii': self.min_ii,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'interaction_hash': self._compute_interaction_hash(),
        }
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatches:
            raise ValueError(f'Frozen split metadata mismatch: {", ".join(mismatches)}')
        columns = ['user_id', 'item_id']
        frames = tuple(pd.DataFrame(payload[key], columns=columns)
                       for key in ('train', 'validation', 'test'))
        for frame in frames:
            if (not frame.empty and
                    (frame['user_id'].min() < 0 or frame['user_id'].max() >= self.n_users or
                     frame['item_id'].min() < 0 or frame['item_id'].max() >= self.n_items)):
                raise ValueError('Frozen split contains out-of-range IDs')
        return frames

    def save_id_mappings(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            'user': {str(k): int(v) for k, v in self.user_id_map.items()},
            'item': {str(k): int(v) for k, v in self.item_id_map.items()},
        }
        with open(path, 'w') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    def _compute_interaction_hash(self):
        values = self.interactions[['user_id', 'item_id']].to_numpy(dtype=np.int64)
        return hashlib.sha256(values.tobytes()).hexdigest()

    def _compute_split_hash(self):
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        digest.update(str(self.data_split_seed).encode())
        digest.update(str(self.min_ui).encode())
        digest.update(str(self.min_ii).encode())
        digest.update(self._compute_interaction_hash().encode())
        for name, frame in (('train', self.train_data), ('validation', self.val_data),
                            ('test', self.test_data)):
            digest.update(name.encode())
            digest.update(str(len(frame)).encode())
            digest.update(frame[['user_id', 'item_id']].to_numpy(dtype=np.int64).tobytes())
        return digest.hexdigest()

    def _build_adj_mat(self):
        users = self.train_data['user_id'].to_numpy(dtype=np.int64)
        items = self.train_data['item_id'].to_numpy(dtype=np.int64) + self.n_users
        rows = np.concatenate([users, items, np.arange(self.n_users + self.n_items)])
        cols = np.concatenate([items, users, np.arange(self.n_users + self.n_items)])
        values = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((values, (rows, cols)),
                            shape=(self.n_users + self.n_items,) * 2).tocsr()
        degree = np.asarray(adj.sum(axis=1)).ravel()
        inv_sqrt = np.power(degree + 1e-10, -0.5)
        return (sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)).tocsr()

    def _build_interact_mat(self):
        users = self.train_data['user_id'].to_numpy(dtype=np.int64)
        items = self.train_data['item_id'].to_numpy(dtype=np.int64)
        values = np.ones(len(users), dtype=np.float32)
        return sp.coo_matrix((values, (users, items)),
                             shape=(self.n_users, self.n_items)).tocsr()

    def get_advantaged_users(self, adv_ratio=0.05):
        frequencies = sorted(self.user_freq.items(), key=lambda pair: (-pair[1], pair[0]))
        n_advantaged = max(1, int(len(frequencies) * adv_ratio))
        return ([user for user, _ in frequencies[:n_advantaged]],
                [user for user, _ in frequencies[n_advantaged:]])

    def get_hot_cold_items(self, hot_ratio=0.2):
        frequencies = sorted(self.item_freq.items(), key=lambda pair: (-pair[1], pair[0]))
        n_hot = max(1, int(len(frequencies) * hot_ratio))
        return ([item for item, _ in frequencies[:n_hot]],
                [item for item, _ in frequencies[n_hot:]])

    def get_train_interactions(self):
        return list(zip(self.train_data['user_id'].values, self.train_data['item_id'].values))

    def get_test_dict(self):
        return dict(zip(self.test_data['user_id'].values, self.test_data['item_id'].values))

    def get_val_dict(self):
        return dict(zip(self.val_data['user_id'].values, self.val_data['item_id'].values))

    def _precompute_eval_negatives(self, eval_dict, seed, n_neg=99):
        rng = np.random.RandomState(seed)
        all_items = set(range(self.n_items))
        validation = self.get_val_dict()
        test = self.get_test_dict()
        result = {}
        for uid in sorted(eval_dict):
            blocked = set(self.user_items.get(uid, set()))
            if uid in validation:
                blocked.add(int(validation[uid]))
            if uid in test:
                blocked.add(int(test[uid]))
            candidates = sorted(all_items - blocked)
            if len(candidates) > n_neg:
                indices = rng.choice(len(candidates), size=n_neg, replace=False)
                result[uid] = [candidates[index] for index in indices]
            else:
                result[uid] = candidates
        return result

    def get_test_neg_items(self):
        return self._test_neg_items

    def get_val_neg_items(self):
        return self._val_neg_items

    def compute_dis_half2half(self, adv_ratio=0.05):
        _, disadvantaged = self.get_advantaged_users(adv_ratio)
        ordered = sorted(disadvantaged, key=lambda u: (-self.user_freq.get(u, 0), u))
        midpoint = len(ordered) // 2
        first, second = ordered[:midpoint], ordered[midpoint:midpoint * 2]
        mapping = {}
        for left, right in zip(first, second):
            mapping[left] = right
            mapping[right] = left
        for user in ordered:
            mapping.setdefault(user, user)
        return mapping

    def compute_static2active(self, adv_ratio=0.05):
        advantaged, disadvantaged = self.get_advantaged_users(adv_ratio)
        if not advantaged:
            return {user: user for user in disadvantaged}
        mapping = {}
        for user in disadvantaged:
            user_items = self.user_items.get(user, set())
            mapping[user] = max(
                advantaged,
                key=lambda candidate: (len(user_items & self.user_items.get(candidate, set())),
                                       -candidate))
        return mapping


def load_dataset(name, data_dir=None, min_ui=5, min_ii=5,
                 data_split_seed=2026, negative_sampling_seed=42,
                 split_path=None):
    return Dataset(name, data_dir, min_ui, min_ii, data_split_seed,
                   negative_sampling_seed, split_path)
