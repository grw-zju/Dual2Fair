import os
import urllib.request
import zipfile
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict


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
}


class Dataset:
    def __init__(self, name, data_dir=None, min_ui=5, min_ii=5):
        self.name = name
        self.data_dir = data_dir or os.path.join(DATA_DIR, name)
        self.min_ui = min_ui
        self.min_ii = min_ii
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        raw_path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        if not os.path.exists(raw_path):
            self._download_data()

        if self.name == 'epinions':
            df = self._load_epinions()
        elif self.name == 'movielens':
            df = self._load_movielens()
        elif self.name == 'gowalla':
            df = self._load_gowalla()
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        df = self._filter(df)
        self.n_users = int(df['user_id'].max() + 1)
        self.n_items = int(df['item_id'].max() + 1)

        self.interactions = df
        self.train_data, self.val_data, self.test_data = self._split_loo(df)

        self.train_user_items = defaultdict(set)
        for u, i in zip(self.train_data['user_id'].values, self.train_data['item_id'].values):
            self.train_user_items[u].add(i)
        self.user_items = self.train_user_items

        self.item_freq = defaultdict(int)
        for i in df['item_id'].values:
            self.item_freq[i] += 1
        self.user_freq = defaultdict(int)
        for u in df['user_id'].values:
            self.user_freq[u] += 1

        self.adj_mat = self._build_adj_mat()
        self.interact_mat = self._build_interact_mat()
        self._val_neg_items = self._precompute_eval_negatives(self.get_val_dict(), n_neg=99)
        self._test_neg_items = self._precompute_eval_negatives(self.get_test_dict(), n_neg=99)

    def _download_data(self):
        info = DATASETS[self.name]
        os.makedirs(self.data_dir, exist_ok=True)
        url = info['url']
        print(f"Downloading {self.name}...")
        if self.name == 'gowalla':
            gz_path = os.path.join(self.data_dir, "gowalla_checkins.txt.gz")
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, 'rb') as f_in:
                with open(os.path.join(self.data_dir, info['file']), 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_path)
        elif url.endswith('.zip'):
            zip_path = os.path.join(self.data_dir, f"{self.name}.zip")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.data_dir)
            os.remove(zip_path)
        else:
            urllib.request.urlretrieve(url, os.path.join(self.data_dir, info['file']))

    def _load_epinions(self):
        raw_path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(raw_path, sep='  ', header=None, engine='python')
        cols_needed = min(df.shape[1], 6)
        df = df.iloc[:, :cols_needed]
        col_names = ['user_id_raw', 'item_id_raw', 'rating', 'timestamp']
        if cols_needed > 4:
            col_names += [f'col{i}' for i in range(4, cols_needed)]
        df.columns = col_names
        df = df[['user_id_raw', 'item_id_raw']].copy()
        df['user_id_raw'] = pd.to_numeric(df['user_id_raw'], errors='coerce')
        df['item_id_raw'] = pd.to_numeric(df['item_id_raw'], errors='coerce')
        df = df.dropna()
        df['user_id_raw'] = df['user_id_raw'].astype(int)
        df['item_id_raw'] = df['item_id_raw'].astype(int)
        return self._remap_ids(df)

    def _load_movielens(self):
        raw_path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(raw_path, sep='::', header=None,
                         names=['user_id_raw', 'item_id_raw', 'rating', 'timestamp'],
                         engine='python')
        df = df[df['rating'] >= 4].copy()
        df = df[['user_id_raw', 'item_id_raw', 'timestamp']].copy()
        df['user_id_raw'] = df['user_id_raw'].astype(int)
        df['item_id_raw'] = df['item_id_raw'].astype(int)
        df['timestamp'] = df['timestamp'].astype(int)
        return self._remap_ids(df)

    def _load_gowalla(self):
        raw_path = os.path.join(self.data_dir, DATASETS[self.name]['file'])
        df = pd.read_csv(raw_path, sep='\t', header=None, engine='python',
                         names=['user_id_raw', 'timestamp', 'latitude', 'longitude', 'item_id_raw'])
        df = df[['user_id_raw', 'item_id_raw']].copy()
        df['user_id_raw'] = pd.to_numeric(df['user_id_raw'], errors='coerce')
        df['item_id_raw'] = pd.to_numeric(df['item_id_raw'], errors='coerce')
        df = df.dropna()
        df['user_id_raw'] = df['user_id_raw'].astype(int)
        df['item_id_raw'] = df['item_id_raw'].astype(int)
        return self._remap_ids(df)

    def _remap_ids(self, df):
        if 'user_id_raw' in df.columns:
            uids = df['user_id_raw'].unique()
            iids = df['item_id_raw'].unique()
            u_map = {old: new for new, old in enumerate(uids)}
            i_map = {old: new for new, old in enumerate(iids)}
            df['user_id'] = df['user_id_raw'].map(u_map)
            df['item_id'] = df['item_id_raw'].map(i_map)
        else:
            uids = df['user_id'].unique()
            iids = df['item_id'].unique()
            u_map = {old: new for new, old in enumerate(uids)}
            i_map = {old: new for new, old in enumerate(iids)}
            df['user_id'] = df['user_id'].map(u_map)
            df['item_id'] = df['item_id'].map(i_map)
        cols = ['user_id', 'item_id']
        if 'timestamp' in df.columns:
            cols.append('timestamp')
        return df[cols].copy()

    def _filter(self, df):
        while True:
            u_cnt = df['user_id'].value_counts()
            i_cnt = df['item_id'].value_counts()
            df = df[df['user_id'].isin(u_cnt[u_cnt >= self.min_ui].index)]
            df = df[df['item_id'].isin(i_cnt[i_cnt >= self.min_ii].index)]
            u_cnt2 = df['user_id'].value_counts()
            i_cnt2 = df['item_id'].value_counts()
            if len(u_cnt) == len(u_cnt2) and len(i_cnt) == len(i_cnt2):
                break
        return self._remap_ids(df)

    def _split_loo(self, df):
        df = df[['user_id', 'item_id']].copy()

        train_rows, val_rows, test_rows = [], [], []
        for uid, group in df.groupby('user_id'):
            if 'timestamp' in group.columns:
                items = group.sort_values('timestamp')['item_id'].values.tolist()
            else:
                items = group['item_id'].values.tolist()
                np.random.shuffle(items)
            if len(items) > 2:
                test_rows.append([uid, items[-1]])
                val_rows.append([uid, items[-2]])
                train_rows.extend([[uid, i] for i in items[:-2]])
            elif len(items) > 1:
                test_rows.append([uid, items[-1]])
                train_rows.extend([[uid, i] for i in items[:-1]])
            else:
                train_rows.extend([[uid, i] for i in items])

        cols = ['user_id', 'item_id']
        train_df = pd.DataFrame(train_rows, columns=cols)
        val_df = pd.DataFrame(val_rows, columns=cols) if val_rows else pd.DataFrame(columns=cols)
        test_df = pd.DataFrame(test_rows, columns=cols) if test_rows else pd.DataFrame(columns=cols)
        return train_df, val_df, test_df

    def _build_adj_mat(self):
        nu, ni = self.n_users, self.n_items
        R = sp.dok_matrix((nu + ni, nu + ni), dtype=np.float32)
        for u in range(nu):
            R[int(u), int(u)] = 1.0
        for i in range(ni):
            R[nu + int(i), nu + int(i)] = 1.0
        for u, i in zip(self.train_data['user_id'].values, self.train_data['item_id'].values):
            R[int(u), nu + int(i)] = 1.0
            R[nu + int(i), int(u)] = 1.0
        adj = R.tocsr()
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum + 1e-10, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        return norm_adj.tocsr()

    def _build_interact_mat(self):
        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u, i in zip(self.train_data['user_id'].values, self.train_data['item_id'].values):
            R[int(u), int(i)] = 1.0
        return R.tocsr()

    def get_advantaged_users(self, adv_ratio=0.05):
        freqs = sorted(self.user_freq.items(), key=lambda x: x[1], reverse=True)
        n_adv = max(1, int(len(freqs) * adv_ratio))
        adv = [u for u, _ in freqs[:n_adv]]
        disadv = [u for u, _ in freqs[n_adv:]]
        return adv, disadv

    def get_hot_cold_items(self, hot_ratio=0.2):
        freqs = sorted(self.item_freq.items(), key=lambda x: x[1], reverse=True)
        n_hot = max(1, int(len(freqs) * hot_ratio))
        hot = [i for i, _ in freqs[:n_hot]]
        cold = [i for i, _ in freqs[n_hot:]]
        return hot, cold

    def get_train_interactions(self):
        return list(zip(self.train_data['user_id'].values, self.train_data['item_id'].values))

    def get_test_dict(self):
        return dict(zip(self.test_data['user_id'].values, self.test_data['item_id'].values))

    def get_val_dict(self):
        return dict(zip(self.val_data['user_id'].values, self.val_data['item_id'].values))

    def _precompute_eval_negatives(self, eval_dict, n_neg=99):
        rng = np.random.RandomState(42)
        all_items = set(range(self.n_items))
        eval_neg = {}
        val_dict = self.get_val_dict()
        test_dict = self.get_test_dict()
        for uid in eval_dict.keys():
            pos_set = self.user_items.get(uid, set())
            for heldout_item in (val_dict.get(uid), test_dict.get(uid)):
                if heldout_item is not None:
                    if isinstance(heldout_item, (int, np.integer)):
                        pos_set = pos_set | {int(heldout_item)}
                    else:
                        pos_set = pos_set | set(heldout_item)
            candidate_pool = list(all_items - pos_set)
            if len(candidate_pool) < n_neg:
                neg_items = candidate_pool
            else:
                neg_indices = rng.choice(len(candidate_pool), size=n_neg, replace=False)
                neg_items = [candidate_pool[i] for i in neg_indices]
            eval_neg[uid] = neg_items
        return eval_neg

    def get_test_neg_items(self):
        return self._test_neg_items

    def get_val_neg_items(self):
        return self._val_neg_items

    def compute_dis_half2half(self, adv_ratio=0.05):
        adv_users, disadv_users = self.get_advantaged_users(adv_ratio)
        disadv_sorted = sorted(disadv_users, key=lambda u: self.user_freq.get(u, 0), reverse=True)
        half_n = len(disadv_sorted) // 2
        if half_n == 0:
            half2half = {u: u for u in disadv_sorted}
            return half2half
        half_1 = disadv_sorted[:half_n]
        half_2 = disadv_sorted[half_n:2*half_n]

        one_hot_1 = np.zeros((half_n, self.n_items), dtype=np.float32)
        one_hot_2 = np.zeros((half_n, self.n_items), dtype=np.float32)
        for idx, u in enumerate(half_1):
            for i in self.user_items.get(u, set()):
                if i < self.n_items:
                    one_hot_1[idx, i] = 1.0
        for idx, u in enumerate(half_2):
            for i in self.user_items.get(u, set()):
                if i < self.n_items:
                    one_hot_2[idx, i] = 1.0

        try:
            import ot
            M = ot.dist(one_hot_2, one_hot_1)
            a = ot.unif(half_n)
            b = ot.unif(half_n)
            P = ot.emd(a, b, M, numItermax=10000000)
            P_argmax = np.argmax(P, axis=1)
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            M = np.sum((one_hot_2[:, np.newaxis, :] - one_hot_1[np.newaxis, :, :]) ** 2, axis=2)
            row_ind, col_ind = linear_sum_assignment(M)
            P_argmax = col_ind

        half2half = {}
        for i in range(half_n):
            half2half[half_2[i]] = half_1[P_argmax[i]]
            half2half[half_1[P_argmax[i]]] = half_2[i]
        for u in disadv_sorted:
            if u not in half2half:
                half2half[u] = u
        return half2half

    def compute_static2active(self, adv_ratio=0.05):
        adv_users, disadv_users = self.get_advantaged_users(adv_ratio)
        static2active = {}
        for dis_u in disadv_users:
            dis_items = self.user_items.get(dis_u, set())
            best_adv = None
            best_intersect = -1
            for adv_u in adv_users:
                adv_items = self.user_items.get(adv_u, set())
                intersect = len(dis_items & adv_items)
                if intersect > best_intersect:
                    best_intersect = intersect
                    best_adv = adv_u
            if best_adv is not None:
                static2active[dis_u] = best_adv
            else:
                static2active[dis_u] = adv_users[0]
        return static2active


def load_dataset(name, data_dir=None, min_ui=5, min_ii=5):
    return Dataset(name, data_dir, min_ui, min_ii)
