import numpy as np
import torch
from sklearn.metrics import ndcg_score as sklearn_ndcg_score

from .metrics import (compute_dif_affine_invariant, compute_dif_legacy,
                      compute_dif_rank, compute_duf, compute_ndcg_full,
                      compute_uif)


class Evaluator:
    def __init__(self, dataset, k=10, num_repeats=1, n_neg_sampled=99,
                 device=None, split='test', max_full_eval_scores=50000000,
                 dif_mode='affine_invariant', uif_w1=0.5, uif_w2=0.5):
        if num_repeats != 1:
            raise ValueError('Evaluator repetition is unsupported; use independent model seeds')
        if dif_mode not in {'legacy', 'affine_invariant', 'rank'}:
            raise ValueError(f'Unknown DIF mode: {dif_mode}')
        self.dataset = dataset
        self.k = k
        self.n_neg_sampled = n_neg_sampled
        self.device = device or torch.device('cpu')
        self.split = split
        self.max_full_eval_scores = max_full_eval_scores
        self.dif_mode = dif_mode
        self.uif_w1 = uif_w1
        self.uif_w2 = uif_w2
        if split == 'val' and dataset.get_val_dict():
            self.test_dict = dataset.get_val_dict()
            self.test_neg_items = dataset.get_val_neg_items()
        else:
            self.test_dict = dataset.get_test_dict()
            self.test_neg_items = dataset.get_test_neg_items()
        self.train_user_items = dataset.user_items
        self.baseline_ndcg = None
        self.baseline_duf = None
        self.baseline_dif = None

    def set_baseline(self, ndcg, duf, dif):
        self.baseline_ndcg = ndcg
        self.baseline_duf = duf
        self.baseline_dif = dif

    def _dif(self, scores):
        functions = {
            'legacy': compute_dif_legacy,
            'affine_invariant': compute_dif_affine_invariant,
            'rank': compute_dif_rank,
        }
        return functions[self.dif_mode](scores, self.test_dict, self.train_user_items, self.k)

    def _result(self, ndcg_scores, mean_ndcg, mean_hit, dif_value):
        duf_value = compute_duf(ndcg_scores)
        uif_value = compute_uif(
            ndcg_scores, dif_value, self.uif_w1, self.uif_w2,
            self.baseline_duf, self.baseline_dif)
        return {
            'NDCG': mean_ndcg,
            'Hit': mean_hit,
            'DUF': duf_value,
            'DIF': dif_value,
            'UIF': uif_value,
            'DIF_protocol': f'{self.dif_mode}_{self.split}',
            'UIF_baseline_DUF': self.baseline_duf,
            'UIF_baseline_DIF': self.baseline_dif,
            'UIF_w1': self.uif_w1,
            'UIF_w2': self.uif_w2,
            'UIF_normalized': self.baseline_duf is not None and self.baseline_dif is not None,
        }

    def evaluate(self, user_embeddings=None, item_embeddings=None, model=None,
                 w1=None, w2=None):
        if w1 is not None:
            self.uif_w1 = w1
        if w2 is not None:
            self.uif_w2 = w2
        n_scores = self.dataset.n_users * self.dataset.n_items
        if self.max_full_eval_scores is not None and n_scores > self.max_full_eval_scores:
            raise ValueError(
                f'Full evaluation would materialize {n_scores:,} scores, exceeding '
                f'evaluation.max_full_eval_scores={self.max_full_eval_scores:,}')
        scores = self._compute_all_scores(user_embeddings, item_embeddings, model)
        ndcg_scores, mean_ndcg, mean_hit = compute_ndcg_full(
            scores, self.test_dict, self.train_user_items, self.k)
        return self._result(ndcg_scores, mean_ndcg, mean_hit, self._dif(scores))

    def sampled_evaluate(self, user_embeddings=None, item_embeddings=None, model=None,
                         w1=None, w2=None):
        if w1 is not None:
            self.uif_w1 = w1
        if w2 is not None:
            self.uif_w2 = w2
        ndcg_scores, mean_ndcg, mean_hit, dif_value = self._sampled_metrics(
            user_embeddings, item_embeddings, model)
        return self._result(ndcg_scores, mean_ndcg, mean_hit, dif_value)

    def _sampled_metrics(self, user_embeddings=None, item_embeddings=None, model=None):
        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.from_numpy(user_embeddings).float()
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float()
        ndcg_scores, hit_scores = {}, {}
        exposure = np.zeros(self.dataset.n_items, dtype=np.float64)
        quality = np.zeros(self.dataset.n_items, dtype=np.float64)
        for uid, test_items in self.test_dict.items():
            positive = int(test_items) if isinstance(test_items, (int, np.integer)) else int(next(iter(test_items)))
            negatives = self.test_neg_items.get(uid, [])
            items = [positive] + list(negatives)
            relevance = np.zeros(len(items), dtype=np.float64)
            relevance[0] = 1.0
            scores = self._score_user_items(uid, items, user_embeddings, item_embeddings, model)
            ndcg_scores[uid] = float(sklearn_ndcg_score([relevance], [scores], k=self.k))
            top_indices = np.argsort(-scores, kind='mergesort')[:self.k]
            hit_scores[uid] = float(0 in top_indices)
            exposure[np.asarray(items)[top_indices]] += 1.0
            if self.dif_mode == 'legacy':
                item_quality = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
            else:
                order = np.argsort(scores, kind='mergesort')
                ranks = np.empty(len(scores), dtype=np.float64)
                ranks[order] = np.arange(len(scores), dtype=np.float64)
                if self.dif_mode == 'rank':
                    descending = np.empty(len(scores), dtype=np.float64)
                    descending[np.argsort(-scores, kind='mergesort')] = np.arange(1, len(scores) + 1)
                    item_quality = 1.0 / np.log2(descending + 1.0)
                else:
                    item_quality = (ranks + 1.0) / max(1, len(scores))
            quality[items] += item_quality
        observed = quality > 0
        dif_value = float(np.var(exposure[observed] / quality[observed])) if np.any(observed) else 0.0
        mean_ndcg = float(np.mean(list(ndcg_scores.values()))) if ndcg_scores else 0.0
        mean_hit = float(np.mean(list(hit_scores.values()))) if hit_scores else 0.0
        return ndcg_scores, mean_ndcg, mean_hit, dif_value

    def _score_user_items(self, uid, item_ids, user_embeddings=None,
                          item_embeddings=None, model=None):
        item_tensor = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
        user_tensor = torch.full_like(item_tensor, uid)
        with torch.no_grad():
            if model is not None:
                return model.forward(user_tensor, item_tensor).detach().cpu().numpy()
            if user_embeddings is None or item_embeddings is None:
                raise ValueError('A model or explicit user/item embeddings are required')
            user = user_embeddings[uid].to(self.device)
            items = item_embeddings[item_ids].to(self.device)
            return (items @ user).detach().cpu().numpy()

    def _compute_all_scores(self, user_embeddings, item_embeddings, model=None):
        if model is not None:
            if not hasattr(model, 'compute_all_scores'):
                raise TypeError('Model must implement compute_all_scores; dot-product fallback is forbidden')
            with torch.no_grad():
                return model.compute_all_scores(self.device).detach().cpu().numpy()
        if user_embeddings is None or item_embeddings is None:
            raise ValueError('A model or explicit user/item embeddings are required')
        users = torch.as_tensor(user_embeddings, dtype=torch.float32)
        items = torch.as_tensor(item_embeddings, dtype=torch.float32)
        return (users @ items.T).cpu().numpy()
