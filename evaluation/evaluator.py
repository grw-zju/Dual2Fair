import numpy as np
import torch
from .metrics import (compute_ndcg_full, compute_sampled_ndcg, compute_duf,
                      compute_dif_from_exposure, compute_sampled_dif_from_exposure,
                      compute_uif)


class Evaluator:
    def __init__(self, dataset, k=10, num_repeats=1, n_neg_sampled=99, device=None, split='test',
                 max_full_eval_scores=50000000):
        self.dataset = dataset
        self.k = k
        self.num_repeats = num_repeats
        self.n_neg_sampled = n_neg_sampled
        self.device = device or torch.device('cpu')
        self.split = split
        self.max_full_eval_scores = max_full_eval_scores
        if split == 'val' and hasattr(dataset, 'get_val_dict') and dataset.get_val_dict():
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

    def evaluate(self, user_embeddings=None, item_embeddings=None, model=None, w1=0.5, w2=0.5):
        n_scores = self.dataset.n_users * self.dataset.n_items
        if self.max_full_eval_scores is not None and n_scores > self.max_full_eval_scores:
            raise ValueError(
                f"Full evaluation would materialize {n_scores:,} user-item scores "
                f"({self.dataset.n_users:,} users x {self.dataset.n_items:,} items), "
                f"which exceeds evaluation.max_full_eval_scores={self.max_full_eval_scores:,}. "
                "Use --eval_mode sampled for this dataset or raise the config limit if you have enough memory."
            )
        all_scores = self._compute_all_scores(user_embeddings, item_embeddings, model)
        results = {'NDCG': 0.0, 'Hit': 0.0, 'DUF': 0.0, 'DIF': 0.0, 'UIF': 0.0}
        for _ in range(self.num_repeats):
            ndcg_scores, mean_ndcg, mean_hit = compute_ndcg_full(
                all_scores, self.test_dict, self.train_user_items, self.k)
            dif_val = compute_dif_from_exposure(all_scores, self.test_dict, self.train_user_items, self.k)
            duf_val = compute_duf(ndcg_scores)
            uif_val = compute_uif(ndcg_scores, dif_val, w1=w1, w2=w2,
                              baseline_duf=self.baseline_duf,
                              baseline_dif=self.baseline_dif)
            results['NDCG'] += mean_ndcg
            results['Hit'] += mean_hit
            results['DUF'] += duf_val
            results['DIF'] += dif_val
            results['UIF'] += uif_val
        for key in results:
            results[key] /= self.num_repeats
        return results

    def sampled_evaluate(self, user_embeddings=None, item_embeddings=None, model=None, w1=0.5, w2=0.5):
        if model is not None and getattr(model, '_backbone_name_hint', None) in {'lightgcn', 'vaecf'}:
            with torch.no_grad():
                user_embeddings = model.get_user_embeddings().detach().cpu()
                item_embeddings = model.get_item_embeddings().detach().cpu()
            model = None
        results = {'NDCG': 0.0, 'Hit': 0.0, 'DUF': 0.0, 'DIF': 0.0, 'UIF': 0.0}
        for _ in range(self.num_repeats):
            ndcg_scores, mean_ndcg, mean_hit, dif_val = self._sampled_metrics(
                user_embeddings=user_embeddings, item_embeddings=item_embeddings, model=model)
            duf_val = compute_duf(ndcg_scores)
            uif_val = compute_uif(ndcg_scores, dif_val, w1=w1, w2=w2,
                              baseline_duf=self.baseline_duf,
                              baseline_dif=self.baseline_dif)
            results['NDCG'] += mean_ndcg
            results['Hit'] += mean_hit
            results['DUF'] += duf_val
            results['DIF'] += dif_val
            results['UIF'] += uif_val
        for key in results:
            results[key] /= self.num_repeats
        return results

    def _sampled_metrics(self, user_embeddings=None, item_embeddings=None, model=None):
        from sklearn.metrics import ndcg_score as sklearn_ndcg_score
        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.from_numpy(user_embeddings).float()
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float()

        ndcg_scores, hit_scores = {}, {}
        n_items = self.dataset.n_items
        exposure = np.zeros(n_items)
        quality = np.zeros(n_items)

        for uid, test_items in self.test_dict.items():
            if isinstance(test_items, (int, np.integer)):
                pos_item = int(test_items)
            elif isinstance(test_items, set):
                pos_item = list(test_items)[0]
            else:
                pos_item = list(test_items)[0]

            neg_items = self.test_neg_items.get(uid)
            if neg_items is None:
                test_items_set = {pos_item} if isinstance(test_items, (int, np.integer)) else set(test_items)
                candidate_pool = list(set(range(n_items)) - self.train_user_items.get(uid, set()) - test_items_set)
                if len(candidate_pool) < self.n_neg_sampled:
                    neg_items = candidate_pool
                else:
                    neg_indices = np.random.choice(len(candidate_pool), size=self.n_neg_sampled, replace=False)
                    neg_items = [candidate_pool[i] for i in neg_indices]

            eval_items = [pos_item] + list(neg_items)
            labels = np.zeros(len(eval_items))
            labels[0] = 1.0
            scores = self._score_user_items(uid, eval_items, user_embeddings, item_embeddings, model)

            ndcg_scores[uid] = sklearn_ndcg_score([labels], [scores], k=self.k)
            top_k_idx = np.argsort(-scores)[:self.k]
            hit_scores[uid] = 1.0 if 0 in top_k_idx else 0.0

            probs = 1.0 / (1.0 + np.exp(-scores))
            for local_idx in top_k_idx:
                exposure[eval_items[local_idx]] += 1
            for local_idx, item in enumerate(eval_items):
                quality[item] += probs[local_idx]

        mean_ndcg = np.mean(list(ndcg_scores.values())) if ndcg_scores else 0.0
        mean_hit = np.mean(list(hit_scores.values())) if hit_scores else 0.0
        observed = quality > 0
        if np.any(observed):
            eq_ratio = exposure[observed] / (quality[observed] + 1e-10)
            dif_val = np.mean((eq_ratio - eq_ratio.mean()) ** 2)
        else:
            dif_val = 0.0
        return ndcg_scores, mean_ndcg, mean_hit, dif_val

    def _score_user_items(self, uid, item_ids, user_embeddings=None, item_embeddings=None, model=None):
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        user_tensor = torch.LongTensor([uid] * len(item_ids)).to(self.device)
        with torch.no_grad():
            if model is not None:
                scores = model.forward(user_tensor, item_tensor)
                return scores.detach().cpu().numpy()
            u = user_embeddings[uid].to(self.device)
            items = item_embeddings[item_ids].to(self.device)
            return (items @ u).detach().cpu().numpy()

    def _compute_all_scores(self, user_embeddings, item_embeddings, model=None):
        if model is not None and hasattr(model, 'compute_all_scores'):
            with torch.no_grad():
                return model.compute_all_scores(self.device).cpu().numpy()
        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.from_numpy(user_embeddings).float()
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float()
        with torch.no_grad():
            return (user_embeddings @ item_embeddings.T).cpu().numpy()
