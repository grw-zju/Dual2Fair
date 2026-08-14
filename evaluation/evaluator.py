import numpy as np
import torch

from .metrics import (compute_duf, compute_rank_dif, compute_uif,
                      hit_ratio_at_k, ndcg_at_k)


class Evaluator:
    def __init__(self, dataset, k=10, device=None, split='test',
                 user_batch_size=64, baseline_duf=None, baseline_dif=None,
                 uif_w1=0.5, uif_w2=0.5, eps0=1e-8,
                 require_uif_reference=False, **deprecated):
        self.dataset = dataset
        self.k = int(k)
        self.device = device or torch.device('cpu')
        self.split = split
        self.user_batch_size = int(user_batch_size)
        self.baseline_duf = baseline_duf
        self.baseline_dif = baseline_dif
        self.uif_w1 = float(uif_w1)
        self.uif_w2 = float(uif_w2)
        self.eps0 = float(eps0)
        self.require_uif_reference = bool(require_uif_reference)
        self.heldout = (dataset.get_val_dict() if split == 'val'
                        else dataset.get_test_dict())
        self.validation = dataset.get_val_dict()
        self.warm_items = np.asarray(sorted(
            item for item, frequency in dataset.item_freq.items() if frequency > 0),
            dtype=np.int64)

    def set_baseline(self, ndcg, duf, dif):
        self.baseline_duf = duf
        self.baseline_dif = dif

    def candidate_items(self, user):
        excluded = set(self.dataset.user_items.get(user, set()))
        if self.split == 'test' and user in self.validation:
            excluded.add(int(self.validation[user]))
        candidates = [item for item in self.warm_items if item not in excluded]
        positive = self.heldout.get(user)
        if positive is not None and int(positive) in self.warm_items and int(positive) not in candidates:
            candidates.append(int(positive))
        return np.asarray(sorted(candidates), dtype=np.int64)

    def _score_pairs(self, model, user, items):
        item_tensor = torch.as_tensor(items, dtype=torch.long, device=self.device)
        user_tensor = torch.full_like(item_tensor, int(user))
        return model.forward(user_tensor, item_tensor).detach().cpu().numpy()

    def evaluate(self, model=None, user_embeddings=None, item_embeddings=None,
                 require_uif_reference=None, **unused):
        if require_uif_reference is None:
            require_uif_reference = self.require_uif_reference
        if model is None:
            if user_embeddings is None or item_embeddings is None:
                raise ValueError('Model or explicit embeddings are required')
            users = torch.as_tensor(user_embeddings, dtype=torch.float32)
            items = torch.as_tensor(item_embeddings, dtype=torch.float32)

            class DotProductModel:
                def forward(self, user_ids, item_ids):
                    return (users[user_ids.cpu()] * items[item_ids.cpu()]).sum(1)
            model = DotProductModel()
        ndcg_scores, hit_scores = {}, {}
        exposure = np.zeros(self.dataset.n_items, dtype=np.float64)
        relevance = np.zeros(self.dataset.n_items, dtype=np.float64)
        evaluated = np.zeros(self.dataset.n_items, dtype=bool)
        for user in sorted(self.heldout):
            candidates = self.candidate_items(user)
            if not len(candidates):
                continue
            scores = self._score_pairs(model, user, candidates)
            order = np.lexsort((candidates, -scores))
            ranked = candidates[order]
            positive = {int(self.heldout[user])}
            ndcg_scores[user] = ndcg_at_k(ranked, positive, self.k)
            hit_scores[user] = hit_ratio_at_k(ranked, positive, self.k)
            exposure[ranked[:self.k]] += 1.0
            count = len(ranked)
            relevance[ranked] += (count - np.arange(1, count + 1) + 1) / count
            evaluated[ranked] = True
        dif = compute_rank_dif(exposure, relevance, evaluated, self.eps0)
        result = {
            'NDCG': float(np.mean(list(ndcg_scores.values()))) if ndcg_scores else 0.0,
            'Hit': float(np.mean(list(hit_scores.values()))) if hit_scores else 0.0,
            'DUF': compute_duf(ndcg_scores), 'DIF': dif,
            'evaluation_protocol': f'full_warm_start_{self.split}'}
        if self.baseline_duf is not None and self.baseline_dif is not None:
            result['UIF'] = compute_uif(
                ndcg_scores, dif, self.baseline_duf, self.baseline_dif,
                self.uif_w1, self.uif_w2, self.eps0)
        elif require_uif_reference:
            raise ValueError('UIF reference constants are not configured')
        else:
            result['UIF'] = None
        return result

    def sampled_evaluate(self, *args, **kwargs):
        import warnings
        warnings.warn('sampled_evaluate is debug-only and deprecated for paper results',
                      DeprecationWarning, stacklevel=2)
        return self.evaluate(*args, **kwargs)
