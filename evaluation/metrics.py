import numpy as np
from sklearn.metrics import ndcg_score as sklearn_ndcg_score


def ndcg_at_k(predicted_items, true_items, k=10):
    gains = [1.0 / np.log2(index + 2) for index, item in enumerate(predicted_items[:k])
             if item in true_items]
    ideal = sum(1.0 / np.log2(index + 2) for index in range(min(len(true_items), k)))
    return sum(gains) / ideal if ideal else 0.0


def hit_ratio_at_k(predicted_items, true_items, k=10):
    return float(any(item in true_items for item in predicted_items[:k]))


def compute_ndcg_full(all_scores, test_dict, train_user_items, k=10):
    ndcg_scores, hit_scores = {}, {}
    for uid, test_items in test_dict.items():
        relevant = {int(test_items)} if isinstance(test_items, (int, np.integer)) else set(test_items)
        scores = np.asarray(all_scores[uid]).copy()
        for item in train_user_items.get(uid, set()):
            if item < len(scores):
                scores[item] = -np.inf
        ranked = np.argsort(-scores, kind='mergesort')
        ndcg_scores[uid] = ndcg_at_k(ranked, relevant, k)
        hit_scores[uid] = hit_ratio_at_k(ranked, relevant, k)
    return (ndcg_scores,
            float(np.mean(list(ndcg_scores.values()))) if ndcg_scores else 0.0,
            float(np.mean(list(hit_scores.values()))) if hit_scores else 0.0)


def compute_duf(ndcg_scores):
    values = np.asarray(list(ndcg_scores.values()), dtype=np.float64)
    return float(np.var(values)) if values.size else 0.0


def _candidate_items(uid, n_items, train_user_items):
    blocked = train_user_items.get(uid, set())
    return np.asarray([item for item in range(n_items) if item not in blocked], dtype=np.int64)


def _percentile_quality(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        return np.ones_like(scores)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(scores.size, dtype=np.float64)
    return (ranks + 1.0) / scores.size


def _rank_quality(scores):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind='mergesort')
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    return 1.0 / np.log2(ranks + 1.0)


def _dif_from_quality(all_scores, test_dict, train_user_items, k, quality_fn):
    n_items = all_scores.shape[1]
    exposure = np.zeros(n_items, dtype=np.float64)
    quality = np.zeros(n_items, dtype=np.float64)
    for uid in sorted(test_dict):
        candidates = _candidate_items(uid, n_items, train_user_items)
        if candidates.size == 0:
            continue
        scores = np.asarray(all_scores[uid, candidates], dtype=np.float64)
        top_indices = np.argsort(-scores, kind='mergesort')[:k]
        exposure[candidates[top_indices]] += 1.0
        quality[candidates] += quality_fn(scores)
    observed = quality > 0
    if not np.any(observed):
        return 0.0
    ratios = exposure[observed] / quality[observed]
    return float(np.var(ratios))


def compute_dif_affine_invariant(all_scores, test_dict, train_user_items, k=10):
    return _dif_from_quality(all_scores, test_dict, train_user_items, k, _percentile_quality)


def compute_dif_rank(all_scores, test_dict, train_user_items, k=10):
    return _dif_from_quality(all_scores, test_dict, train_user_items, k, _rank_quality)


def compute_dif_legacy(all_scores, test_dict, train_user_items, k=10):
    def sigmoid_quality(scores):
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    return _dif_from_quality(all_scores, test_dict, train_user_items, k, sigmoid_quality)


def compute_dif_from_exposure(all_scores, test_dict, train_user_items, k=10):
    return compute_dif_legacy(all_scores, test_dict, train_user_items, k)


def compute_dif_from_ranked_lists(ranked_lists, all_scores, test_dict, train_user_items, k=10):
    n_items = all_scores.shape[1]
    exposure = np.zeros(n_items, dtype=np.float64)
    quality = np.zeros(n_items, dtype=np.float64)
    for uid in sorted(test_dict):
        candidates = _candidate_items(uid, n_items, train_user_items)
        if candidates.size == 0:
            continue
        quality[candidates] += _percentile_quality(all_scores[uid, candidates])
        ranked = [item for item in ranked_lists.get(uid, [])[:k] if item < n_items]
        exposure[ranked] += 1.0
    observed = quality > 0
    return float(np.var(exposure[observed] / quality[observed])) if np.any(observed) else 0.0


def compute_sampled_ndcg(all_scores, test_dict, train_user_items, n_neg=99, k=10,
                         test_neg_items=None):
    ndcg_scores, hit_scores = {}, {}
    n_items = all_scores.shape[1]
    rng = np.random.RandomState(42)
    for uid, test_items in test_dict.items():
        positive = int(test_items) if isinstance(test_items, (int, np.integer)) else int(next(iter(test_items)))
        negatives = test_neg_items.get(uid) if test_neg_items is not None else None
        if negatives is None:
            pool = sorted(set(range(n_items)) - train_user_items.get(uid, set()) - {positive})
            negatives = pool if len(pool) <= n_neg else [pool[index] for index in rng.choice(len(pool), n_neg, replace=False)]
        items = [positive] + list(negatives)
        relevance = np.zeros(len(items), dtype=np.float64)
        relevance[0] = 1.0
        scores = np.asarray(all_scores[uid, items], dtype=np.float64)
        ndcg_scores[uid] = float(sklearn_ndcg_score([relevance], [scores], k=k))
        hit_scores[uid] = float(0 in np.argsort(-scores, kind='mergesort')[:k])
    return (ndcg_scores,
            float(np.mean(list(ndcg_scores.values()))) if ndcg_scores else 0.0,
            float(np.mean(list(hit_scores.values()))) if hit_scores else 0.0)


def compute_sampled_dif_from_exposure(all_scores, test_dict, train_user_items,
                                      n_neg=99, k=10, test_neg_items=None,
                                      mode='affine_invariant'):
    n_items = all_scores.shape[1]
    exposure = np.zeros(n_items, dtype=np.float64)
    quality = np.zeros(n_items, dtype=np.float64)
    rng = np.random.RandomState(42)
    quality_fn = {'legacy': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))),
                  'affine_invariant': _percentile_quality,
                  'rank': _rank_quality}[mode]
    for uid, test_items in test_dict.items():
        positive = int(test_items) if isinstance(test_items, (int, np.integer)) else int(next(iter(test_items)))
        negatives = test_neg_items.get(uid) if test_neg_items is not None else None
        if negatives is None:
            pool = sorted(set(range(n_items)) - train_user_items.get(uid, set()) - {positive})
            negatives = pool if len(pool) <= n_neg else [pool[index] for index in rng.choice(len(pool), n_neg, replace=False)]
        items = np.asarray([positive] + list(negatives), dtype=np.int64)
        scores = np.asarray(all_scores[uid, items], dtype=np.float64)
        exposure[items[np.argsort(-scores, kind='mergesort')[:k]]] += 1.0
        quality[items] += quality_fn(scores)
    observed = quality > 0
    return float(np.var(exposure[observed] / quality[observed])) if np.any(observed) else 0.0


def exposure_gini(exposure):
    values = np.sort(np.asarray(exposure, dtype=np.float64))
    if values.size == 0 or values.sum() == 0:
        return 0.0
    indices = np.arange(1, values.size + 1)
    return float((2.0 * np.sum(indices * values) / (values.size * values.sum()))
                 - (values.size + 1.0) / values.size)


def exposure_coefficient_of_variation(exposure):
    values = np.asarray(exposure, dtype=np.float64)
    mean = values.mean() if values.size else 0.0
    return float(values.std() / mean) if mean > 0 else 0.0


def catalog_coverage(exposure):
    values = np.asarray(exposure)
    return float(np.mean(values > 0)) if values.size else 0.0


def head_mid_tail_exposure(exposure, item_frequency):
    values = np.asarray(exposure, dtype=np.float64)
    order = sorted(range(len(values)), key=lambda item: (-item_frequency.get(item, 0), item))
    n_items = len(order)
    groups = {
        'head': order[:max(1, int(0.2 * n_items))],
        'mid': order[max(1, int(0.2 * n_items)):max(1, int(0.8 * n_items))],
        'tail': order[max(1, int(0.8 * n_items)):max(1, int(0.95 * n_items))],
        'extreme_tail': order[max(1, int(0.95 * n_items)):],
    }
    return {name: float(values[indices].sum()) if indices else 0.0
            for name, indices in groups.items()}


def compute_uif(ndcg_scores, dif_value, w1=0.5, w2=0.5,
                 baseline_duf=None, baseline_dif=None, require_baseline=False):
    if w1 < 0 or w2 < 0 or not np.isclose(w1 + w2, 1.0):
        raise ValueError('UIF weights must be nonnegative and sum to one')
    if require_baseline and (baseline_duf is None or baseline_dif is None):
        raise ValueError('Frozen validation-derived UIF constants are required')
    duf_value = compute_duf(ndcg_scores)
    mean_ndcg = float(np.mean(list(ndcg_scores.values()))) if ndcg_scores else 0.0
    if mean_ndcg <= 0:
        return 1e12
    normalized_duf = duf_value / baseline_duf if baseline_duf is not None and baseline_duf > 0 else duf_value
    normalized_dif = dif_value / baseline_dif if baseline_dif is not None and baseline_dif > 0 else dif_value
    return float((w1 * normalized_duf + w2 * normalized_dif) / mean_ndcg)
