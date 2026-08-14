import numpy as np


def ndcg_at_k(ranked_items, relevant_items, k=10):
    relevant = set(relevant_items)
    dcg = sum(1.0 / np.log2(rank + 2)
              for rank, item in enumerate(ranked_items[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(rank + 2)
                for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def hit_ratio_at_k(ranked_items, relevant_items, k=10):
    relevant = set(relevant_items)
    return float(any(item in relevant for item in ranked_items[:k]))


def compute_duf(ndcg_scores):
    values = np.asarray(list(ndcg_scores.values()), dtype=np.float64)
    return float(np.var(values)) if values.size else 0.0


def compute_rank_dif(exposure, cumulative_relevance, evaluated_mask, eps0=1e-8):
    mask = np.asarray(evaluated_mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    ratios = np.asarray(exposure, dtype=np.float64)[mask] / (
        np.asarray(cumulative_relevance, dtype=np.float64)[mask] + eps0)
    return float(np.var(ratios))


def rank_relevance_from_order(ranked_items, n_items=None):
    ranked = np.asarray(ranked_items, dtype=np.int64)
    count = len(ranked)
    width = int(n_items) if n_items is not None else (int(ranked.max()) + 1 if count else 0)
    relevance = np.zeros(width, dtype=np.float64)
    if count:
        relevance[ranked] = (count - np.arange(1, count + 1) + 1) / count
    return relevance


def compute_uif(ndcg_scores, dif_value, baseline_duf, baseline_dif,
                 w1=0.5, w2=0.5, eps0=1e-8):
    if baseline_duf is None or baseline_dif is None:
        raise ValueError('Split-specific Standard DUF/DIF references are required')
    if not np.isclose(w1 + w2, 1.0):
        raise ValueError('UIF weights must sum to one')
    mean_ndcg = float(np.mean(list(ndcg_scores.values()))) if ndcg_scores else 0.0
    return float((w1 * compute_duf(ndcg_scores) / (baseline_duf + eps0)
                  + w2 * dif_value / (baseline_dif + eps0))
                 / (mean_ndcg + eps0))


def average_per_run_uif(run_metrics):
    return float(np.mean([run['UIF'] for run in run_metrics]))


def compute_dif_affine_invariant(all_scores, test_dict, train_user_items, k=10):
    n_users, n_items = all_scores.shape
    exposure = np.zeros(n_items)
    relevance = np.zeros(n_items)
    evaluated = np.zeros(n_items, dtype=bool)
    for user in test_dict:
        candidates = [item for item in range(n_items)
                      if item not in train_user_items.get(user, set())]
        scores = np.asarray(all_scores[user, candidates])
        order = np.lexsort((np.asarray(candidates), -scores))
        ranked = np.asarray(candidates)[order]
        exposure[ranked[:k]] += 1
        count = len(ranked)
        relevance[ranked] += (count - np.arange(1, count + 1) + 1) / count
        evaluated[ranked] = True
    return compute_rank_dif(exposure, relevance, evaluated)


compute_dif_rank = compute_dif_affine_invariant
