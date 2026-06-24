import numpy as np
import torch
from sklearn.metrics import ndcg_score as sklearn_ndcg_score, roc_auc_score


def ndcg_at_k(predicted_items, true_items, k=10):
    dcg = 0.0
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = 0.0
    n_relevant = min(len(true_items), k)
    for i in range(n_relevant):
        ideal_dcg += 1.0 / np.log2(i + 2)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def hit_ratio_at_k(predicted_items, true_items, k=10):
    for item in predicted_items[:k]:
        if item in true_items:
            return 1.0
    return 0.0


def compute_ndcg_full(all_scores, test_dict, train_user_items, k=10):
    ndcg_scores, hit_scores = {}, {}
    for uid, test_items in test_dict.items():
        if isinstance(test_items, (int, np.integer)):
            test_items = {int(test_items)}
        elif not isinstance(test_items, set):
            test_items = set(test_items)
        scores = all_scores[uid].copy()
        train_items = train_user_items.get(uid, set())
        for ti in train_items:
            if ti < len(scores):
                scores[ti] = -np.inf
        ranked = np.argsort(-scores)
        ndcg_scores[uid] = ndcg_at_k(ranked, test_items, k)
        hit_scores[uid] = hit_ratio_at_k(ranked, test_items, k)
    mean_ndcg = np.mean(list(ndcg_scores.values())) if ndcg_scores else 0.0
    mean_hit = np.mean(list(hit_scores.values())) if hit_scores else 0.0
    return ndcg_scores, mean_ndcg, mean_hit


def compute_duf(ndcg_scores):
    values = list(ndcg_scores.values())
    if len(values) == 0:
        return 0.0
    mean_ndcg = np.mean(values)
    return np.mean([(v - mean_ndcg) ** 2 for v in values])


def compute_dif_from_exposure(all_scores, test_dict, train_user_items, k=10):
    epsilon = 1e-10
    n_items = all_scores.shape[1]
    exposure = np.zeros(n_items)
    prob_scores = 1.0 / (1.0 + np.exp(-all_scores))
    quality = np.zeros(n_items)
    for uid in test_dict.keys():
        scores = all_scores[uid].copy()
        train_items = train_user_items.get(uid, set())
        for ti in train_items:
            if ti < len(scores):
                scores[ti] = -np.inf
                prob_scores[uid, ti] = 0.0
        ranked = np.argsort(-scores)[:k]
        for item in ranked:
            exposure[item] += 1
        quality += prob_scores[uid]
    eq_ratio = exposure / (quality + epsilon)
    mean_eq = np.mean(eq_ratio)
    return np.mean([(r - mean_eq) ** 2 for r in eq_ratio])


def compute_dif_from_ranked_lists(ranked_lists, all_scores, test_dict, train_user_items, k=10):
    epsilon = 1e-10
    n_items = all_scores.shape[1]
    exposure = np.zeros(n_items)
    prob_scores = 1.0 / (1.0 + np.exp(-all_scores))
    quality = np.zeros(n_items)

    for uid in test_dict.keys():
        scores = all_scores[uid].copy()
        for ti in train_user_items.get(uid, set()):
            if ti < len(scores):
                prob_scores[uid, ti] = 0.0
        ranked = ranked_lists.get(uid, [])[:k]
        for item in ranked:
            if item < n_items:
                exposure[item] += 1
        quality += prob_scores[uid]

    eq_ratio = exposure / (quality + epsilon)
    mean_eq = np.mean(eq_ratio)
    return np.mean([(r - mean_eq) ** 2 for r in eq_ratio])


def compute_sampled_ndcg(all_scores, test_dict, train_user_items, n_neg=99, k=10,
                         test_neg_items=None):
    ndcg_scores = {}
    hit_scores = {}
    n_items = all_scores.shape[1]

    for uid, test_items in test_dict.items():
        if isinstance(test_items, (int, np.integer)):
            pos_item = int(test_items)
        elif isinstance(test_items, set):
            pos_item = list(test_items)[0]
        else:
            pos_item = list(test_items)[0]

        if test_neg_items is not None and uid in test_neg_items:
            neg_items = test_neg_items[uid]
        else:
            test_items_set = {pos_item} if isinstance(test_items, (int, np.integer)) else set(test_items)
            train_set = train_user_items.get(uid, set())
            candidate_pool = list(set(range(n_items)) - train_set - test_items_set)
            if len(candidate_pool) < n_neg:
                neg_items = candidate_pool
            else:
                neg_indices = np.random.choice(len(candidate_pool), size=n_neg, replace=False)
                neg_items = [candidate_pool[i] for i in neg_indices]

        eval_items = [pos_item] + list(neg_items)
        true_relevance = np.zeros(len(eval_items))
        true_relevance[0] = 1.0

        pred_scores = all_scores[uid, eval_items]

        ndcg_scores[uid] = sklearn_ndcg_score([true_relevance], [pred_scores], k=k)
        top_k_items = np.argsort(-pred_scores)[:k]
        hit_scores[uid] = 1.0 if 0 in top_k_items else 0.0

    mean_ndcg = np.mean(list(ndcg_scores.values())) if ndcg_scores else 0.0
    mean_hit = np.mean(list(hit_scores.values())) if hit_scores else 0.0
    return ndcg_scores, mean_ndcg, mean_hit


def compute_sampled_dif_from_exposure(all_scores, test_dict, train_user_items,
                                      n_neg=99, k=10, test_neg_items=None):
    epsilon = 1e-10
    n_items = all_scores.shape[1]
    exposure = np.zeros(n_items)
    quality = np.zeros(n_items)

    for uid, test_items in test_dict.items():
        if isinstance(test_items, (int, np.integer)):
            pos_item = int(test_items)
        elif isinstance(test_items, set):
            pos_item = list(test_items)[0]
        else:
            pos_item = list(test_items)[0]

        if test_neg_items is not None and uid in test_neg_items:
            neg_items = test_neg_items[uid]
        else:
            test_items_set = {pos_item} if isinstance(test_items, (int, np.integer)) else set(test_items)
            train_set = train_user_items.get(uid, set())
            candidate_pool = list(set(range(n_items)) - train_set - test_items_set)
            if len(candidate_pool) < n_neg:
                neg_items = candidate_pool
            else:
                neg_indices = np.random.choice(len(candidate_pool), size=n_neg, replace=False)
                neg_items = [candidate_pool[i] for i in neg_indices]

        eval_items = [pos_item] + list(neg_items)
        pred_scores = all_scores[uid, eval_items]
        top_k_indices = np.argsort(-pred_scores)[:k]
        prob_scores = 1.0 / (1.0 + np.exp(-pred_scores))

        for local_idx in top_k_indices:
            exposure[eval_items[local_idx]] += 1
        for local_idx, item in enumerate(eval_items):
            quality[item] += prob_scores[local_idx]

    observed = quality > 0
    if not np.any(observed):
        return 0.0
    eq_ratio = exposure[observed] / (quality[observed] + epsilon)
    mean_eq = np.mean(eq_ratio)
    return np.mean([(r - mean_eq) ** 2 for r in eq_ratio])


def compute_uif(ndcg_scores, dif_value, w1=0.5, w2=0.5,
                 baseline_duf=None, baseline_dif=None):
    duf_value = compute_duf(ndcg_scores)
    mean_ndcg = np.mean(list(ndcg_scores.values())) if ndcg_scores else 0.0
    if mean_ndcg == 0:
        return 1e12
    duf_normalized = duf_value / baseline_duf if baseline_duf and baseline_duf > 0 else duf_value
    dif_normalized = dif_value / baseline_dif if baseline_dif and baseline_dif > 0 else dif_value
    return (w1 * duf_normalized + w2 * dif_normalized) / mean_ndcg
