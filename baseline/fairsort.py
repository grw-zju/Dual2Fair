import numpy as np
import torch


class FairSort:
    """
    FairSort: Learning to fair rank for personalized recommendations in two-sided platforms.
    Analogizes each recommendation list to a runway and employs binary-search
    velocity assignment to guarantee minimum user utility while balancing provider exposure.
    
    Ref: Wu et al., "FairSort: Learning to fair rank for personalized recommendations 
    in two-sided platforms", TKDE 2025.
    """
    
    def __init__(self, k=10, min_utility=0.3, search_steps=6, device=None):
        self.k = k
        self.min_utility = min_utility
        self.search_steps = search_steps
        self.device = device or torch.device('cpu')
    
    def rerank(self, user_embeddings, item_embeddings, test_dict, train_user_items, k=10):
        """
        FairSort re-ranking using binary-search velocity assignment.
        
        Each item gets a "velocity" (position weight) that balances
        user utility (relevance) with producer exposure (fairness).
        """
        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.from_numpy(user_embeddings).float().to(self.device)
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float().to(self.device)
        
        reranked_lists = {}
        
        item_exposure = np.zeros(item_embeddings.shape[0])
        
        for uid in test_dict.keys():
            with torch.no_grad():
                score_tensor = item_embeddings @ user_embeddings[uid]
            train_items = train_user_items.get(uid, set())
            if train_items:
                idx = torch.LongTensor([i for i in train_items if i < score_tensor.shape[0]]).to(score_tensor.device)
                score_tensor[idx] = -torch.inf
            window = min(max(10 * k, k), score_tensor.shape[0])
            top_scores, top_idx = torch.topk(score_tensor, k=window)
            ranked_items = top_idx.cpu().numpy().tolist()
            scores = np.full(score_tensor.shape[0], -np.inf, dtype=np.float32)
            scores[top_idx.cpu().numpy()] = top_scores.cpu().numpy()
            reranked = self._fair_sort(ranked_items, scores, item_exposure, k)
            reranked_lists[uid] = reranked
            
            for pos, item in enumerate(reranked):
                item_exposure[item] += 1.0 / (pos + 1)
        
        return reranked_lists
    
    def _compute_velocities(self, ranked_items, item_exposure, strength, k):
        """
        Binary-search velocity assignment for each item.
        Items with lower exposure get higher velocity (earlier position).
        """
        velocities = np.ones(len(ranked_items))
        
        target_exp = item_exposure.mean() + 1e-10
        for idx, item in enumerate(ranked_items):
            current_exp = item_exposure[item]
            if current_exp < target_exp:
                velocities[idx] = 1.0 + strength * (target_exp - current_exp) / target_exp
            else:
                velocities[idx] = 1.0
        
        return velocities
    
    def _fair_sort(self, items, scores, item_exposure, k):
        base_list = items[:k]
        base_utility = self._utility(base_list, scores, items[:max(10 * k, k)])
        min_allowed = self.min_utility * base_utility

        best = base_list
        lo, hi = 0.0, 2.0
        window = items[:max(10 * k, k)]
        for _ in range(self.search_steps):
            mid = (lo + hi) / 2.0
            velocities = self._compute_velocities(window, item_exposure, mid, k)
            candidate = self._velocity_sort(window, scores, velocities, k)
            if self._utility(candidate, scores, window) >= min_allowed:
                best = candidate
                lo = mid
            else:
                hi = mid
        return best

    def _velocity_sort(self, items, scores, velocities, k):
        """
        Sort items by velocity-adjusted scores.
        """
        item_scores = []
        finite_scores = np.array([scores[item] for item in items if np.isfinite(scores[item])])
        min_score = finite_scores.min() if len(finite_scores) else 0.0
        max_score = finite_scores.max() if len(finite_scores) else 1.0
        span = max(max_score - min_score, 1e-8)
        for idx, item in enumerate(items):
            if idx < len(velocities):
                relevance = (scores[item] - min_score) / span
                adjusted_score = relevance * velocities[idx]
            else:
                adjusted_score = scores[item]
            item_scores.append((item, adjusted_score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        reranked = [item for item, _ in item_scores[:k]]
        return reranked

    def _utility(self, items, scores, reference_items):
        finite_scores = np.array([scores[item] for item in reference_items if np.isfinite(scores[item])])
        min_score = finite_scores.min() if len(finite_scores) else 0.0
        max_score = finite_scores.max() if len(finite_scores) else 1.0
        span = max(max_score - min_score, 1e-8)
        utility = 0.0
        for pos, item in enumerate(items):
            if np.isfinite(scores[item]):
                utility += ((scores[item] - min_score) / span) / np.log2(pos + 2)
        return max(utility, 1e-8)
