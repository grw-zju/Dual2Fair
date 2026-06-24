import numpy as np
import torch


class CPFair:
    """
    CPFair: Personalized Consumer and Producer Fairness re-ranking.
    Post-processing method that directly modifies recommendation results.
    
    Ref: Naghiaei et al., "CPFair: Personalized consumer and producer fairness
    re-ranking for recommender systems", SIGIR 2022.
    """
    
    def __init__(self, alpha=0.5, k=10, utility_weight=0.8, device=None):
        self.alpha = alpha
        self.k = k
        self.utility_weight = utility_weight
        self.device = device or torch.device('cpu')
    
    def rerank(self, user_embeddings, item_embeddings, test_dict, train_user_items,
               item_groups, k=10):
        """
        Re-rank to balance user utility (consumer) and item exposure (producer).
        
        Args:
            item_groups: dict item_id -> 'hot' or 'cold'
        """
        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.from_numpy(user_embeddings).float().to(self.device)
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float().to(self.device)
        
        reranked_lists = {}
        
        n_items = item_embeddings.shape[0]
        cold_mask = np.zeros(n_items, dtype=bool)
        for i, g in item_groups.items():
            if g == 'cold' and i < n_items:
                cold_mask[i] = True
        exposure_counter = np.zeros(n_items, dtype=np.float32)
        
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
            
            reranked = self._cp_rerank(ranked_items, scores, exposure_counter, cold_mask, k)
            reranked_lists[uid] = reranked[:k]
            
            for item in reranked[:k]:
                exposure_counter[item] += 1
        
        return reranked_lists
    
    def _cp_rerank(self, ranked_items, scores, exposure_counter, cold_mask, k):
        """
        Greedy utility-preserving re-ranking with producer exposure compensation.
        """
        result = []
        remaining = np.array(ranked_items[:max(10 * k, k)], dtype=np.int64)
        finite_scores = scores[remaining][np.isfinite(scores[remaining])]
        if len(finite_scores) == 0:
            return ranked_items[:k]
        max_score = np.nanmax(finite_scores)
        min_score = np.nanmin(finite_scores)
        score_span = max(max_score - min_score, 1e-8)

        for pos in range(k):
            if len(remaining) == 0:
                break
            desired_cold = (pos + 1) * self.alpha
            current_cold = sum(1 for item in result if cold_mask[item])
            cold_deficit = max(0.0, desired_cold - current_cold)
            mean_exposure = float(exposure_counter.mean()) + 1e-8

            relevance = (scores[remaining] - min_score) / score_span
            exposure_bonus = np.maximum(0.0, mean_exposure - exposure_counter[remaining]) / mean_exposure
            producer_bonus = exposure_bonus + cold_deficit * cold_mask[remaining].astype(np.float32)
            rank_penalty = np.arange(len(remaining), dtype=np.float32) / max(1, len(remaining))
            cp_scores = self.utility_weight * relevance + (1 - self.utility_weight) * producer_bonus - 0.01 * rank_penalty
            best_pos = int(np.argmax(cp_scores))
            best_item = int(remaining[best_pos])
            result.append(best_item)
            remaining = np.delete(remaining, best_pos)
        
        if len(result) < k:
            seen = set(result)
            result.extend([item for item in ranked_items if item not in seen][:k - len(result)])
        return result
