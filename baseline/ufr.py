import numpy as np
import torch


class UFR:
    """
    User-oriented Fairness via post-processing re-ranking.
    Maximizes utility while bounding inter-group performance gaps.
    
    Ref: Li et al., "User-oriented fairness in recommendation", WWW 2021.
    """
    
    OFFICIAL_REPO = "external_baselines/user-fairness"

    def __init__(self, k=10, delta=0.05, fairness_metric='hit'):
        self.k = k
        self.delta = delta
        self.fairness_metric = fairness_metric
    
    def rerank(self, user_embeddings, item_embeddings, test_dict, train_user_items, 
               user_groups, k=10, device=None):
        """
        Re-rank recommendations to bound performance gap between user groups.
        
        Args:
            user_embeddings: (n_users, d)
            item_embeddings: (n_items, d)
            test_dict: dict user_id -> test_item_id
            train_user_items: dict user_id -> set of train items
            user_groups: dict mapping user_id -> 'advantaged' or 'disadvantaged'
            k: top-K
        
        Returns:
            reranked_lists: dict user_id -> list of reranked item_ids
        """
        if device is None:
            device = torch.device('cpu')
        
        if isinstance(user_embeddings, np.ndarray):
            user_embeddings = torch.from_numpy(user_embeddings).float().to(device)
        if isinstance(item_embeddings, np.ndarray):
            item_embeddings = torch.from_numpy(item_embeddings).float().to(device)
        
        reranked_lists = {}
        
        adv_users = [u for u, g in user_groups.items() if g == 'advantaged']
        disadv_users = [u for u, g in user_groups.items() if g == 'disadvantaged']
        all_scores_for_gurobi = {}
        item_pop = {}
        for items in train_user_items.values():
            for item in items:
                item_pop[item] = item_pop.get(item, 0) + 1
        
        for uid in test_dict.keys():
            with torch.no_grad():
                score_tensor = item_embeddings @ user_embeddings[uid]
                train_items = train_user_items.get(uid, set())
                if train_items:
                    idx = torch.LongTensor([i for i in train_items if i < score_tensor.shape[0]]).to(score_tensor.device)
                    score_tensor[idx] = -torch.inf
                window = min(max(k * 20, k), score_tensor.shape[0])
                top_scores, top_idx = torch.topk(score_tensor, k=window)
                ranked_items = top_idx.cpu().numpy().tolist()
                scores = np.full(score_tensor.shape[0], -np.inf, dtype=np.float32)
                scores[top_idx.cpu().numpy()] = top_scores.cpu().numpy()
            all_scores_for_gurobi[uid] = scores

            reranked = self._fair_rerank(uid, ranked_items, scores, item_pop, user_groups, k)
            reranked_lists[uid] = reranked[:k]

        optimized = self._try_gurobi_optimize(reranked_lists, all_scores_for_gurobi,
                                              test_dict, user_groups, k)
        if optimized is not None:
            return optimized
        
        return reranked_lists

    def _try_gurobi_optimize(self, reranked_lists, all_scores, test_dict, user_groups, k):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return None

        users = list(test_dict.keys())
        model = gp.Model("UFR")
        model.Params.OutputFlag = 0
        q = {}
        objective_terms = []
        metric_terms = {'advantaged': [], 'disadvantaged': []}

        for uid in users:
            candidates = reranked_lists[uid][:max(k, len(reranked_lists[uid]))]
            true_item = int(test_dict[uid]) if not isinstance(test_dict[uid], (set, list, tuple)) else list(test_dict[uid])[0]
            vars_for_user = []
            for item in candidates:
                var = model.addVar(vtype=GRB.BINARY, name=f"{uid}_{item}")
                q[(uid, item)] = var
                vars_for_user.append(var)
                objective_terms.append(float(all_scores[uid][item]) * var)
                if item == true_item:
                    metric_terms[user_groups.get(uid, 'disadvantaged')].append(var)
            model.addConstr(gp.quicksum(vars_for_user) == min(k, len(candidates)))

        adv_count = max(1, sum(1 for u in users if user_groups.get(u) == 'advantaged'))
        disadv_count = max(1, sum(1 for u in users if user_groups.get(u) == 'disadvantaged'))
        adv_metric = gp.quicksum(metric_terms['advantaged']) / adv_count
        disadv_metric = gp.quicksum(metric_terms['disadvantaged']) / disadv_count
        model.addConstr(adv_metric - disadv_metric <= self.delta)
        model.addConstr(disadv_metric - adv_metric <= self.delta)
        model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            return None

        optimized = {}
        for uid in users:
            selected = []
            for item in reranked_lists[uid]:
                var = q.get((uid, item))
                if var is not None and var.X > 0.5:
                    selected.append(item)
            if len(selected) < k:
                selected.extend([item for item in reranked_lists[uid] if item not in selected][:k - len(selected)])
            optimized[uid] = selected[:k]
        return optimized
    
    def _fair_rerank(self, uid, ranked_items, scores, item_pop, user_groups, k):
        """
        Utility-preserving re-ranking for disadvantaged users.
        """
        if user_groups.get(uid) != 'disadvantaged':
            return ranked_items

        candidate_window = ranked_items[:max(k * 5, k)]
        max_pop = max(item_pop.values()) if item_pop else 1

        adjusted = []
        for rank, item in enumerate(candidate_window):
            relevance = scores[item]
            novelty = 1.0 - (item_pop.get(item, 0) / max_pop)
            utility_guard = 1.0 / np.log2(rank + 2)
            adjusted_score = relevance + self.delta * novelty * utility_guard
            adjusted.append((item, adjusted_score))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        reranked = [item for item, _ in adjusted]
        seen = set(reranked)
        reranked.extend([item for item in ranked_items if item not in seen])
        return reranked
