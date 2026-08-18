import torch
import torch.nn as nn
from .utils import group_mean_gap, soft_exposure_disparity


class MultiFR:
    """
    MultiFR: Multi-objective Optimization for Multi-stakeholder Fairness-aware Recommendation.
    In-processing method that balances accuracy, user fairness, and item fairness
    through multi-objective optimization.
    
    Ref: Wu et al., "A multi-objective optimization framework for multi-stakeholder
    fairness-aware recommendation", TOIS 2023.
    """
    
    def __init__(self, lambda_user=0.1, lambda_item=0.1, n_users=0, n_items=0,
                 embedding_dim=64, device=None):
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.device = device or torch.device('cpu')
    
    def compute_user_fairness_loss(self, user_embeddings, adv_users, disadv_users):
        """
        Group disparity loss: minimize the gap between advantaged and
        disadvantaged user group performance.
        """
        return self.lambda_user * group_mean_gap(user_embeddings, adv_users, disadv_users)
    
    def compute_item_fairness_loss(self, item_embeddings, item_freq, n_items):
        """
        Item exposure fairness: minimize exposure disparity between hot and cold items.
        """
        # If user embeddings are unavailable, fall back to hot/cold representation gap.
        hot_items = torch.argsort(torch.tensor(
            [item_freq.get(i, 0) for i in range(n_items)], device=item_embeddings.device),
            descending=True)[:max(1, int(n_items * 0.2))]
        cold_mask = torch.ones(n_items, dtype=torch.bool, device=item_embeddings.device)
        cold_mask[hot_items] = False
        cold_items = torch.arange(n_items, device=item_embeddings.device)[cold_mask]
        if len(cold_items) == 0:
            return item_embeddings.new_tensor(0.0)
        return self.lambda_item * torch.norm(
            item_embeddings[hot_items].mean(dim=0) - item_embeddings[cold_items].mean(dim=0),
            p=2).pow(2)

    def compute_exposure_fairness_loss(self, user_embeddings, item_embeddings, item_freq, n_items):
        return self.lambda_item * soft_exposure_disparity(
            user_embeddings, item_embeddings, item_freq, n_items, self.device)
    
    def compute_total_loss(self, rec_loss, user_embeddings, item_embeddings,
                           adv_users, disadv_users, item_freq, n_items):
        """
        L_total = L_rec + λ1 * L_user + λ2 * L_item
        """
        user_fair_loss = self.compute_user_fairness_loss(user_embeddings, adv_users, disadv_users)
        item_fair_loss = self.compute_item_fairness_loss(item_embeddings, item_freq, n_items)
        
        total_loss = rec_loss + user_fair_loss + item_fair_loss
        return total_loss, user_fair_loss, item_fair_loss
