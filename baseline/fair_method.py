import torch
import numpy as np
from .utils import group_mean_gap, soft_exposure_disparity


class FAIRMethod:
    """
    FAIR: Interpolating item and user fairness in multi-sided recommendations.
    Online algorithm with sub-linear regret that relaxes two-sided fairness constraints.
    
    Ref: Chen et al., "Interpolating item and user fairness in multi-sided 
    recommendations", NeurIPS 2024.
    """
    
    def __init__(self, lambda_user=0.1, lambda_item=0.1, eta=0.01, device=None):
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.eta = eta
        self.device = device or torch.device('cpu')
        self.cumulative_user_fairness = 0.0
        self.cumulative_item_fairness = 0.0
    
    def compute_online_fairness_loss(self, user_embeddings, item_embeddings,
                                      adv_users, disadv_users, item_freq, n_items, k=10):
        """
        Online fairness loss with sub-linear regret bounds.
        Uses cumulative fairness violations as penalties.
        """
        user_embeddings = user_embeddings.to(self.device)
        item_embeddings = item_embeddings.to(self.device)

        user_gap = group_mean_gap(user_embeddings, adv_users, disadv_users)
        
        self.cumulative_user_fairness += user_gap.item()
        
        item_gap = soft_exposure_disparity(
            user_embeddings, item_embeddings, item_freq, n_items, self.device)
        
        self.cumulative_item_fairness += item_gap.item()
        
        # Online penalty based on cumulative violations
        user_penalty = self.eta * min(self.cumulative_user_fairness, 100.0) * user_gap
        item_penalty = self.eta * min(self.cumulative_item_fairness, 100.0) * item_gap
        
        total_loss = self.lambda_user * user_gap + user_penalty + \
                     self.lambda_item * item_gap + item_penalty
        
        return total_loss
