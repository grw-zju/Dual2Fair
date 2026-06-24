import torch
import torch.nn as nn
from .utils import soft_exposure_disparity


class FairDual:
    """
    FairDual: Bridging Jensen gap for max-min group fairness optimization.
    Reformulates group max-min fairness objective into dual form and learns
    per-group shadow prices via mini-batch dual optimization.
    
    Ref: Xu et al., "Bridging Jensen gap for max-min group fairness optimization
    in recommendation", ICLR 2025.
    """
    
    OFFICIAL_REPO = "external_baselines/FairDual"

    def __init__(self, n_groups=2, lambda_dual=0.1, device=None):
        self.n_groups = n_groups
        self.lambda_dual = lambda_dual
        self.device = device or torch.device('cpu')
        self.shadow_prices = nn.Parameter(torch.ones(n_groups, device=self.device))
    
    def compute_dual_loss(self, group_ndcg_scores, group_exposure_scores):
        """
        Compute dual fairness loss.
        
        L_dual = Σ_g λ_g * (NDCG_g - exposure_g)
        
        where λ_g are shadow prices (learnable parameters)
        """
        ndcg_tensor = torch.tensor(group_ndcg_scores, device=self.device)
        exposure_tensor = torch.tensor(group_exposure_scores, device=self.device)
        
        fairness_gap = torch.abs(ndcg_tensor - exposure_tensor)
        
        dual_loss = torch.sum(self.shadow_prices * fairness_gap)
        
        return self.lambda_dual * dual_loss
    
    def compute_item_fairness_loss(self, item_embeddings, item_freq, n_items):
        """
        Group-based item fairness: balance exposure between hot and cold items.
        """
        freq_tensor = torch.tensor([item_freq.get(i, 0) for i in range(n_items)],
                                   device=item_embeddings.device, dtype=torch.float32)
        sorted_indices = torch.argsort(freq_tensor, descending=True)
        n_hot = max(1, int(n_items * 0.2))
        hot_items = sorted_indices[:n_hot]
        cold_items = sorted_indices[n_hot:]
        if len(cold_items) == 0:
            return item_embeddings.new_tensor(0.0)

        hot_score = item_embeddings[hot_items].norm(dim=1).mean()
        cold_score = item_embeddings[cold_items].norm(dim=1).mean()
        group_gap = torch.stack([hot_score, cold_score])
        prices = torch.softmax(self.shadow_prices, dim=0)
        fairness_loss = torch.sum(prices * torch.abs(group_gap - group_gap.mean()))
        return self.lambda_dual * fairness_loss

    def compute_exposure_fairness_loss(self, user_embeddings, item_embeddings, item_freq, n_items):
        disparity = soft_exposure_disparity(
            user_embeddings.to(self.device), item_embeddings.to(self.device),
            item_freq, n_items, self.device)
        return self.lambda_dual * torch.relu(self.shadow_prices).mean() * disparity
