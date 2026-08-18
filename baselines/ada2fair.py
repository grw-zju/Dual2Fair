import torch
import torch.nn as nn
from .utils import group_mean_gap, soft_exposure_disparity


class Ada2Fair:
    """
    Ada2Fair: Promoting two-sided fairness with adaptive weights.
    In-processing method that dynamically adjusts optimization through an
    adaptive weight generator, assigning adaptive importance to interaction samples.
    
    Ref: Xu et al., "Promoting two-sided fairness with adaptive weights for 
    providers and customers in recommendation", RecSys 2024.
    """
    
    OFFICIAL_REPO = "external_baselines/Ada2Fair"

    def __init__(self, lambda_user=0.1, lambda_item=0.1, hidden_dim=32,
                 n_users=0, n_items=0, embedding_dim=64, device=None):
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.device = device or torch.device('cpu')
        
        self.weight_generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)
        self.provider_fairness_weight = None
        self.user_fairness_weight = None

    def update_fairness_weights(self, user_embs, item_embs, train_user_items, item_freq,
                                top_k=100, delta=1e-6, provider_eta=1.0, user_eta=1.0,
                                user_batch_size=512):
        with torch.no_grad():
            user_embs = user_embs.to(self.device)
            item_embs = item_embs.to(self.device)
            k = min(top_k, item_embs.shape[0])
            discount = 1.0 / torch.log2(torch.arange(2, 2 + k, device=self.device).float())

            exposure = torch.zeros(item_embs.shape[0], device=self.device)
            user_utility = torch.zeros(user_embs.shape[0], device=self.device)
            for start in range(0, user_embs.shape[0], user_batch_size):
                end = min(start + user_batch_size, user_embs.shape[0])
                scores = user_embs[start:end] @ item_embs.T
                top_items = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
                for local_uid, items in enumerate(top_items):
                    uid = start + local_uid
                    exposure[items] += discount
                    history = train_user_items.get(uid, set())
                    if history:
                        hits = torch.tensor([1.0 if item in history else 0.0 for item in items],
                                            device=self.device)
                        user_utility[uid] = (hits * discount).sum() / len(history)

            provider_weight = 1.0 / (exposure.clamp_min(0).pow(provider_eta) + delta)
            provider_weight = provider_weight / provider_weight.mean().clamp_min(1e-8)
            user_weight = 1.0 / (user_utility.clamp_min(0).pow(user_eta) + delta)
            user_weight = user_weight / user_weight.mean().clamp_min(1e-8)
            self.provider_fairness_weight = provider_weight.detach()
            self.user_fairness_weight = user_weight.detach()
    
    def compute_adaptive_weights(self, user_embs, item_embs, user_ids, item_ids):
        """
        Generate adaptive sample weights based on user-item pair representations.
        """
        u_emb = user_embs[user_ids.to(self.device)]
        i_emb = item_embs[item_ids.to(self.device)]
        
        pair_repr = torch.cat([u_emb, i_emb], dim=-1)
        weights = 0.5 + self.weight_generator(pair_repr).squeeze(-1)
        if self.provider_fairness_weight is not None:
            weights = weights * self.provider_fairness_weight[item_ids.to(self.device)]
        if self.user_fairness_weight is not None:
            weights = weights * self.user_fairness_weight[user_ids.to(self.device)]
        weights = weights / weights.mean().detach().clamp_min(1e-8)
        
        return weights
    
    def compute_weighted_bpr_loss(self, backbone, user_ids, pos_ids, neg_ids, weights):
        """
        Weighted BPR loss with adaptive sample importance.
        """
        pos_scores = backbone.forward(user_ids, pos_ids)
        neg_scores = backbone.forward(user_ids, neg_ids)
        
        bpr_loss = -torch.log(nn.functional.sigmoid(pos_scores - neg_scores) + 1e-10)
        weighted_loss = (weights * bpr_loss).mean()
        
        return weighted_loss
    
    def compute_user_fairness_loss(self, user_embeddings, adv_users, disadv_users):
        """
        User-side fairness: align disadvantaged users toward advantaged distribution.
        """
        return self.lambda_user * group_mean_gap(user_embeddings.to(self.device),
                                                 adv_users, disadv_users)
    
    def compute_item_fairness_loss(self, item_embeddings, item_freq, n_items):
        """
        Item-side fairness: reduce exposure disparity.
        """
        hot_items = torch.argsort(torch.tensor(
            [item_freq.get(i, 0) for i in range(n_items)], device=self.device),
            descending=True)[:max(1, int(n_items * 0.2))]
        cold_mask = torch.ones(n_items, dtype=torch.bool, device=self.device)
        cold_mask[hot_items] = False
        cold_items = torch.arange(n_items, device=self.device)[cold_mask]
        if len(cold_items) == 0:
            return item_embeddings.new_tensor(0.0)
        item_embeddings = item_embeddings.to(self.device)
        loss = torch.norm(item_embeddings[hot_items].mean(dim=0) -
                          item_embeddings[cold_items].mean(dim=0), p=2).pow(2)
        return self.lambda_item * loss

    def compute_exposure_fairness_loss(self, user_embeddings, item_embeddings, item_freq, n_items):
        return self.lambda_item * soft_exposure_disparity(
            user_embeddings.to(self.device), item_embeddings.to(self.device),
            item_freq, n_items, self.device)
