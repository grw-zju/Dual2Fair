import torch
import torch.nn as nn


class DPR:
    """
    Disparity Penalization for Regularization.
    Utility-aware regularization strategy that penalizes large disparities
    between item utility and exposure.
    
    Ref: Zhu et al., "Measuring and mitigating item under-recommendation bias
    in personalized ranking systems", SIGIR 2020.
    """
    
    OFFICIAL_REPO = "external_baselines/Item-Underrecommendation-Bias"

    def __init__(self, lambda_dpr=0.1, reg_s=1e-4, alpha_adv=0.01, device=None):
        self.lambda_dpr = lambda_dpr
        self.reg_s = reg_s
        self.alpha_adv = alpha_adv
        self.device = device or torch.device('cpu')
    
    def compute_dpr_loss(self, user_embeddings, item_embeddings, item_freq, n_items, k=10):
        """
        DPR loss: penalize the gap between item exposure and quality.
        
        L_DPR = Σ_v |e_v / q_v - μ_eq|
        
        Approximation: for each item, compute expected exposure vs quality ratio.
        """
        if not isinstance(user_embeddings, torch.Tensor):
            user_embeddings = torch.from_numpy(user_embeddings).float().to(self.device)
        if not isinstance(item_embeddings, torch.Tensor):
            item_embeddings = torch.from_numpy(item_embeddings).float().to(self.device)
        
        freq_tensor = torch.zeros(n_items, device=self.device)
        for iid, freq in item_freq.items():
            if iid < n_items:
                freq_tensor[iid] = freq
        
        expected_quality = (freq_tensor + 1.0) / (freq_tensor.sum() + n_items)
        
        scores = user_embeddings @ item_embeddings.T
        user_limit = min(scores.shape[0], 1000)
        sampled_scores = scores[:user_limit]
        exposure = torch.softmax(sampled_scores, dim=1).mean(dim=0)
        
        eq_ratio = exposure / (expected_quality + 1e-10)
        mean_eq = eq_ratio.mean()
        
        dpr_loss = torch.mean(torch.abs(eq_ratio - mean_eq))
        score_mean = sampled_scores.mean(dim=1)
        score_std = sampled_scores.std(dim=1).clamp_min(1e-6)
        score_distribution_loss = torch.mean(score_mean.pow(2) + score_std.pow(2) -
                                             2.0 * torch.log(score_std) - 1.0)

        freq_threshold = torch.quantile(freq_tensor, 0.8)
        group_label = (freq_tensor >= freq_threshold).float()
        group_prior = group_label.mean().clamp(1e-6, 1 - 1e-6)
        pred_group = torch.sigmoid(item_embeddings.mean(dim=1))
        adv_loss = -torch.mean(group_label * torch.log(pred_group + 1e-8) +
                               (1 - group_label) * torch.log(1 - pred_group + 1e-8))
        confusion_loss = -adv_loss + (group_prior * torch.log(group_prior) +
                                      (1 - group_prior) * torch.log(1 - group_prior))

        return self.lambda_dpr * dpr_loss + self.reg_s * score_distribution_loss + self.alpha_adv * confusion_loss
    
    def add_to_training(self, backbone, rec_loss, item_freq, n_items, k=10):
        """
        Add DPR regularization to BPR training.
        
        Returns combined loss: L_rec + λ * L_DPR
        """
        user_embs = backbone.get_user_embeddings()
        item_embs = backbone.get_item_embeddings()
        dpr_loss = self.compute_dpr_loss(user_embs, item_embs, item_freq, n_items, k)
        total_loss = rec_loss + dpr_loss
        return total_loss
