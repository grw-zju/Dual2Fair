import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from .sinkhorn import sinkhorn_nystrom, compute_cosine_cost_matrix


class UserRepresentationCalibration:
    def __init__(self, epsilon=0.1, fusion_alpha=0.5, n_clusters=64,
                 max_disadv_users=4096, device=None, **kwargs):
        self.epsilon = epsilon
        self.fusion_alpha = fusion_alpha
        self.n_clusters = n_clusters
        self.max_disadv_users = max_disadv_users
        self.device = device or torch.device('cpu')
        self.adv_users = None
        self.disadv_users = None
        self.gmm_weights = None
        self.prototypes = None
        self.user_items_dict = None
        self._cached_gamma_u = None

    def set_user_groups(self, adv_users, disadv_users, user_items_dict):
        self.adv_users = adv_users
        self.disadv_users = disadv_users
        self.user_items_dict = user_items_dict

    def _compute_interest_embs(self, user_ids, item_embs):
        if len(user_ids) == 0:
            return torch.zeros(0, item_embs.shape[1], device=self.device)
        interest_embs = torch.zeros(len(user_ids), item_embs.shape[1], device=self.device)
        counts = torch.zeros(len(user_ids), 1, device=self.device)
        item_embs_ref = item_embs
        for i, u in enumerate(user_ids):
            items = [iid for iid in self.user_items_dict.get(u, set()) if iid < item_embs_ref.shape[0]]
            if len(items) > 0:
                interest_embs[i] = item_embs_ref[items].mean(dim=0)
                counts[i] = len(items)
        mask = counts > 0
        interest_embs = interest_embs * mask.float()
        return interest_embs

    def fit_gmm(self, item_embs):
        adv_interest_embs = self._compute_interest_embs(self.adv_users, item_embs)
        if len(adv_interest_embs) == 0:
            return
        X = adv_interest_embs.detach().cpu().numpy()
        n_components = min(self.n_clusters, len(X))
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                               random_state=42, max_iter=100)
        gmm.fit(X)
        self.gmm_weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=self.device)
        self.prototypes = torch.tensor(gmm.means_, dtype=torch.float32, device=self.device)

    def compute_user_ot_loss(self, item_embs):
        if self.disadv_users is None or len(self.disadv_users) == 0:
            return torch.tensor(0.0, device=self.device)
        if self.prototypes is None:
            return torch.tensor(0.0, device=self.device)
        if len(self.disadv_users) > self.max_disadv_users:
            idx = torch.randperm(len(self.disadv_users), device=self.device)[:self.max_disadv_users]
            sampled_disadv_users = [self.disadv_users[int(i)] for i in idx.cpu().tolist()]
        else:
            sampled_disadv_users = self.disadv_users
        disadv_interest_embs = self._compute_interest_embs(sampled_disadv_users, item_embs)
        n_disadv = len(sampled_disadv_users)
        n_proto = self.prototypes.shape[0]
        source_weights = torch.ones(n_disadv, device=self.device) / n_disadv
        target_weights = self.gmm_weights[:n_proto].clone()
        C_u = compute_cosine_cost_matrix(disadv_interest_embs, self.prototypes, self.device)
        gamma_u = sinkhorn_nystrom(C_u, source_weights, target_weights, epsilon=self.epsilon,
                                    max_iter=100, convergence_tol=1e-3,
                                    nystrom_rank=32, rank_increment=8, nystrom_tol=1e-3,
                                    tikhonov_lambda=1e-4, device=self.device)
        ot_cost = (gamma_u * C_u).sum()
        entropy = -torch.sum(gamma_u * torch.log(gamma_u + 1e-10))
        L_user = ot_cost - self.epsilon * entropy
        self._cached_gamma_u = gamma_u.detach()
        self._cached_disadv_users = sampled_disadv_users
        self._cached_disadv_interest_embs = disadv_interest_embs.detach()
        return L_user

    def calibrate_disadvantaged_users(self, user_embs, item_embs):
        if self.disadv_users is None or len(self.disadv_users) == 0:
            return user_embs
        if self.prototypes is None:
            return user_embs
        disadv_to_idx = {u: i for i, u in enumerate(self.disadv_users)}
        gamma_u = self._cached_gamma_u
        if gamma_u is None:
            disadv_interest_embs = self._compute_interest_embs(self.disadv_users, item_embs)
            source_weights = torch.ones(len(self.disadv_users), device=self.device) / len(self.disadv_users)
            target_weights = self.gmm_weights[:self.prototypes.shape[0]].clone()
            C_u = compute_cosine_cost_matrix(disadv_interest_embs, self.prototypes, self.device)
            gamma_u = sinkhorn_nystrom(C_u, source_weights, target_weights, epsilon=self.epsilon,
                                        max_iter=100, convergence_tol=1e-3,
                                        nystrom_rank=32, rank_increment=8, nystrom_tol=1e-3,
                                        tikhonov_lambda=1e-4, device=self.device)
        cached_users = getattr(self, '_cached_disadv_users', self.disadv_users)
        calibrated_interest = gamma_u @ self.prototypes
        calibrated_user_embs = user_embs.clone()
        alpha = self.fusion_alpha
        for i, u in enumerate(cached_users):
            calibrated_interest_d = calibrated_interest[i]
            calibrated_user_embs[u] = alpha * user_embs[u] + (1 - alpha) * calibrated_interest_d
        return calibrated_user_embs
