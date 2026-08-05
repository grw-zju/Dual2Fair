import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from .transport import compute_cosine_cost_matrix, dense_log_sinkhorn


class UserRepresentationCalibration(nn.Module):
    def __init__(self, epsilon=0.1, fusion_alpha=0.5, rho_u=None,
                 n_clusters=64, user_chunk_size=4096,
                 random_state=42, sinkhorn_max_iter=100,
                 sinkhorn_convergence_tol=1e-3, device=None, **kwargs):
        super().__init__()
        self.epsilon = float(epsilon)
        self.rho_u = float(fusion_alpha if rho_u is None else rho_u)
        self.n_clusters = int(n_clusters)
        self.user_chunk_size = int(user_chunk_size)
        self.random_state = int(random_state)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.sinkhorn_convergence_tol = float(sinkhorn_convergence_tol)
        self.device_hint = torch.device(device or 'cpu')
        self.projection = nn.Identity()
        self.adv_users = []
        self.disadv_users = []
        self.user_items_dict = {}
        self.register_buffer('gmm_weights', None)
        self.register_buffer('gmm_covariances', None)
        self.register_buffer('prototypes', None)
        self.register_buffer('cached_gamma_u', None)
        self.register_buffer('cached_user_ids', None)

    def set_user_groups(self, adv_users, disadv_users, user_items_dict):
        self.adv_users = [int(user) for user in adv_users]
        self.disadv_users = [int(user) for user in disadv_users]
        self.user_items_dict = user_items_dict

    def _compute_interest_embs(self, user_ids, item_embs):
        interests = torch.zeros(len(user_ids), item_embs.shape[1],
                                dtype=item_embs.dtype, device=item_embs.device)
        valid = torch.zeros(len(user_ids), dtype=torch.bool, device=item_embs.device)
        for row, user in enumerate(user_ids):
            items = sorted(item for item in self.user_items_dict.get(int(user), set())
                           if 0 <= item < item_embs.shape[0])
            if items:
                interests[row] = item_embs[items].mean(dim=0)
                valid[row] = True
        return interests, valid

    def fit_gmm(self, item_embs):
        interests, valid = self._compute_interest_embs(self.adv_users, item_embs)
        samples = interests[valid]
        if samples.numel() == 0:
            self.gmm_weights = None
            self.gmm_covariances = None
            self.prototypes = None
            return
        if len(samples) == 1:
            self.gmm_weights = torch.ones(1, dtype=item_embs.dtype, device=item_embs.device)
            self.gmm_covariances = torch.zeros_like(samples.detach())
            self.prototypes = samples.detach().clone()
            return
        n_components = min(self.n_clusters, len(samples))
        model = GaussianMixture(n_components=n_components, covariance_type='diag',
                                random_state=self.random_state, max_iter=100)
        model.fit(samples.detach().cpu().numpy())
        self.gmm_weights = torch.as_tensor(model.weights_, dtype=item_embs.dtype,
                                           device=item_embs.device)
        self.gmm_covariances = torch.as_tensor(model.covariances_, dtype=item_embs.dtype,
                                               device=item_embs.device)
        self.prototypes = torch.as_tensor(model.means_, dtype=item_embs.dtype,
                                          device=item_embs.device)

    @staticmethod
    def barycentric_projection(plan, targets, eps=1e-12):
        normalized = plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)
        return normalized @ targets

    def compute_user_ot_loss(self, item_embs):
        if not self.disadv_users or self.prototypes is None:
            return item_embs.sum() * 0.0
        plans, losses, covered = [], [], []
        all_interests, valid = self._compute_interest_embs(self.disadv_users, item_embs)
        valid_rows = torch.where(valid)[0]
        if len(valid_rows) == 0:
            return item_embs.sum() * 0.0
        valid_interests = all_interests[valid_rows]
        n_valid = len(valid_interests)
        for start in range(0, n_valid, self.user_chunk_size):
            end = min(start + self.user_chunk_size, n_valid)
            chunk = valid_interests[start:end]
            source = torch.full((len(chunk),), 1.0 / len(chunk),
                                device=item_embs.device, dtype=item_embs.dtype)
            target = self.gmm_weights / self.gmm_weights.sum().clamp_min(1e-12)
            cost = compute_cosine_cost_matrix(chunk, self.prototypes)
            plan = dense_log_sinkhorn(cost, source, target, self.epsilon,
                                      self.sinkhorn_max_iter,
                                      self.sinkhorn_convergence_tol)
            entropy = -(plan * torch.log(plan.clamp_min(1e-12))).sum()
            losses.append((plan * cost).sum() - self.epsilon * entropy)
            plans.append(plan.detach())
            covered.append(valid_rows[start:end])
        self.cached_gamma_u = torch.cat(plans, dim=0)
        covered_rows = torch.cat(covered)
        self.cached_user_ids = torch.as_tensor(
            [self.disadv_users[index] for index in covered_rows.cpu().tolist()],
            dtype=torch.long, device=item_embs.device)
        return torch.stack(losses).mean()

    def calibrate_disadvantaged_users(self, user_embs, item_embs, refresh=True):
        calibrated = user_embs.clone()
        if not self.disadv_users or self.prototypes is None:
            return calibrated
        if refresh or self.cached_gamma_u is None or self.cached_user_ids is None:
            self.compute_user_ot_loss(item_embs)
        if self.cached_gamma_u is None or self.cached_user_ids is None:
            return calibrated
        targets = self.projection(self.prototypes)
        barycenters = self.barycentric_projection(self.cached_gamma_u, targets)
        ids = self.cached_user_ids
        calibrated[ids] = self.rho_u * user_embs[ids] + (1.0 - self.rho_u) * barycenters
        return calibrated

    def extra_state(self):
        return {'adv_users': self.adv_users, 'disadv_users': self.disadv_users,
                'random_state': self.random_state}
