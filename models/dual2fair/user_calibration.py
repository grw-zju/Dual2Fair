import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from .alignment import hard_user_match, linear_mmd_loss, linear_mmd_state


class UserRepresentationCalibration(nn.Module):
    def __init__(self, representation_dim, gmm_clusters=64, gmm_max_iter=50,
                 gmm_tol=1e-4, gmm_covariance_floor=1e-6, rho_u_init=-2.0,
                 epsilon_u=0.1, tau_u=0.2, sinkhorn_max_iter=100,
                 sinkhorn_tol=1e-3, user_kernel_chunk_size=4096,
                 eps0=1e-8, log_eps=1e-20, random_state=42, **deprecated):
        super().__init__()
        if deprecated:
            warnings.warn(f'Deprecated user-calibration options ignored: {sorted(deprecated)}',
                          DeprecationWarning, stacklevel=2)
        self.gmm_clusters = int(gmm_clusters)
        self.gmm_max_iter = int(gmm_max_iter)
        self.gmm_tol = float(gmm_tol)
        self.gmm_covariance_floor = float(gmm_covariance_floor)
        self.epsilon_u = float(epsilon_u)
        self.tau_u = float(tau_u)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.user_kernel_chunk_size = int(user_kernel_chunk_size)
        self.eps0 = float(eps0)
        self.log_eps = float(log_eps)
        self.random_state = int(random_state)

        self.W_u_c = nn.Linear(representation_dim, representation_dim, bias=False)
        self.W_u_h = nn.Linear(representation_dim, representation_dim, bias=False)
        self.P_u = nn.Linear(representation_dim, representation_dim, bias=False)
        nn.init.eye_(self.W_u_c.weight)
        nn.init.eye_(self.W_u_h.weight)
        nn.init.eye_(self.P_u.weight)
        self.rho_u_tilde = nn.Parameter(torch.tensor(float(rho_u_init)))
        self.layer_norm = nn.LayerNorm(representation_dim)

        self.higher_activity_users = []
        self.sparse_history_users = []
        self.user_items = {}
        self.register_buffer('gmm_weights', None)
        self.register_buffer('gmm_covariances', None)
        self.register_buffer('prototypes', None)
        self.register_buffer('user_transport', None)
        self.register_buffer('sparse_user_ids', None)
        self.register_buffer('sparse_calibration_x', None)
        self.register_buffer('alignment_targets', None)
        self.register_buffer('hard_user_indices', None)
        self.alignment_mode = 'ot'
        self.mmd_state = None

    @property
    def rho_u(self):
        return torch.sigmoid(self.rho_u_tilde)

    def set_user_groups(self, higher_activity_users, sparse_history_users, user_items):
        self.higher_activity_users = [int(user) for user in higher_activity_users]
        self.sparse_history_users = [int(user) for user in sparse_history_users]
        self.user_items = user_items

    def set_legacy_user_groups(self, adv_users, disadv_users, user_items):
        warnings.warn('set_legacy_user_groups is deprecated', DeprecationWarning, stacklevel=2)
        self.set_user_groups(adv_users, disadv_users, user_items)

    def history_representations(self, user_ids, item_representations):
        histories = torch.zeros(len(user_ids), item_representations.shape[1],
                                dtype=item_representations.dtype,
                                device=item_representations.device)
        valid = torch.zeros(len(user_ids), dtype=torch.bool,
                            device=item_representations.device)
        detached_items = item_representations.detach()
        for row, user in enumerate(user_ids):
            items = sorted(item for item in self.user_items.get(int(user), set())
                           if 0 <= item < detached_items.shape[0])
            if items:
                histories[row] = detached_items[items].mean(dim=0)
                valid[row] = True
        return histories, valid

    def calibration_representations(self, user_ids, user_representations,
                                    item_representations):
        ids = torch.as_tensor(user_ids, dtype=torch.long,
                              device=user_representations.device)
        histories, valid = self.history_representations(user_ids, item_representations)
        values = self.layer_norm(self.W_u_c(user_representations[ids])
                                 + self.W_u_h(histories))
        return values, valid

    def fit_gmm(self, higher_x, valid):
        samples = higher_x[valid].detach()
        if samples.numel() == 0:
            raise ValueError('No higher-activity users have training histories')
        if len(samples) == 1:
            self.gmm_weights = torch.ones(1, device=samples.device, dtype=samples.dtype)
            self.prototypes = samples.clone()
            self.gmm_covariances = torch.full_like(samples, self.gmm_covariance_floor)
            return
        components = min(self.gmm_clusters, len(samples))
        model = GaussianMixture(
            n_components=components, covariance_type='diag',
            random_state=self.random_state, max_iter=self.gmm_max_iter,
            tol=self.gmm_tol, reg_covar=self.gmm_covariance_floor)
        model.fit(samples.cpu().numpy())
        self.gmm_weights = torch.as_tensor(model.weights_, device=samples.device,
                                           dtype=samples.dtype)
        self.prototypes = torch.as_tensor(model.means_, device=samples.device,
                                          dtype=samples.dtype)
        self.gmm_covariances = torch.as_tensor(model.covariances_, device=samples.device,
                                               dtype=samples.dtype)

    def _kernel_matvec(self, sparse_x, prototypes, vector, transpose=False):
        outputs = []
        if transpose:
            result = torch.zeros(prototypes.shape[0], dtype=sparse_x.dtype,
                                 device=sparse_x.device)
        for start in range(0, len(sparse_x), self.user_kernel_chunk_size):
            chunk = sparse_x[start:start + self.user_kernel_chunk_size]
            cost = 1.0 - F.normalize(chunk, dim=1) @ F.normalize(prototypes, dim=1).T
            kernel = torch.exp(-cost / self.epsilon_u)
            if transpose:
                result += kernel.T @ vector[start:start + len(chunk)]
            else:
                outputs.append(kernel @ vector)
        return result if transpose else torch.cat(outputs)

    def solve_global_user_ot(self, sparse_x):
        n_sparse = len(sparse_x)
        if n_sparse == 0:
            raise ValueError('Sparse-history user set is empty')
        prototypes = self.prototypes.detach()
        source = torch.full((n_sparse,), 1.0 / n_sparse, dtype=sparse_x.dtype,
                            device=sparse_x.device)
        target = self.gmm_weights / self.gmm_weights.sum().clamp_min(self.eps0)
        a = torch.ones_like(source)
        b = torch.ones_like(target)
        for _ in range(self.sinkhorn_max_iter):
            kb = self._kernel_matvec(sparse_x, prototypes, b)
            a = source / kb.clamp_min(self.log_eps)
            kta = self._kernel_matvec(sparse_x, prototypes, a, transpose=True)
            b = target / kta.clamp_min(self.log_eps)
            rows = a * self._kernel_matvec(sparse_x, prototypes, b)
            cols = b * self._kernel_matvec(sparse_x, prototypes, a, transpose=True)
            violation = max((rows - source).abs().max().item(),
                            (cols - target).abs().max().item())
            if violation <= self.sinkhorn_tol:
                break
        chunks = []
        for start in range(0, n_sparse, self.user_kernel_chunk_size):
            chunk = sparse_x[start:start + self.user_kernel_chunk_size]
            cost = 1.0 - F.normalize(chunk, dim=1) @ F.normalize(prototypes, dim=1).T
            kernel = torch.exp(-cost / self.epsilon_u)
            chunks.append(a[start:start + len(chunk), None] * kernel * b[None, :])
        plan = torch.cat(chunks, dim=0)
        self.user_transport = plan.detach()
        return plan

    def refresh(self, user_representations, item_representations, alignment_mode='ot'):
        self.alignment_mode = alignment_mode
        higher_x, higher_valid = self.calibration_representations(
            self.higher_activity_users, user_representations, item_representations)
        self.fit_gmm(higher_x, higher_valid)
        sparse_x, sparse_valid = self.calibration_representations(
            self.sparse_history_users, user_representations, item_representations)
        valid_rows = torch.where(sparse_valid)[0]
        valid_x = sparse_x[valid_rows]
        self.sparse_user_ids = torch.as_tensor(
            [self.sparse_history_users[index] for index in valid_rows.cpu().tolist()],
            dtype=torch.long, device=user_representations.device)
        self.sparse_calibration_x = valid_x.detach()
        self.user_transport = None
        self.alignment_targets = None
        self.hard_user_indices = None
        self.mmd_state = None
        if alignment_mode == 'ot':
            plan = self.solve_global_user_ot(valid_x)
            self.alignment_targets = self.barycentric_targets().detach()
            return plan
        if alignment_mode == 'hard':
            target = self.gmm_weights / self.gmm_weights.sum().clamp_min(self.eps0)
            indices = hard_user_match(valid_x.detach(), self.prototypes.detach(),
                                      target.detach(), self.epsilon_u, self.eps0)
            self.hard_user_indices = indices.detach()
            self.alignment_targets = self.prototypes.detach()[self.hard_user_indices].detach()
            return self.hard_user_indices
        if alignment_mode == 'mmd':
            source = torch.full((len(valid_x),), 1.0 / len(valid_x),
                                dtype=valid_x.dtype, device=valid_x.device)
            target = self.gmm_weights / self.gmm_weights.sum().clamp_min(self.eps0)
            self.mmd_state = linear_mmd_state(valid_x.detach(), source,
                                              self.prototypes.detach(),
                                              target.detach(), self.eps0)
            self.alignment_targets = (valid_x.detach() + self.mmd_state.delta).detach()
            return self.mmd_state
        raise ValueError(f'Unknown alignment_mode: {alignment_mode}')

    def barycentric_targets(self):
        row_masses = self.user_transport.sum(dim=1, keepdim=True)
        normalized = self.user_transport / row_masses.clamp_min(self.eps0)
        return normalized @ self.prototypes.detach()

    def calibrate(self, user_representations, enable_confidence=True):
        calibrated = user_representations.clone()
        if self.sparse_user_ids is None:
            return calibrated
        if self.alignment_mode == 'ot':
            if self.user_transport is None:
                return calibrated
            z = self.barycentric_targets()
        else:
            if self.alignment_targets is None:
                return calibrated
            z = self.alignment_targets
        x = self.sparse_calibration_x
        confidence = torch.exp(-(1.0 - F.cosine_similarity(x, z, dim=1)) / self.tau_u)
        if not enable_confidence:
            confidence = torch.ones_like(confidence)
        residual = self.P_u(z - x)
        ids = self.sparse_user_ids
        calibrated[ids] = (user_representations[ids]
                           + self.rho_u * confidence[:, None] * residual)
        return calibrated

    def fixed_coupling_loss(self, user_representations, item_representations):
        x, valid = self.calibration_representations(
            self.sparse_history_users, user_representations, item_representations)
        x = x[valid]
        if self.alignment_mode == 'ot':
            cost = 1.0 - F.normalize(x, dim=1) @ F.normalize(self.prototypes.detach(), dim=1).T
            return (self.user_transport.detach() * cost).sum()
        if self.alignment_mode == 'hard':
            cost = 1.0 - F.cosine_similarity(
                x, self.prototypes.detach()[self.hard_user_indices.detach()], dim=1)
            source = torch.full_like(cost, 1.0 / len(cost))
            return (source * cost).sum()
        if self.alignment_mode == 'mmd':
            source = torch.full((len(x),), 1.0 / len(x), dtype=x.dtype, device=x.device)
            target = self.gmm_weights.detach() / self.gmm_weights.detach().sum().clamp_min(self.eps0)
            return linear_mmd_loss(x, source, self.prototypes.detach(), target, self.eps0)
        raise ValueError(f'Unknown alignment_mode: {self.alignment_mode}')

    def fairness_parameters(self):
        return list(self.W_u_c.parameters()) + list(self.W_u_h.parameters())
