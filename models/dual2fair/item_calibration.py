import torch
import torch.nn as nn

from .transport import (compute_cosine_cost_matrix, dense_log_sinkhorn,
                        deterministic_landmark_indices, landmark_transport)


class ItemRepresentationCalibration(nn.Module):
    VALID_TARGET_MODES = {'uniform', 'merit', 'merit_uniform_mixture'}

    def __init__(self, epsilon=0.1, alpha_smoothing=1.0, beta_scaling=1.0,
                 sinkhorn_max_iter=100, sinkhorn_convergence_tol=1e-3,
                 max_ot_items=4096, item_anchor_count=256,
                 item_target_mode='merit_uniform_mixture', merit_uniform_gamma=0.5,
                 rho_v=0.5, random_state=42, device=None, **kwargs):
        super().__init__()
        if item_target_mode not in self.VALID_TARGET_MODES:
            raise ValueError(f'Unknown item target mode: {item_target_mode}')
        self.epsilon = float(epsilon)
        self.alpha_smoothing = float(alpha_smoothing)
        self.beta_scaling = float(beta_scaling)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.sinkhorn_convergence_tol = float(sinkhorn_convergence_tol)
        self.max_ot_items = int(max_ot_items)
        self.item_anchor_count = int(item_anchor_count)
        self.item_target_mode = item_target_mode
        self.merit_uniform_gamma = float(merit_uniform_gamma)
        self.rho_v = float(rho_v)
        self.random_state = int(random_state)
        self.device_hint = torch.device(device or 'cpu')
        self.projection = nn.Identity()
        self.register_buffer('anchor_indices', None)
        self.register_buffer('target_distribution', None)
        self.register_buffer('cached_item_projection', None)
        self.cached_item_plans = None

    def _frequency_values(self, n_items, item_freq, dtype, device):
        values = torch.zeros(n_items, dtype=dtype, device=device)
        for item, frequency in item_freq.items():
            if 0 <= item < n_items:
                values[item] = float(frequency)
        return values

    def _source_weights(self, frequencies):
        weights = 1.0 / (frequencies + self.alpha_smoothing).pow(self.beta_scaling)
        return weights / weights.sum().clamp_min(1e-12)

    def _target_weights(self, merit):
        uniform = torch.full_like(merit, 1.0 / merit.numel())
        if self.item_target_mode == 'uniform':
            return uniform
        normalized = merit.clamp_min(0)
        normalized = normalized / normalized.sum().clamp_min(1e-12)
        if self.item_target_mode == 'merit':
            return normalized
        gamma = self.merit_uniform_gamma
        return (1.0 - gamma) * normalized + gamma * uniform

    def _select_anchors(self, frequencies):
        count = min(self.item_anchor_count, frequencies.numel())
        return deterministic_landmark_indices(frequencies, count)

    def update_state(self, item_embeddings, item_freq, merit=None):
        frequencies = self._frequency_values(item_embeddings.shape[0], item_freq,
                                              item_embeddings.dtype, item_embeddings.device)
        self.anchor_indices = self._select_anchors(frequencies)
        anchor_merit = (frequencies[self.anchor_indices] + 1.0 if merit is None
                        else merit.detach()[self.anchor_indices].clamp_min(1e-12))
        self.target_distribution = self._target_weights(anchor_merit)

    @staticmethod
    def barycentric_projection(plan, anchors, eps=1e-12):
        normalized = plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)
        return normalized @ anchors

    def _transport(self, item_embeddings, item_freq, merit=None):
        if self.anchor_indices is None or self.target_distribution is None:
            self.update_state(item_embeddings, item_freq, merit)
        frequencies = self._frequency_values(item_embeddings.shape[0], item_freq,
                                              item_embeddings.dtype, item_embeddings.device)
        source = self._source_weights(frequencies)
        anchors = item_embeddings[self.anchor_indices].detach()
        if len(anchors) == len(item_embeddings):
            cost = compute_cosine_cost_matrix(item_embeddings, anchors)
            plan = dense_log_sinkhorn(cost, source, self.target_distribution,
                                      self.epsilon, self.sinkhorn_max_iter,
                                      self.sinkhorn_convergence_tol)
            projection = self.barycentric_projection(plan, anchors)
            plans = [plan]
        else:
            projection, plans = landmark_transport(
                item_embeddings, source, anchors, self.target_distribution,
                self.epsilon, self.sinkhorn_max_iter,
                self.sinkhorn_convergence_tol,
                chunk_size=min(self.max_ot_items, item_embeddings.shape[0]))
        return projection, plans

    def compute_item_ot_loss(self, item_embeddings, item_freq, merit=None):
        projection, plans = self._transport(item_embeddings, item_freq, merit)
        self.cached_item_projection = projection.detach()
        self.cached_item_plans = [plan.detach() for plan in plans]
        return (1.0 - torch.nn.functional.cosine_similarity(
            item_embeddings, projection.detach(), dim=1)).mean()

    def calibrate_items(self, item_embeddings, item_freq, merit=None, refresh=True):
        if refresh or self.cached_item_projection is None:
            projection, plans = self._transport(item_embeddings, item_freq, merit)
            self.cached_item_projection = projection.detach()
            self.cached_item_plans = [plan.detach() for plan in plans]
        projected = self.projection(self.cached_item_projection.to(item_embeddings.device))
        calibrated = self.rho_v * item_embeddings + (1.0 - self.rho_v) * projected
        loss = (1.0 - torch.nn.functional.cosine_similarity(
            item_embeddings, projected.detach(), dim=1)).mean()
        return calibrated, loss
