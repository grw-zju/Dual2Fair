import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import hard_item_match_blockwise, linear_mmd_loss, linear_mmd_state
from .sinkhorn import (build_adaptive_nystrom_state,
                       compute_lowrank_barycenter,
                       compute_lowrank_fixed_coupling_cost,
                       solve_item_sinkhorn_lowrank)


class ItemRepresentationCalibration(nn.Module):
    def __init__(self, representation_dim, epsilon_v=0.1, tau_v=0.2,
                 kappa=1.0, beta_pop=0.5, delta_m=1e-6, omega=0.2,
                 rho_v_init=-2.0, nystrom_initial_rank=32,
                 nystrom_max_rank=256, nystrom_tol=1e-3,
                 nystrom_num_strata=5, nystrom_pinv_rtol=1e-6,
                 sinkhorn_max_iter=100, sinkhorn_tol=1e-3,
                 eps0=1e-8, random_state=42, **deprecated):
        super().__init__()
        self.epsilon_v = float(epsilon_v)
        self.tau_v = float(tau_v)
        self.kappa = float(kappa)
        self.beta_pop = float(beta_pop)
        self.delta_m = float(delta_m)
        self.omega = float(omega)
        self.nystrom_initial_rank = int(nystrom_initial_rank)
        self.nystrom_max_rank = int(nystrom_max_rank)
        self.nystrom_tol = float(nystrom_tol)
        self.nystrom_num_strata = int(nystrom_num_strata)
        self.nystrom_pinv_rtol = float(nystrom_pinv_rtol)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.eps0 = float(eps0)
        self.random_state = int(random_state)
        self.W_v = nn.Linear(representation_dim, representation_dim, bias=False)
        self.P_v = nn.Linear(representation_dim, representation_dim, bias=False)
        nn.init.eye_(self.W_v.weight)
        nn.init.eye_(self.P_v.weight)
        self.layer_norm = nn.LayerNorm(representation_dim)
        self.rho_v_tilde = nn.Parameter(torch.tensor(float(rho_v_init)))
        self.transport_state = None
        self.alignment_mode = 'ot'
        self.mmd_state = None
        self.register_buffer('target_distribution', None)
        self.register_buffer('source_distribution', None)
        self.register_buffer('target_anchors', None)
        self.register_buffer('warm_item_ids', None)
        self.register_buffer('calibration_x', None)
        self.register_buffer('barycentric_targets_cache', None)
        self.register_buffer('hard_item_indices', None)

    @property
    def rho_v(self):
        return torch.sigmoid(self.rho_v_tilde)

    def calibration_space(self, item_representations):
        return F.normalize(self.layer_norm(self.W_v(item_representations)),
                           dim=1, eps=self.eps0)

    def source_marginal(self, frequencies):
        masses = (frequencies + self.kappa).pow(-self.beta_pop)
        return masses / masses.sum().clamp_min(self.eps0)

    def relevance_target(self, merit, omega=None):
        mixture = self.omega if omega is None else float(omega)
        smoothed = merit.clamp_min(0) + self.delta_m
        merit_target = smoothed / smoothed.sum().clamp_min(self.eps0)
        uniform = torch.full_like(merit_target, 1.0 / len(merit_target))
        target = (1.0 - mixture) * merit_target + mixture * uniform
        return target / target.sum().clamp_min(self.eps0)

    def refresh(self, item_representations, frequencies, merit, alignment_mode='ot'):
        self.alignment_mode = alignment_mode
        x = self.calibration_space(item_representations)
        source = self.source_marginal(frequencies)
        target = self.relevance_target(merit)
        anchors = x.detach()
        self.transport_state = None
        self.mmd_state = None
        self.hard_item_indices = None
        self.source_distribution = source.detach()
        self.target_distribution = target.detach()
        self.target_anchors = anchors
        self.calibration_x = x.detach()
        if alignment_mode == 'ot':
            state = build_adaptive_nystrom_state(
                anchors, frequencies, self.epsilon_v,
                self.nystrom_initial_rank, self.nystrom_max_rank,
                self.nystrom_tol, self.nystrom_num_strata,
                self.nystrom_pinv_rtol, self.random_state)
            state = solve_item_sinkhorn_lowrank(
                state, source, target, self.sinkhorn_max_iter,
                self.sinkhorn_tol, self.eps0)
            barycenters = compute_lowrank_barycenter(state, anchors, self.eps0)
            self.transport_state = state
            self.barycentric_targets_cache = barycenters.detach()
            return state
        if alignment_mode == 'hard':
            indices = hard_item_match_blockwise(
                anchors, anchors, target.detach(), self.epsilon_v,
                eps0=self.eps0)
            self.hard_item_indices = indices.detach()
            self.barycentric_targets_cache = anchors[self.hard_item_indices].detach()
            return self.hard_item_indices
        if alignment_mode == 'mmd':
            self.mmd_state = linear_mmd_state(anchors, source.detach(), anchors,
                                              target.detach(), self.eps0)
            self.barycentric_targets_cache = (anchors + self.mmd_state.delta).detach()
            return self.mmd_state
        raise ValueError(f'Unknown alignment_mode: {alignment_mode}')

    def calibrate(self, item_representations, enable_confidence=True):
        if self.barycentric_targets_cache is None:
            return item_representations
        x = self.calibration_space(item_representations)
        z = self.barycentric_targets_cache
        normalized_z = F.normalize(z, dim=1, eps=self.eps0)
        confidence = torch.exp(-(1.0 - (x * normalized_z).sum(1)) / self.tau_v)
        if not enable_confidence:
            confidence = torch.ones_like(confidence)
        return (item_representations + self.rho_v * confidence[:, None]
                * self.P_v(z - x))

    def fixed_coupling_loss(self, item_representations):
        x = self.calibration_space(item_representations)
        if self.alignment_mode == 'ot':
            return compute_lowrank_fixed_coupling_cost(
                self.transport_state, x, self.target_anchors.detach())
        if self.alignment_mode == 'hard':
            targets = self.target_anchors.detach()[self.hard_item_indices.detach()]
            cost = 1.0 - (x * targets).sum(dim=1)
            source = self.source_distribution.detach()
            return (source * cost).sum()
        if self.alignment_mode == 'mmd':
            return linear_mmd_loss(x, self.source_distribution.detach(),
                                   self.target_anchors.detach(),
                                   self.target_distribution.detach(), self.eps0)
        raise ValueError(f'Unknown alignment_mode: {self.alignment_mode}')

    def fairness_parameters(self):
        return list(self.W_v.parameters())
