import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_adapter
from .item_calibration import ItemRepresentationCalibration
from .state import CalibrationOutput, CalibrationState
from .user_calibration import UserRepresentationCalibration


class Dual2Fair(nn.Module):
    def __init__(self, backbone, dataset, config, device=None, backbone_name=None):
        super().__init__()
        self.backbone = backbone
        self.dataset = dataset
        self.config = config
        self.device_hint = torch.device(device or 'cpu')
        self.backbone_name = (backbone_name or getattr(backbone, '_backbone_name_hint', None)
                              or backbone.__class__.__name__.lower())
        self.adapter = get_adapter(self.backbone_name, backbone)
        settings = config['dual2fair']
        users, items = self.get_native_embeddings()
        dimension = items.shape[1]
        seed = config.get('seeds', {}).get('ot_sampling', 42)
        self.user_calibration = UserRepresentationCalibration(
            dimension, gmm_clusters=settings['gmm_clusters'],
            gmm_max_iter=settings['gmm_max_iter'], gmm_tol=settings['gmm_tol'],
            gmm_covariance_floor=settings['gmm_covariance_floor'],
            rho_u_init=settings['rho_u_init'], epsilon_u=settings['epsilon_u'],
            tau_u=settings['tau_u'], sinkhorn_max_iter=settings['sinkhorn_max_iter'],
            sinkhorn_tol=settings['sinkhorn_tol'],
            user_kernel_chunk_size=settings['user_kernel_chunk_size'],
            eps0=settings['eps0'], log_eps=settings['log_eps'], random_state=seed)
        higher, sparse = dataset.get_user_activity_groups(settings['sparse_user_ratio'])
        self.user_calibration.set_user_groups(higher, sparse, dataset.user_items)
        self.item_calibration = ItemRepresentationCalibration(
            dimension, epsilon_v=settings['epsilon_v'], tau_v=settings['tau_v'],
            kappa=settings['kappa'], beta_pop=settings['beta_pop'],
            delta_m=settings['delta_m'], omega=settings['omega'],
            rho_v_init=settings['rho_v_init'],
            nystrom_initial_rank=settings['nystrom_initial_rank'],
            nystrom_max_rank=settings['nystrom_max_rank'],
            nystrom_tol=settings['nystrom_tol'],
            nystrom_num_strata=settings['nystrom_num_strata'],
            nystrom_pinv_rtol=settings['nystrom_pinv_rtol'],
            sinkhorn_max_iter=settings['sinkhorn_max_iter'],
            sinkhorn_tol=settings['sinkhorn_tol'], eps0=settings['eps0'],
            item_solver_mode=settings.get('item_solver_mode', 'lowrank'),
            dense_max_items=settings.get('dense_max_items', 5000),
            random_state=seed)
        self.enable_user_calibration = settings.get('enable_user_calibration', True)
        self.enable_item_calibration = settings.get('enable_item_calibration', True)
        self.enable_confidence = settings.get('enable_confidence', True)
        self.alignment_mode = settings.get('alignment_mode', 'ot')
        if self.alignment_mode not in {'ot', 'hard', 'mmd'}:
            raise ValueError(f"Unknown alignment_mode: {self.alignment_mode}")
        self.ema_decay = float(settings['ema_decay'])
        self.training_candidate_size = int(settings['training_candidate_size'])
        self.register_buffer('training_candidates', self._build_training_candidates(seed))
        self.ema_state = {name: parameter.detach().clone()
                          for name, parameter in self.named_parameters()}
        self.calibration_state = None
        self.scoring_state = None
        self.refresh_index = 0

    def get_native_embeddings(self):
        return self.adapter.get_raw_user_repr(), self.adapter.get_raw_item_repr()

    def _build_training_candidates(self, seed):
        generator = torch.Generator().manual_seed(int(seed))
        warm_items = sorted(item for item, frequency in self.dataset.item_freq.items()
                            if frequency > 0)
        rows = []
        for user in range(self.dataset.n_users):
            pool = [item for item in warm_items
                    if item not in self.dataset.user_items.get(user, set())]
            if not pool:
                rows.append(torch.empty(0, dtype=torch.long))
                continue
            count = min(self.training_candidate_size, len(pool))
            permutation = torch.randperm(len(pool), generator=generator)[:count]
            rows.append(torch.tensor([pool[index] for index in permutation], dtype=torch.long))
        width = max(len(row) for row in rows)
        candidates = torch.full((len(rows), width), -1, dtype=torch.long)
        for user, row in enumerate(rows):
            candidates[user, :len(row)] = row
        return candidates

    def _ema_parameter_context(self):
        current = {name: parameter.detach().clone()
                   for name, parameter in self.named_parameters()}
        self.load_state_dict({**self.state_dict(), **self.ema_state}, strict=False)
        return current

    def _restore_parameters(self, current):
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(current[name])

    def compute_training_merit(self):
        merit = torch.zeros(self.dataset.n_items, device=self.device_hint)
        current = self._ema_parameter_context()
        try:
            users, items = self.get_native_embeddings()
            raw_state = {'raw_user_repr': users, 'raw_item_repr': items,
                         'calibrated_user_repr': users, 'calibrated_item_repr': items}
            with torch.no_grad():
                for user in range(self.dataset.n_users):
                    candidates = self.training_candidates[user]
                    candidates = candidates[candidates >= 0].to(users.device)
                    if not len(candidates):
                        continue
                    user_ids = torch.full_like(candidates, user)
                    scores = self.adapter.score_pairs_with_calibrated_state(
                        user_ids, candidates, raw_state)
                    order = torch.argsort(candidates, stable=True)
                    candidates, scores = candidates[order], scores[order]
                    ranking = torch.argsort(scores, descending=True, stable=True)
                    ranks = torch.empty(len(candidates), device=scores.device)
                    ranks[ranking] = torch.arange(1, len(candidates) + 1,
                                                  device=scores.device,
                                                  dtype=scores.dtype)
                    relevance = (len(candidates) - ranks + 1) / len(candidates)
                    merit.index_add_(0, candidates, relevance)
        finally:
            self._restore_parameters(current)
        return merit.detach()

    def refresh_calibration_state(self):
        users, items = self.get_native_embeddings()
        if self.enable_user_calibration:
            self.user_calibration.refresh(users, items, self.alignment_mode)
        frequencies = torch.as_tensor(
            [self.dataset.item_freq.get(item, 0) for item in range(self.dataset.n_items)],
            dtype=items.dtype, device=items.device)
        merit = self.compute_training_merit()
        if self.enable_item_calibration:
            self.item_calibration.refresh(items, frequencies, merit, self.alignment_mode)
        self.refresh_index += 1
        higher = torch.as_tensor(self.user_calibration.higher_activity_users,
                                 dtype=torch.long, device=users.device)
        sparse = (self.user_calibration.sparse_user_ids.detach().clone()
                  if self.enable_user_calibration else higher[:0])
        user_transport = (self.user_calibration.user_transport
                          if self.enable_user_calibration else None)
        user_bary = (self.user_calibration.barycentric_targets()
                     if self.enable_user_calibration and self.alignment_mode == 'ot'
                     else (self.user_calibration.alignment_targets
                           if self.enable_user_calibration else None))
        user_cal_x = (self.user_calibration.sparse_calibration_x.detach().clone()
                      if self.enable_user_calibration else None)
        item_target = (self.item_calibration.target_distribution
                       if self.enable_item_calibration else None)
        item_source = (self.item_calibration.source_distribution
                       if self.enable_item_calibration else None)
        item_transport = (self.item_calibration.transport_state
                          if self.enable_item_calibration else None)
        item_bary = (self.item_calibration.barycentric_targets_cache
                     if self.enable_item_calibration else None)
        item_anchors = (self.item_calibration.target_anchors
                        if self.enable_item_calibration else None)
        self.calibration_state = CalibrationState(
            self.refresh_index, higher, sparse,
            user_transport, item_target, item_transport,
            user_bary, user_cal_x, item_bary,
            self.training_candidates.detach().clone(),
            alignment_mode=self.alignment_mode,
            user_hard_indices=(self.user_calibration.hard_user_indices
                               if self.enable_user_calibration else None),
            item_hard_indices=(self.item_calibration.hard_item_indices
                               if self.enable_item_calibration else None),
            user_alignment_state=(self.user_calibration.mmd_state
                                  if self.enable_user_calibration else None),
            item_alignment_state=(self.item_calibration.mmd_state
                                  if self.enable_item_calibration else None),
            item_source_marginal=item_source,
            item_target_anchors=item_anchors)
        self.build_calibrated_embeddings()
        return self.calibration_state

    def build_calibrated_embeddings(self, calibration_state=None):
        users, items = self.get_native_embeddings()
        calibrated_users = (self.user_calibration.calibrate(
            users, self.enable_confidence) if self.enable_user_calibration else users)
        calibrated_items = (self.item_calibration.calibrate(
            items, self.enable_confidence) if self.enable_item_calibration else items)
        self.scoring_state = {
            'raw_user_repr': users, 'raw_item_repr': items,
            'calibrated_user_repr': calibrated_users,
            'calibrated_item_repr': calibrated_items}
        return CalibrationOutput(users, items, calibrated_users, calibrated_items,
                                 self.calibration_state)

    def forward(self, user_ids, item_ids):
        if self.scoring_state is None:
            self.refresh_calibration_state()
        return self.adapter.score_pairs_with_calibrated_state(
            user_ids, item_ids, self.scoring_state)

    def bpr_loss(self, users, positive_items, negative_items):
        return -F.logsigmoid(self.forward(users, positive_items)
                             - self.forward(users, negative_items)).mean()

    def bce_loss(self, users, items, labels):
        return F.binary_cross_entropy_with_logits(
            self.forward(users, items), labels.float())

    def compute_all_scores(self, device=None):
        if self.scoring_state is None:
            self.build_calibrated_embeddings()
        return self.adapter.score_all_with_calibrated_state(self.scoring_state)

    def compute_user_fixed_coupling_loss(self):
        if not self.enable_user_calibration:
            users, _ = self.get_native_embeddings()
            return torch.tensor(0.0, device=users.device)
        users, items = self.get_native_embeddings()
        return self.user_calibration.fixed_coupling_loss(users, items)

    def compute_item_fixed_coupling_loss(self):
        if not self.enable_item_calibration:
            _, items = self.get_native_embeddings()
            return torch.tensor(0.0, device=items.device)
        _, items = self.get_native_embeddings()
        return self.item_calibration.fixed_coupling_loss(items)

    def accuracy_parameters(self):
        return list(self.parameters())

    def fairness_correction_parameters(self):
        return (self.user_calibration.fairness_parameters()
                + self.item_calibration.fairness_parameters())

    def update_ema(self):
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                self.ema_state[name].mul_(self.ema_decay).add_(
                    parameter.detach(), alpha=1.0 - self.ema_decay)

    def clear_scoring_state(self):
        self.scoring_state = None
        if hasattr(self.backbone, '_clear_cache'):
            self.backbone._clear_cache()

    def load_checkpoint_state(self, checkpoint, accuracy_optimizer=None,
                              fairness_optimizer=None):
        if checkpoint.get('split_hash') not in {None, getattr(self.dataset, 'split_hash', None)}:
            raise ValueError('Checkpoint split hash mismatch')
        self.load_state_dict(checkpoint['model'])
        self.ema_state = {name: value.detach().clone().to(self.device_hint)
                          for name, value in checkpoint['ema_state'].items()}
        self.calibration_state = checkpoint['calibration_state']
        state = self.calibration_state
        self.alignment_mode = getattr(state, 'alignment_mode', self.alignment_mode)
        self.user_calibration.alignment_mode = self.alignment_mode
        self.item_calibration.alignment_mode = self.alignment_mode
        self.user_calibration.user_transport = state.user_transport
        self.user_calibration.sparse_user_ids = state.sparse_history_users
        self.user_calibration.sparse_calibration_x = state.user_calibration_x
        self.user_calibration.alignment_targets = state.user_barycentric_targets
        self.user_calibration.hard_user_indices = getattr(state, 'user_hard_indices', None)
        self.user_calibration.mmd_state = getattr(state, 'user_alignment_state', None)
        self.item_calibration.transport_state = state.item_transport
        self.item_calibration.target_distribution = state.item_target_marginal
        self.item_calibration.source_distribution = getattr(state, 'item_source_marginal', None)
        self.item_calibration.target_anchors = getattr(state, 'item_target_anchors', None)
        self.item_calibration.barycentric_targets_cache = state.item_barycentric_targets
        self.item_calibration.hard_item_indices = getattr(state, 'item_hard_indices', None)
        self.item_calibration.mmd_state = getattr(state, 'item_alignment_state', None)
        if accuracy_optimizer is not None and checkpoint.get('accuracy_optimizer'):
            accuracy_optimizer.load_state_dict(checkpoint['accuracy_optimizer'])
        if fairness_optimizer is not None and checkpoint.get('fairness_optimizer'):
            fairness_optimizer.load_state_dict(checkpoint['fairness_optimizer'])
        self.clear_scoring_state()
        return checkpoint

    def checkpoint_state(self, accuracy_optimizer=None, fairness_optimizer=None,
                         epoch=None, iteration=None, selection_metadata=None):
        return {
            'model': {name: value.detach().clone()
                      for name, value in self.state_dict().items()},
            'ema_state': {name: value.detach().clone()
                          for name, value in self.ema_state.items()},
            'calibration_state': copy.deepcopy(self.calibration_state),
            'accuracy_optimizer': (copy.deepcopy(accuracy_optimizer.state_dict())
                                   if accuracy_optimizer else None),
            'fairness_optimizer': (copy.deepcopy(fairness_optimizer.state_dict())
                                   if fairness_optimizer else None),
            'epoch': epoch, 'iteration': iteration,
            'selection_metadata': copy.deepcopy(selection_metadata or {}),
            'split_hash': getattr(self.dataset, 'split_hash', None),
            'config': copy.deepcopy(self.config)}
