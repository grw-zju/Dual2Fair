import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_adapter
from .item_calibration import ItemRepresentationCalibration
from .state import CalibrationOutput
from .user_calibration import UserRepresentationCalibration


class Dual2Fair(nn.Module):
    def __init__(self, backbone, dataset, config, device=None, backbone_name=None):
        super().__init__()
        self.backbone = backbone
        self.dataset = dataset
        self.config = config
        self.device_hint = torch.device(device or 'cpu')
        name = backbone_name or getattr(backbone, '_backbone_name_hint', None)
        if name is None:
            name = backbone.__class__.__name__.lower()
        self.backbone_name = name
        self.adapter = get_adapter(name, backbone)
        settings = config.get('dual2fair', {})
        seed = config.get('seeds', {}).get('ot_sampling', 42)
        raw_user_repr, raw_item_repr = self._raw_representations()
        representation_dim = raw_item_repr.shape[1]
        self.user_calibration = UserRepresentationCalibration(
            epsilon=settings.get('sinkhorn_epsilon', 0.1),
            rho_u=settings.get('rho_u', settings.get('fusion_alpha', 0.5)),
            n_clusters=settings.get('n_clusters', 64),
            user_chunk_size=settings.get('user_chunk_size', 4096),
            representation_dim=representation_dim,
            random_state=seed,
            sinkhorn_max_iter=settings.get('sinkhorn_max_iter', 100),
            sinkhorn_convergence_tol=settings.get('sinkhorn_convergence_tol', 1e-3),
            device=self.device_hint)
        advantaged, disadvantaged = dataset.get_advantaged_users(settings.get('adv_ratio', 0.05))
        self.user_calibration.set_user_groups(advantaged, disadvantaged, dataset.user_items)
        self.item_calibration = ItemRepresentationCalibration(
            epsilon=settings.get('sinkhorn_epsilon', 0.1),
            alpha_smoothing=settings.get('alpha_smoothing', 1.0),
            beta_scaling=settings.get('beta_scaling', 1.0),
            sinkhorn_max_iter=settings.get('sinkhorn_max_iter', 100),
            sinkhorn_convergence_tol=settings.get('sinkhorn_convergence_tol', 1e-3),
            max_ot_items=settings.get('max_ot_items', 4096),
            item_anchor_count=settings.get('item_anchor_count', 256),
            item_target_mode=settings.get('item_target_mode', 'merit_uniform_mixture'),
            merit_uniform_gamma=settings.get('merit_uniform_gamma', 0.5),
            rho_v=settings.get('rho_v', 0.5), representation_dim=representation_dim,
            random_state=seed,
            device=self.device_hint)
        self.lambda1 = settings.get('lambda1', 0.1)
        self.lambda2 = settings.get('lambda2', 0.1)
        self.calibration_state = None
        self.last_output = None

    def _raw_representations(self):
        return self.adapter.get_raw_user_repr(), self.adapter.get_raw_item_repr()

    def _merit(self, raw_users, raw_items):
        settings = self.config.get('dual2fair', {})
        generator = torch.Generator(device='cpu').manual_seed(
            self.config.get('seeds', {}).get('ot_sampling', 42))
        n_users = raw_users.shape[0]
        n_items = raw_items.shape[0]
        user_count = min(settings.get('merit_user_sample_size', 1024), n_users)
        item_count = min(settings.get('merit_item_sample_size', 2048), n_items)
        sampled_users = torch.randperm(n_users, generator=generator)[:user_count].to(raw_users.device)
        if self.item_calibration.anchor_indices is not None:
            sampled_items = self.item_calibration.anchor_indices.to(raw_items.device)
        else:
            sampled_items = torch.randperm(n_items, generator=generator)[:item_count].to(raw_items.device)
        merit = torch.ones(n_items, dtype=raw_items.dtype, device=raw_items.device)
        accumulated = torch.zeros(len(sampled_items), dtype=raw_items.dtype,
                                  device=raw_items.device)
        batch_size = settings.get('merit_user_batch_size', 64)
        state = {
            'raw_user_repr': raw_users.detach(),
            'raw_item_repr': raw_items.detach(),
            'calibrated_user_repr': raw_users.detach(),
            'calibrated_item_repr': raw_items.detach(),
        }
        with torch.no_grad():
            for start in range(0, len(sampled_users), batch_size):
                users = sampled_users[start:start + batch_size]
                user_ids = users[:, None].expand(-1, len(sampled_items)).reshape(-1)
                item_ids = sampled_items[None, :].expand(len(users), -1).reshape(-1)
                scores = self.adapter.score_pairs_with_calibrated_state(
                    user_ids, item_ids, state).reshape(len(users), len(sampled_items))
                order = torch.argsort(scores, dim=1)
                ranks = torch.empty_like(scores)
                values = torch.arange(1, len(sampled_items) + 1, device=scores.device,
                                      dtype=scores.dtype).expand_as(scores)
                ranks.scatter_(1, order, values)
                accumulated += (ranks / max(1, len(sampled_items))).sum(dim=0)
        merit[sampled_items] = accumulated / max(1, len(sampled_users))
        return merit

    def update_calibration_state(self):
        raw_users, raw_items = self._raw_representations()
        self.user_calibration.fit_gmm(raw_items.detach())
        user_loss = self.user_calibration.compute_user_ot_loss(raw_items)
        frequencies = self.item_calibration._frequency_values(
            raw_items.shape[0], self.dataset.item_freq, raw_items.dtype, raw_items.device)
        self.item_calibration.anchor_indices = self.item_calibration._select_anchors(frequencies)
        merit = self._merit(raw_users, raw_items)
        self.item_calibration.update_state(raw_items.detach(), self.dataset.item_freq, merit)
        item_loss = self.item_calibration.compute_item_ot_loss(raw_items, self.dataset.item_freq, merit)
        calibrated_users = self.user_calibration.calibrate_disadvantaged_users(
            raw_users, raw_items, refresh=False)
        calibrated_items, _ = self.item_calibration.calibrate_items(
            raw_items, self.dataset.item_freq, merit, refresh=False)
        self.calibration_state = {
            'raw_user_repr': raw_users,
            'raw_item_repr': raw_items,
            'calibrated_user_repr': calibrated_users,
            'calibrated_item_repr': calibrated_items,
        }
        self.last_output = CalibrationOutput(
            raw_users, raw_items, calibrated_users, calibrated_items,
            self.user_calibration.cached_gamma_u,
            self.item_calibration.cached_item_plans[0] if self.item_calibration.cached_item_plans else None,
            user_ot_objective=user_loss, item_ot_objective=item_loss,
            fusion_rho_u=self.user_calibration.rho_u,
            fusion_rho_v=self.item_calibration.rho_v,
            scorer_state=self.calibration_state)
        return self.last_output

    def refresh_scoring_state(self):
        raw_users, raw_items = self._raw_representations()
        calibrated_users = self.user_calibration.calibrate_disadvantaged_users(
            raw_users, raw_items, refresh=False)
        merit = self._merit(raw_users, raw_items)
        calibrated_items, _ = self.item_calibration.calibrate_items(
            raw_items, self.dataset.item_freq, merit, refresh=False)
        self.calibration_state = {
            'raw_user_repr': raw_users,
            'raw_item_repr': raw_items,
            'calibrated_user_repr': calibrated_users,
            'calibrated_item_repr': calibrated_items,
        }
        return self.calibration_state

    def training_fairness_losses(self, refresh=True):
        state = self.refresh_scoring_state() if refresh else self._state()
        user_ids = self.user_calibration.cached_user_ids
        if user_ids is None or user_ids.numel() == 0:
            user_loss = state['raw_user_repr'].sum() * 0.0
        else:
            barycenters = self.user_calibration.barycentric_projection(
                self.user_calibration.cached_gamma_u,
                self.user_calibration.prototypes.detach())
            projected = self.user_calibration.projection(barycenters)
            raw_users = state['raw_user_repr'][user_ids].detach()
            calibrated_users = (self.user_calibration.rho_u * raw_users
                                + (1.0 - self.user_calibration.rho_u) * projected)
            user_loss = (1.0 - F.cosine_similarity(
                raw_users, calibrated_users, dim=1)).mean()
        raw_items = state['raw_item_repr'].detach()
        projected_items = self.item_calibration.projection(
            self.item_calibration.cached_item_projection.detach())
        calibrated_items = (self.item_calibration.rho_v * raw_items
                            + (1.0 - self.item_calibration.rho_v) * projected_items)
        item_loss = (1.0 - F.cosine_similarity(
            raw_items, calibrated_items, dim=1)).mean()
        return user_loss, item_loss

    def _state(self):
        if self.calibration_state is None:
            self.update_calibration_state()
        return self.calibration_state

    def forward(self, user_ids, item_ids):
        return self.adapter.score_pairs_with_calibrated_state(user_ids, item_ids, self._state())

    def bpr_loss(self, user_ids, positive_item_ids, negative_item_ids):
        positive = self.forward(user_ids, positive_item_ids)
        negative = self.forward(user_ids, negative_item_ids)
        return -F.logsigmoid(positive - negative).mean()

    def bce_loss(self, user_ids, item_ids, labels):
        return F.binary_cross_entropy_with_logits(self.forward(user_ids, item_ids), labels.float())

    def vaecf_reconstruction_loss(self, user_ids):
        if self.backbone_name != 'vaecf':
            raise TypeError('VAECF reconstruction loss requires the VAECF adapter')
        state = self._state()
        logits = (state['calibrated_user_repr'][user_ids]
                  @ state['calibrated_item_repr'].T
                  + self.backbone.get_item_bias().unsqueeze(0))
        rows = self.backbone._get_interact_batch(
            int(user_ids.min()), int(user_ids.max()) + 1).to(logits.device)
        if not torch.equal(user_ids, torch.arange(int(user_ids.min()),
                                                  int(user_ids.max()) + 1,
                                                  device=user_ids.device)):
            rows = torch.stack([
                self.backbone._get_interact_batch(int(user), int(user) + 1)[0]
                for user in user_ids.cpu().tolist()]).to(logits.device)
        mean, log_variance = self.backbone._encode(rows)
        reconstruction = -torch.sum(rows * F.log_softmax(logits, dim=1)) / rows.shape[0]
        anneal = min(self.backbone.anneal_cap,
                     self.backbone.update_count / self.backbone.total_anneal_steps)
        return reconstruction + anneal * self.backbone.kl_loss(mean, log_variance) / rows.shape[0]

    def compute_all_scores(self, device=None):
        return self.adapter.score_all_with_calibrated_state(self._state())

    def compute_fairness_losses(self, refresh=False):
        if refresh or self.last_output is None:
            self.update_calibration_state()
        return self.last_output.user_ot_objective, self.last_output.item_ot_objective

    def get_calibrated_embeddings(self):
        state = self._state()
        return state['calibrated_user_repr'], state['calibrated_item_repr']

    def checkpoint_state(self, optimizer=None, scheduler=None, epoch=None, global_step=None):
        return {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
            'global_step': global_step,
            'config': self.config,
            'dataset': self.dataset.name,
            'split_hash': getattr(self.dataset, 'split_hash', None),
            'backbone_name': self.backbone_name,
            'calibration_metadata': {
                'adv_users': self.user_calibration.adv_users,
                'disadv_users': self.user_calibration.disadv_users,
                'item_target_mode': self.item_calibration.item_target_mode,
            },
        }

    def save_checkpoint(self, path, optimizer=None, scheduler=None,
                        epoch=None, global_step=None):
        torch.save(self.checkpoint_state(optimizer, scheduler, epoch, global_step), path)

    def load_checkpoint(self, path, optimizer=None, scheduler=None, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device_hint,
                                weights_only=False)
        if checkpoint.get('split_hash') not in {None, getattr(self.dataset, 'split_hash', None)}:
            raise ValueError('Checkpoint split hash does not match the loaded dataset')
        model_state = checkpoint['model']
        dynamic_buffers = {
            'user_calibration.gmm_weights': (self.user_calibration, 'gmm_weights'),
            'user_calibration.gmm_covariances': (self.user_calibration, 'gmm_covariances'),
            'user_calibration.prototypes': (self.user_calibration, 'prototypes'),
            'user_calibration.cached_gamma_u': (self.user_calibration, 'cached_gamma_u'),
            'user_calibration.cached_user_ids': (self.user_calibration, 'cached_user_ids'),
            'item_calibration.anchor_indices': (self.item_calibration, 'anchor_indices'),
            'item_calibration.target_distribution': (self.item_calibration, 'target_distribution'),
            'item_calibration.cached_item_projection': (self.item_calibration, 'cached_item_projection'),
        }
        for key, (module, name) in dynamic_buffers.items():
            if key in model_state:
                setattr(module, name, torch.empty_like(model_state[key], device=self.device_hint))
        self.load_state_dict(model_state)
        if optimizer is not None and checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        self.calibration_state = None
        self.last_output = None
        if hasattr(self.backbone, '_clear_cache'):
            self.backbone._clear_cache()
        self.refresh_scoring_state()
        return checkpoint

    def clear_calibration_state(self):
        self.calibration_state = None
        self.last_output = None
        if hasattr(self.backbone, '_clear_cache'):
            self.backbone._clear_cache()

    @staticmethod
    def _gradient_norm(loss, parameters):
        params = [parameter for parameter in parameters if parameter.requires_grad]
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        values = [gradient.norm().pow(2) for gradient in gradients if gradient is not None]
        return torch.sqrt(torch.stack(values).sum()) if values else loss.new_tensor(0.0)

    def compute_gradient_diagnostics(self):
        user_loss, item_loss = self.compute_fairness_losses(refresh=True)
        user_params = self.adapter.user_side_parameters()
        item_params = self.adapter.item_side_parameters()
        return {
            'user_loss_on_user_params': float(self._gradient_norm(user_loss, user_params).detach().cpu()),
            'user_loss_on_item_params': float(self._gradient_norm(user_loss, item_params).detach().cpu()),
            'item_loss_on_user_params': float(self._gradient_norm(item_loss, user_params).detach().cpu()),
            'item_loss_on_item_params': float(self._gradient_norm(item_loss, item_params).detach().cpu()),
        }
