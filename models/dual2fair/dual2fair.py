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
        self.user_calibration = UserRepresentationCalibration(
            epsilon=settings.get('sinkhorn_epsilon', 0.1),
            rho_u=settings.get('rho_u', settings.get('fusion_alpha', 0.5)),
            n_clusters=settings.get('n_clusters', 64),
            user_chunk_size=settings.get('user_chunk_size', 4096),
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
            rho_v=settings.get('rho_v', 0.5), random_state=seed,
            device=self.device_hint)
        self.lambda1 = settings.get('lambda1', 0.1)
        self.lambda2 = settings.get('lambda2', 0.1)
        self.calibration_state = None
        self.last_output = None

    def _raw_representations(self):
        return self.adapter.get_raw_user_repr(), self.adapter.get_raw_item_repr()

    def _merit(self, raw_users, raw_items):
        with torch.no_grad():
            scores = raw_users.detach() @ raw_items.detach().T
            order = torch.argsort(scores, dim=1)
            ranks = torch.empty_like(scores)
            rank_values = torch.arange(1, scores.shape[1] + 1, device=scores.device,
                                       dtype=scores.dtype).expand_as(scores)
            ranks.scatter_(1, order, rank_values)
            return (ranks / scores.shape[1]).mean(dim=0)

    def update_calibration_state(self):
        raw_users, raw_items = self._raw_representations()
        self.user_calibration.fit_gmm(raw_items.detach())
        user_loss = self.user_calibration.compute_user_ot_loss(raw_items)
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
        self.clear_calibration_state()
        self.update_calibration_state()
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
