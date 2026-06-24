import torch
import torch.nn as nn
from .user_calibration import UserRepresentationCalibration
from .item_calibration import ItemRepresentationCalibration
from .bilevel_opt import BiLevelOptimizer


class Dual2Fair:
    def __init__(self, backbone, dataset, config, device=None):
        self.backbone = backbone
        self.dataset = dataset
        self.config = config
        self.device = device or torch.device('cpu')

        d2f_config = config.get('dual2fair', {})

        self.user_calibration = UserRepresentationCalibration(
            epsilon=d2f_config.get('sinkhorn_epsilon', 0.1),
            fusion_alpha=d2f_config.get('fusion_alpha', 0.5),
            n_clusters=d2f_config.get('n_clusters', 64),
            max_disadv_users=d2f_config.get('max_disadv_users', 4096),
            device=self.device,
        )

        adv_users, disadv_users = dataset.get_advantaged_users(
            d2f_config.get('adv_ratio', 0.05))
        self.user_calibration.set_user_groups(adv_users, disadv_users, dataset.user_items)

        self.item_calibration = ItemRepresentationCalibration(
            epsilon=d2f_config.get('sinkhorn_epsilon', 0.1),
            alpha_smoothing=d2f_config.get('alpha_smoothing', 1.0),
            beta_scaling=d2f_config.get('beta_scaling', 1.0),
            nystrom_rank=d2f_config.get('nystrom_rank', 32),
            rank_increment=d2f_config.get('nystrom_rank_increment', 8),
            nystrom_tol=d2f_config.get('nystrom_tol', 1e-3),
            tikhonov_lambda=d2f_config.get('nystrom_lambda', 1e-4),
            sinkhorn_max_iter=d2f_config.get('sinkhorn_max_iter', 100),
            sinkhorn_convergence_tol=d2f_config.get('sinkhorn_convergence_tol', 1e-3),
            max_ot_items=d2f_config.get('max_ot_items', 4096),
            device=self.device,
        )

        self.bilevel_optimizer = BiLevelOptimizer(
            beta=d2f_config.get('bilevel_beta', 3),
            alpha1=d2f_config.get('mirror_alpha1', 1.0),
            alpha2=d2f_config.get('mirror_alpha2', 0.1),
            learning_rate=config.get('model', {}).get('learning_rate', 0.001),
        )

        self.lambda1 = d2f_config.get('lambda1', 0.1)
        self.lambda2 = d2f_config.get('lambda2', 0.1)
        self.adv_ratio = d2f_config.get('adv_ratio', 0.05)

    def _get_raw_embs(self):
        return self.backbone.get_user_embeddings().to(self.device), \
               self.backbone.get_item_embeddings().to(self.device)

    def compute_fairness_losses(self):
        user_embs, item_embs = self._get_raw_embs()
        self.user_calibration.fit_gmm(item_embs)
        L_user = self.user_calibration.compute_user_ot_loss(item_embs)
        L_item = self.item_calibration.compute_item_ot_loss(item_embs, self.dataset.item_freq)
        return L_user, L_item

    def get_calibrated_embeddings(self):
        user_embs, item_embs = self._get_raw_embs()
        calibrated_user_embs = self.user_calibration.calibrate_disadvantaged_users(user_embs, item_embs)
        calibrated_item_embs, _ = self.item_calibration.calibrate_items(item_embs, self.dataset.item_freq)
        return calibrated_user_embs, calibrated_item_embs
