import torch
import numpy as np
from .sinkhorn import sinkhorn_nystrom, compute_cosine_cost_matrix


class ItemRepresentationCalibration:
    def __init__(self, epsilon=0.1, alpha_smoothing=1.0, beta_scaling=1.0,
                 nystrom_rank=32, rank_increment=8, nystrom_tol=1e-3,
                 tikhonov_lambda=1e-4, sinkhorn_max_iter=100,
                 sinkhorn_convergence_tol=1e-3, max_ot_items=4096, device=None):
        self.epsilon = epsilon
        self.alpha_smoothing = alpha_smoothing
        self.beta_scaling = beta_scaling
        self.nystrom_rank = nystrom_rank
        self.rank_increment = rank_increment
        self.nystrom_tol = nystrom_tol
        self.tikhonov_lambda = tikhonov_lambda
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_convergence_tol = sinkhorn_convergence_tol
        self.max_ot_items = max_ot_items
        self.device = device or torch.device('cpu')

    def _frequency_values(self, n_items, item_freq):
        freq_values = torch.zeros(n_items, device=self.device)
        for iid, freq in item_freq.items():
            if iid < n_items:
                freq_values[iid] = freq
        return freq_values

    def _select_ot_items(self, freq_values):
        n_items = freq_values.shape[0]
        if n_items <= self.max_ot_items:
            return torch.arange(n_items, device=self.device)
        weights = 1.0 / (freq_values + self.alpha_smoothing) ** self.beta_scaling
        weights = weights / weights.sum().clamp_min(1e-12)
        return torch.multinomial(weights, self.max_ot_items, replacement=False)

    def compute_item_ot_loss(self, item_embeddings, item_freq):
        """
        Module B: Item Representation Calibration via OT.
        
        Args:
            item_embeddings: (n_items, d) item embeddings tensor
            item_freq: dict mapping item_id -> interaction frequency
        
        Returns:
            item_ot_loss: scalar OT loss value (L_item)
        """
        n_items = item_embeddings.shape[0]
        
        freq_values = self._frequency_values(n_items, item_freq)
        selected = self._select_ot_items(freq_values)
        selected_item_embeddings = item_embeddings[selected]
        selected_freq_values = freq_values[selected]
        
        weights = 1.0 / (selected_freq_values + self.alpha_smoothing) ** self.beta_scaling
        weights = weights / weights.sum()
        
        n_selected = selected_item_embeddings.shape[0]
        uniform_weights = torch.ones(n_selected, device=self.device) / n_selected
        
        C_v = compute_cosine_cost_matrix(selected_item_embeddings, selected_item_embeddings, self.device)
        
        gamma_v = sinkhorn_nystrom(C_v, weights, uniform_weights, epsilon=self.epsilon,
                                    max_iter=self.sinkhorn_max_iter,
                                    convergence_tol=self.sinkhorn_convergence_tol,
                                    nystrom_rank=self.nystrom_rank,
                                    rank_increment=self.rank_increment,
                                    nystrom_tol=self.nystrom_tol,
                                    tikhonov_lambda=self.tikhonov_lambda,
                                    device=self.device)

        item_ot_loss = (gamma_v * C_v).sum() - self.epsilon * self._entropy(gamma_v)

        return item_ot_loss

    def calibrate_items(self, item_embeddings, item_freq):
        """
        Apply OT calibration to item embeddings and return calibrated embeddings.
        
        Args:
            item_embeddings: (n_items, d) item embeddings
            item_freq: dict mapping item_id -> frequency
        
        Returns:
            calibrated_item_embs: (n_items, d) calibrated item embeddings
            item_ot_loss: scalar loss
        """
        n_items = item_embeddings.shape[0]
        freq_values = self._frequency_values(n_items, item_freq)
        selected = self._select_ot_items(freq_values)
        selected_item_embeddings = item_embeddings[selected]
        selected_freq_values = freq_values[selected]
        
        weights = 1.0 / (selected_freq_values + self.alpha_smoothing) ** self.beta_scaling
        weights = weights / weights.sum()
        
        n_selected = selected_item_embeddings.shape[0]
        uniform_weights = torch.ones(n_selected, device=self.device) / n_selected
        
        C_v = compute_cosine_cost_matrix(selected_item_embeddings, selected_item_embeddings, self.device)
        
        gamma_v = sinkhorn_nystrom(C_v, weights, uniform_weights, epsilon=self.epsilon,
                                    max_iter=self.sinkhorn_max_iter,
                                    convergence_tol=self.sinkhorn_convergence_tol,
                                    nystrom_rank=self.nystrom_rank,
                                    rank_increment=self.rank_increment,
                                    nystrom_tol=self.nystrom_tol,
                                    tikhonov_lambda=self.tikhonov_lambda,
                                    device=self.device)

        item_ot_loss = (gamma_v * C_v).sum() - self.epsilon * self._entropy(gamma_v)

        calibrated_selected = gamma_v.detach() @ selected_item_embeddings
        calibrated_item_embs = item_embeddings.clone()
        calibrated_item_embs[selected] = calibrated_selected

        return calibrated_item_embs, item_ot_loss

    def _entropy(self, gamma):
        return -torch.sum(gamma * torch.log(gamma + 1e-10))
