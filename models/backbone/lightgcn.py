import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from .base_backbone import BaseBackbone


class LightGCN(BaseBackbone):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, dropout=0.0):
        super().__init__(n_users, n_items, embedding_dim)
        self.n_layers = n_layers
        self.dropout = dropout
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self._init_weights()
        self._cached_user_embs = None
        self._cached_item_embs = None
        self._adj_tensor = None
        self._adj_tensor_device = None
        self._adj_mat = None
        self._propagate_mode = 'mean'

    def set_adj_mat(self, adj_mat):
        self._adj_mat = adj_mat
        self._adj_tensor = None
        self._adj_tensor_device = None

    def _get_adj_tensor(self, device):
        if self._adj_tensor is None or self._adj_tensor_device != device:
            self._adj_tensor = self._sparse_mat_to_tensor(self._adj_mat, device)
            self._adj_tensor_device = device
        return self._adj_tensor

    def _propagate(self):
        device = self.user_embedding.weight.device
        adj = self._get_adj_tensor(device)
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        if self._propagate_mode == 'weighted':
            result = ego_embeddings
            for layer_idx in range(self.n_layers):
                ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
                result = result + ego_embeddings * (1.0 / (layer_idx + 2))
            user_all, item_all = torch.split(result, [self.n_users, self.n_items])
        else:
            all_embeddings = [ego_embeddings]
            for _ in range(self.n_layers):
                ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
                all_embeddings.append(ego_embeddings)
            all_embeddings = torch.stack(all_embeddings, dim=1)
            light_out = torch.mean(all_embeddings, dim=1)
            user_all, item_all = torch.split(light_out, [self.n_users, self.n_items])
        return user_all, item_all

    def get_user_embeddings(self):
        if self._cached_user_embs is not None:
            return self._cached_user_embs
        user_all, _ = self._propagate()
        return user_all

    def get_item_embeddings(self):
        if self._cached_item_embs is not None:
            return self._cached_item_embs
        _, item_all = self._propagate()
        return item_all

    def _update_cache(self):
        user_all, item_all = self._propagate()
        self._cached_user_embs = user_all.detach()
        self._cached_item_embs = item_all.detach()

    def _clear_cache(self):
        self._cached_user_embs = None
        self._cached_item_embs = None

    def _sparse_mat_to_tensor(self, sp_mat, device):
        sp_mat = sp_mat.tocoo()
        indices = torch.LongTensor(np.vstack([sp_mat.row, sp_mat.col]))
        values = torch.FloatTensor(sp_mat.data)
        shape = sp_mat.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device).to(torch.float32)

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        user_all, item_all = self._propagate()
        user_all = user_all.to(user_ids.device)
        item_all = item_all.to(user_ids.device)
        u_emb = user_all[user_ids]
        p_emb = item_all[pos_item_ids]
        n_emb = item_all[neg_item_ids]
        pos_scores = torch.sum(u_emb * p_emb, dim=-1)
        neg_scores = torch.sum(u_emb * n_emb, dim=-1)
        loss = -torch.log(F.sigmoid(pos_scores - neg_scores) + 1e-10)
        return loss.mean()

    def bce_loss(self, user_ids, item_ids, labels):
        user_all, item_all = self._propagate()
        user_all = user_all.to(user_ids.device)
        item_all = item_all.to(user_ids.device)
        u_emb = user_all[user_ids]
        i_emb = item_all[item_ids]
        scores = torch.sum(u_emb * i_emb, dim=-1)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        return loss

    def forward(self, user_ids, item_ids):
        user_all, item_all = self._propagate()
        user_all = user_all.to(user_ids.device)
        item_all = item_all.to(user_ids.device)
        u_emb = user_all[user_ids]
        i_emb = item_all[item_ids]
        scores = torch.sum(u_emb * i_emb, dim=-1)
        return scores

    def compute_all_scores(self, device=None):
        with torch.no_grad():
            user_all, item_all = self._propagate()
            scores = user_all @ item_all.T
        return scores
