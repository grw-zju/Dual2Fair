import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_backbone import BaseBackbone


class VAECF(BaseBackbone):
    def __init__(self, n_users, n_items, embedding_dim=64, encoder_dims=None, decoder_dims=None,
                 encoder_hidden_dims=None, decoder_hidden_dims=None, dropout=0.5, anneal_cap=0.2, total_anneal_steps=200000):
        super().__init__(n_users, n_items, embedding_dim)
        if encoder_dims is None:
            if encoder_hidden_dims is not None:
                encoder_dims = [n_items] + list(encoder_hidden_dims) + [embedding_dim]
            else:
                encoder_dims = [n_items, 600, 200, embedding_dim]
        if decoder_dims is None:
            if decoder_hidden_dims is not None:
                decoder_dims = [embedding_dim] + list(decoder_hidden_dims) + [n_items]
            else:
                decoder_dims = [embedding_dim, 200, n_items]

        self.dropout = dropout
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.update_count = 0

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            if i == len(encoder_dims) - 2:
                encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1] * 2))
            else:
                encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
        self.decoder = nn.ModuleList(decoder_layers)

        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self._cached_user_embeddings = None

        self._init_weights()

    def get_user_embeddings(self):
        if self._cached_user_embeddings is not None and not self.training:
            return self._cached_user_embeddings.to(self.item_emb.weight.device)

        device = self.item_emb.weight.device
        was_training = self.training
        self.eval()
        user_chunks = []
        batch_size = getattr(self, 'embedding_batch_size', 1024)
        with torch.no_grad():
            for start in range(0, self.n_users, batch_size):
                end = min(start + batch_size, self.n_users)
                x = self._get_interact_batch(start, end).to(device)
                mu, _ = self._encode(x)
                user_chunks.append(mu.detach().cpu())
        if was_training:
            self.train()
        user_embs = torch.cat(user_chunks, dim=0)
        if not was_training:
            self._cached_user_embeddings = user_embs
        return user_embs.to(device)

    def get_item_embeddings(self):
        return self.item_emb.weight

    def _get_interact_mat(self):
        if isinstance(self._interact_mat, torch.Tensor):
            return self._interact_mat
        raise RuntimeError("VAECF interaction matrix is sparse; use _get_interact_batch for memory-safe access.")

    def set_interact_mat(self, interact_mat):
        self._interact_mat = interact_mat
        self._cached_user_embeddings = None

    def _get_interact_batch(self, start, end):
        import numpy as np
        mat = self._interact_mat[start:end]
        arr = mat.toarray() if hasattr(mat, 'toarray') else np.array(mat)
        return torch.from_numpy(arr).float()

    def _clear_cache(self):
        self._cached_user_embeddings = None

    def _encode(self, x):
        h = F.dropout(x, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            h = torch.tanh(h)
        mu = h[:, :self.embedding_dim]
        logvar = h[:, self.embedding_dim:]
        return mu, logvar

    def _decode(self, z):
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i < len(self.decoder) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, user_ids, item_ids):
        user_embs = self.get_user_embeddings()
        item_embs = self.get_item_embeddings()
        u_emb = user_embs[user_ids]
        i_emb = item_embs[item_ids]
        scores = torch.sum(u_emb * i_emb, dim=-1)
        return scores

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def kl_loss(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def train_step(self, interact_mat, device):
        self.update_count += 1
        anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)

        x = interact_mat.to(device)

        mu, logvar = self._encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self._decode(z)

        softmax_recon = F.softmax(reconstructed, dim=1)
        ce_loss = -torch.sum(x * torch.log(softmax_recon + 1e-10)) / x.shape[0]
        kl_loss = self.kl_loss(mu, logvar) / x.shape[0]

        total_loss = ce_loss + anneal * kl_loss
        return total_loss, reconstructed

    def train_batch(self, user_rows, device):
        self._clear_cache()
        self.update_count += 1
        anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)

        x = user_rows.to(device)
        mu, logvar = self._encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self._decode(z)

        log_softmax_recon = F.log_softmax(reconstructed, dim=1)
        ce_loss = -torch.sum(x * log_softmax_recon) / x.shape[0]
        kl_loss = self.kl_loss(mu, logvar) / x.shape[0]
        return ce_loss + anneal * kl_loss

    def compute_all_scores(self, device=None):
        with torch.no_grad():
            user_embs = self.get_user_embeddings().cpu()
            item_embs = self.get_item_embeddings().cpu()
            scores = user_embs @ item_embs.T
        return scores

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)
        loss = -torch.log(F.sigmoid(pos_scores - neg_scores) + 1e-10)
        return loss.mean()
