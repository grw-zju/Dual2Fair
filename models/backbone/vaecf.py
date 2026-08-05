import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone


class VAECF(BaseBackbone):
    def __init__(self, n_users, n_items, embedding_dim=64, encoder_dims=None,
                 decoder_dims=None, encoder_hidden_dims=None,
                 decoder_hidden_dims=None, dropout=0.5, anneal_cap=0.2,
                 total_anneal_steps=200000):
        super().__init__(n_users, n_items, embedding_dim)
        if encoder_dims is None:
            encoder_dims = ([n_items] + list(encoder_hidden_dims) + [embedding_dim]
                            if encoder_hidden_dims is not None
                            else [n_items, 600, 200, embedding_dim])
        if decoder_dims is None:
            decoder_dims = ([embedding_dim] + list(decoder_hidden_dims) + [n_items]
                            if decoder_hidden_dims is not None
                            else [embedding_dim, 200, n_items])
        self.dropout = dropout
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.update_count = 0
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.encoder = nn.ModuleList([
            nn.Linear(encoder_dims[index],
                      encoder_dims[index + 1] * 2 if index == len(encoder_dims) - 2
                      else encoder_dims[index + 1])
            for index in range(len(encoder_dims) - 1)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(decoder_dims[index], decoder_dims[index + 1])
            for index in range(len(decoder_dims) - 1)
        ])
        self._cached_user_embeddings = None
        self._init_weights()

    def set_interact_mat(self, interact_mat):
        self._interact_mat = interact_mat
        self._clear_cache()

    def _get_interact_batch(self, start, end):
        import numpy as np
        matrix = self._interact_mat[start:end]
        values = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
        return torch.from_numpy(values).float()

    def _clear_cache(self):
        self._cached_user_embeddings = None

    def _encode(self, x):
        hidden = F.dropout(x, p=self.dropout, training=self.training)
        for layer in self.encoder:
            hidden = torch.tanh(layer(hidden))
        return hidden[:, :self.embedding_dim], hidden[:, self.embedding_dim:]

    def _decode_hidden(self, z):
        hidden = z
        for layer in self.decoder[:-1]:
            hidden = torch.tanh(layer(hidden))
        return hidden

    def _decode(self, z):
        return self.decoder[-1](self._decode_hidden(z))

    def _all_latent_means(self):
        device = next(self.parameters()).device
        if self.training:
            chunks = []
            for start in range(0, self.n_users, 1024):
                rows = self._get_interact_batch(start, min(start + 1024, self.n_users)).to(device)
                mean, _ = self._encode(rows)
                chunks.append(mean)
            return torch.cat(chunks, dim=0)
        chunks = []
        with torch.no_grad():
            for start in range(0, self.n_users, 1024):
                rows = self._get_interact_batch(start, min(start + 1024, self.n_users)).to(device)
                mean, _ = self._encode(rows)
                chunks.append(mean)
        return torch.cat(chunks, dim=0)

    def get_user_embeddings(self):
        if self._cached_user_embeddings is not None and not self.training:
            return self._cached_user_embeddings
        latent = self._all_latent_means()
        representations = self._decode_hidden(latent)
        if not self.training:
            self._cached_user_embeddings = representations.detach()
        return representations

    def get_item_embeddings(self):
        return self.decoder[-1].weight

    def get_item_bias(self):
        bias = self.decoder[-1].bias
        if bias is None:
            return torch.zeros(self.n_items, device=self.get_item_embeddings().device)
        return bias

    def forward(self, user_ids, item_ids):
        users = self.get_user_embeddings()[user_ids]
        items = self.get_item_embeddings()[item_ids]
        return torch.sum(users * items, dim=-1) + self.get_item_bias()[item_ids]

    def reparameterize(self, mean, log_variance):
        if not self.training:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def kl_loss(self, mean, log_variance):
        return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

    def train_step(self, interact_mat, device):
        self.update_count += 1
        anneal = min(self.anneal_cap, self.update_count / self.total_anneal_steps)
        x = interact_mat.to(device)
        mean, log_variance = self._encode(x)
        logits = self._decode(self.reparameterize(mean, log_variance))
        reconstruction = -torch.sum(x * F.log_softmax(logits, dim=1)) / x.shape[0]
        return reconstruction + anneal * self.kl_loss(mean, log_variance) / x.shape[0], logits

    def train_batch(self, user_rows, device):
        self._clear_cache()
        loss, _ = self.train_step(user_rows, device)
        return loss

    def compute_all_scores(self, device=None):
        with torch.no_grad():
            return (self.get_user_embeddings() @ self.get_item_embeddings().T
                    + self.get_item_bias().unsqueeze(0))

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        positive = self.forward(user_ids, pos_item_ids)
        negative = self.forward(user_ids, neg_item_ids)
        return -F.logsigmoid(positive - negative).mean()

    def bce_loss(self, user_ids, item_ids, labels):
        return F.binary_cross_entropy_with_logits(self.forward(user_ids, item_ids), labels.float())
