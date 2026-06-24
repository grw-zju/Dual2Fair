import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_backbone import BaseBackbone


class NeuMF(BaseBackbone):
    def __init__(self, n_users, n_items, embedding_dim=64, mlp_layers=None):
        super().__init__(n_users, n_items, embedding_dim)
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)

        mlp_input_dim = embedding_dim * 2
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)

        self.mlp_layers = nn.ModuleList()
        for out_dim in mlp_layers:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, out_dim))
            mlp_input_dim = out_dim

        self.pred_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)

        self._init_weights()

    def get_user_embeddings(self):
        gmf_u = self.gmf_user_emb.weight
        mlp_u = self.mlp_user_emb.weight
        return gmf_u + mlp_u

    def get_item_embeddings(self):
        gmf_i = self.gmf_item_emb.weight
        mlp_i = self.mlp_item_emb.weight
        return gmf_i + mlp_i

    def forward(self, user_ids, item_ids):
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i

        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = mlp_input
        for layer in self.mlp_layers:
            mlp_out = nn.ReLU()(layer(mlp_out))

        concat_out = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.pred_layer(concat_out).squeeze(-1)
        return score

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)
        loss = -torch.log(nn.functional.sigmoid(pos_scores - neg_scores) + 1e-10)
        return loss.mean()

    def bce_loss(self, user_ids, item_ids, labels):
        scores = self.forward(user_ids, item_ids)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        return loss

    def compute_all_scores(self, device=None):
        cpu = torch.device('cpu')
        n_users = self.n_users
        n_items = self.n_items
        batch_size = 256
        all_scores = []
        with torch.no_grad():
            gmf_u_w = self.gmf_user_emb.weight.detach().to(cpu)
            gmf_i_w = self.gmf_item_emb.weight.detach().to(cpu)
            mlp_u_w = self.mlp_user_emb.weight.detach().to(cpu)
            mlp_i_w = self.mlp_item_emb.weight.detach().to(cpu)
            mlp_params = [(l.weight.detach().to(cpu), (l.bias.detach().to(cpu) if l.bias is not None else None)) for l in self.mlp_layers]
            pred_w = self.pred_layer.weight.detach().to(cpu)
            pred_b = self.pred_layer.bias.detach().to(cpu) if self.pred_layer.bias is not None else None
            for u_start in range(0, n_users, batch_size):
                u_end = min(u_start + batch_size, n_users)
                n_u = u_end - u_start
                u_ids = torch.arange(u_start, u_end)
                i_ids = torch.arange(n_items)
                u_expand = u_ids.unsqueeze(1).expand(n_u, n_items).reshape(-1)
                i_expand = i_ids.unsqueeze(0).expand(n_u, n_items).reshape(-1)
                gmf_u = gmf_u_w[u_expand]
                gmf_i = gmf_i_w[i_expand]
                gmf_out = gmf_u * gmf_i
                mlp_u = mlp_u_w[u_expand]
                mlp_i = mlp_i_w[i_expand]
                mlp_input = torch.cat([mlp_u, mlp_i], dim=-1)
                mlp_out = mlp_input
                for w, b in mlp_params:
                    mlp_out = F.relu(F.linear(mlp_out, w, b))
                concat_out = torch.cat([gmf_out, mlp_out], dim=-1)
                scores = F.linear(concat_out, pred_w, pred_b).squeeze(-1)
                scores = scores.reshape(n_u, n_items)
                all_scores.append(scores)
        return torch.cat(all_scores, dim=0)
