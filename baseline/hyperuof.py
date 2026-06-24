import torch
import torch.nn as nn
import numpy as np


class HyperUOF:
    """
    Hypergraph Convolutional Network for User-Oriented Fairness.
    In-processing method that lets disadvantaged users exploit high-order
    correlations learned from advantaged users.
    
    Ref: Han et al., "Hypergraph convolutional network for user-oriented fairness 
    in recommender systems", SIGIR 2024.
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2, device=None):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device or torch.device('cpu')
        
        self.user_emb = nn.Embedding(n_users, embedding_dim).to(self.device)
        self.item_emb = nn.Embedding(n_items, embedding_dim).to(self.device)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
    
    def set_hypergraph_adj(self, H_inv_sqrt, device=None):
        """
        Set hypergraph adjacency for convolution.
        H_inv_sqrt: normalized hypergraph incidence matrix (sparse tensor)
        """
        if device is None:
            device = self.device
        self.H_inv_sqrt = H_inv_sqrt.to(device)
    
    def hypergraph_conv(self, user_embs, item_embs):
        """
        Perform hypergraph convolution.
        e_u = Σ_{e∈E} 1/|e| * h(u,e) * Σ_{v∈e} 1/|e| * h(v,e) * e_v
        """
        all_embs = torch.cat([user_embs, item_embs], dim=0)
        
        if hasattr(self, 'H_inv_sqrt') and self.H_inv_sqrt is not None:
            convolved = torch.sparse.mm(self.H_inv_sqrt, all_embs)
            convolved = torch.sparse.mm(self.H_inv_sqrt.T, convolved)
        else:
            convolved = all_embs
        
        user_conv, item_conv = torch.split(convolved, [self.n_users, self.n_items])
        return user_conv, item_conv
    
    def forward(self, user_ids, item_ids):
        user_embs = self.user_emb(user_ids.to(self.device))
        item_embs = self.item_emb(item_ids.to(self.device))
        
        user_conv, item_conv = self.hypergraph_conv(
            self.user_emb.weight, self.item_emb.weight
        )
        
        u_final = user_embs + user_conv[user_ids.to(self.device)]
        i_final = item_embs + item_conv[item_ids.to(self.device)]
        
        scores = torch.sum(u_final * i_final, dim=-1)
        return scores
    
    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)
        loss = -torch.log(nn.functional.sigmoid(pos_scores - neg_scores) + 1e-10)
        return loss.mean()
    
    def fair_loss(self, adv_users, disadv_users):
        """
        Fairness regularization: align disadvantaged user embeddings toward
        advantaged user distribution.
        """
        adv_embs = self.user_emb.weight[adv_users].to(self.device)
        disadv_embs = self.user_emb.weight[disadv_users].to(self.device)
        
        adv_mean = adv_embs.mean(dim=0)
        disadv_mean = disadv_embs.mean(dim=0)
        
        loss = torch.norm(disadv_mean - adv_mean, p=2) ** 2
        return loss
    
    def get_user_embeddings(self):
        user_conv, _ = self.hypergraph_conv(self.user_emb.weight, self.item_emb.weight)
        return self.user_emb.weight + user_conv
    
    def get_item_embeddings(self):
        _, item_conv = self.hypergraph_conv(self.user_emb.weight, self.item_emb.weight)
        return self.item_emb.weight + item_conv
