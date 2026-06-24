import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseBackbone(ABC, nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

    @abstractmethod
    def get_user_embeddings(self):
        pass

    @abstractmethod
    def get_item_embeddings(self):
        pass

    @abstractmethod
    def forward(self, user_ids, item_ids):
        pass

    @abstractmethod
    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        pass

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            scores = self.forward(user_ids, item_ids)
        return scores

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
