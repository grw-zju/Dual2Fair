import torch

from .base import BackboneAdapter


class LightGCNAdapter(BackboneAdapter):
    def get_raw_user_repr(self):
        users, _ = self.backbone._propagate()
        return users

    def get_raw_item_repr(self):
        _, items = self.backbone._propagate()
        return items

    def score_pairs_with_calibrated_state(self, user_ids, item_ids, state):
        users = state['calibrated_user_repr'][user_ids]
        items = state['calibrated_item_repr'][item_ids]
        return torch.sum(users * items, dim=-1)

    def score_all_with_calibrated_state(self, state):
        return state['calibrated_user_repr'] @ state['calibrated_item_repr'].T

    def user_side_parameters(self):
        return list(self.backbone.user_embedding.parameters())

    def item_side_parameters(self):
        return list(self.backbone.item_embedding.parameters())
