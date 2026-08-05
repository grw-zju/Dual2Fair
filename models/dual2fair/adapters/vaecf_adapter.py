import torch

from .base import BackboneAdapter


class VAECFAdapter(BackboneAdapter):
    def get_raw_user_repr(self):
        return self.backbone.get_user_embeddings()

    def get_raw_item_repr(self):
        return self.backbone.get_item_embeddings()

    def score_pairs_with_calibrated_state(self, user_ids, item_ids, state):
        users = state['calibrated_user_repr'][user_ids]
        items = state['calibrated_item_repr'][item_ids]
        bias = self.backbone.get_item_bias()[item_ids]
        return torch.sum(users * items, dim=-1) + bias

    def score_all_with_calibrated_state(self, state):
        return (state['calibrated_user_repr'] @ state['calibrated_item_repr'].T
                + self.backbone.get_item_bias().unsqueeze(0))

    def user_side_parameters(self):
        return list(self.backbone.encoder.parameters())

    def item_side_parameters(self):
        return list(self.backbone.decoder.parameters())
