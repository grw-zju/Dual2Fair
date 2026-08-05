import torch
import torch.nn.functional as F

from .base import BackboneAdapter


class NeuMFAdapter(BackboneAdapter):
    def get_raw_user_repr(self):
        return 0.5 * (self.backbone.gmf_user_emb.weight + self.backbone.mlp_user_emb.weight)

    def get_raw_item_repr(self):
        return 0.5 * (self.backbone.gmf_item_emb.weight + self.backbone.mlp_item_emb.weight)

    def _branches(self, state):
        raw_users = state['raw_user_repr']
        raw_items = state['raw_item_repr']
        user_delta = state['calibrated_user_repr'] - raw_users
        item_delta = state['calibrated_item_repr'] - raw_items
        return {
            'gmf_user': self.backbone.gmf_user_emb.weight + user_delta,
            'mlp_user': self.backbone.mlp_user_emb.weight + user_delta,
            'gmf_item': self.backbone.gmf_item_emb.weight + item_delta,
            'mlp_item': self.backbone.mlp_item_emb.weight + item_delta,
        }

    def score_pairs_with_calibrated_state(self, user_ids, item_ids, state):
        branches = self._branches(state)
        gmf_output = branches['gmf_user'][user_ids] * branches['gmf_item'][item_ids]
        mlp_output = torch.cat([branches['mlp_user'][user_ids],
                                branches['mlp_item'][item_ids]], dim=-1)
        for layer in self.backbone.mlp_layers:
            mlp_output = F.relu(layer(mlp_output))
        return self.backbone.pred_layer(torch.cat([gmf_output, mlp_output], dim=-1)).squeeze(-1)

    def score_all_with_calibrated_state(self, state, user_batch_size=256):
        rows = []
        device = self.backbone.gmf_user_emb.weight.device
        item_ids = torch.arange(self.backbone.n_items, device=device)
        for start in range(0, self.backbone.n_users, user_batch_size):
            end = min(start + user_batch_size, self.backbone.n_users)
            users = torch.arange(start, end, device=device)
            user_ids = users[:, None].expand(-1, self.backbone.n_items).reshape(-1)
            expanded_items = item_ids[None, :].expand(len(users), -1).reshape(-1)
            scores = self.score_pairs_with_calibrated_state(user_ids, expanded_items, state)
            rows.append(scores.reshape(len(users), self.backbone.n_items))
        return torch.cat(rows, dim=0)

    def user_side_parameters(self):
        return list(self.backbone.gmf_user_emb.parameters()) + list(self.backbone.mlp_user_emb.parameters())

    def item_side_parameters(self):
        return list(self.backbone.gmf_item_emb.parameters()) + list(self.backbone.mlp_item_emb.parameters())
