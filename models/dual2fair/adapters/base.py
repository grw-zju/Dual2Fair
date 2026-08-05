from abc import ABC, abstractmethod

import torch.nn as nn


class BackboneAdapter(nn.Module, ABC):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @abstractmethod
    def get_raw_user_repr(self):
        raise NotImplementedError

    @abstractmethod
    def get_raw_item_repr(self):
        raise NotImplementedError

    @abstractmethod
    def score_pairs_with_calibrated_state(self, user_ids, item_ids, state):
        raise NotImplementedError

    @abstractmethod
    def score_all_with_calibrated_state(self, state):
        raise NotImplementedError

    def user_side_parameters(self):
        return []

    def item_side_parameters(self):
        return []
