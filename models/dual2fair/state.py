from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .sinkhorn import LowRankTransportState


@dataclass
class CalibrationState:
    refresh_index: int
    higher_activity_users: torch.Tensor
    sparse_history_users: torch.Tensor
    user_transport: Optional[torch.Tensor]
    item_target_marginal: Optional[torch.Tensor]
    item_transport: Optional[LowRankTransportState]
    user_barycentric_targets: Optional[torch.Tensor]
    user_calibration_x: Optional[torch.Tensor]
    item_barycentric_targets: Optional[torch.Tensor]
    training_candidates: torch.Tensor
    candidate_version: int = 1
    alignment_mode: str = 'ot'
    user_hard_indices: Optional[torch.Tensor] = None
    item_hard_indices: Optional[torch.Tensor] = None
    user_alignment_state: Any = None
    item_alignment_state: Any = None
    item_source_marginal: Optional[torch.Tensor] = None
    item_target_anchors: Optional[torch.Tensor] = None


@dataclass
class CalibrationOutput:
    raw_user_representations: torch.Tensor
    raw_item_representations: torch.Tensor
    calibrated_user_representations: torch.Tensor
    calibrated_item_representations: torch.Tensor
    calibration_state: Optional[CalibrationState] = None
    diagnostics: Optional[Dict[str, float]] = None
