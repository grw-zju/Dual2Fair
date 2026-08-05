from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class CalibrationOutput:
    raw_user_representations: torch.Tensor
    raw_item_representations: torch.Tensor
    calibrated_user_representations: torch.Tensor
    calibrated_item_representations: torch.Tensor
    user_ot_plan: Optional[torch.Tensor] = None
    item_ot_plan: Optional[torch.Tensor] = None
    normalized_coupling_diagnostics: Dict[str, float] = field(default_factory=dict)
    user_ot_objective: Optional[torch.Tensor] = None
    item_ot_objective: Optional[torch.Tensor] = None
    fusion_rho_u: float = 1.0
    fusion_rho_v: float = 1.0
    scorer_state: Dict[str, Any] = field(default_factory=dict)
