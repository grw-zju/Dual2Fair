from .dense_sinkhorn import (compute_cosine_cost_matrix, dense_log_sinkhorn,
                             marginal_errors)
from .landmark_sinkhorn import deterministic_landmark_indices, landmark_transport

__all__ = [
    'compute_cosine_cost_matrix', 'dense_log_sinkhorn', 'marginal_errors',
    'deterministic_landmark_indices', 'landmark_transport',
]
