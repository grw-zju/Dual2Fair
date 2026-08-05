import torch

from .transport import dense_log_sinkhorn


def sinkhorn_nystrom(cost_matrix, p, q, epsilon=0.1, max_iter=100,
                     convergence_tol=1e-3, device=None, **kwargs):
    if kwargs:
        unsupported = ', '.join(sorted(kwargs))
        raise TypeError(f'Nyström is not implemented; unsupported arguments: {unsupported}')
    return dense_log_sinkhorn(cost_matrix, p, q, epsilon, max_iter,
                              convergence_tol, device=device)


from .transport import compute_cosine_cost_matrix

__all__ = ['sinkhorn_nystrom', 'compute_cosine_cost_matrix']
