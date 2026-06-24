import torch
import numpy as np


def logsumexp(a, dim, keepdim=False):
    max_a = torch.max(a, dim=dim, keepdim=True)[0]
    result = max_a + torch.log(torch.sum(torch.exp(a - max_a), dim=dim, keepdim=True))
    if not keepdim:
        result = result.squeeze(dim)
    return result


def sinkhorn_nystrom(cost_matrix, p, q, epsilon=0.1, max_iter=100, convergence_tol=1e-3,
                     nystrom_rank=32, rank_increment=8, nystrom_tol=1e-3, tikhonov_lambda=1e-4, device=None):
    """
    Stable log-domain Sinkhorn solver.

    The function keeps the historical name and Nystrom-related parameters for
    compatibility with the rest of the project, but the broken large-matrix
    Nystrom branch is intentionally disabled. Large OT problems should be
    reduced with max_ot_items/max_disadv_users before calling this solver.
    """
    if device is None:
        device = torch.device('cpu')

    m, n = cost_matrix.shape

    if isinstance(cost_matrix, np.ndarray):
        C = torch.from_numpy(cost_matrix).float().to(device)
    else:
        C = cost_matrix.float().to(device)

    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p).float().to(device)
    else:
        p = p.float().to(device)

    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float().to(device)
    else:
        q = q.float().to(device)

    log_p = torch.log(p.clamp(min=1e-20))
    log_q = torch.log(q.clamp(min=1e-20))

    f = torch.zeros(m, device=device)
    g = torch.zeros(n, device=device)

    max_dense_size = 10000
    if max(m, n) > max_dense_size:
        raise ValueError(
            f"Sinkhorn cost matrix shape {tuple(C.shape)} is too large for the stable dense solver. "
            "Reduce dual2fair.max_ot_items or dual2fair.max_disadv_users so each OT problem is "
            f"at most {max_dense_size} rows/columns."
        )

    transport_plan = _sinkhorn_dense_log(C, log_p, log_q, f, g, epsilon,
                                         max_iter, convergence_tol, device)

    return transport_plan


def _sinkhorn_dense_log(C, log_p, log_q, f, g, epsilon, max_iter, convergence_tol, device):
    m, n = C.shape

    for it in range(max_iter):
        log_K = -C / epsilon

        f_new = log_p - logsumexp(log_K + g.unsqueeze(0), dim=1)

        g_new = log_q - logsumexp(log_K.T + f_new.unsqueeze(0), dim=1)

        f_diff = torch.abs(f_new - f).max().item()
        g_diff = torch.abs(g_new - g).max().item()

        f = f_new
        g = g_new

        if f_diff < convergence_tol and g_diff < convergence_tol:
            break

    log_K = -C / epsilon
    log_transport = f.unsqueeze(1) + log_K + g.unsqueeze(0)
    transport_plan = torch.exp(log_transport)

    return transport_plan


def compute_cosine_cost_matrix(source_embs, target_embs, device=None):
    if device is None:
        device = torch.device('cpu')

    if isinstance(source_embs, np.ndarray):
        source_embs = torch.from_numpy(source_embs).float().to(device)
    else:
        source_embs = source_embs.float().to(device)

    if isinstance(target_embs, np.ndarray):
        target_embs = torch.from_numpy(target_embs).float().to(device)
    else:
        target_embs = target_embs.float().to(device)

    source_norm = source_embs / (source_embs.norm(dim=1, keepdim=True) + 1e-10)
    target_norm = target_embs / (target_embs.norm(dim=1, keepdim=True) + 1e-10)

    cosine_sim = source_norm @ target_norm.T
    cost = 1.0 - cosine_sim
    cost = torch.clamp(cost, min=0.0)
    return cost
