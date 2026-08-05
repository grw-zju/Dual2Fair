import numpy as np
import torch


def logsumexp(values, dim, keepdim=False):
    maximum = torch.max(values, dim=dim, keepdim=True).values
    result = maximum + torch.log(torch.sum(torch.exp(values - maximum), dim=dim, keepdim=True))
    return result if keepdim else result.squeeze(dim)


def dense_log_sinkhorn(cost_matrix, source_weights, target_weights, epsilon=0.1,
                       max_iter=100, convergence_tol=1e-3, max_dense_size=10000,
                       device=None):
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    device = device or (cost_matrix.device if isinstance(cost_matrix, torch.Tensor)
                        else torch.device('cpu'))
    cost = torch.as_tensor(cost_matrix, dtype=torch.float32, device=device)
    source = torch.as_tensor(source_weights, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_weights, dtype=torch.float32, device=device)
    if cost.ndim != 2 or cost.shape != (source.numel(), target.numel()):
        raise ValueError('Cost shape must match source and target marginal sizes')
    if max(cost.shape) > max_dense_size:
        raise ValueError(f'Dense Sinkhorn shape {tuple(cost.shape)} exceeds {max_dense_size}')
    source = source / source.sum().clamp_min(1e-20)
    target = target / target.sum().clamp_min(1e-20)
    log_source = torch.log(source.clamp_min(1e-20))
    log_target = torch.log(target.clamp_min(1e-20))
    log_kernel = -cost / epsilon
    f = torch.zeros_like(source)
    g = torch.zeros_like(target)
    for _ in range(max_iter):
        next_f = log_source - torch.logsumexp(log_kernel + g.unsqueeze(0), dim=1)
        next_g = log_target - torch.logsumexp(log_kernel.T + next_f.unsqueeze(0), dim=1)
        difference = max((next_f - f).abs().max().item(), (next_g - g).abs().max().item())
        f, g = next_f, next_g
        if difference < convergence_tol:
            break
    return torch.exp(f.unsqueeze(1) + log_kernel + g.unsqueeze(0))


def marginal_errors(plan, source_weights, target_weights):
    source = torch.as_tensor(source_weights, dtype=plan.dtype, device=plan.device)
    target = torch.as_tensor(target_weights, dtype=plan.dtype, device=plan.device)
    return {
        'row_max_error': float((plan.sum(dim=1) - source).abs().max().detach().cpu()),
        'column_max_error': float((plan.sum(dim=0) - target).abs().max().detach().cpu()),
    }


def compute_cosine_cost_matrix(source_embeddings, target_embeddings, device=None):
    device = device or (source_embeddings.device if isinstance(source_embeddings, torch.Tensor)
                        else torch.device('cpu'))
    source = torch.as_tensor(source_embeddings, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_embeddings, dtype=torch.float32, device=device)
    source = source / source.norm(dim=1, keepdim=True).clamp_min(1e-10)
    target = target / target.norm(dim=1, keepdim=True).clamp_min(1e-10)
    return (1.0 - source @ target.T).clamp_min(0.0)
