import torch

from .dense_sinkhorn import compute_cosine_cost_matrix, dense_log_sinkhorn


def deterministic_landmark_indices(frequencies, n_landmarks):
    frequencies = torch.as_tensor(frequencies, dtype=torch.float32)
    n_items = frequencies.numel()
    if n_landmarks >= n_items:
        return torch.arange(n_items, device=frequencies.device)
    order = torch.argsort(frequencies, descending=True, stable=True)
    positions = torch.linspace(0, n_items - 1, steps=n_landmarks, device=frequencies.device)
    return order[positions.round().long()].unique(sorted=True)


def landmark_transport(source_embeddings, source_weights, anchor_embeddings,
                       target_weights, epsilon=0.1, max_iter=100,
                       convergence_tol=1e-3, chunk_size=4096):
    if anchor_embeddings.shape[0] >= source_embeddings.shape[0]:
        raise ValueError('Landmark backend requires fewer anchors than source items')
    plans = []
    projected = []
    detached_anchors = anchor_embeddings.detach()
    total_source = torch.as_tensor(source_weights, device=source_embeddings.device,
                                   dtype=source_embeddings.dtype)
    total_source = total_source / total_source.sum().clamp_min(1e-12)
    for start in range(0, source_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, source_embeddings.shape[0])
        chunk_weights = total_source[start:end]
        chunk_mass = chunk_weights.sum().clamp_min(1e-12)
        local_source = chunk_weights / chunk_mass
        cost = compute_cosine_cost_matrix(source_embeddings[start:end], detached_anchors)
        plan = dense_log_sinkhorn(cost, local_source, target_weights, epsilon,
                                  max_iter, convergence_tol)
        normalized = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        plans.append(plan * chunk_mass)
        projected.append(normalized @ detached_anchors)
    return torch.cat(projected, dim=0), plans
