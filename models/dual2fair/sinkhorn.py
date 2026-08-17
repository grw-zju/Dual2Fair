from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DenseTransportState:
    plan: torch.Tensor
    row_masses: torch.Tensor


@dataclass
class LowRankTransportState:
    F: torch.Tensor
    A_pinv: torch.Tensor
    gamma: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    row_masses: torch.Tensor
    rank: int
    certificate: float
    landmark_indices: torch.Tensor


def _stratified_landmarks(frequencies, rank, num_strata, seed):
    n_items = len(frequencies)
    order = torch.argsort(frequencies, stable=True)
    strata = torch.tensor_split(order, min(num_strata, n_items))
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    selected = []
    base = rank // len(strata)
    remainder = rank % len(strata)
    for index, stratum in enumerate(strata):
        count = min(len(stratum), base + int(index < remainder))
        if count:
            permutation = torch.randperm(len(stratum), generator=generator)[:count]
            selected.append(stratum.cpu()[permutation])
    landmarks = torch.cat(selected) if selected else order[:rank].cpu()
    if len(landmarks) < rank:
        used = set(landmarks.tolist())
        extra = [item for item in order.cpu().tolist() if item not in used]
        landmarks = torch.cat([landmarks, torch.tensor(extra[:rank - len(landmarks)])])
    return landmarks[:rank].to(frequencies.device)


def _kernel_columns(normalized_items, landmark_indices, epsilon):
    landmarks = normalized_items[landmark_indices]
    similarities = normalized_items @ landmarks.T
    return torch.exp(-(1.0 - similarities) / epsilon)


def lowrank_matvec(state, vector):
    core = state.F @ (state.A_pinv @ (state.F.T @ vector))
    return core + state.gamma * torch.ones_like(core) * vector.sum()


def build_adaptive_nystrom_state(normalized_items, frequencies, epsilon_v=0.1,
                                  initial_rank=32, max_rank=256, tolerance=1e-3,
                                  num_strata=5, pinv_rtol=1e-6, seed=42):
    items = F.normalize(normalized_items, dim=1)
    n_items = len(items)
    rank = min(initial_rank, n_items)
    k_min = torch.exp(torch.tensor(-2.0 / epsilon_v, dtype=items.dtype,
                                   device=items.device))
    while True:
        indices = _stratified_landmarks(frequencies, rank, num_strata, seed)
        factors = _kernel_columns(items, indices, epsilon_v)
        anchor_kernel = factors[indices]
        anchor_pinv = torch.linalg.pinv(anchor_kernel, rtol=pinv_rtol)
        diagonal = (factors @ anchor_pinv * factors).sum(dim=1)
        error = (1.0 - diagonal.min()).clamp_min(0.0)
        gamma = (error - k_min / 2.0).clamp_min(0.0)
        certificate = float((error + gamma).detach().cpu())
        if certificate <= tolerance or rank >= min(max_rank, n_items):
            ones = torch.ones(n_items, dtype=items.dtype, device=items.device)
            return LowRankTransportState(
                F=factors, A_pinv=anchor_pinv, gamma=gamma,
                a=ones.clone(), b=ones.clone(), row_masses=ones.clone(),
                rank=rank, certificate=certificate, landmark_indices=indices)
        rank = min(rank * 2, max_rank, n_items)


def solve_item_sinkhorn_lowrank(state, source, target, max_iter=100,
                                tolerance=1e-3, eps0=1e-8):
    source = source / source.sum().clamp_min(eps0)
    target = target / target.sum().clamp_min(eps0)
    a = torch.ones_like(source)
    b = torch.ones_like(target)
    for _ in range(max_iter):
        a = source / lowrank_matvec(state, b).clamp_min(eps0)
        b = target / lowrank_matvec(state, a).clamp_min(eps0)
        rows = a * lowrank_matvec(state, b)
        columns = b * lowrank_matvec(state, a)
        violation = max((rows - source).abs().max().item(),
                        (columns - target).abs().max().item())
        if violation <= tolerance:
            break
    state.a = a.detach()
    state.b = b.detach()
    state.row_masses = (a * lowrank_matvec(state, b)).detach()
    return state


def compute_lowrank_barycenter(state, target_anchors, eps0=1e-8):
    weighted_targets = state.b[:, None] * target_anchors
    transported = state.a[:, None] * torch.stack(
        [lowrank_matvec(state, weighted_targets[:, dim])
         for dim in range(weighted_targets.shape[1])], dim=1)
    return transported / state.row_masses[:, None].clamp_min(eps0)


def compute_lowrank_fixed_coupling_cost(state, source_items, target_anchors):
    weighted_targets = state.b[:, None] * target_anchors
    transported = state.a[:, None] * torch.stack(
        [lowrank_matvec(state, weighted_targets[:, dim])
         for dim in range(weighted_targets.shape[1])], dim=1)
    return state.row_masses.sum() - (source_items * transported).sum()


def solve_item_sinkhorn_dense(normalized_items, source, target, epsilon_v=0.1,
                              max_iter=100, tolerance=1e-3, eps0=1e-8):
    items = F.normalize(normalized_items, dim=1)
    source = source / source.sum().clamp_min(eps0)
    target = target / target.sum().clamp_min(eps0)
    kernel = torch.exp(-(1.0 - items @ items.T) / epsilon_v)
    a = torch.ones_like(source)
    b = torch.ones_like(target)
    for _ in range(max_iter):
        a = source / (kernel @ b).clamp_min(eps0)
        b = target / (kernel.T @ a).clamp_min(eps0)
        plan = a[:, None] * kernel * b[None, :]
        violation = max((plan.sum(1) - source).abs().max().item(),
                        (plan.sum(0) - target).abs().max().item())
        if violation <= tolerance:
            break
    plan = (a[:, None] * kernel * b[None, :]).detach()
    return DenseTransportState(plan=plan, row_masses=plan.sum(1))


def compute_dense_barycenter(state, target_anchors, eps0=1e-8):
    return (state.plan @ target_anchors) / state.row_masses[:, None].clamp_min(eps0)


def compute_dense_fixed_coupling_cost(state, source_items, target_anchors):
    transported = state.plan @ target_anchors
    return state.row_masses.sum() - (source_items * transported).sum()


def solve_user_sinkhorn_chunked(*args, **kwargs):
    raise DeprecationWarning('User OT is implemented by UserRepresentationCalibration')


def sinkhorn_nystrom(*args, **kwargs):
    import warnings
    warnings.warn('sinkhorn_nystrom is deprecated; use solve_item_sinkhorn_lowrank',
                  DeprecationWarning, stacklevel=2)
    return solve_item_sinkhorn_lowrank(*args, **kwargs)
