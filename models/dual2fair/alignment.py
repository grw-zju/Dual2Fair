from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class AlignmentResult:
    targets: torch.Tensor
    fairness_loss: Optional[torch.Tensor]
    state: Any


@dataclass
class HardAlignmentState:
    indices: torch.Tensor


@dataclass
class LinearMMDAlignmentState:
    delta: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor


def normalized_cosine_scores(source, target):
    return F.normalize(source, dim=1) @ F.normalize(target, dim=1).T


def hard_user_match(source_x, prototypes, target_mass, epsilon, eps0=1e-8):
    target = target_mass / target_mass.sum().clamp_min(eps0)
    scores = normalized_cosine_scores(source_x, prototypes.detach())
    scores = scores + float(epsilon) * torch.log(target.clamp_min(eps0))[None, :]
    return torch.argmax(scores, dim=1)


def hard_item_match_blockwise(source_x, target_y, target_mass, epsilon,
                              source_chunk_size=512, target_chunk_size=4096,
                              eps0=1e-8):
    n_source = source_x.shape[0]
    n_target = target_y.shape[0]
    target = target_mass / target_mass.sum().clamp_min(eps0)
    prior = float(epsilon) * torch.log(target.clamp_min(eps0))
    result = torch.empty(n_source, dtype=torch.long, device=source_x.device)
    for source_start in range(0, n_source, int(source_chunk_size)):
        source_chunk = source_x[source_start:source_start + int(source_chunk_size)]
        best_score = torch.full((len(source_chunk),), -torch.inf,
                                dtype=source_chunk.dtype, device=source_chunk.device)
        best_index = torch.full((len(source_chunk),), n_target,
                                dtype=torch.long, device=source_chunk.device)
        for target_start in range(0, n_target, int(target_chunk_size)):
            target_chunk = target_y[target_start:target_start + int(target_chunk_size)]
            scores = source_chunk @ target_chunk.T
            scores = scores + prior[target_start:target_start + len(target_chunk)][None, :]
            local_score, local_position = scores.max(dim=1)
            local_index = local_position + target_start
            better = ((local_score > best_score)
                      | ((local_score == best_score) & (local_index < best_index)))
            best_score = torch.where(better, local_score, best_score)
            best_index = torch.where(better, local_index, best_index)
        result[source_start:source_start + len(source_chunk)] = best_index
    return result


def weighted_mean(points, weights, eps0=1e-8):
    normalized = weights / weights.sum().clamp_min(eps0)
    return (normalized[:, None] * points).sum(dim=0)


def linear_mmd_state(source_x, source_mass, target_y, target_mass, eps0=1e-8):
    source = source_mass / source_mass.sum().clamp_min(eps0)
    target = target_mass / target_mass.sum().clamp_min(eps0)
    source_mean = weighted_mean(source_x, source, eps0)
    target_mean = weighted_mean(target_y.detach(), target, eps0)
    delta = (target_mean - source_mean).detach()
    return LinearMMDAlignmentState(delta=delta,
                                   source_mean=source_mean.detach(),
                                   target_mean=target_mean.detach())


def linear_mmd_loss(source_x, source_mass, target_y, target_mass, eps0=1e-8):
    source = source_mass / source_mass.sum().clamp_min(eps0)
    target = target_mass / target_mass.sum().clamp_min(eps0)
    source_mean = weighted_mean(source_x, source, eps0)
    target_mean = weighted_mean(target_y.detach(), target, eps0)
    return (source_mean - target_mean).pow(2).sum()
