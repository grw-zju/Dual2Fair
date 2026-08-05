# Manuscript Patches

## User calibration

Let `Γᵘ∈R^(N_d×K)` be the entropic OT coupling between disadvantaged-user interests and advantaged-user prototypes. We use its conditional rows,

`Γ̄ᵘ_ik = Γᵘ_ik / (Σ_l Γᵘ_il + ε)`,

and define `zᵘ_i=Σ_k Γ̄ᵘ_ik p_k`. The calibrated user representation is

`ẽᵘ_i=ρ_u eᵘ_i+(1-ρ_u)P_u(zᵘ_i)`.

## Item calibration

For detached anchor representations `a_j`,

`Γ̄ᵛ_ij = Γᵛ_ij / (Σ_l Γᵛ_il + ε)`,
`zᵛ_i=Σ_j Γ̄ᵛ_ij stopgrad(a_j)`,
`ẽᵛ_i=ρ_v eᵛ_i+(1-ρ_v)P_v(zᵛ_i)`.

The merit–uniform target is

`t_j=(1-γ)m_j/Σ_l m_l+γ/A`.

`γ=1` is the uniform-target ablation. Merit is constructed from training interactions and detached, user-wise percentile-normalized predictions.

## Backbone-specific scorers

LightGCN retains propagated-representation dot products. NeuMF retains separate GMF and MLP branches and its prediction layer. VAECF uses decoder logits, including output bias. The recommendation loss and evaluation invoke the same calibrated scorer.

## Optimization terminology

The default is hierarchical mirror-style alternating leader–follower optimization, not strict bi-level optimization:

`θ'=θ-α₁η∇L_fair(θ)`,
`θ_new=θ'+α₂η∇L_fair(θ')`.

The second gradient is recomputed at `θ'`. The associated first-order surrogate is

`L_mirror=(α₁-α₂)L_fair+0.5α₁α₂η||∇L_fair||²`.

## UIF

Standard validation DUF and DIF are frozen for each dataset/backbone/evaluation-protocol tuple. All methods, including Standard, use these constants. With normalized Standard terms and `w₁+w₂=1`, Standard UIF is approximately `1/NDCG`.

## DIF

The affine-invariant metric forms quality from within-user percentile ranks. Consequently, any positive affine transformation `a f(u,v)+b`, `a>0`, preserves ranking and quality. Rank DIF uses discounted ranks. The previous sigmoid-score metric is retained only as `DIF_legacy`.

## Scalability

User transport costs `O(N_dK)` and is processed in chunks. Item inference uses `A` deterministic anchors, requiring `O(N_iA)` costs rather than an `N_i×N_i` matrix. Exact time and memory are reported by executable benchmarks.

## Limitations

The method calibrates users and items represented by training interactions. Strict zero-interaction cold start is unsupported without side information. Streaming updates are outside the current scope.

## Training pseudocode

1. Build train-only groups, frequencies, and frozen split.
2. At configured intervals, fit prototypes and update deterministic item anchors/targets.
3. Build calibrated backbone state.
4. Compute recommendation loss through the calibrated scorer.
5. Apply the configured joint, alternating, or mirror-alternating update.
6. Select checkpoints using validation metrics only.

## Inference pseudocode

1. Restore backbone and complete calibration state from the selected checkpoint.
2. Clear stale caches and deterministically rebuild representations when required.
3. Calibrate all eligible users and map every item through deterministic anchors.
4. Score with the backbone-native calibrated scorer.
