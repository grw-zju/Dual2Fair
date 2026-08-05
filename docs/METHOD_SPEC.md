# Executable Method Specification

## Scope and terminology

The corrected default is `mirror_alternating`, described as **hierarchical mirror-style alternating leader–follower optimization**. It is not strict bi-level optimization. The name `unrolled_bilevel` is reserved for a tested hypergradient implementation and otherwise fails explicitly.

## Notation

- `N_u`: users; `N_i`: items; `N_d`: disadvantaged users; `K`: user prototypes; `A`: item anchors; `d`: calibration dimension.
- `E_u ∈ R^(N_u×d)`, `E_v ∈ R^(N_i×d)`: raw calibration representations supplied by a backbone adapter.
- `rho_u,rho_v ∈ [0,1]`: residual fusion coefficients.
- `eps`: positive numerical floor.

## User calibration

Advantaged-user interests are means of item calibration representations over training interactions only. Empty-interaction users are excluded from GMM fitting and retain their raw representation.

A deterministic diagonal-covariance GMM, seeded by `ot_sampling_seed`, produces prototypes `P ∈ R^(K×d)` and weights `q ∈ Δ^K`.

For source weights `p ∈ Δ^(N_d)`:

```
Gamma_u ∈ R^(N_d×K)
Gamma_u 1 = p
Gamma_u^T 1 = q
GammaBar_u[i,k] = Gamma_u[i,k] / (sum_l Gamma_u[i,l] + eps)
z_u[i] = sum_k GammaBar_u[i,k] P[k]
E_u_cal[i] = rho_u E_u[i] + (1-rho_u) user_projection(z_u[i])
```

All eligible disadvantaged users are processed in deterministic chunks. Each local plan is scaled by its global source mass; this is a deterministic chunk-wise approximation with repeated prototype marginals, not an exact single global OT solve. Advantaged users and ineligible empty-interaction users remain unchanged by default. Fairness updates detach backbone representations and update the dedicated `user_projection` module.

## Item calibration

Deterministic popularity-stratified anchors `H ∈ R^(A×d)` are selected from training-state representations. Target anchors are detached when constructing barycenters.

```
Gamma_v ∈ R^(N_i×A)
GammaBar_v[i,j] = Gamma_v[i,j] / (sum_l Gamma_v[i,l] + eps)
z_v[i] = sum_j GammaBar_v[i,j] stop_gradient(H[j])
E_v_cal[i] = rho_v E_v[i] + (1-rho_v) item_projection(z_v[i])
```

Every item receives an anchor-based mapping at inference. Evaluation never samples a new subset. The trained item objective is a projection-alignment surrogate, `mean_i[1-cos(e_i, stopgrad(z_i))]`, while the Sinkhorn plan constructs the target mapping. Fairness updates detach backbone representations and update the dedicated `item_projection` module.

## Item target distributions

Supported modes:

- `uniform`: `t_j = 1/A`
- `merit`: `t_j = merit_j / sum_l merit_l`
- `merit_uniform_mixture`:

```
t_j = (1-gamma) merit_j / sum_l merit_l + gamma/A
```

Merit uses a deterministic, stratified subset of training users and deterministic item anchors, scored in chunks through the backbone adapter. Scores are normalized per user by percentile rank before aggregation. The sampling seed and sample sizes are configuration values. `gamma=1` exactly reproduces the uniform target. The corrected default is `merit_uniform_mixture`.

## Backbone adapter contract

Each supported backbone implements:

```
get_calibration_state()
get_raw_user_repr()
get_raw_item_repr()
calibrate_users(...)
calibrate_items(...)
score_pairs_with_calibrated_state(...)
score_all_with_calibrated_state(...)
```

The recommendation loss and evaluation call the same adapter scoring functions. Generic dot-product fallback is forbidden.

### LightGCN

Calibration operates on propagated user/item representations. Native dot-product scoring is preserved.

### NeuMF

GMF and MLP user/item branches remain separate. OT is built in a defined shared calibration space; the resulting mappings are applied consistently to both branches. Scores use element-wise GMF interaction, the original MLP stack, and the original prediction layer. No dot-product fallback is allowed.

### VAECF

Native scores are decoder logits. Item scoring representations are derived from the final decoder output weights and include the decoder bias. The independent untrained item embedding is unsupported and removed.

## Training and update frequency

- GMM and transport state update once per epoch by default, controlled by `ot_update_interval`.
- Recommendation batches use the current calibrated state through the adapter scorer.
- Merit predictions and item target anchors are detached.
- Dense user-to-prototype plans may remain differentiable during their update; persisted inference plans are detached.
- Backbone and calibration caches are cleared after relevant parameter updates.

## Optimization modes

### `joint_weighted_sum`

```
L = L_rec + lambda1 L_user + lambda2 L_item
```

### `alternating`

An accuracy update is followed by a standard fairness update.

### `mirror_alternating` (default)

An accuracy update is followed periodically by:

```
theta' = theta - alpha1 eta grad L_fair(theta)
recompute L_fair(theta')
theta_new = theta' + alpha2 eta grad L_fair(theta')
```

The documented surrogate is:

```
L_mirror = (alpha1-alpha2) L_fair
         + 0.5 alpha1 alpha2 eta ||grad L_fair||^2
```

No `.data` mutation or stale-gradient reuse is permitted.

### `unrolled_bilevel`

Unavailable unless a tested one-step unrolled or implicit hypergradient is implemented. Requests otherwise raise `NotImplementedError`.

## Deterministic inference and checkpoint state

Checkpoints include backbone parameters, calibration-module parameters, GMM weights/means/covariances, prototypes, anchor indices, target distribution, fusion coefficients, OT state, optimizer/scheduler state, epoch/global step, resolved configuration, and dataset/split hashes.

After restoration, stale caches are cleared and calibration state is restored or rebuilt from that checkpoint. Two repeated inference calls must match within numerical tolerance.

## Evaluation protocol

- Candidate sets are fixed per split and `evaluation_seed`.
- User/item groups and all frequency-derived values use training data only.
- UIF normalization constants are frozen from the Standard model's validation results for each dataset/backbone/protocol tuple.
- `DIF_legacy`, affine-invariant DIF, and rank DIF are reported with protocol metadata.
- Independent runs train fresh models under explicit model seeds while sharing one frozen split.
