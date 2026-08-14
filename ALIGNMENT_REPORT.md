# Dual2Fair Revision Alignment Report

**Audited repository state:** latest working tree (post-alignment)

This report tracks implementation alignment against the TKDE major-revision specification.
It does not claim that manuscript results have been reproduced.

## Component Status Summary

| Paper component | Current implementation | Status | Files |
|---|---|---|---|
| User grouping | `get_user_activity_groups(sparse_user_ratio=0.95)` with deterministic ties; bottom 95% = sparse-history, top 5% = higher-activity; deprecated `get_advantaged_users` alias retained | Aligned | `data/dataset_utils.py`, `models/dual2fair/dual2fair.py` |
| User calibration representation | `LN(W_u_c * e_u + W_u_h * detached_history)` with trainable W_u_c, W_u_h, LayerNorm; stop-gradient on item history | Aligned | `models/dual2fair/user_calibration.py` |
| GMM | sklearn diagonal GMM on detached higher-activity x_u; 64 clusters, max_iter 50, tol 1e-4, cov floor 1e-6, deterministic seed; persists means and weights | Aligned | `models/dual2fair/user_calibration.py` |
| User OT | Global Sinkhorn over all sparse-history users with chunked kernel-vector products; source = 1/N_s, target = pi_k; no user sampling/capping | Aligned | `models/dual2fair/user_calibration.py` |
| User barycentric correction | Row-normalized transport -> barycentric target z; confidence = exp(-(1-cos(x,z))/tau_u); residual: e + rho_u * c * P_u(z - x) with sigmoid(rho_u_tilde) and identity-init P_u | Aligned | `models/dual2fair/user_calibration.py` |
| Item calibration space | `h_v = LN(W_v * e_v)`, `x_v = h_v / |h_v|_2` with trainable W_v (identity init) and L2 normalization | Aligned | `models/dual2fair/item_calibration.py` |
| Item source marginal | `chi_v = 1/(n_v + kappa)^beta_pop`, normalized; all warm-start items participate; no item sampling/capping | Aligned | `models/dual2fair/item_calibration.py` |
| Item merit | Per-user fixed training candidate sets (size 500, uniform without replacement, exclude training history, frozen at init); rank-based merit with lagged EMA state | Aligned | `models/dual2fair/dual2fair.py` |
| Relevance-aware target | `nu = (1-omega)*(m+delta_m)/sum(m+delta_m) + omega*1/N_v` with omega=0.2, delta_m=1e-6; endpoints omega=0 (merit only) and omega=1 (uniform) supported for ablation | Aligned | `models/dual2fair/item_calibration.py` |
| Item OT | Adaptive Nyström low-rank Sinkhorn with stratified landmarks, implicit matvec, certificate-based rank doubling (32->256) | Aligned | `models/dual2fair/sinkhorn.py`, `models/dual2fair/item_calibration.py` |
| Adaptive Nyström | `build_adaptive_nystrom_state` with error certificate `1 - min(diag(K_tilde))`, positivity safeguard gamma, doubling 32->64->128->256, max 256; no dense Nv x Nv kernel on paper path | Aligned | `models/dual2fair/sinkhorn.py` |
| Implicit coupling | `LowRankTransportState` with F, A_pinv, gamma, a, b, row_masses; `lowrank_matvec` implements F(A_pinv(F^T v)) + gamma*1*(1^T v) | Aligned | `models/dual2fair/sinkhorn.py` |
| Confidence | User: exp(-(1-cos(x,z))/tau_u); Item: exp(-(1-cos(x,z/|z|))/tau_v); disable_confidence toggle for ablation | Aligned | calibration modules |
| Residual calibration | User: e + rho_u * c_u * P_u(z_u - x_u); Item: e + rho_v * c_v * P_v(z_v - x_v); sigmoid rho, identity-init P | Aligned | calibration modules |
| Hierarchical optimization | `HierarchicalAlternatingOptimizer` with separate Adam optimizers for accuracy (all params, lr=1e-3) and fairness (W_u_c/W_u_h/W_v only, lr=5e-4); mirror step recomputes gradient | Aligned | `models/dual2fair/hierarchical_opt.py` |
| EMA | Complete trainable parameter state EMA with xi=0.99; lagged EMA used for merit computation; checkpoint save/restore | Aligned | `models/dual2fair/dual2fair.py` |
| Calibration refresh | Per-epoch refresh: raw repr -> calibration space -> GMM -> EMA merit -> nu -> user OT -> item OT -> cache -> fairness correction | Aligned | `models/dual2fair/dual2fair.py` |
| Mirror second gradient | First step: theta' = theta - alpha1*eta*grad_L(theta); Second step: theta = theta' + alpha2*eta*grad_L(theta') with recomputed forward/backward on same frozen transport state | Aligned | `models/dual2fair/hierarchical_opt.py` |
| Fairness parameter restriction | Fairness correction only updates W_u_c, W_u_h, W_v; backbone, P_u, P_v, rho_u_tilde, rho_v_tilde frozen | Aligned | `models/dual2fair/hierarchical_opt.py`, `models/dual2fair/dual2fair.py` |
| DUF | Per-user NDCG variance in full-catalog warm-start evaluation | Aligned | `evaluation/metrics.py`, `evaluation/evaluator.py` |
| DIF | Rank-based DIF: exposure/relevance ratio variance over V_eval; invariant to monotonic score transforms; raw-score sigmoid DIF removed from paper path | Aligned | `evaluation/metrics.py`, `evaluation/evaluator.py`, `run.py` |
| UIF | Split-specific Standard baseline DUF/DIF references required by default; missing references raise instead of silently returning paper-path UIF; per-run UIF computed then averaged; w1=w2=0.5 enforced | Aligned | `config/default.yaml`, `evaluation/metrics.py`, `evaluation/evaluator.py`, `run.py` |
| Validation/test protocol | Full-catalog warm-start evaluation is paper default; sampled mode retained as deprecated debug alias; blockwise per-user scoring, no dense U×V matrix | Aligned | `evaluation/evaluator.py`, `run.py` |
| Five independent runs | `scripts/run_five_seeds.py` launches 5 subprocesses with seeds [42,43,44,45,46]; `run.py --seeds` also requires exactly 5 distinct seeds and reuses the same aggregation validation; per-seed JSON + aggregate JSON with mean/std/ci95; split_hash metadata validation | Aligned | `scripts/run_five_seeds.py`, `run.py` |
| Checkpointing | Full checkpoint: deep-cloned model state, calibration state, EMA state, optimizer states, epoch/iteration, split hash, selection metadata; restore helper reloads calibration caches | Aligned | `models/dual2fair/dual2fair.py` |
| Ablations | w/o URC, w/o IRC, Uniform Target (omega=1), Merit Only (omega=0), w/o Confidence, Joint Weighted Sum, Standard Alternating, Hard Matching, and MMD Alignment all supported via config toggles | Aligned | `config/default.yaml`, `models/dual2fair/ablations.py`, `models/dual2fair/alignment.py` |
| Hard Matching | Historical exact implementation not found. Implemented from explicit spec: independent target-prior-aware MAP assignments for users/items; no OT coupling, no global column-capacity constraint, one target per source; cached assignments define residual targets | Aligned; Table III reproduction not rerun | `models/dual2fair/alignment.py`, `models/dual2fair/user_calibration.py`, `models/dual2fair/item_calibration.py`, `tests/test_hard_alignment.py` |
| MMD Alignment | Historical exact implementation not found. Implemented weighted linear-kernel MMD: first-moment matching with detached global mean-shift executable correction; no sample-to-target correspondence, no Sinkhorn/Nyström/nearest-neighbor assignment | Aligned; Table III reproduction not rerun | `models/dual2fair/alignment.py`, `models/dual2fair/user_calibration.py`, `models/dual2fair/item_calibration.py`, `tests/test_mmd_alignment.py` |
| Baselines (ESAM, MGL) | Provenance-checked external official-code wrappers: ESAM uses the SIGIR 2020 paper-linked `https://github.com/A-bone1/ESAM`; MGL uses the KDD 2023 official `https://github.com/weicy15/MGL`. Wrappers export the frozen Dual2Fair split, validate required official files/commit, run a configured external command, and require external metrics or score matrices; no surrogate implementation is fabricated | Aligned as external wrappers | `baseline/external_official.py`, `baseline/__init__.py`, `run.py`, `config/default.yaml` |
| Other baselines | UFR, HyperUOF, DPR, FairDual, CPFair, MultiFR, Ada2Fair, FAIR, FairSort, PopularityIPS all implemented as local adapters | Aligned | `baseline/` |
| Dataset preprocessing | min_user_interactions=3, min_item_interactions=1, chronological LOO, implicit binary feedback, dedup; preprocessing audit script exists | Aligned — requires result reproduction check | `data/dataset_utils.py`, `scripts/audit_preprocessing.py` |
| README | Rewritten with paper-aligned terminology, method description, scope limitations, paper-reported results (not claimed as reproduced) | Aligned | `README.md` |
| Project structure | `hierarchical_opt.py` as paper path; `bilevel_opt.py` as deprecated shim with DeprecationWarning | Aligned | `models/dual2fair/` |
| Cross-side interference diagnostic | `scripts/analyze_cross_side_interference.py` computes gradient leakage ratios | Aligned | `scripts/` |
| Optimization trajectory logging | Validation epoch logging with NDCG/DUF/DIF/UIF/strategy/seed | Aligned | `run.py` |
| Efficiency benchmark | `scripts/benchmark_efficiency.py` measures time/memory for standard and low-rank methods | Aligned | `scripts/` |
| Scaling benchmark | `scripts/benchmark_scaling.py` scaffolding for catalog fraction scaling | Aligned | `scripts/` |
| Static audit | `scripts/audit_revision_alignment.py` checks for forbidden patterns (fusion_alpha, max_disadv_users, max_ot_items, BiLevelOptimizer, disabled Nyström) | Aligned | `scripts/` |

## Prohibited paper-path remnants — status

| Pattern | Status |
|---|---|
| `fusion_alpha` | Not present in paper-path code |
| `max_disadv_users` | Not present in paper-path code |
| `max_ot_items` | Not present in paper-path code |
| `adv_ratio` (as paper-path default) | Removed from paper-path calls; deprecated alias only |
| `uniform_weights` as default target | Not present |
| `BiLevelOptimizer` | Deprecated alias only; paper path uses `HierarchicalAlternatingOptimizer` |
| raw sigmoid-quality DIF | Removed from paper path; post-processing baselines use rank-based DIF |
| sampled evaluation as CLI default | Full-catalog is default; sampled is deprecated alias |
| `advantaged/disadvantaged` in Dual2Fair terminology | Dual2Fair uses higher-activity/sparse-history; baselines retain their own terminology |
| disabled Nyström branch | Nyström solver is active on paper path |
| `sinkhorn_nystrom` stub | Deprecated alias with DeprecationWarning |
| `tikhonov_lambda` | Not present; pinv tolerance named `nystrom_pinv_rtol` |

## Alternative-design alignment ablations

### Complete Dual2Fair

- Mechanism: global soft entropy-regularized OT.
- User side: sparse-history users transport to higher-activity GMM prototypes with source marginal `p_i=1/N_s` and target marginal `q_k=pi_k`; barycentric projection gives `z_i^u`.
- Item side: inverse-popularity source marginal `s_i` transports to relevance-aware target `nu_j` over detached item anchors; adaptive Nyström keeps item coupling implicit.
- Distinction: global soft correspondence + explicit source/target marginal constraints + barycentric target.

### Hard Matching

- Existing historical implementation found? no.
- Final mathematical definition:
  - User assignment: `k_i* = argmax_k cos(x_i^s, mu_k) + epsilon_u log(q_k + eps0)` with deterministic lowest-index tie break from `torch.argmax` over sorted prototype indices.
  - User executable target: `z_i^u = mu_{k_i*}`.
  - User fairness surrogate: `sum_i (1/N_s) * (1 - cos(x_i^s, mu_{k_i*}))` with detached assignments.
  - Item assignment: `j_i* = argmax_j x_i^T y_j + epsilon_v log(nu_j + eps0)` computed by exact blockwise top-1 search; no dense `N_v x N_v` matrix.
  - Item executable target: `z_i^v = y_{j_i*}`.
  - Item fairness surrogate: `sum_i s_i * (1 - x_i^T y_{j_i*})` with detached assignments.
- Distinction from OT: independent hard correspondence + no global column-capacity constraint + one target per source.
- Files modified: `models/dual2fair/alignment.py`, `models/dual2fair/user_calibration.py`, `models/dual2fair/item_calibration.py`, `models/dual2fair/dual2fair.py`, `models/dual2fair/state.py`, `config/default.yaml`, `run.py`, `README.md`, tests.
- Tests: `tests/test_hard_alignment.py`, `tests/test_alignment_modes_smoke.py`.
- Paper reproduction status: IMPLEMENTATION ALIGNED; TABLE III VALUES NOT RERUN.

### MMD Alignment

- Existing historical implementation found? no.
- Final mathematical definition:
  - Weighted linear-kernel MMD: `|| sum_i p_i x_i - sum_j q_j y_j ||_2^2`.
  - User source: `p_i=1/N_s` over sparse-history calibration representations; user target: `q_k=pi_k` over detached GMM prototypes.
  - User executable correction: `Delta_u = mu_Q - mu_P`, cached detached at refresh; `z_i^u = x_i^s + Delta_u`.
  - Item source: inverse-popularity `s_i`; item target: relevance-aware `nu_j` over detached anchors `y_j`.
  - Item executable correction: `Delta_v = mu_Q - mu_P`, cached detached at refresh; `z_i^v = x_i + Delta_v`.
- Distinction from OT: distribution-level first-moment matching + no sample-to-target correspondence + global mean-shift correction.
- Files modified: `models/dual2fair/alignment.py`, `models/dual2fair/user_calibration.py`, `models/dual2fair/item_calibration.py`, `models/dual2fair/dual2fair.py`, `models/dual2fair/state.py`, `config/default.yaml`, `run.py`, `README.md`, tests.
- Tests: `tests/test_mmd_alignment.py`, `tests/test_alignment_modes_smoke.py`.
- Paper reproduction status: IMPLEMENTATION ALIGNED; TABLE III VALUES NOT RERUN.

The repository previously did not contain an implementation of the Hard Matching / MMD Alignment ablations. The implementations added in this revision follow the explicit mathematical specifications documented here. The manuscript-reported Table III values have not been reproduced by this code change.

## Follow-up fixes in this pass

| Issue found | Fix | Status |
|---|---|---|
| Hard Matching and MMD Alignment were explicit unavailable ablations | Added `alignment_mode={ot,hard,mmd}` with Hard independent MAP matching and MMD weighted linear first-moment alignment while preserving all other Dual2Fair modules, optimizer, and evaluation protocol | Fixed |
| ESAM/MGL were explicit unavailable stubs despite paper-linked official code | Added external official-code wrappers with provenance, repo validation, split export, configurable command execution, and metrics/score collection | Fixed |
| UIF references were optional in the default evaluator path | Added `evaluation.require_uif_reference: true`, explicit null val/test placeholders, evaluator enforcement, and smoke-only `--allow-missing-uif-reference` override | Fixed |
| `run.py --seeds` accepted arbitrary seed counts and used weaker aggregation | Now requires exactly five distinct seeds and delegates aggregation to `scripts.run_five_seeds.aggregate_runs` | Fixed |
| Multi-seed aggregation accepted non-finite UIF values | Aggregator now rejects missing/non-finite metrics before computing means | Fixed |
| Dual2Fair validation epoch logging only printed NDCG | Added per-epoch validation NDCG/DUF/DIF/UIF/strategy/seed/selected checkpoint trajectory in run outputs | Fixed |
| Deprecated optimization package imported a non-existent local module and helpers called a removed cache-clear method | Fixed compatibility import and switched helpers to `clear_scoring_state()` | Fixed |
| `rank_relevance_from_order` assumed item ids were dense positions | Fixed helper to size by item id catalog or explicit `n_items` | Fixed |
| Checkpoint snapshots were shallow for model/EMA/optimizer/calibration state | Checkpoints now deep-clone tensors/state and restore via `load_checkpoint_state()` | Fixed |

## Known unresolved items

1. **Hard Matching ablation**: Historical implementation not found in current tree or visible git history. Implemented from the explicit specification above; manuscript Table III values not rerun.
2. **MMD Alignment ablation**: Historical implementation not found in current tree or visible git history. Implemented from the explicit weighted linear-MMD specification above; manuscript Table III values not rerun.
3. **ESAM and MGL baselines**: External official-code wrappers are implemented and registered. They require local checkouts of the official repositories and configured commands that produce metrics or score matrices; no local simplified surrogate is used.
4. **Dataset preprocessing statistics**: The manuscript min-3 preprocessing rule is implemented. This follow-up intentionally skipped preprocessing audit per user request. `scripts/audit_preprocessing.py` reports missing raw data without downloading; if raw files are supplied and observed statistics differ from paper-reported values, the discrepancy is flagged as "requires result reproduction check".
5. **Full paper result reproduction**: Not run during this alignment task. Manuscript-reported values are labeled as "Paper-reported results" in README, not as "reproduced by current commit".

## IMPLEMENTATION ALIGNED; FULL RESULT REPRODUCTION NOT RUN
