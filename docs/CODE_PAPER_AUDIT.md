# Historical Code–Paper Audit

**Audit Date**: 2026-08-05
**Audited Commit**: `dc3849a`
**Status**: Historical audit of the pre-revision implementation. The defects below describe that audited commit and are retained as revision evidence; they are not claims about the current `main` branch.

---

## Defect 1 (Critical): OT barycentric projection missing row normalization — User side

**File**: `models/dual2fair/user_calibration.py:97`
**Code**: `calibrated_interest = gamma_u @ self.prototypes`

The OT transport plan `gamma_u` has row sums ≈ `p_i = 1/n_disadv`. Using `gamma_u @ prototypes`
without dividing by row mass produces vectors shrunk by ~1/n factor instead of conditional barycenters.

**Fix**: `gamma_bar = gamma_u / gamma_u.sum(dim=1, keepdim=True).clamp_min(eps); z = gamma_bar @ prototypes`

---

## Defect 2 (Critical): OT barycentric projection missing row normalization — Item side

**File**: `models/dual2fair/item_calibration.py:92`
**Code**: `calibrated_selected = gamma_v.detach() @ selected_item_embeddings`

Same issue: no division by source row mass.

**Fix**: Same row normalization as above.

---

## Defect 3 (Critical): Calibrated embeddings not used in training loss

**File**: `models/dual2fair/dual2fair.py:48-49`, `run.py:365-620`
**Issue**: `compute_fairness_losses()` only computes OT losses on `item_embs`.
`get_calibrated_embeddings()` is called only during validation/test (run.py:528-533).
BPR/BCE training always uses the backbone's raw scorer.

The paper claims a "fusion interface" between calibration and the backbone scorer/loss,
but this interface is never exercised during training.

**Fix**: Training loss must be evaluated through the calibrated scoring path.

---

## Defect 4 (Critical): User fairness loss does not update user representations

**File**: `models/dual2fair/dual2fair.py:48-49`
**Issue**: Both `L_user` and `L_item` are derived from `item_embs`. The user OT
interest vector is computed as the mean of item embeddings. In models like NeuMF,
`L_user` primarily updates item parameters, not disadvantaged-user parameters.

**Fix**: The user calibration path must connect to user-side parameters through
the calibrated scoring interface.

---

## Defect 5 (Critical): No bi-level parameter separation

**File**: `run.py:365-620`
**Issue**: Training uses a single backbone optimizer. Recommendation loss and
fairness loss update the same `backbone.parameters()`. There is no Θ_acc / Θ_fair
separation and no hypergradient. `BiLevelOptimizer` (bilevel_opt.py) is instantiated
in `dual2fair.py:33` but never called in the training loop.

**Fix**: Implement explicit optimization modes (joint, alternating, mirror_alternating).
Remove the unused/broken `BiLevelOptimizer`.

---

## Defect 6 (Critical): BiLevelOptimizer mirror step is broken

**File**: `models/dual2fair/bilevel_opt.py:52-60`
**Issue**: After `fair_optimizer.zero_grad()`, the code tries to use `param.grad`
which was just zeroed. The second mirror step is ineffective — it does nothing.

**Fix**: Replace with correctly implemented optimization modes that recompute
the fairness loss at intermediate parameters θ'.

---

## Defect 7 (Critical): Nyström not implemented

**File**: `models/dual2fair/sinkhorn.py:14-17`
**Issue**: The function is named `sinkhorn_nystrom` and accepts Nyström parameters
(`nystrom_rank`, `rank_increment`, `nystrom_tol`, `tikhonov_lambda`) but the docstring
explicitly states the Nyström branch is disabled. All OT uses dense log-domain Sinkhorn.

**Fix**: Rename to `dense_log_sinkhorn`. Remove Nyström parameters from the dense API.
Implement a genuine landmark-based scalable backend if scalability is needed.

---

## Defect 8 (Critical): NeuMF evaluation bypasses MLP and prediction layer

**File**: `models/backbone/neumf.py:20-26`
**Issue**: `get_user_embeddings()` returns `gmf_user_emb + mlp_user_emb` and
`get_item_embeddings()` returns `gmf_item_emb + mlp_item_emb`. Dual2Fair evaluation
uses `dot(gmf+mlp_user, gmf+mlp_item)`, completely bypassing the MLP layers and
prediction layer that NeuMF training uses.

**Fix**: Implement a NeuMF calibration adapter that preserves the native scorer.
Calibrate GMF and MLP branch tensors, then pass through the original MLP and pred_layer.

---

## Defect 9 (Critical): VAECF uses untrained independent item embedding

**File**: `models/backbone/vaecf.py:35`
**Issue**: `self.item_emb = nn.Embedding(n_items, embedding_dim)` is separate from
the decoder. VAE reconstruction loss only updates encoder/decoder. `item_emb` receives
no gradient from training. Evaluation uses `dot(encoder_mu, item_emb.weight)` instead
of decoder logits.

**Fix**: Use decoder logits as the scorer. Tie item representations to decoder
output weights. Remove the untrained `item_emb`.

---

## Defect 10 (High): Large datasets only calibrate a small subset

**File**: `models/dual2fair/user_calibration.py:75-76`, `item_calibration.py:53-57`
**Issue**: User OT samples at most 4096 disadvantaged users; item OT samples at most
4096 items. At evaluation, only cached/sampled users are calibrated; others stay raw.
Items are randomly re-sampled each evaluation call.

**Fix**: User OT with K=64 prototypes should process ALL disadvantaged users in chunks
(O(N_d × K)). Item inference must map every item to calibrated anchors.

---

## Defect 11 (High): Best checkpoint restoration leaves stale OT/GMM caches

**File**: `run.py:561-563`
**Issue**: `backbone.load_state_dict(best_state)` restores backbone parameters but
does not refit GMM or recompute transport plans. Final test uses best backbone with
last training epoch's calibration caches.

**Fix**: Save and restore full calibration state (GMM, prototypes, OT plans) alongside
backbone parameters.

---

## Defect 12 (High): Dataset statistics do not match paper

**File**: `README.md`, `data/dataset_utils.py`
**Issue**: Current README reports Epinions 20,382/30,989/542,856, MovieLens
6,034/3,125/574,376, Gowalla 29,495/40,358/2,001,700. Paper reports different numbers.

**Decision**: Keep current statistics; update paper to match.

---

## Defect 13 (High): User/item frequency computed before train/val/test split

**File**: `data/dataset_utils.py:75-80`
**Issue**: `item_freq` and `user_freq` are computed on the full `df` (before splitting)
and used for advantaged/disadvantaged grouping and training weights.

**Fix**: Compute frequencies from `train_data` only.

---

## Defect 14 (High): `num_repeats=5` is not five independent runs

**File**: `config/default.yaml:79`, `evaluation/evaluator.py:34-37`
**Issue**: `num_repeats: 5` repeats the evaluator's deterministic metric computation
5 times. It does not train 5 independent models.

**Fix**: Replace with `--seeds 42 43 44 45 46` for independent training runs.

---

## Defect 15 (High): DIF not affine-invariant

**File**: `evaluation/metrics.py:51-77`
**Issue**: `compute_dif_from_exposure` applies sigmoid to scores. Sampled mode only
accumulates quality on ~100 candidates per user. DIF is not invariant to
positive affine transformations of scores.

**Fix**: Add `compute_dif_affine_invariant` using per-user percentile ranks.

---

## Defect 16 (High): UIF normalization protocol is incorrect

**File**: `run_experiment.py:55-60`, `evaluation/metrics.py:185-195`
**Issue**: `run_experiment.py` uses Standard's **test** DUF/DIF as normalization
constants for other methods. `compute_uif` falls back to raw values when constants
are missing. Standard's UIF should equal ~1/NDCG with frozen validation constants
and w1+w2=1.

**Fix**: Freeze Standard's **validation** constants. Read w1/w2 from config.

---

## Defect 17 (Medium): Only backbone parameters are checkpointed

**File**: `run.py:561`
**Issue**: Checkpoint only saves `backbone.state_dict()`. No GMM, prototypes, sampling
indices, transport plans, or calibration configuration are saved.

**Fix**: Save and restore full calibration state.

---

## Defect 18 (Medium): README claims `external_baselines/` directory

**File**: `README.md`
**Issue**: README references `external_baselines/` directory but the directory does
not exist in the repository (it is gitignored).

**Fix**: Remove the reference or clarify that baselines are local implementations.
