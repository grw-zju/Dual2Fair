# Reviewer Evidence Matrix

| Concern | Code | Test / evidence | Output |
|---|---|---|---|
| OT row normalization | `user_calibration.py`, `item_calibration.py` | `test_sinkhorn.py` | pytest report |
| Training/evaluation scorer mismatch | `dual2fair.py`, adapters | `test_training_evaluation_scorer_consistency` | pytest report |
| NeuMF fallback scorer | `neumf_adapter.py` | `test_neumf_disabled_calibration_preserves_scorer` | pytest report |
| VAECF untrained item embedding | `vaecf.py`, `vaecf_adapter.py` | `test_vaecf_decoder_equivalence_and_gradient` | pytest report |
| Broken bi-level claim | `optimization/` | `test_mirror_second_gradient_at_intermediate_parameters` | pytest report |
| False Nyström claim | `transport/dense_sinkhorn.py`, `landmark_sinkhorn.py` | benchmark scripts | JSON benchmark output |
| Partial calibration | calibration modules | `test_deterministic_full_user_and_item_calibration` | pytest report |
| Stale checkpoint caches | `Dual2Fair.load_checkpoint` | `test_checkpoint_restores_calibration` | pytest report |
| Data leakage | `dataset_utils.py` | `test_train_only_frequencies` | pytest report |
| DIF scale sensitivity | `metrics.py` | `test_dif_positive_affine_invariance` | pytest report |
| UIF normalization | `metrics.py`, evaluator | UIF tests | result metadata |
| Fake five runs | `run.py --seeds` | independent initialization test | aggregate JSON |
| Scalability | landmark backend | benchmark scripts | scaling JSON |
| Reproducibility | split hashes, seeds, checkpoints | CI + manifest | checkpoint/result JSON |
