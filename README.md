# Dual2Fair

Corrected research implementation of **Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation**.

## Correctness status

This revision fixes row-normalized OT barycentric projection, integrates calibrated representations into training and evaluation, preserves native LightGCN/NeuMF/VAECF scorers, removes the false Nyström and strict bi-level claims, eliminates train/validation/test frequency leakage, and replaces evaluator repetition with independent model seeds.

The default optimization mode is **hierarchical mirror-style alternating leader–follower optimization**. Strict unrolled bilevel optimization is not implemented or claimed.

## Installation

```bash
pip install -r requirements.txt
pytest
```

## Demo

A deterministic synthetic dataset is bundled at `data/demo/interactions.csv`.

```bash
python run.py --dataset demo --backbone lightgcn --method standard --gpu -1
python run.py --dataset demo --backbone lightgcn --method dual2fair --gpu -1
python run.py --dataset demo --backbone neumf --method dual2fair --gpu -1
python run.py --dataset demo --backbone vaecf --method dual2fair --gpu -1
```

## Independent runs

```bash
python run.py --dataset movielens --backbone lightgcn --method dual2fair \
  --seeds 42 43 44 45 46 --split-seed 2026
```

Each seed trains a fresh model; the split is frozen and reused. Results include mean, standard deviation, and 95% confidence interval.

## Datasets

Raw datasets download on demand. Verify exact source checksums and corrected statistics before reporting:

```bash
python scripts/verify_dataset.py --dataset movielens --min-user-interactions 5 --min-item-interactions 5
python scripts/verify_dataset.py --dataset epinions --min-user-interactions 5 --min-item-interactions 5
python scripts/verify_dataset.py --dataset gowalla --min-user-interactions 15 --min-item-interactions 20
```

Historical workspace counts were MovieLens 6,034/3,125/574,376, Epinions 20,382/30,989/542,856, and Gowalla 29,495/40,358/2,001,700. Do not reuse them unless the verifier reproduces them for the exact source checksum.

## Model semantics

- **LightGCN**: propagated user/item representations and native dot product.
- **NeuMF**: separate GMF and MLP branches, original MLP stack, and prediction layer.
- **VAECF**: decoder hidden user representation, decoder output weights, and decoder bias.
- **User OT**: all disadvantaged users are processed in chunks against GMM prototypes.
- **Item OT**: all items map through deterministic landmarks; no random evaluation subset.
- **Item target**: merit–uniform mixture by default; uniform and merit modes are ablations.

## Metrics

- NDCG@K and Hit@K
- DUF (per-user NDCG variance)
- `DIF_legacy`
- affine-invariant DIF (default)
- rank DIF
- UIF using frozen Standard validation DUF/DIF constants

UIF weights are configured by `evaluation.uif_w1/uif_w2`. UIF is a summary indicator, not the sole selection criterion.

## Scalability

The repository implements dense log-domain Sinkhorn and a deterministic landmark backend. It does not claim Nyström Sinkhorn.

```bash
python scripts/benchmark_efficiency.py --items 10000 --anchors 256
python scripts/benchmark_scaling.py --items 10000 --anchors 256
```

## Reproducibility

Checkpoints include backbone/calibration parameters, GMM/prototype buffers, item anchors and target distribution, optimizer/scheduler state when supplied, resolved configuration, and split hash. Frequencies, groups, GMM inputs, and merit targets use training interactions only.

## Documentation

- `docs/CODE_PAPER_AUDIT.md`
- `docs/METHOD_SPEC.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/DATASET_MANIFEST.md`
- `docs/BASELINE_MANIFEST.md`
- `docs/REVIEWER_EVIDENCE_MATRIX.md`
- `docs/MANUSCRIPT_PATCHES.md`

Baseline files in `baseline/` are local compatible implementations unless a pinned official source is explicitly recorded in `docs/BASELINE_MANIFEST.md`.

## Citation

```bibtex
@article{geng2026dual2fair,
  title={Decoupled User--Item Representation Calibration for Two-Sided Fairness in Recommendation},
  author={Geng, Renwu and Liu, Weiming and Wang, Fan and Hu, Liang and Chen, Chaochao},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```
