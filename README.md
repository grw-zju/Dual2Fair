# Dual2Fair

Corrected research implementation of **Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation**.

## Correctness status

This revision fixes row-normalized OT barycentric projection, integrates calibrated representations into training and evaluation, preserves native LightGCN/NeuMF/VAECF scorers, removes the false Nyström and strict bi-level claims, eliminates train/validation/test frequency leakage, and replaces evaluator repetition with independent model seeds.

The default optimization mode is **hierarchical mirror-style alternating leader–follower optimization**. Strict unrolled bilevel optimization is not implemented or claimed.

## Installation

```bash
pip install -r requirements.txt
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

Raw datasets download on demand. The bundled `demo` dataset is intended for a quick runnable check.

## Model semantics

- **LightGCN**: propagated user/item representations and native dot product.
- **NeuMF**: separate GMF and MLP branches, original MLP stack, and prediction layer.
- **VAECF**: decoder hidden user representation, decoder output weights, and decoder bias.
- **User OT**: all disadvantaged users are processed in chunks against GMM prototypes.
- **Item OT**: all items map through deterministic landmarks; no random evaluation subset.
- **Item target**: merit–uniform mixture estimated from deterministic sampled training users and item anchors through the native scorer; uniform and merit modes are ablations.
- **Popularity baseline**: `popularity_ips` provides inverse-propensity popularity reweighting.

## Metrics

- NDCG@K and Hit@K
- DUF (per-user NDCG variance)
- `DIF_legacy`
- affine-invariant DIF (default)
- rank DIF
- UIF using frozen Standard validation DUF/DIF constants

UIF weights are configured by `evaluation.uif_w1/uif_w2`. UIF is a summary indicator, not the sole selection criterion.

## Reproducibility

Checkpoints include backbone/calibration parameters, GMM/prototype buffers, item anchors and target distribution, optimizer/scheduler state when supplied, resolved configuration, and split hash. Frequencies, groups, GMM inputs, and merit targets use training interactions only.

Baseline files in `baseline/` are local compatible implementations unless their source is explicitly stated.

## Citation

```bibtex
@article{geng2026dual2fair,
  title={Decoupled User--Item Representation Calibration for Two-Sided Fairness in Recommendation},
  author={Geng, Renwu and Liu, Weiming and Wang, Fan and Hu, Liang and Chen, Chaochao},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```
