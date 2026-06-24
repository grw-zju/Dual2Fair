# Dual2Fair: Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation

PyTorch implementation of the paper "Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation" (TKDE 2026).

## Overview

Dual2Fair is a model-agnostic in-processing framework that addresses two-sided fairness in recommender systems through:
1. **Dual-Path Decoupled Optimization**: Separate calibration paths for user-side (disadvantaged user representation) and item-side (long-tail item exposure) fairness
2. **Hierarchical Bi-level Optimization**: Accuracy as leader objective, fairness as cooperative follower with mirror-style regularization
3. **Optimal Transport Alignment**: Entropy-regularized OT with bounded sampled user/item calibration for large datasets

## Project Structure

```
Dual2Fair/
├── config/default.yaml         # Default hyperparameters
├── data/dataset_utils.py       # Dataset download, preprocessing, train/val/test LOO split
├── models/backbone/
│   ├── base_backbone.py        # Backbone base class
│   ├── neumf.py                # NeuMF (Neural Collaborative Filtering)
│   ├── vaecf.py                # VAECF (Variational Auto-Encoders for CF)
│   ├── lightgcn.py             # LightGCN (Light Graph Convolution Network)
├── models/dual2fair/
│   ├── sinkhorn.py             # Stable log-domain Sinkhorn OT solver
│   ├── user_calibration.py     # GMM prototype construction + OT alignment
│   ├── item_calibration.py     # Inverse-frequency OT alignment
│   ├── bilevel_opt.py          # Bi-level mirror-style optimization
│   ├── dual2fair.py            # Full Dual2Fair framework
├── baseline/
│   ├── ufr.py                  # UFR (User-side, post-processing)
│   ├── hyperuof.py             # HyperUOF (User-side, in-processing)
│   ├── dpr.py                  # DPR (Item-side, regularization)
│   ├── fairdual.py             # FairDual (Item-side, dual optimization)
│   ├── cpfair.py               # CPFair (Two-sided, re-ranking)
│   ├── multifr.py              # MultiFR (Two-sided, multi-objective)
│   ├── ada2fair.py             # Ada2Fair (Two-sided, adaptive weights)
│   ├── fair_method.py          # FAIR (Two-sided, online algorithm)
│   ├── fairsort.py             # FairSort (Two-sided, velocity assignment)
├── external_baselines/         # Cloned official baseline repositories and notes
├── evaluation/
│   ├── metrics.py              # NDCG@K, DUF, DIF, UIF metrics
│   ├── evaluator.py            # Evaluation orchestrator
├── run.py                      # Main training script
├── run_experiment.py           # Batch experiment runner
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Environment Setup

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.9+
- PyTorch >= 1.12
- numpy, scipy, scikit-learn, pandas, pyyaml
- Optional: `gurobipy` only if you want UFR to use the official Gurobi optimizer path. Without it, UFR uses the built-in greedy fallback.

## Dataset Preparation

Datasets are automatically downloaded when first loaded if network access is available. You can also place the raw files under `data/` manually:

With the raw files currently in this workspace and the thresholds in `config/default.yaml`, the loader produces:

| Dataset | Users | Items | Interactions | Train | Val | Test |
|---------|-------|-------|-------------|-------|-----|------|
| MovieLens | 6,034 | 3,125 | 574,376 | 562,308 | 6,034 | 6,034 |
| Epinions | 20,382 | 30,989 | 542,856 | 502,092 | 20,382 | 20,382 |
| Gowalla | 29,495 | 40,358 | 2,001,700 | 1,942,710 | 29,495 | 29,495 |

Use these statistics to verify that preprocessing finished correctly before running experiments.

For MovieLens-1M, download from https://grouplens.org/datasets/movielens/1m/ and extract `ml-1m/ratings.dat` to `data/movielens/ml-1m/`. The current workspace already contains MovieLens files.

The loader filters users/items by the thresholds in `config/default.yaml`, then creates a leave-one-out style split:
- timestamped data: last item = test, second-last item = validation
- untimestamped data: shuffled last item = test, second-last item = validation

Training and early stopping use validation. The final JSON result is computed on test.

## How to Run

### Single Experiment

```bash
python3 run.py --dataset movielens --backbone lightgcn --method standard --loss_type bpr
python3 run.py --dataset movielens --backbone lightgcn --method dual2fair --lambda1 0.1 --lambda2 0.1 --loss_type bpr
python3 run.py --dataset movielens --backbone lightgcn --method dpr --loss_type bpr
```

Options:
- `--dataset`: epinions, movielens, gowalla
- `--backbone`: neumf, vaecf, lightgcn
- `--method`: standard, dual2fair, ufr, hyperuof, dpr, fairdual, cpfair, multifr, ada2fair, fair, fairsort
- `--lambda1, --lambda2`: Dual2Fair fairness weights (default: 0.1)
- `--eval_mode`: `sampled` (99 negatives, default) or `full`
- `--loss_type`: `bpr` or `bce`; paper-style top-K runs should use `bpr`
- `--gpu`: GPU device index (default: 0)
- `--seed`: Random seed (default: 42)

Outputs:
- model checkpoints: `saved_models/{dataset}_{backbone}_{method}.pt`
- metrics JSON: `results/{dataset}_{backbone}_{method}.json`

### Batch Experiments

```bash
python3 run_experiment.py --datasets movielens --backbones lightgcn --methods standard dual2fair dpr
python3 run_experiment.py --datasets epinions movielens gowalla --backbones lightgcn neumf vaecf
```

`run_experiment.py` first trains/evaluates the standard backbone for each dataset/backbone pair. Its test DUF/DIF are then used as the normalization constants for UIF when running the remaining methods on that same dataset/backbone.

Convenience shell commands:

```bash
bash run.sh quick
bash run.sh standard
bash run.sh dual2fair
bash run.sh baseline
bash run.sh all
```

For large server runs:

```bash
bash scripts/run_all.sh 0 42 sampled
```

`run.sh all` and `scripts/run_all.sh` can take a long time because they cover all datasets, backbones, and baselines.

## VAECF User/Item Embedding Extraction

VAECF is a VAE-based collaborative filtering model that uses multinomial likelihood. The user and item embeddings are obtained as follows:

**User Embedding**: The encoder takes each user's interaction vector (binary/soft count) as input and outputs a latent representation. The encoder mean vector `μ` serves as the user embedding:
```python
# In vaecf.py
user_embedding = encoder_mu  # Shape: (n_users, embedding_dim)
```

**Item Embedding**: VAECF maintains a separate `item_emb` embedding layer. Each item's embedding is directly accessed:
```python
# In vaecf.py
item_embedding = item_emb.weight  # Shape: (n_items, embedding_dim)
```

The decoder reconstructs the full item space from the latent representation, and the item embeddings provide the item-side representation that aligns with the encoder's latent space for scoring.

## Key Hyperparameters (matching paper Table II)

| Parameter | Value | Description |
|-----------|-------|-------------|
| embedding_dim | 64 | User/item embedding dimension |
| learning_rate | 0.001 | Adam optimizer learning rate |
| batch_size | 4096 | Mini-batch size |
| max_epochs | 200 | Maximum training epochs |
| early_stop | 50 | Patience for early stopping |
| gmm_clusters | 64 | Number of GMM interest prototypes |
| sinkhorn_epsilon | 0.1 | OT entropic regularization |
| nystrom_rank | 32 | Initial Nystrom sketch rank |
| nystrom_tol | 1e-3 | Nystrom approximation tolerance |
| max_ot_items | 4096 | Maximum sampled items for item-side OT on large datasets |
| max_disadv_users | 4096 | Maximum sampled disadvantaged users for user-side OT on large datasets |
| tikhonov_lambda | 1e-4 | Tikhonov regularization |
| sinkhorn_max_iter | 100 | Max Sinkhorn iterations |
| bilevel_beta | 3 | Fairness update frequency interval |
| mirror_alpha1 | 1.0 | First mirror coefficient |
| mirror_alpha2 | 0.1 | Second mirror coefficient |
| adv_ratio | 0.05 | Advantaged user group ratio (top 5%) |
| λ1, λ2 | 0.1 | Fairness trade-off weights (best from grid search) |

## Evaluation Metrics

- **NDCG@10**: Recommendation accuracy
- **DUF**: Deviation from User-Side Fairness (lower = more fair)
- **DIF**: Deviation from Item-Side Fairness (lower = more fair)
- **UIF**: Unified User-Item Fairness (lower = more fair)

UIF = (w1 * DUF/μ_DUF + w2 * DIF/μ_DIF) / μ_NDCG, with w1=w2=0.5

In `sampled` mode, NDCG and Hit follow the SIGIR2025 reference style: predictions and binary labels are evaluated within a fixed candidate set per user using `sklearn.metrics.ndcg_score`, then averaged over users. DUF is computed from those same sampled per-user NDCG values, so sampled accuracy and sampled user fairness use the same candidate protocol. In `full` mode, NDCG/Hit/DUF are computed over all items after masking train interactions.

For large datasets, use `--eval_mode sampled`. The sampled evaluator scores only each user's sampled candidate set instead of materializing the full user-item score matrix. Dual2Fair also samples large user/item OT problems via `dual2fair.max_disadv_users` and `dual2fair.max_ot_items` to avoid quadratic memory use on Epinions and Gowalla.

`full` evaluation is guarded by `evaluation.max_full_eval_scores` (default 50,000,000). If a dataset/backbone combination would require a larger dense user-item score matrix, the run stops with a clear error instead of exhausting memory. Raise this limit only when you have enough RAM/GPU memory.

For single `run.py` calls, UIF uses raw DUF/DIF unless `baseline_duf` and `baseline_dif` are set in the config. For fair method-to-method tables, prefer `run_experiment.py`, which fills those constants from the standard backbone automatically.

## Baselines and Official Sources

The project includes runnable baseline implementations under `baseline/`. Official repositories used for baseline alignment are stored under `external_baselines/`:

| Method | Official source | Local behavior |
|--------|------------------------|----------------|
| UFR | `external_baselines/user-fairness` | Uses official-style Gurobi constrained optimizer if `gurobipy` is installed; otherwise uses a deterministic greedy fallback. |
| DPR | `external_baselines/Item-Underrecommendation-Bias` | PyTorch implementation of score distribution and adversarial group regularization. |
| Ada2Fair | `external_baselines/Ada2Fair` | PyTorch implementation of dynamic provider/user fairness weights. |
| FairDual | `external_baselines/FairDual` | Shadow-price exposure-disparity implementation. |
| CPFair, MultiFR, FAIR, FairSort | Local implementation | Self-contained implementations in the unified PyTorch/numpy pipeline. |

## Dual2Fair Algorithm

1. **User-side Path (Module A)**:
   - A1: Fit GMM on advantaged users' interest embeddings → interest prototypes
   - A2: Align disadvantaged users to prototypes via OT → calibrate representations
   - Fuse: e_final = α * e_original + (1-α) * e_calibrated

2. **Item-side Path (Module B)**:
   - Source: inverse-frequency weighted distribution (emphasize cold items)
   - Target: uniform distribution over all items
   - OT alignment between source and target

3. **Bi-level Optimization (Module C)**:
   - Leader: Θ_acc ← Θ_acc - η∇ L_rec (every step)
   - Follower: Mirror-style update every β=3 steps:
     - Θ' ← Θ - α1η∇ L_fair
     - Θ ← Θ' + α2η∇ L_fair

## Paper Results Reference (Table II - LightGCN backbone)

| Method | Epinions NDCG | DUF | DIF | UIF |
|--------|-------------|-----|-----|-----|
| Standard | 0.4005 | 0.0395 | 0.0185 | 2.4969 |
| Dual2Fair | 0.4122 | 0.0227 | 0.0117 | 1.4642 |

| Method | MovieLens NDCG | DUF | DIF | UIF |
|--------|--------------|-----|-----|-----|
| Standard | 0.4862 | 0.0585 | 0.0255 | 2.0568 |
| Dual2Fair | 0.5012 | 0.0417 | 0.0142 | 1.2666 |

## Citation

```bibtex
@article{geng2026dual2fair,
  title={Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation},
  author={Geng, Renwu and Liu, Weiming and Wang, Fan and Hu, Liang and Chen, Chaochao},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```
