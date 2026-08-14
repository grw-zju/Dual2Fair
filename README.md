# Dual2Fair

PyTorch implementation of Dual2Fair for fairness-aware recommendation on sparse-history users and long-tail items.

Dual2Fair targets sparse-but-observed users and long-tail-but-observed items. Strict zero-interaction cold start is outside the current scope. The reported scalability experiments cover public catalogs up to 40,358 items and do not establish industrial-scale streaming deployment.

## Installation

```bash
pip install -r requirements.txt
```

## Data

### Download

Complete processed datasets can be downloaded from:

```text
https://drive.google.com/drive/folders/1nbI95AFsZG2Oq0cZAYVEBKK8spGpqJ3p?usp=sharing
```

Place the downloaded files under the corresponding directories in `data/`.

A runnable demo dataset is included at:

```text
data/demo/interactions.csv
```

### Dataset Statistics

| Dataset | Users | Items | Interactions | Sparsity |
|---|---:|---:|---:|---:|
| MovieLens | 6,034 | 3,125 | 574,376 | 96.9539% |
| Epinions | 20,382 | 30,989 | 542,856 | 99.9141% |
| Gowalla | 29,495 | 40,358 | 2,001,700 | 99.8318% |

## Quick Demo

The demo dataset can be used for fast CPU smoke tests:

```bash
python run.py --dataset demo --backbone lightgcn --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
python run.py --dataset demo --backbone neumf --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
python run.py --dataset demo --backbone vaecf --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
```

`--allow-missing-uif-reference` is intended only for smoke tests or for generating Standard reference constants.

## Running Experiments

### Main Model

```bash
python run.py --dataset movielens --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config config/default.yaml
python run.py --dataset epinions --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config config/default.yaml
python run.py --dataset gowalla --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config config/default.yaml
```

Supported backbones:

```text
lightgcn, neumf, vaecf
```

Full-catalog warm-start evaluation is the paper default. `sampled` evaluation is retained only as a deprecated fast sanity-check mode.

### Five Independent Runs

```bash
python scripts/run_five_seeds.py --dataset movielens --backbone lightgcn --method dual2fair \
  --split-seed 2026 --gpu 0 --config config/default.yaml
```

Paper-style UIF requires split-specific Standard reference constants. Fill `evaluation.uif_references.val` and `evaluation.uif_references.test` in the config before comparing fairness-aware methods.

### Ablation Studies

Dual2Fair ablations are controlled by config fields:

| Ablation | Config |
|---|---|
| w/o URC | `dual2fair.enable_user_calibration: false` |
| w/o IRC | `dual2fair.enable_item_calibration: false` |
| Uniform Target | `dual2fair.omega: 1.0` |
| Merit Only | `dual2fair.omega: 0.0` |
| w/o Confidence | `dual2fair.enable_confidence: false` |
| Joint Weighted Sum | `dual2fair.optimization_strategy: joint_weighted_sum` |
| Standard Alternating | `dual2fair.optimization_strategy: standard_alternating` |
| Hard Matching | `dual2fair.alignment_mode: hard` |
| MMD Alignment | `dual2fair.alignment_mode: mmd` |
| Dual2Fair | `dual2fair.alignment_mode: ot` and `dual2fair.optimization_strategy: hierarchical_mirror` |

Command-line examples:

```bash
python run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode hard --eval_mode full --gpu 0
python run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode mmd --eval_mode full --gpu 0
```

Hard Matching replaces globally constrained soft OT with independent target-prior-aware hard assignments. MMD Alignment replaces sample-level transport correspondence with weighted linear-kernel first-moment alignment and uses the resulting mean shift as the calibration direction.

### Baselines

Local baseline adapters are registered under `baseline/`. Some baselines require external official-code wrappers; configure the corresponding `baseline.<method>` fields only when running those methods.

### Efficiency and Analysis

```bash
python scripts/benchmark_efficiency.py
python scripts/benchmark_scaling.py
python scripts/analyze_cross_side_interference.py
```

## Paper-Reported Results

The following values are manuscript-reported five-run means and are not claimed as results reproduced by the current commit.

| Dataset | Method | NDCG | DUF | DIF | UIF |
|---|---|---:|---:|---:|---:|
| Epinions | Standard | 0.4005 | 0.0395 | 0.0185 | 2.4969 |
| Epinions | Dual2Fair | 0.4140 | 0.0212 | 0.0108 | 1.3533 |
| MovieLens | Standard | 0.4862 | 0.0585 | 0.0255 | 2.0568 |
| MovieLens | Dual2Fair | 0.5031 | 0.0392 | 0.0133 | 1.1843 |
| Gowalla | Standard | 0.4011 | 0.0552 | 0.0323 | 2.4931 |
| Gowalla | Dual2Fair | 0.4164 | 0.0375 | 0.0146 | 1.3585 |

## Notes

- Do not treat demo or sampled-evaluation outputs as paper reproduction results.
- Full paper result reproduction has not been rerun by this code update.
- See `ALIGNMENT_REPORT.md` for implementation alignment details and unresolved reproduction notes.
