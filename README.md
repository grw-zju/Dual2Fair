# Dual2Fair

PyTorch implementation of Dual2Fair for fairness-aware recommendation on sparse-history users and long-tail items.

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

## Project Structure

```text
Dual2Fair/
  config/
    default.yaml
  data/
    dataset_utils.py
    demo/
  models/
    backbone/
    dual2fair/
      alignment.py
      user_calibration.py
      item_calibration.py
      sinkhorn.py
      hierarchical_opt.py
      state.py
      dual2fair.py
  baseline/
  evaluation/
  scripts/
  tests/
  run.py
```

## Quick Demo

The demo dataset can be used for fast CPU smoke tests:

```bash
python run.py --dataset demo --backbone lightgcn --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
python run.py --dataset demo --backbone neumf --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
python run.py --dataset demo --backbone vaecf --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
```

## Running Experiments

### Main Model

```bash
python run.py --dataset movielens --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config config/default.yaml
python run.py --dataset epinions --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config config/default.yaml
python run.py --dataset gowalla --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config config/default.yaml
```

Supported datasets:

```text
movielens, epinions, gowalla, demo
```

Supported backbones:

```text
lightgcn, neumf, vaecf
```

### Five Independent Runs

```bash
python scripts/run_five_seeds.py --dataset movielens --backbone lightgcn --method dual2fair \
  --split-seed 2026 --gpu 0 --config config/default.yaml
```

### Ablation Studies

```bash
python run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode hard --eval_mode full --gpu 0
python run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode mmd --eval_mode full --gpu 0
```

Common config switches:

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

### Baselines

```bash
python run.py --dataset movielens --backbone lightgcn --method standard --eval_mode full --gpu 0
python run.py --dataset movielens --backbone lightgcn --method dpr --eval_mode full --gpu 0
python run.py --dataset movielens --backbone lightgcn --method fairdual --eval_mode full --gpu 0
```

Available method choices are listed in `run.py`.

### Efficiency and Analysis

```bash
python scripts/benchmark_efficiency.py
python scripts/benchmark_scaling.py
python scripts/analyze_cross_side_interference.py
```
