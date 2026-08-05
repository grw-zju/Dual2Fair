# Dual2Fair

PyTorch implementation of Dual2Fair.

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

Download the complete processed datasets from:

https://drive.google.com/drive/folders/1nbI95AFsZG2Oq0cZAYVEBKK8spGpqJ3p?usp=sharing

Place the downloaded files under the corresponding directories in `data/`.

| Dataset | Users | Items | Interactions | Sparsity | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 6,034 | 3,125 | 574,376 | 96.9539% | 562,308 | 6,034 | 6,034 |
| Epinions | 20,382 | 30,989 | 542,856 | 99.9141% | 502,092 | 20,382 | 20,382 |
| Gowalla | 29,495 | 40,358 | 2,001,700 | 99.8318% | 1,942,710 | 29,495 | 29,495 |

A small runnable dataset is included at `data/demo/interactions.csv`.

## Run the Demo

```bash
python run.py --dataset demo --backbone lightgcn --method standard --gpu -1
python run.py --dataset demo --backbone lightgcn --method dual2fair --gpu -1
python run.py --dataset demo --backbone neumf --method dual2fair --gpu -1
python run.py --dataset demo --backbone vaecf --method dual2fair --gpu -1
```

## Run on Complete Datasets

```bash
python run.py --dataset movielens --backbone lightgcn --method dual2fair --gpu 0
python run.py --dataset epinions --backbone neumf --method dual2fair --gpu 0
python run.py --dataset gowalla --backbone lightgcn --method dual2fair --gpu 0
```

Available backbones:

- `lightgcn`
- `neumf`
- `vaecf`

Available methods:

- `standard`
- `dual2fair`
- `ufr`
- `hyperuof`
- `dpr`
- `fairdual`
- `cpfair`
- `multifr`
- `ada2fair`
- `fair`
- `fairsort`
- `popularity_ips`

## Multiple Seeds

```bash
python run.py --dataset movielens --backbone lightgcn --method dual2fair \
  --seeds 42 43 44 45 46 --split-seed 2026 --gpu 0
```

Model checkpoints are written to `saved_models/` and experiment outputs are written to `results/`. Both directories are excluded from Git.
