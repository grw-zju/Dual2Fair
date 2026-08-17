# Dual2Fair

PyTorch implementation of Dual2Fair for fairness-aware recommendation.

## Installation

```bash
pip install -r requirements.txt
```

## Data

The complete processed datasets are not stored in this repository. Download the processed data package from Google Drive and place the files under `data/`:

```text
https://drive.google.com/drive/folders/1nbI95AFsZG2Oq0cZAYVEBKK8spGpqJ3p?usp=sharing
```

A lightweight demo dataset is included at `data/demo/interactions.csv` for interface validation.

| Dataset | Users | Items | Interactions | Sparsity |
|---|---:|---:|---:|---:|
| MovieLens | 6,034 | 3,125 | 574,376 | 96.9539% |
| Epinions | 20,382 | 30,989 | 542,856 | 99.9141% |
| Gowalla | 29,495 | 40,358 | 2,001,700 | 99.8318% |

## Project Structure

```text
Dual2Fair/
  data/
  models/
  baseline/
  evaluation/
  scripts/
  tests/
  run.py
```

## Quick Demo

```bash
python3 run.py --dataset demo --backbone lightgcn --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
python3 run.py --dataset demo --backbone neumf --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
python3 run.py --dataset demo --backbone vaecf --method dual2fair --eval_mode full --gpu -1 --allow-missing-uif-reference
```

## Run Full Experiments

```bash
python3 run.py --dataset movielens --backbone lightgcn --method dual2fair --eval_mode full --gpu 0
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --eval_mode full --gpu 0
python3 run.py --dataset gowalla --backbone lightgcn --method dual2fair --eval_mode full --gpu 0
```

Supported datasets:

```text
movielens, epinions, gowalla, demo
```

Supported backbones:

```text
lightgcn, neumf, vaecf
```

## Baselines and Ablations

```bash
python3 run.py --dataset movielens --backbone lightgcn --method standard --eval_mode full --gpu 0
python3 run.py --dataset movielens --backbone lightgcn --method dpr --eval_mode full --gpu 0
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode hard --eval_mode full --gpu 0
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode mmd --eval_mode full --gpu 0
```

Available method choices and command-line options are listed in `run.py`.

## Tests

```bash
python -m pytest -q
```
