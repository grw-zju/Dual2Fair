# Dual2Fair

PyTorch implementation for **Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation**.

## Installation

```bash
pip install -r requirements.txt
```

## Data

### Download

The complete processed datasets are not stored in this repository. Download the processed data package from Google Drive and place the files under `data/`:

```text
https://drive.google.com/drive/folders/1nbI95AFsZG2Oq0cZAYVEBKK8spGpqJ3p?usp=sharing
```

A lightweight demo dataset is included at:

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

## Running Experiments

### Main Model

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

### Baselines

```bash
python3 run.py --dataset movielens --backbone lightgcn --method standard --eval_mode full --gpu 0
python3 run.py --dataset movielens --backbone lightgcn --method dpr --eval_mode full --gpu 0
python3 run.py --dataset movielens --backbone lightgcn --method fairdual --eval_mode full --gpu 0
```

### Ablations

```bash
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode hard --eval_mode full --gpu 0
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode mmd --eval_mode full --gpu 0
```

### Five Runs

```bash
python3 scripts/run_five_seeds.py --dataset movielens --backbone lightgcn --method dual2fair --gpu 0
```

## Tests

```bash
python3 -m pytest -q
```
