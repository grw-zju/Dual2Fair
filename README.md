# Dual2Fair

PyTorch implementation for **Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation**.

## Installation

```bash
pip install -r requirements.txt
```

## Data

The complete processed datasets are distributed separately and are not stored in GitHub.
Download the processed package from Google Drive and place it under `data/`:

```text
https://drive.google.com/drive/folders/1nbI95AFsZG2Oq0cZAYVEBKK8spGpqJ3p?usp=sharing
```

Expected layout:

```text
data/movielens/ml-1m/ratings.dat
data/epinions/epinion_with_rating_timestamp_txt/rating_with_timestamp.txt
data/gowalla/loc-gowalla_totalCheckins.txt
data/demo/interactions.csv
```

The bundled demo data are for demo runs only and do not reproduce reported tables.

| Dataset | Users | Items | Interactions | Sparsity |
|---|---:|---:|---:|---:|
| MovieLens | 6,034 | 3,125 | 574,376 | 96.9539% |
| Epinions | 20,382 | 30,989 | 542,856 | 99.9141% |
| Gowalla | 29,495 | 40,358 | 2,001,700 | 99.8318% |

Verify a downloaded data package:

```bash
python3 scripts/verify_data_package.py --data-root data
```

## Quick Demo

```bash
python3 run.py --dataset demo --backbone lightgcn --method dual2fair --eval_mode full --gpu -1 --config configs/demo.yaml
```

## Reproducing the paper experiments

The default configuration is `configs/default.yaml`.

### Main model

```bash
python3 run.py --dataset movielens --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config configs/default.yaml
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config configs/default.yaml
python3 run.py --dataset gowalla --backbone lightgcn --method dual2fair --eval_mode full --gpu 0 --config configs/default.yaml
```

Supported datasets: `movielens`, `epinions`, `gowalla`, `demo`.
Supported backbones: `lightgcn`, `neumf`, `vaecf`.

### UIF reference construction

Build split-specific Standard references from five Standard runs:

```bash
python3 scripts/build_uif_reference.py results/standard_seed_42.json results/standard_seed_43.json results/standard_seed_44.json results/standard_seed_45.json results/standard_seed_46.json --output results/uif_reference.json
```

Use the reference file in later runs:

```bash
python3 run.py --dataset movielens --backbone lightgcn --method dual2fair --uif-reference-file results/uif_reference.json --config configs/default.yaml
```

### Five-seed execution

```bash
python3 scripts/run_multi_seed.py --dataset movielens --backbone lightgcn --method dual2fair --config configs/default.yaml
```

### Hyperparameter selection

```bash
python3 scripts/select_hyperparameters.py --standard-aggregate results/standard_val_aggregate.json --candidate results/dual2fair_l1_0.1_l2_0.1_val_aggregate.json --output results/selected_config.json
```

### Baselines and ablations

```bash
python3 run.py --dataset movielens --backbone lightgcn --method standard --eval_mode full --gpu 0 --config configs/default.yaml
python3 run.py --dataset movielens --backbone lightgcn --method dpr --eval_mode full --gpu 0 --config configs/default.yaml
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode hard --eval_mode full --gpu 0 --config configs/default.yaml
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode mmd --eval_mode full --gpu 0 --config configs/default.yaml
```

### Table IV efficiency benchmark

```bash
python3 scripts/run_efficiency.py --dataset movielens --backbone lightgcn --method standard --gpu 0 --config configs/default.yaml --output-json results/table4_standard.json
python3 scripts/run_efficiency.py --dataset movielens --backbone lightgcn --method dual2fair_lowrank --gpu 0 --config configs/default.yaml --output-json results/table4_lowrank.json
```

Dense mode is an explicit reference/benchmark path and should only be used when the catalog is small enough:

```bash
python3 scripts/run_efficiency.py --dataset movielens --backbone lightgcn --method dual2fair_dense --gpu 0 --config configs/default.yaml
```

### Table V Gowalla scaling benchmark

Exact Gowalla scaling inputs must come from the external processed-data package. The script fails if exact subsets or manifests are missing.

```bash
python3 scripts/run_scaling.py --subset-dir data/gowalla_scaling --gpu 0 --config configs/default.yaml --output-json results/table5_scaling.json
```

## Tests

```bash
python3 -m pytest -q
```
