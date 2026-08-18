# Dual2Fair

PyTorch implementation for **Decoupled User–Item Representation Calibration for Two-Sided Fairness in Recommendation**.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA GPU is recommended for full datasets; CPU is sufficient for the bundled demo.

```bash
pip install -r requirements.txt
```

## Data

Download the processed data package from Google Drive and place it under `data/`:

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

The bundled demo data are only for checking that the code executes.

| Dataset | Users | Items | Interactions | Sparsity |
|---|---:|---:|---:|---:|
| MovieLens | 6,034 | 3,125 | 574,376 | 96.9539% |
| Epinions | 20,382 | 30,989 | 542,856 | 99.9141% |
| Gowalla | 29,495 | 40,358 | 2,001,700 | 99.8318% |

## Quick Demo

```bash
python3 run.py --dataset demo --backbone lightgcn --method dual2fair --config configs/demo.yaml --gpu -1
```

## Running Experiments

### Main Model

```bash
python3 run.py --dataset movielens --backbone lightgcn --method dual2fair --config configs/default.yaml --gpu 0
python3 run.py --dataset movielens --backbone neumf --method dual2fair --config configs/default.yaml --gpu 0
python3 run.py --dataset movielens --backbone vaecf --method dual2fair --config configs/default.yaml --gpu 0
```

Supported datasets: `movielens`, `epinions`, `gowalla`, `demo`.
Supported backbones: `lightgcn`, `neumf`, `vaecf`.

### Baselines

Generic command:

```bash
python3 run.py --dataset movielens --backbone lightgcn --method METHOD_NAME --config configs/default.yaml --gpu 0
```

Supported `METHOD_NAME` values:

```text
standard, ufr, hyperuof, dpr, fairdual, cpfair, multifr, ada2fair, fair, fairsort, popularity_ips, esam, mgl
```

ESAM and MGL use external official-repository adapters. Supply their `repo_path` and `command` fields in the corresponding baseline configuration before running them.

### Ablations

Generic command:

```bash
python3 run.py --dataset epinions --backbone lightgcn --method dual2fair --alignment_mode hard --config configs/default.yaml --gpu 0
```

Supported executable variants:

```text
w/o URC:              dual2fair.enable_user_calibration = false
w/o IRC:              dual2fair.enable_item_calibration = false
Uniform Target:       dual2fair.omega = 1.0
Merit Only:           dual2fair.omega = 0.0
w/o Confidence:       dual2fair.enable_confidence = false
Joint Weighted Sum:   dual2fair.optimization_strategy = joint_weighted_sum
Standard Alternating: dual2fair.optimization_strategy = standard_alternating
Hard Matching:        --alignment_mode hard
MMD Alignment:        --alignment_mode mmd
```
