# Experiment Protocol

## Splits and seeds

One split is frozen per dataset with `data_split_seed=2026`. Model runs reuse that split and use independent model seeds 42–46. Negative sampling, OT sampling, and evaluation have separate configured seeds.

## Training information boundary

User activity, item frequency, advantaged/disadvantaged groups, hot/cold groups, inverse-frequency weights, GMM inputs, and merit targets use training interactions only.

## Validation and model selection

Hyperparameters are selected on validation metrics. The recommended rule minimizes validation UIF subject to validation NDCG dropping no more than the configured tolerance from Standard. Test metrics are never used for selection.

## UIF

For each dataset/backbone/candidate-protocol tuple, Standard validation DUF and DIF are frozen before comparison methods are evaluated. Standard and all methods use the same constants. Report weights (0.25,0.75), (0.5,0.5), and (0.75,0.25).

## DIF

Report metric names with candidate protocol: `DIF_legacy`, `DIF_affine_invariant`, and `DIF_rank`. Candidate sets are fixed across methods.

## Independent runs

Use `--seeds 42 43 44 45 46 --split-seed 2026`. Each seed initializes and trains a fresh model. Aggregate mean, sample standard deviation, and 95% confidence interval.

## Efficiency

Use `scripts/benchmark_efficiency.py` and `scripts/benchmark_scaling.py`. Report item/user counts, backend, anchors, Sinkhorn iterations, runtime, and peak CPU memory.
