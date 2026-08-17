# Data Preparation

This repository does not store the complete datasets. Prepare the original files under the following paths before running full-data experiments.

## Expected raw files

```text
data/movielens/ml-1m/ratings.dat
data/epinions/epinion_with_rating_timestamp_txt/rating_with_timestamp.txt
data/gowalla/loc-gowalla_totalCheckins.txt
data/demo/interactions.csv
```

## Original sources

- MovieLens-1M: `https://files.grouplens.org/datasets/movielens/ml-1m.zip`
- Epinions: `https://www.cse.msu.edu/~tangjili/datasetcode/epinions_with_rating_timestamp_txt.zip`
- Gowalla: `https://huggingface.co/datasets/habedi/gowalla-dataset/resolve/main/original_data/loc-gowalla_totalCheckins.txt.gz?download=true`

An external processed-data package may also be used if it matches the expected directory structure.

## Processing protocol

`data/dataset_utils.py` performs the preprocessing used by the code path:

1. implicit binary feedback;
2. duplicate user-item interactions removed;
3. users with fewer than three interactions removed;
4. chronological leave-one-out split;
5. the latest interaction is test, the second latest is validation, and the rest are training;
6. evaluation uses the warm-start catalog induced by the training split.

Frozen split files are written under the configured results directory when `run.py` is executed. Reusing the same split seed reuses the same split metadata.

## Demo data

`data/demo/interactions.csv` is a lightweight interface-validation dataset. It is intended only for smoke tests and does not reproduce the numerical results in Tables IV-V.
