#!/bin/bash
# Dual2Fair: Data Preparation Script
# Downloads and preprocesses MovieLens-1M, Epinions, and Gowalla datasets
# Run this script once before experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"

echo "=============================================="
echo "Dual2Fair Data Preparation"
echo "=============================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Data dir: ${DATA_DIR}"

# Install dependencies
echo "[1/4] Installing Python dependencies..."
pip install torch numpy scipy scikit-learn pandas pyyaml

# MovieLens-1M
echo "[2/4] Preparing MovieLens-1M..."
ML_DIR="${DATA_DIR}/movielens"
mkdir -p "${ML_DIR}"
if [ ! -f "${ML_DIR}/ml-1m/ratings.dat" ]; then
    echo "  Downloading MovieLens-1M..."
    if [ ! -f "${ML_DIR}/ml-1m.zip" ]; then
        wget -q "https://files.grouplens.org/datasets/movielens/ml-1m.zip" -O "${ML_DIR}/ml-1m.zip"
    fi
    echo "  Extracting..."
    unzip -q -o "${ML_DIR}/ml-1m.zip" -d "${ML_DIR}"
    rm -f "${ML_DIR}/ml-1m.zip"
    echo "  MovieLens-1M ready."
else
    echo "  MovieLens-1M already exists, skipping."
fi

# Epinions
echo "[3/4] Preparing Epinions..."
EP_DIR="${DATA_DIR}/epinions"
mkdir -p "${EP_DIR}"
if [ ! -f "${EP_DIR}/epinion_with_rating_timestamp_txt/rating_with_timestamp.txt" ]; then
    echo "  Downloading Epinions..."
    if [ ! -f "${EP_DIR}/epinions_with_rating_timestamp_txt.zip" ]; then
        wget -q "https://www.cse.msu.edu/~tangjili/datasetcode/epinions_with_rating_timestamp_txt.zip" \
            -O "${EP_DIR}/epinions_with_rating_timestamp_txt.zip"
    fi
    echo "  Extracting..."
    unzip -q -o "${EP_DIR}/epinions_with_rating_timestamp_txt.zip" -d "${EP_DIR}"
    rm -f "${EP_DIR}/epinions_with_rating_timestamp_txt.zip"
    echo "  Epinions ready."
else
    echo "  Epinions already exists, skipping."
fi

# Gowalla
echo "[4/4] Preparing Gowalla..."
GW_DIR="${DATA_DIR}/gowalla"
mkdir -p "${GW_DIR}"
if [ ! -f "${GW_DIR}/loc-gowalla_totalCheckins.txt" ]; then
    echo "  Downloading Gowalla..."
    wget -q "https://huggingface.co/datasets/habedi/gowalla-dataset/resolve/main/original_data/loc-gowalla_totalCheckins.txt.gz?download=true" \
        -O "${GW_DIR}/loc-gowalla_totalCheckins.txt.gz" || {
        echo "  HuggingFace download failed, trying Stanford SNAP..."
        wget -q "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz" \
            -O "${GW_DIR}/loc-gowalla_totalCheckins.txt.gz"
    }
    echo "  Decompressing..."
    gunzip -f "${GW_DIR}/loc-gowalla_totalCheckins.txt.gz"
    echo "  Gowalla ready."
else
    echo "  Gowalla already exists, skipping."
fi

# Preprocess all datasets (run Python to verify data loading)
echo ""
echo "Verifying data loading..."
cd "${PROJECT_DIR}"
python -c "
from data.dataset_utils import load_dataset
for name in ['movielens', 'epinions', 'gowalla']:
    ds_config = {
        'movielens': {'min_ui': 5, 'min_ii': 5},
        'epinions': {'min_ui': 5, 'min_ii': 5},
        'gowalla': {'min_ui': 15, 'min_ii': 20},
    }[name]
    print(f'Loading {name}...')
    ds = load_dataset(name, min_ui=ds_config['min_ui'], min_ii=ds_config['min_ii'])
    print(f'  {name}: {ds.n_users} users, {ds.n_items} items, {len(ds.interactions)} interactions')
print('All datasets loaded successfully!')
"

echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
