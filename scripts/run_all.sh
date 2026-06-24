#!/bin/bash
# Dual2Fair: Full Experiment Script for GPU Server
# Usage: bash scripts/run_all.sh [GPU_ID] [SEED] [EVAL_MODE]
# Example: bash scripts/run_all.sh 0 42 sampled

set -e

GPU_ID=${1:-0}
SEED=${2:-42}
EVAL_MODE=${3:-sampled}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/saved_models"

echo "=============================================="
echo "Dual2Fair Full Experiments"
echo "GPU: ${GPU_ID}, Seed: ${SEED}, Eval: ${EVAL_MODE}"
echo "Project: ${PROJECT_DIR}"
echo "=============================================="

cd "${PROJECT_DIR}"

DATASETS="movielens epinions gowalla"
BACKBONES="neumf vaecf lightgcn"

echo ""
echo "[Phase 1] Training standard backbones..."
for ds in ${DATASETS}; do
    for bb in ${BACKBONES}; do
        LOG="${LOG_DIR}/${ds}_${bb}_standard.log"
        echo "  Standard ${bb} on ${ds}..."
        python run.py --dataset ${ds} --backbone ${bb} --method standard \
            --eval_mode ${EVAL_MODE} --gpu ${GPU_ID} --seed ${SEED} 2>&1 | tee "${LOG}"
    done
done

echo ""
echo "[Phase 2] Training Dual2Fair..."
for ds in ${DATASETS}; do
    for bb in ${BACKBONES}; do
        for l1 in 0.01 0.05 0.1 0.5 1.0; do
            for l2 in 0.01 0.05 0.1 0.5 1.0; do
                LOG="${LOG_DIR}/${ds}_${bb}_dual2fair_l1${l1}_l2${l2}.log"
                echo "  Dual2Fair ${bb}/${ds}, λ1=${l1}, λ2=${l2}..."
                python run.py --dataset ${ds} --backbone ${bb} --method dual2fair \
                    --lambda1 ${l1} --lambda2 ${l2} \
                    --eval_mode ${EVAL_MODE} --gpu ${GPU_ID} --seed ${SEED} 2>&1 | tee "${LOG}"
            done
        done
    done
done

echo ""
echo "[Phase 3] In-processing baselines..."
INPROC_METHODS="dpr fairdual multifr ada2fair fair hyperuof"
for ds in ${DATASETS}; do
    for bb in ${BACKBONES}; do
        for method in ${INPROC_METHODS}; do
            LOG="${LOG_DIR}/${ds}_${bb}_${method}.log"
            echo "  ${method} on ${bb}/${ds}..."
            python run.py --dataset ${ds} --backbone ${bb} --method ${method} \
                --eval_mode ${EVAL_MODE} --gpu ${GPU_ID} --seed ${SEED} 2>&1 | tee "${LOG}"
        done
    done
done

echo ""
echo "[Phase 4] Post-processing baselines..."
POSTPROC_METHODS="ufr cpfair fairsort"
for ds in ${DATASETS}; do
    for bb in ${BACKBONES}; do
        for method in ${POSTPROC_METHODS}; do
            LOG="${LOG_DIR}/${ds}_${bb}_${method}.log"
            echo "  ${method} on ${bb}/${ds}..."
            python run.py --dataset ${ds} --backbone ${bb} --method ${method} \
                --eval_mode ${EVAL_MODE} --gpu ${GPU_ID} --seed ${SEED} 2>&1 | tee "${LOG}"
        done
    done
done

echo ""
echo "[Phase 5] Aggregating results..."
python -c "
import json, os, glob
results_dir = '${RESULTS_DIR}'
all_results = {}
for fpath in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    with open(fpath) as f:
        r = json.load(f)
    key = os.path.basename(fpath).replace('.json', '')
    all_results[key] = r
out_path = os.path.join(results_dir, 'all_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Aggregated {len(all_results)} results to {out_path}')
"

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results: ${RESULTS_DIR}/all_results.json"
echo "Logs: ${LOG_DIR}/"
echo "=============================================="
