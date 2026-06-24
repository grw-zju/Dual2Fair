#!/bin/bash

# Dual2Fair experiment runner
# Usage: bash run.sh [experiment_type]
#   experiment_type: standard | dual2fair | baseline | all

DATASETS="movielens epinions gowalla"
BACKBONES="lightgcn neumf vaecf"
METHODS="standard dual2fair"
BASELINES="ufr hyperuof dpr fairdual cpfair multifr ada2fair fair fairsort"
CONFIG="config/default.yaml"
GPU=0
EVAL_MODE="sampled"
LOSS_TYPE="bpr"
LAMBDA1=0.1
LAMBDA2=0.1
SEED=42

mkdir -p saved_models results logs

run_standard() {
    for dataset in $DATASETS; do
        for backbone in $BACKBONES; do
            echo "[Standard] $dataset/$backbone"
            python3 run.py \
                --dataset $dataset \
                --backbone $backbone \
                --method standard \
                --eval_mode $EVAL_MODE \
                --loss_type $LOSS_TYPE \
                --config $CONFIG \
                --seed $SEED \
                --gpu $GPU \
                2>&1 | tee "logs/standard_${dataset}_${backbone}.log"
        done
    done
}

run_dual2fair() {
    for dataset in $DATASETS; do
        for backbone in $BACKBONES; do
            echo "[Dual2Fair] $dataset/$backbone lambda1=$LAMBDA1 lambda2=$LAMBDA2"
            python3 run.py \
                --dataset $dataset \
                --backbone $backbone \
                --method dual2fair \
                --lambda1 $LAMBDA1 \
                --lambda2 $LAMBDA2 \
                --eval_mode $EVAL_MODE \
                --loss_type $LOSS_TYPE \
                --config $CONFIG \
                --seed $SEED \
                --gpu $GPU \
                2>&1 | tee "logs/dual2fair_${dataset}_${backbone}.log"
        done
    done
}

run_baselines() {
    for dataset in $DATASETS; do
        for backbone in $BACKBONES; do
            for baseline in $BASELINES; do
                echo "[Baseline] $dataset/$backbone/$baseline"
                python3 run.py \
                    --dataset $dataset \
                    --backbone $backbone \
                    --method $baseline \
                    --eval_mode $EVAL_MODE \
                    --loss_type $LOSS_TYPE \
                    --config $CONFIG \
                    --seed $SEED \
                    --gpu $GPU \
                    2>&1 | tee "logs/${baseline}_${dataset}_${backbone}.log"
            done
        done
    done
}

run_grid_search() {
    for dataset in $DATASETS; do
        for backbone in $BACKBONES; do
            for l1 in 0.01 0.05 0.1 0.5 1.0; do
                for l2 in 0.01 0.05 0.1 0.5 1.0; do
                    echo "[GridSearch] $dataset/$backbone lambda1=$l1 lambda2=$l2"
                    python3 run.py \
                        --dataset $dataset \
                        --backbone $backbone \
                        --method dual2fair \
                        --lambda1 $l1 \
                        --lambda2 $l2 \
                        --eval_mode $EVAL_MODE \
                        --loss_type $LOSS_TYPE \
                        --config $CONFIG \
                        --seed $SEED \
                        --gpu $GPU \
                        2>&1 | tee "logs/grid_${dataset}_${backbone}_l1${l1}_l2${l2}.log"
                done
            done
        done
    done
}

case "$1" in
    standard)
        run_standard
        ;;
    dual2fair)
        run_dual2fair
        ;;
    baseline|baselines)
        run_baselines
        ;;
    grid|grid_search)
        run_grid_search
        ;;
    all)
        run_standard
        run_dual2fair
        run_baselines
        ;;
    quick)
        python3 run.py \
            --dataset movielens \
            --backbone lightgcn \
            --method dual2fair \
            --lambda1 $LAMBDA1 \
            --lambda2 $LAMBDA2 \
            --eval_mode $EVAL_MODE \
            --loss_type $LOSS_TYPE \
            --config $CONFIG \
            --seed $SEED \
            --gpu $GPU \
            2>&1 | tee "logs/quick_test.log"
        ;;
    *)
        echo "Usage: bash run.sh [standard|dual2fair|baseline|grid|all|quick]"
        echo ""
        echo "  standard   - Run standard backbone training on all datasets/backbones"
        echo "  dual2fair  - Run Dual2Fair on all datasets/backbones"
        echo "  baseline   - Run all baselines on all datasets/backbones"
        echo "  grid       - Grid search over lambda1/lambda2"
        echo "  all        - Run complete experiment matrix"
        echo "  quick      - Quick test: Dual2Fair on ML/LightGCN only"
        echo ""
        echo "Variables (edit in script):"
        echo "  DATASETS=$DATASETS"
        echo "  BACKBONES=$BACKBONES"
        echo "  LAMBDA1=$LAMBDA1  LAMBDA2=$LAMBDA2"
        echo "  GPU=$GPU  SEED=$SEED  EVAL_MODE=$EVAL_MODE  LOSS_TYPE=$LOSS_TYPE"
        ;;
esac
