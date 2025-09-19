#!/bin/bash

set -e

#CONFIGS to load in (manual parameters based on a previous sweep)

CONFIG_0="configs/config_uniref50_hpo0.yaml"
CONFIG_1="configs/config_uniref50_hpo1.yaml"
CONFIG_2="configs/config_uniref50_hpo2.yaml"
CONFIG_3="configs/config_uniref50_hpo3.yaml"

# Set default parameters
WORK_DIR="."
DATAFILE="./input_data/subset_uniref50.pt"
WANDB_PROJECT="uniref50_sedd_optimized"
DEVICE="auto"  # Auto-detect best device
SEED=42

# Generate unique run name
RUN_NAME="manual_opt$(date +%Y%m%d_%H%M%S)"

# Set PYTHONPATH to include current directory
export PYTHONPATH="$PWD:$PYTHONPATH"
FRESH_FLAG="--fresh"

# Run 4 jobs on each Polaris device

CUDA_VISIBLE_DEVICES=0 python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_0" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${RUN_NAME}_hpo0" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $FRESH_FLAG > logs/train_uniref50_hpo0_test.log 2> logs/train_uniref50_hpo0_test.err &

CUDA_VISIBLE_DEVICES=1 python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_1" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${RUN_NAME}_hpo1" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $FRESH_FLAG > logs/train_uniref50_hpo1_test.log 2> logs/train_uniref50_hpo1_test.err &

CUDA_VISIBLE_DEVICES=2 python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_2" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${RUN_NAME}_hpo2" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $FRESH_FLAG >logs/train_uniref50_hpo2_test.log 2> logs/train_uniref50_hpo2_test.err &

CUDA_VISIBLE_DEVICES=3 python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_3" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${RUN_NAME}_hpo3" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $FRESH_FLAG >logs/train_uniref50_hpo3_test.log 2> logs/train_uniref50_hpo3_test.err
