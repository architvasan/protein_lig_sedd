#!/bin/bash

set -e

#CONFIGS to load in (manual parameters based on a previous sweep)

CONFIG_0="configs/config_uniref50_hpo4.yaml"

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

CUDA_VISIBLE_DEVICES=3 python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_0" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${RUN_NAME}_hpo0" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $FRESH_FLAG > logs/train_uniref50_hpo4_large.log 2> logs/train_uniref50_hpo4_large.err
