#!/bin/bash

# Example script to run training without wandb logging
# This will use file logging instead

python protlig_dd/training/run_train_uniref_ddp_polaris.py \
    --work_dir ./experiments/test_no_wandb \
    --config ./configs/uniref50_optimized.yaml \
    --datafile ./input_data/processed_uniref50.pt \
    --device cuda:0 \
    --seed 42 \
    --sampling_method rigorous \
    --epochs 5 \
    --no_wandb

echo "Training completed! Check logs in ./experiments/test_no_wandb/logs/"
