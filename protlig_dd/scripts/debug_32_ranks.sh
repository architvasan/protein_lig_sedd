#!/bin/bash

# Debug script for 32-rank DDP training
# This uses minimal mode to disable evaluations and complex logging

echo "🚀 Starting 32-rank DDP debugging with minimal mode..."

python protlig_dd/training/run_train_uniref_ddp_polaris.py \
    --work_dir ./experiments/debug_32_ranks \
    --config ./configs/uniref50_optimized.yaml \
    --datafile ./input_data/processed_uniref50.pt \
    --device cuda:0 \
    --seed 42 \
    --sampling_method rigorous \
    --epochs 1 \
    --no_wandb \
    --minimal_mode

echo "✅ 32-rank debugging completed! Check logs in ./experiments/debug_32_ranks/logs/"
echo "📊 Each rank should have its own log file: training_rank{0-31}_*.log"
