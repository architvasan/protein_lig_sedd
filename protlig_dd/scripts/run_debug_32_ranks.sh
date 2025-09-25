#!/bin/bash

# Complete MPI script for 32-rank DDP debugging
# Replace these variables with your actual values

# Set your environment variables
export NUM_GPUS=32
export PPN=4  # Processes per node (adjust based on your setup)
export VENV_PATH="/path/to/your/venv"  # Replace with your actual venv path
export DATAFILE="/path/to/your/processed_uniref50.pt"  # Replace with actual path
export DEVICE="cuda:0"

# Create minimal debug config
cat > /tmp/minimal_debug_config.yaml << EOF
model:
  d_model: 256
  n_layers: 4
  n_heads: 8
  
training:
  batch_size: 4  # Very small batch
  n_iters: 10    # Only 10 steps
  log_freq: 1    # Log every step
  eval_freq: 999999  # Never evaluate
  accum: 1
  
optim:
  lr: 1e-4
  grad_clip: 1.0
  
data:
  max_protein_len: 64  # Very short sequences
EOF

echo "🚀 Starting SUPER MINIMAL 32-rank DDP debugging..."
echo "📊 Running only 10 training steps with maximum debugging"
echo "📝 Config: batch_size=4, max_len=64, 10 steps only"
echo ""

# Run with MPI
mpirun -np $NUM_GPUS -ppn $PPN \
        --cpu-bind depth -d 16 \
        ./shell_scripts/set_affinity_gpu_polaris.sh \
        $VENV_PATH/bin/python protlig_dd/training/run_train_uniref_ddp_polaris.py \
        --work_dir ./experiments/debug_32_ranks_super_minimal \
        --config /tmp/minimal_debug_config.yaml \
        --datafile "$DATAFILE" \
        --device "$DEVICE" \
        --seed 42 \
        --sampling_method rigorous \
        --epochs 1 \
        --no_wandb \
        --minimal_mode \
        --tokenize_on_fly 2>&1 | tee debug_32_ranks_output.log

echo ""
echo "✅ Debug run completed!"
echo "📊 Check debug_32_ranks_output.log for detailed output"
echo "🔍 Look for patterns in the rank outputs to see where it hangs"
