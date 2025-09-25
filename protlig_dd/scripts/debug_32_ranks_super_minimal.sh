#!/bin/bash

# Super minimal debug script for 32-rank DDP training
# This runs only 10 steps with maximum debugging

echo "🚀 Starting SUPER MINIMAL 32-rank DDP debugging..."
echo "📊 This will run only 10 training steps with maximum debugging output"

# Override config to run minimal steps
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

echo "📝 Created minimal config with:"
echo "   - 10 training steps only"
echo "   - Batch size 4"
echo "   - Max length 64"
echo "   - No evaluations"

# This script just creates the config - you need to run with MPI
echo "🚨 IMPORTANT: This script only creates the config file."
echo "🚨 You need to run with your MPI command like this:"
echo ""
echo "mpirun -np \$NUM_GPUS -ppn \$PPN \\"
echo "        --cpu-bind depth -d 16 \\"
echo "        ./shell_scripts/set_affinity_gpu_polaris.sh \\"
echo "        \$VENV_PATH/bin/python protlig_dd/training/run_train_uniref_ddp_polaris.py \\"
echo "        --work_dir ./experiments/debug_32_ranks_super_minimal \\"
echo "        --config /tmp/minimal_debug_config.yaml \\"
echo "        --datafile \$DATAFILE \\"
echo "        --device \$DEVICE \\"
echo "        --seed 42 \\"
echo "        --sampling_method rigorous \\"
echo "        --epochs 1 \\"
echo "        --no_wandb \\"
echo "        --minimal_mode \\"
echo "        --tokenize_on_fly 2>&1 | tee debug_32_ranks.log"
echo ""
echo "📝 Minimal config created at: /tmp/minimal_debug_config.yaml"

echo "✅ Super minimal debugging completed!"
echo "📊 Check output above to see exactly where it hangs"
