#!/bin/bash

# 🆕 START FRESH TRAINING SCRIPT
# Multiple options to start training from scratch

echo "🆕 FRESH TRAINING OPTIONS"
echo "========================="
echo ""

# Check if checkpoints exist
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
    echo "📂 Existing checkpoints found:"
    ls -la checkpoints/
    echo ""
    
    echo "Choose an option:"
    echo "1) Remove checkpoints and start fresh"
    echo "2) Backup checkpoints and start fresh"
    echo "3) Use --fresh flag to ignore checkpoints"
    echo "4) Cancel"
    echo ""
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            echo "🗑️  Removing existing checkpoints..."
            rm -rf checkpoints/
            echo "✅ Checkpoints removed"
            ;;
        2)
            backup_dir="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
            echo "📦 Backing up checkpoints to $backup_dir..."
            mv checkpoints/ "$backup_dir"
            echo "✅ Checkpoints backed up to $backup_dir"
            ;;
        3)
            echo "🚀 Using --fresh flag to ignore existing checkpoints"
            FRESH_FLAG="--fresh"
            ;;
        4)
            echo "❌ Cancelled"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
else
    echo "✅ No existing checkpoints found - ready for fresh training"
fi

echo ""
echo "🚀 Starting fresh training..."
echo "================================"

# Set default parameters
WORK_DIR="."
CONFIG_FILE="configs/config_uniref50_stable.yaml"
DATAFILE="./input_data/subset_uniref50.pt"
WANDB_PROJECT="uniref50_sedd_optimized"
DEVICE="auto"  # Auto-detect best device
SEED=42

# Generate unique run name
RUN_NAME="fresh_start_$(date +%Y%m%d_%H%M%S)"

echo "📋 Training Configuration:"
echo "   Work Dir: $WORK_DIR"
echo "   Config: $CONFIG_FILE"
echo "   Data: $DATAFILE"
echo "   Wandb Project: $WANDB_PROJECT"
echo "   Run Name: $RUN_NAME"
echo "   Device: $DEVICE"
echo "   Seed: $SEED"
echo "   Fresh Start: ${FRESH_FLAG:-No (checkpoints removed)}"
echo ""

# Confirm before starting
read -p "🤔 Start training with these settings? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "🎬 STARTING TRAINING..."
echo "======================"

# Set PYTHONPATH to include current directory
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run the training script as a module (same as main script)
python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_FILE" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$RUN_NAME" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $FRESH_FLAG >logs/train_uniref50_opt.log 2> logs/train_uniref50_opt.err

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TRAINING COMPLETED SUCCESSFULLY!"
    echo "=================================="
    echo "📊 Check your Wandb dashboard for results"
    echo "📁 New checkpoints saved in: checkpoints/"
else
    echo ""
    echo "❌ TRAINING FAILED"
    echo "=================="
    echo "💡 Check the error messages above"
fi

echo ""
echo "=== Script finished ==="
