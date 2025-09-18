#!/bin/bash

# Super lazy wandb sweep launcher
# Just run: ./lazy_sweep.sh

echo "🚀 SUPER LAZY WANDB SWEEP"
echo "========================="

# Default values - modify these if needed
WORK_DIR="./sweep_work"
CONFIG="./configs/uniref50_optimized.yaml"  # Adjust path as needed
DATAFILE="./input_data/processed_uniref50.pt"
PROJECT="uniref50-lazy-sweep"
DEVICE="cuda:0"
COUNT=20

echo "📁 Work dir: $WORK_DIR"
echo "⚙️  Config: $CONFIG"
echo "💾 Data: $DATAFILE"
echo "🏷️  Project: $PROJECT"
echo "🖥️  Device: $DEVICE"
echo "🔢 Runs: $COUNT"
echo "========================="

# Create work directory
mkdir -p $WORK_DIR

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    echo "💡 Please update the CONFIG variable in this script"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATAFILE" ]; then
    echo "❌ Data file not found: $DATAFILE"
    echo "💡 Please update the DATAFILE variable in this script"
    exit 1
fi

# Launch the sweep
echo "🚀 Launching sweep..."
cd "$(dirname "$0")"
python launch_sweep.py \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG" \
    --datafile "$DATAFILE" \
    --project "$PROJECT" \
    --device "$DEVICE" \
    --count $COUNT

echo "🎉 Done! Check your wandb dashboard for results."
