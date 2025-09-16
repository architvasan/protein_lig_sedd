#!/bin/bash

###########################
### OPTIMIZED UNIREF50 TRAINING SCRIPT ###
##########################

# Set strict error handling
set -e
set -u
set -o pipefail

###########################
### CONFIGURATION VARIABLES ###
##########################
WORK_DIR="/Users/ramanathana/Work/Protein-Ligand-SEDD/protein_lig_sedd"
CONFIG_FILE="$WORK_DIR/configs/config_uniref50_stable.yaml"
DATAFILE="$WORK_DIR/input_data/processed_uniref50.pt"

# Wandb configuration
WANDBPROJ="uniref50_sedd_optimized"
WANDBNAME="uniref50_optimized_$(date +'%Y%m%d_%H%M%S')"

# Hardware configuration with cross-platform support
DEVICE="auto"  # Auto-detect best device (cuda:0, mps, or cpu)
SEED=42

# Parse command line arguments for device override
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --mps)
            DEVICE="mps"
            shift
            ;;
        --cuda)
            DEVICE="cuda:0"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device DEVICE] [--cpu] [--mps] [--cuda]"
            echo "  --device DEVICE  Specify device (auto, cpu, mps, cuda:0, etc.)"
            echo "  --cpu           Force CPU training"
            echo "  --mps           Force Apple Silicon MPS training"
            echo "  --cuda          Force CUDA GPU training"
            exit 1
            ;;
    esac
done

# Memory optimization (CUDA-specific)
if [[ "$DEVICE" == "cuda"* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_LAUNCH_BLOCKING=0
    echo "🚀 CUDA memory optimizations enabled"
elif [[ "$DEVICE" == "mps" ]]; then
    echo "🍎 Apple Silicon MPS training enabled"
else
    echo "💻 CPU training enabled"
fi

#########################
### ENVIRONMENT SETUP ###
########################

echo "=== Setting up environment ==="
cd "$WORK_DIR"

# Activate virtual environment (adjust path as needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
elif [ -f "pl_dd_venv/bin/activate" ]; then
    source pl_dd_venv/bin/activate
    echo "Activated virtual environment: pl_dd_venv"
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Install/check required packages
echo "=== Checking required packages ==="
python -c "import wandb; print(f'Wandb: {wandb.__version__}')" || {
    echo "Installing wandb..."
    pip install wandb
}

python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')" || {
    echo "Installing matplotlib..."
    pip install matplotlib seaborn
}

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DATAFILE" ]; then
    echo "Warning: Data file not found: $DATAFILE"
    echo "You may need to process UniRef50 data first."
    echo "Run: ./download_uniref50_data.sh"
fi

#########################
### SYSTEM CHECKS ###
########################

echo "=== System checks ==="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "CUDA device name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
fi

#########################
### MEMORY OPTIMIZATION ###
########################

echo "=== Applying memory optimizations ==="

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Set memory fraction (adjust based on your GPU)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

#########################
### WANDB SETUP ###
########################

echo "=== Setting up Wandb ==="

# Check if wandb is logged in
if ! wandb status >/dev/null 2>&1; then
    echo "⚠️  Wandb not logged in. Please log in to track your experiments."
    echo "Run: wandb login"
    echo "Or set WANDB_API_KEY environment variable"
    echo "You can also run in offline mode by setting WANDB_MODE=offline"

    # Ask user if they want to continue without wandb
    read -p "Continue without wandb login? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please set up wandb and try again."
        exit 1
    else
        echo "Running with wandb in offline mode..."
        export WANDB_MODE=offline
    fi
else
    echo "✅ Wandb is logged in and ready"
fi

#########################
### TRAINING EXECUTION ###
########################

echo "=== Starting training ==="
echo "Work directory: $WORK_DIR"
echo "Config file: $CONFIG_FILE"
echo "Data file: $DATAFILE"
echo "Device: $DEVICE"
echo "Wandb project: $WANDBPROJ"
echo "Wandb name: $WANDBNAME"

# Create logs directory
mkdir -p logs

# Run training with optimizations
echo "🚀 Starting optimized training..."
echo "📊 Wandb tracking will be available at: https://wandb.ai"
echo "🔗 Look for the web interface link in the output below!"
echo ""

python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_FILE" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDBPROJ" \
    --wandb_name "$WANDBNAME" \
    --device "$DEVICE" \
    --seed "$SEED" \
    2>&1 | tee "logs/uniref50_optimized_$(date +'%Y%m%d_%H%M%S').log"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📊 Check your Wandb dashboard for results: https://wandb.ai"
else
    echo ""
    echo "❌ Training failed. Check the log file for details."
    echo "💡 Common issues:"
    echo "   - Missing data file (run ./download_uniref50_data.sh)"
    echo "   - Wandb not logged in (run wandb login)"
    echo "   - GPU memory issues (reduce batch size in config)"
fi

echo "=== Training completed ==="

#########################
### POST-TRAINING CLEANUP ###
########################

echo "=== Cleaning up ==="
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
echo "GPU memory cleared"

echo "=== Script finished ==="
