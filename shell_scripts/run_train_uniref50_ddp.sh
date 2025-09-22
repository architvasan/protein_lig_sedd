#!/bin/bash

# UniRef50 Distributed Data Parallel (DDP) Training Launcher
# Supports multi-GPU training with automatic hyperparameter scaling

set -e

# Default configuration
WORK_DIR="./experiments_ddp"
CONFIG_FILE="configs/config_uniref50_ddp.yaml"
DATAFILE="./input_data/processed_uniref50.pt"
WANDB_PROJECT="uniref50-sedd-ddp"
WANDB_NAME=""
SEED=42
SAMPLING_METHOD="rigorous"
FRESH_START=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
🚀 UniRef50 Distributed Data Parallel (DDP) Training

Usage: $0 [OPTIONS]

OPTIONS:
    -w, --work-dir PATH     Working directory (default: $WORK_DIR)
    -c, --config PATH       Configuration file (default: $CONFIG_FILE)
    -d, --datafile PATH     Data file path (default: $DATAFILE)
    -p, --project NAME      Wandb project name (default: $WANDB_PROJECT)
    -n, --name NAME         Wandb run name (auto-generated if not provided)
    -s, --seed N            Random seed (default: $SEED)
    -m, --method METHOD     Sampling method: rigorous|simple (default: $SAMPLING_METHOD)
    --fresh                 Force fresh start (ignore checkpoints)
    -h, --help              Show this help message

EXAMPLES:
    # Basic DDP training with default settings
    $0

    # Custom configuration and data
    $0 --config my_config.yaml --datafile my_data.pt

    # Fresh start with custom project name
    $0 --fresh --project my_experiment

HYPERPARAMETER SCALING FOR DDP:
    - Learning rate: Scaled linearly by number of GPUs
    - Batch size: Effective batch size = batch_size × accum × num_gpus
    - Warmup steps: Scaled by number of GPUs to maintain same effective warmup

REQUIREMENTS:
    - Multiple CUDA GPUs
    - NCCL backend support
    - Processed UniRef50 data file
    - Wandb account (wandb login)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--datafile)
            DATAFILE="$2"
            shift 2
            ;;
        -p|--project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -n|--name)
            WANDB_NAME="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -m|--method)
            SAMPLING_METHOD="$2"
            shift 2
            ;;
        --fresh)
            FRESH_START=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Generate run name if not provided
if [[ -z "$WANDB_NAME" ]]; then
    WANDB_NAME="ddp_$(date +'%Y%m%d_%H%M%S')"
fi

# Print banner
echo "🚀 UNIREF50 DDP TRAINING"
echo "========================"
echo ""

# Validate environment
print_status "Validating environment..."

# Check CUDA
if ! nvidia-smi > /dev/null 2>&1; then
    print_error "CUDA not available. DDP training requires CUDA GPUs."
    exit 1
fi

# Count GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ $NUM_GPUS -lt 2 ]]; then
    print_warning "Only $NUM_GPUS GPU(s) detected. DDP is most beneficial with 2+ GPUs."
    print_warning "Consider using run_train_uniref50_optimized.sh for single-GPU training."
fi

# Check files
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$DATAFILE" ]]; then
    print_error "Data file not found: $DATAFILE"
    print_warning "Run ./download_uniref50_data.sh to prepare data"
    exit 1
fi

# Check wandb
if ! wandb status > /dev/null 2>&1; then
    print_error "Wandb not logged in. Run: wandb login"
    exit 1
fi

# Validate sampling method
if [[ "$SAMPLING_METHOD" != "rigorous" && "$SAMPLING_METHOD" != "simple" ]]; then
    print_error "Invalid sampling method: $SAMPLING_METHOD (must be 'rigorous' or 'simple')"
    exit 1
fi

print_success "Environment validation passed!"
echo ""

# Show configuration
print_status "DDP Training Configuration:"
echo "  🖥️  GPUs detected: $NUM_GPUS"
echo "  📁 Work directory: $WORK_DIR"
echo "  ⚙️  Config file: $CONFIG_FILE"
echo "  💾 Data file: $DATAFILE"
echo "  🏷️  Wandb project: $WANDB_PROJECT"
echo "  🏷️  Wandb run: $WANDB_NAME"
echo "  🎲 Seed: $SEED"
echo "  🧬 Sampling method: $SAMPLING_METHOD"
echo "  🔄 Fresh start: $FRESH_START"
echo ""

# Create directories
mkdir -p "$WORK_DIR"
mkdir -p logs

# Show hyperparameter scaling info
print_status "Hyperparameter Scaling for $NUM_GPUS GPUs:"
echo "  📈 Learning rate will be scaled by ${NUM_GPUS}x"
echo "  📦 Effective batch size = batch_size × accum × $NUM_GPUS"
echo "  🔥 Warmup steps will be scaled by ${NUM_GPUS}x"
echo ""

# Build command
CMD_ARGS=(
    "--work_dir" "$WORK_DIR"
    "--config" "$CONFIG_FILE"
    "--datafile" "$DATAFILE"
    "--wandb_project" "$WANDB_PROJECT"
    "--wandb_name" "$WANDB_NAME"
    "--seed" "$SEED"
    "--sampling_method" "$SAMPLING_METHOD"
)

if [[ "$FRESH_START" == "true" ]]; then
    CMD_ARGS+=("--fresh")
fi

# Show command
print_status "Executing DDP training:"
echo "  python -m protlig_dd.training.run_train_uniref50_ddp ${CMD_ARGS[*]}"
echo ""

# Confirm execution
echo "⚠️  This will start distributed training across $NUM_GPUS GPUs."
echo "📊 Results will be logged to Wandb project: $WANDB_PROJECT"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Cancelled by user."
    exit 0
fi

# Set environment variables for better DDP performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

# Create log file
LOG_FILE="logs/uniref50_ddp_$(date +'%Y%m%d_%H%M%S').log"

print_status "Starting DDP training..."
print_status "Logs will be saved to: $LOG_FILE"
echo ""

# Execute training
if python -m protlig_dd.training.run_train_uniref50_ddp "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
    print_success "DDP training completed successfully!"
    echo ""
    echo "📊 Next steps:"
    echo "  1. Check Wandb dashboard: https://wandb.ai"
    echo "  2. Review logs: $LOG_FILE"
    echo "  3. Find checkpoints in: $WORK_DIR"
else
    print_error "DDP training failed!"
    echo ""
    echo "💡 Troubleshooting:"
    echo "  - Check GPU availability: nvidia-smi"
    echo "  - Verify NCCL installation: python -c 'import torch; print(torch.distributed.is_nccl_available())'"
    echo "  - Check logs: $LOG_FILE"
    echo "  - Ensure sufficient GPU memory for larger effective batch size"
    echo "  - Try reducing batch size in config if OOM errors occur"
    exit 1
fi
