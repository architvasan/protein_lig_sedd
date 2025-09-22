#!/bin/bash

# UniRef50 Hyperparameter Sweep Launcher
# This script makes it easy to run hyperparameter sweeps

set -e

# Default values
BASE_CONFIG="configs/config_uniref50_sweeps.yaml"
WORK_DIR="./hyperparam_experiments"
DATAFILE="./input_data/subset_uniref50.pt"
SWEEP_TYPE="predefined"
NUM_RANDOM=10
GPUS="0,1,2,3"
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to show usage
show_usage() {
    cat << EOF
🧬 UniRef50 Hyperparameter Sweep Launcher

Usage: $0 [OPTIONS]

OPTIONS:
    -c, --config PATH       Base configuration file (default: $BASE_CONFIG)
    -w, --work-dir PATH     Working directory (default: $WORK_DIR)
    -d, --datafile PATH     Data file path (default: $DATAFILE)
    -t, --type TYPE         Sweep type: predefined|random (default: $SWEEP_TYPE)
    -n, --num-random N      Number of random configs (default: $NUM_RANDOM)
    -g, --gpus LIST         Comma-separated GPU IDs (default: $GPUS)
    --dry-run               Print commands without executing
    -h, --help              Show this help message

EXAMPLES:
    # Run predefined sweep with default settings
    $0

    # Run random sweep with 20 configurations
    $0 --type random --num-random 20

    # Dry run to see what would be executed
    $0 --dry-run

    # Use custom config and data files
    $0 --config my_config.yaml --datafile my_data.pt

    # Use specific GPUs
    $0 --gpus 0,1,2,3

    # Use only 2 GPUs
    $0 --gpus 0,1

PREDEFINED CONFIGURATIONS:
    - small_fast: Quick iteration config (512 hidden, simple sampling)
    - medium_rigorous: Balanced config (768 hidden, rigorous sampling)
    - large_quality: High-quality config (1024 hidden, rigorous sampling)
    - high_lr_experiment: High learning rate experiment
    - curriculum_focus: Curriculum learning focused config

REQUIREMENTS:
    - Python environment with required packages
    - Wandb account and login (wandb login)
    - CUDA-capable GPU
    - Processed UniRef50 data file

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        -w|--work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -d|--datafile)
            DATAFILE="$2"
            shift 2
            ;;
        -t|--type)
            SWEEP_TYPE="$2"
            shift 2
            ;;
        -n|--num-random)
            NUM_RANDOM="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

# Print banner
echo "🧬 UNIREF50 HYPERPARAMETER SWEEP"
echo "=================================="
echo ""

# Validate inputs
print_status "Validating inputs..."

if [[ ! -f "$BASE_CONFIG" ]]; then
    print_error "Base config file not found: $BASE_CONFIG"
    exit 1
fi

if [[ ! -f "$DATAFILE" ]]; then
    print_warning "Sweep dataset not found: $DATAFILE"

    # Check if original dataset exists
    ORIGINAL_DATAFILE="./input_data/processed_uniref50.pt"
    if [[ -f "$ORIGINAL_DATAFILE" ]]; then
        print_status "Creating 10k sample sweep dataset from original..."
        if python create_sweep_dataset.py --input "$ORIGINAL_DATAFILE" --output "$DATAFILE"; then
            print_success "Sweep dataset created successfully!"
        else
            print_error "Failed to create sweep dataset"
            exit 1
        fi
    else
        print_error "Original data file not found: $ORIGINAL_DATAFILE"
        print_warning "You may need to run: ./download_uniref50_data.sh"
        exit 1
    fi
fi

if [[ "$SWEEP_TYPE" != "predefined" && "$SWEEP_TYPE" != "random" ]]; then
    print_error "Invalid sweep type: $SWEEP_TYPE (must be 'predefined' or 'random')"
    exit 1
fi

# Check if wandb is logged in
if ! wandb status > /dev/null 2>&1; then
    print_warning "Wandb not logged in. Please run: wandb login"
    exit 1
fi

# Check GPU availability and validate specified GPUs
if ! nvidia-smi > /dev/null 2>&1; then
    print_warning "nvidia-smi not found. Make sure CUDA is available."
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_success "Found $GPU_COUNT GPU(s)"

    # Validate specified GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        if [[ $gpu_id -ge $GPU_COUNT ]]; then
            print_error "GPU $gpu_id not available. Only GPUs 0-$((GPU_COUNT-1)) are available."
            exit 1
        fi
    done

    print_success "Using GPUs: $GPUS (${#GPU_ARRAY[@]} concurrent jobs)"
fi

print_success "All validations passed!"
echo ""

# Show configuration
print_status "Sweep Configuration:"
echo "  📁 Base config: $BASE_CONFIG"
echo "  📁 Work directory: $WORK_DIR"
echo "  💾 Data file: $DATAFILE"
echo "  🎯 Sweep type: $SWEEP_TYPE"
if [[ "$SWEEP_TYPE" == "random" ]]; then
    echo "  🎲 Random configs: $NUM_RANDOM"
fi
echo "  🖥️  Available GPUs: $GPUS"
echo "  ⚡ Max concurrent jobs: $(echo $GPUS | tr ',' '\n' | wc -l)"
echo "  🔍 Dry run: $DRY_RUN"
echo ""

# Create work directory
mkdir -p "$WORK_DIR"

# Build command
CMD="python hyperparameter_sweep.py"
CMD="$CMD --base_config '$BASE_CONFIG'"
CMD="$CMD --work_dir '$WORK_DIR'"
CMD="$CMD --datafile '$DATAFILE'"
CMD="$CMD --sweep_type '$SWEEP_TYPE'"
CMD="$CMD --gpus '$GPUS'"

if [[ "$SWEEP_TYPE" == "random" ]]; then
    CMD="$CMD --num_random $NUM_RANDOM"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    CMD="$CMD --dry_run"
fi

# Show command
print_status "Executing command:"
echo "  $CMD"
echo ""

# Confirm execution (unless dry run)
if [[ "$DRY_RUN" != "true" ]]; then
    echo "⚠️  This will start training jobs that may run for hours."
    echo "📊 Results will be logged to Wandb project: uniref50_hyperparam_sweep"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cancelled by user."
        exit 0
    fi
fi

# Execute command
print_status "Starting hyperparameter sweep..."
echo ""

if eval $CMD; then
    print_success "Hyperparameter sweep completed!"
    echo ""
    echo "📊 Next steps:"
    echo "  1. Check Wandb dashboard: https://wandb.ai"
    echo "  2. Analyze results: python analyze_sweep_results.py"
    echo "  3. Review logs in: $WORK_DIR/sweep_*/"
else
    print_error "Hyperparameter sweep failed!"
    echo ""
    echo "💡 Troubleshooting:"
    echo "  - Check GPU availability: nvidia-smi"
    echo "  - Check Wandb login: wandb status"
    echo "  - Check data file: ls -la $DATAFILE"
    echo "  - Check logs in: $WORK_DIR/sweep_*/"
    exit 1
fi
