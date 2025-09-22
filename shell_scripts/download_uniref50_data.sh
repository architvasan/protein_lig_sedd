#!/bin/bash

###########################
### DOWNLOAD UNIREF50 DATA ###
##########################

set -e
set -u

echo "=== UniRef50 Data Download Script ==="

# Configuration
WORK_DIR="/Users/ramanathana/Work/Protein-Ligand-SEDD/protein_lig_sedd"
OUTPUT_DIR="$WORK_DIR/input_data"
NUM_SEQUENCES=10000

echo "Work directory: $WORK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Target sequences: $NUM_SEQUENCES"

# Change to work directory
cd "$WORK_DIR"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p scripts
mkdir -p logs

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment: venv"
elif [ -f "pl_dd_venv/bin/activate" ]; then
    source pl_dd_venv/bin/activate
    echo "✅ Activated virtual environment: pl_dd_venv"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Check Python version
echo "Python version: $(python --version)"

# Function to install package if not available
install_if_missing() {
    local package=$1
    local import_name=${2:-$package}
    
    if python -c "import $import_name" 2>/dev/null; then
        echo "✅ $package is available"
    else
        echo "📦 Installing $package..."
        pip install "$package"
    fi
}

# Install required packages
echo "=== Checking and installing required packages ==="
install_if_missing "torch"
install_if_missing "transformers"
install_if_missing "datasets"
install_if_missing "tqdm"
install_if_missing "numpy"

# Try to download real UniRef50 data first
echo "=== Attempting to download real UniRef50 data ==="
if python scripts/download_real_uniref50.py \
    --output_dir "$OUTPUT_DIR" \
    --num_sequences "$NUM_SEQUENCES" \
    --filename "processed_uniref50.pt" \
    2>&1 | tee "logs/uniref50_download.log"; then
    
    echo "✅ Successfully downloaded real UniRef50 data!"
    DATASET_FILE="$OUTPUT_DIR/processed_uniref50.pt"
    
else
    echo "⚠️  Real UniRef50 download failed, creating test dataset..."
    
    # Fallback to test dataset
    if python scripts/create_test_protein_dataset.py \
        --output_dir "$OUTPUT_DIR" \
        --num_sequences "$NUM_SEQUENCES" \
        --filename "uniref50_subset.pt" \
        2>&1 | tee "logs/test_dataset_creation.log"; then
        
        echo "✅ Successfully created test dataset!"
        DATASET_FILE="$OUTPUT_DIR/uniref50_subset.pt"
        
    else
        echo "❌ Both download methods failed!"
        exit 1
    fi
fi

# Verify the dataset file
if [ -f "$DATASET_FILE" ]; then
    echo "✅ Dataset file created: $DATASET_FILE"
    echo "📊 File size: $(du -h "$DATASET_FILE" | cut -f1)"
    
    # Show metadata if available
    METADATA_FILE="${DATASET_FILE%.pt}_metadata.json"
    if [ -f "$METADATA_FILE" ]; then
        echo "📋 Dataset metadata:"
        cat "$METADATA_FILE" | python -m json.tool
    fi
    
    # Test loading the dataset
    echo "🧪 Testing dataset loading..."
    python -c "
import torch
print('Loading dataset...')
data = torch.load('$DATASET_FILE')
print(f'✅ Successfully loaded {len(data)} sequences')
print(f'📊 Sample sequence length: {data[0][\"length\"]}')
print(f'🔤 Sample sequence: {data[0][\"protein_seq\"][:50]}...')
print(f'🎯 Token shape: {data[0][\"prot_tokens\"].shape}')
"
    
else
    echo "❌ Dataset file not found: $DATASET_FILE"
    exit 1
fi

# Create a simple config for testing
echo "=== Creating test configuration ==="
TEST_CONFIG="$WORK_DIR/configs/config_uniref50_test.yaml"

cat > "$TEST_CONFIG" << EOF
work_dir: $WORK_DIR

model:
  name: small
  type: ddit
  hidden_size: 512
  cond_dim: 128
  length: 512
  n_blocks: 6
  n_heads: 8
  scale_by_sigma: True
  dropout: 0.1
  device: cuda:0

defaults:
  - _self_
  - model: small

ngpus: 1
tokens: 25  # 20 amino acids + 5 special tokens

training:
  batch_size: 16
  accum: 2
  epochs: 5
  max_samples: $NUM_SEQUENCES
  num_workers: 2
  seed: 42
  force_reprocess: False
  n_iters: 5000
  snapshot_freq: 500
  log_freq: 50
  eval_freq: 500
  snapshot_freq_for_preemption: 500
  weight: standard
  snapshot_sampling: True
  ema: 0.999
  task: protein_only

data:
  train: uniref50
  valid: uniref50
  cache_dir: data
  train_ratio: 0.9
  val_ratio: 0.1
  max_protein_len: 512
  max_ligand_len: 128
  use_structure: False
  vocab_size_protein: 25
  vocab_size_ligand: 2364

graph:
  type: absorb
  file: data
  report_all: False

noise:
  type: cosine
  sigma_min: !!float "1e-4"
  sigma_max: 0.5
  eps: !!float "0.05"

sampling:
  predictor: euler
  steps: 50
  noise_removal: True

eval:
  batch_size: 8
  perplexity: True
  perplexity_batch_size: 4

optim:
  weight_decay: 0.01
  optimizer: AdamW
  lr: !!float "5e-5"
  beta1: 0.9
  beta2: 0.95
  eps: !!float "1e-8"
  warmup: 1000
  grad_clip: 1.0

hydra:
  run:
    dir: exp_local/uniref50_test/\${now:%Y.%m.%d}/\${now:%H%M%S}
EOF

echo "✅ Created test configuration: $TEST_CONFIG"

# Summary
echo ""
echo "="*60
echo "🎉 DATASET PREPARATION COMPLETE!"
echo "="*60
echo "📁 Dataset file: $DATASET_FILE"
echo "⚙️  Test config: $TEST_CONFIG"
echo ""
echo "🚀 To start training, run:"
echo "   python -m protlig_dd.training.run_train_uniref50_optimized \\"
echo "     --work_dir '$WORK_DIR' \\"
echo "     --config '$TEST_CONFIG' \\"
echo "     --datafile '$DATASET_FILE' \\"
echo "     --wandb_project 'uniref50_test' \\"
echo "     --wandb_name 'test_run_$(date +%Y%m%d_%H%M%S)'"
echo ""
echo "📊 Or use the optimized training script:"
echo "   ./run_train_uniref50_optimized.sh"
echo "="*60
