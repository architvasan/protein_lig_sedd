#!/bin/bash

###########################
### CREATE TEST DATASET ###
##########################

set -e
set -u

echo "=== Creating Test Protein Dataset ==="

# Configuration
WORK_DIR="/Users/ramanathana/Work/Protein-Ligand-SEDD/protein_lig_sedd"
OUTPUT_DIR="$WORK_DIR/input_data"
NUM_SEQUENCES=5000
FILENAME="uniref50_subset.pt"

echo "Work directory: $WORK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of sequences: $NUM_SEQUENCES"

# Change to work directory
cd "$WORK_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p scripts

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
elif [ -f "pl_dd_venv/bin/activate" ]; then
    source pl_dd_venv/bin/activate
    echo "Activated virtual environment: pl_dd_venv"
else
    echo "No virtual environment found, using system Python"
fi

# Check Python and required packages
echo "Python version: $(python --version)"

# Install required packages if not available
echo "Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "Installing PyTorch..."
    pip install torch
}

python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo "Installing transformers..."
    pip install transformers
}

python -c "import tqdm; print('tqdm available')" || {
    echo "Installing tqdm..."
    pip install tqdm
}

python -c "import numpy; print(f'NumPy: {numpy.__version__}')" || {
    echo "Installing numpy..."
    pip install numpy
}

# Run the dataset creation script
echo "=== Running dataset creation ==="
python scripts/create_test_protein_dataset.py \
    --output_dir "$OUTPUT_DIR" \
    --num_sequences "$NUM_SEQUENCES" \
    --filename "$FILENAME"

# Check if file was created
if [ -f "$OUTPUT_DIR/$FILENAME" ]; then
    echo "✅ Dataset created successfully!"
    echo "📁 File: $OUTPUT_DIR/$FILENAME"
    echo "📊 File size: $(du -h "$OUTPUT_DIR/$FILENAME" | cut -f1)"
    
    # Show metadata if available
    METADATA_FILE="$OUTPUT_DIR/${FILENAME%.pt}_metadata.json"
    if [ -f "$METADATA_FILE" ]; then
        echo "📋 Metadata:"
        cat "$METADATA_FILE"
    fi
else
    echo "❌ Dataset creation failed!"
    exit 1
fi

echo "=== Dataset creation completed ==="
