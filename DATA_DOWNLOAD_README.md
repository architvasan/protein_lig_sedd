# UniRef50 Data Download Guide

This guide helps you download and prepare UniRef50 data for testing the SEDD protein model.

## Quick Start

### Option 1: Automated Download (Recommended)
```bash
# Run the comprehensive download script
./download_uniref50_data.sh
```

This script will:
1. Try to download real UniRef50 data from Hugging Face
2. Fall back to creating a test dataset if download fails
3. Create a test configuration file
4. Verify the dataset works

### Option 2: Manual Steps

#### Step 1: Create Test Dataset
```bash
# Create a synthetic test dataset (5,000 sequences)
./create_test_dataset.sh
```

#### Step 2: Download Real UniRef50 (Optional)
```bash
# Download real UniRef50 data (10,000 sequences)
python scripts/download_real_uniref50.py \
    --output_dir ./input_data \
    --num_sequences 10000 \
    --filename processed_uniref50.pt
```

## What You'll Get

After running the download script, you'll have:

### Dataset Files
- `input_data/processed_uniref50.pt` or `input_data/uniref50_subset.pt` - Main dataset
- `input_data/*_metadata.json` - Dataset statistics
- `input_data/vocab.json` - Protein tokenizer vocabulary
- `input_data/merges.txt` - Tokenizer merge rules

### Configuration
- `configs/config_uniref50_test.yaml` - Test configuration for training

### Dataset Structure
Each dataset contains a list of dictionaries with:
```python
{
    'protein_seq': str,           # Original amino acid sequence
    'prot_tokens': torch.Tensor,  # Tokenized sequence [512]
    'length': int                 # Original sequence length
}
```

## Dataset Options

### 1. Real UniRef50 Data
- **Source**: Hugging Face `agemagician/uniref50`
- **Pros**: Authentic protein sequences
- **Cons**: Requires internet, larger download
- **Size**: ~10,000 sequences for testing

### 2. Synthetic Test Data
- **Source**: Generated from known protein templates
- **Pros**: Fast, no internet required, controlled
- **Cons**: Less diverse than real data
- **Size**: ~5,000 sequences

## Troubleshooting

### Common Issues

1. **"datasets not found"**
   ```bash
   pip install datasets
   ```

2. **"transformers not found"**
   ```bash
   pip install transformers
   ```

3. **Download fails**
   - The script automatically falls back to synthetic data
   - Check internet connection for real UniRef50 download

4. **Memory issues**
   - Reduce `num_sequences` parameter
   - Use smaller batch sizes in training config

### Verification

Test that your dataset works:
```python
import torch

# Load dataset
data = torch.load('input_data/processed_uniref50.pt')
print(f"Loaded {len(data)} sequences")

# Check first sequence
sample = data[0]
print(f"Sequence: {sample['protein_seq'][:50]}...")
print(f"Tokens shape: {sample['prot_tokens'].shape}")
print(f"Length: {sample['length']}")
```

## Training with the Data

Once you have the dataset, you can start training:

### Quick Test Training
```bash
python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir . \
    --config configs/config_uniref50_test.yaml \
    --datafile input_data/processed_uniref50.pt \
    --wandb_project "uniref50_test" \
    --wandb_name "test_run"
```

### Full Optimized Training
```bash
./run_train_uniref50_optimized.sh
```

## Dataset Statistics

Typical dataset will have:
- **Sequences**: 5,000-10,000 protein sequences
- **Length range**: 30-500 amino acids
- **Average length**: ~150 amino acids
- **Vocabulary**: 25 tokens (20 amino acids + 5 special tokens)
- **File size**: ~50-100 MB

## Next Steps

1. **Download the data**: Run `./download_uniref50_data.sh`
2. **Verify the dataset**: Check the output files
3. **Start training**: Use the provided training scripts
4. **Monitor progress**: Check Wandb for training metrics

## Support

If you encounter issues:
1. Check the log files in `logs/`
2. Verify all dependencies are installed
3. Try the synthetic dataset if real download fails
4. Reduce dataset size if memory is limited

The synthetic dataset is perfectly fine for testing the training pipeline and model architecture!
