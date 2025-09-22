# 🧬 Processing Existing .pt Files with Protein Sequences

## Overview

The `RealUniRef50Downloader` class now supports processing existing `.pt` files containing protein sequences instead of downloading from UniRef50. This is useful when you already have protein sequence data in the expected format.

## ✨ New Features

### 1. **Process from Existing File**
- Load protein sequences from `.pt` files
- Support multiple dictionary formats
- Validate and filter sequences
- Generate tokenized output compatible with training pipeline

### 2. **Flexible Input Formats**
The processor supports `.pt` files containing:
```python
# Format 1: List of dictionaries with 'protein_seq' key
[
    {'protein_seq': 'MATRIFV...'},
    {'protein_seq': 'MFFFVHR...'},
    ...
]

# Format 2: List of dictionaries with 'sequence' key
[
    {'sequence': 'MATRIFV...'},
    {'sequence': 'MFFFVHR...'},
    ...
]

# Format 3: List of dictionaries with 'text' key
[
    {'text': 'MATRIFV...'},
    {'text': 'MFFFVHR...'},
    ...
]

# Format 4: List of strings
[
    'MATRIFV...',
    'MFFFVHR...',
    ...
]
```

## 🚀 Usage

### Command Line Usage

```bash
# Process existing .pt file
python scripts/download_real_uniref50.py \
    --input_file your_proteins.pt \
    --output_dir ./processed_data \
    --num_sequences 5000 \
    --filename processed_proteins.pt

# Still works: Download from UniRef50 (original functionality)
python scripts/download_real_uniref50.py \
    --output_dir ./processed_data \
    --num_sequences 10000
```

### Python API Usage

```python
from scripts.download_real_uniref50 import RealUniRef50Downloader

# Initialize processor
downloader = RealUniRef50Downloader(
    output_dir="./output",
    num_sequences=1000
)

# Process existing file
output_file = downloader.process_from_file(
    input_file="my_proteins.pt",
    output_filename="processed_proteins.pt"
)

print(f"Processed data saved to: {output_file}")
```

## 📊 Output Format

The processed output maintains the same format as the original UniRef50 processor:

```python
[
    {
        'protein_seq': 'MATRIFV...',           # Original sequence
        'prot_tokens': torch.tensor([...]),    # Tokenized sequence
        'length': 156                          # Sequence length
    },
    ...
]
```

## 🔧 Features

### **Sequence Validation**
- Filters sequences by length (30-1000 amino acids)
- Validates amino acid composition (≥95% standard amino acids)
- Removes invalid or malformed sequences

### **Tokenization**
- Uses protein-specific tokenizer
- Handles special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`)
- Supports standard 20 amino acids
- Pads/truncates to consistent length (512 tokens)

### **Metadata Generation**
- Tracks sequence statistics
- Records data source information
- Saves processing metadata as JSON

## 🧪 Testing

Run the test script to verify functionality:

```bash
python test_process_pt_file.py
```

This will:
1. Create a sample `.pt` file with protein sequences
2. Process it using the new functionality
3. Verify the output format
4. Clean up test files

## 📁 File Structure

```
your_project/
├── scripts/
│   └── download_real_uniref50.py    # Updated with new functionality
├── test_process_pt_file.py          # Test script
├── PT_FILE_PROCESSING_README.md     # This documentation
└── your_proteins.pt                 # Your input file
```

## ⚙️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_file` | Path to existing .pt file | None (downloads from UniRef50) |
| `--output_dir` | Output directory | `./input_data` |
| `--num_sequences` | Max sequences to process | 10000 |
| `--filename` | Output filename | `processed_uniref50.pt` |

## 🎯 Use Cases

1. **Custom Protein Datasets**: Process your own curated protein sequences
2. **Filtered Data**: Use pre-filtered or domain-specific protein sequences  
3. **Offline Processing**: Work with existing data without internet connection
4. **Data Pipeline Integration**: Integrate with existing data processing workflows

## 🔍 Example Workflow

```bash
# 1. Prepare your protein sequences in .pt format
# your_proteins.pt contains: [{'protein_seq': 'MATRIFV...'}, ...]

# 2. Process the file
python scripts/download_real_uniref50.py \
    --input_file your_proteins.pt \
    --output_dir ./training_data \
    --num_sequences 5000 \
    --filename my_processed_proteins.pt

# 3. Use in training pipeline
# The output format is compatible with existing training scripts
```

## ✅ Validation

The processor validates:
- File existence and format
- Dictionary structure
- Sequence validity (amino acid composition)
- Sequence length constraints
- Data type consistency

## 🚨 Error Handling

Common issues and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Input file doesn't exist | Check file path |
| `ValueError: Expected list` | Wrong data format | Ensure .pt contains list of dicts/strings |
| `No valid sequences found` | All sequences filtered out | Check sequence format and length |

## 🎉 Benefits

- ✅ **Flexible Input**: Supports multiple data formats
- ✅ **Robust Validation**: Filters invalid sequences automatically  
- ✅ **Compatible Output**: Works with existing training pipeline
- ✅ **Metadata Tracking**: Maintains processing information
- ✅ **Error Handling**: Clear error messages and validation
- ✅ **Backward Compatible**: Original UniRef50 download still works
