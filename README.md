# 🧬 Protein-Ligand SEDD: Score-based Enhanced Diffusion for Discrete Data

A PyTorch implementation of SEDD (Score-based Enhanced Diffusion for Discrete data) for protein and protein-ligand generation, with support for distributed training and cross-platform compatibility.

## ✨ Key Features

- 🧬 **Protein Generation**: Generate realistic protein sequences using diffusion models
- 🚀 **Cross-Platform**: Supports CUDA, Apple Silicon (MPS), Intel XPU, and CPU
- 🔄 **Distributed Training**: Multi-GPU training with PyTorch DDP
- 📊 **Experiment Tracking**: Integrated Weights & Biases logging
- 🎯 **Curriculum Learning**: Advanced noise scheduling for stable training
- 🔧 **Memory Efficient**: Gradient checkpointing and optimized attention
- 📈 **Comprehensive Evaluation**: Built-in sequence analysis and foldability metrics
- ⚡ **Aurora Ready**: Optimized for Intel XPU supercomputer training

## 🏗️ Model Architecture

- **Base Model**: Transformer-based diffusion model (SEDD)
- **Attention**: V100-compatible attention mechanism (no Flash Attention required)
- **Graph Type**: Absorbing diffusion process for discrete sequences
- **Noise Schedule**: Cosine scheduling with curriculum learning
- **Sampling**: Both rigorous CTMC and fast heuristic sampling methods

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/architvasan/protein_lig_sedd.git
cd protein_lig_sedd

# Create virtual environment (recommended)
python -m venv protein_sedd_env
source protein_sedd_env/bin/activate  # On Windows: protein_sedd_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download UniRef50 subset for training (10k sequences)
./shell_scripts/download_uniref50_data.sh
```

### 3. Start Training

```bash
# Fresh training with interactive setup
./shell_scripts/start_fresh_training.sh

# Or direct training
./shell_scripts/run_train_uniref50_optimized.sh
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU (recommended) or CPU
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: 2GB+ for data and checkpoints

### Platform Support
- ✅ **CUDA GPUs** (NVIDIA)
- ✅ **Apple Silicon** (MPS)
- ✅ **Intel XPU** (Aurora supercomputer)
- ✅ **CPU** (slower but functional)

## 🔧 Installation Options

### Standard Installation
```bash
pip install -r requirements.txt
```

### Minimal Installation (basic training only)
```bash
pip install -r requirements-minimal.txt
```

### Aurora/Intel XPU Installation
```bash
pip install -r requirements.txt
pip install -r requirements-aurora.txt
```

## 🎯 Training Options

### Single GPU Training
```bash
# Auto-detect best device
./shell_scripts/run_train_uniref50_optimized.sh

# Specify device
./shell_scripts/run_train_uniref50_optimized.sh --device cuda:0
./shell_scripts/run_train_uniref50_optimized.sh --device mps     # Apple Silicon
./shell_scripts/run_train_uniref50_optimized.sh --device cpu
```

### Multi-GPU Training (DDP)
```bash
# 4 GPU training
./shell_scripts/run_train_uniref50_ddp.sh --gpus 4

# Custom configuration
python -m torch.distributed.launch --nproc_per_node=4 \
    protlig_dd/training/run_train_uniref50_ddp.py \
    --work_dir ./experiments/ddp_run \
    --config configs/config_uniref50_ddp.yaml
```

### Aurora Supercomputer Training
```bash
# Submit job on Aurora
sbatch shell_scripts/run_train_protonly_polaris.sh
```

## 📊 Experiment Tracking

The training automatically logs to [Weights & Biases](https://wandb.ai):

```bash
# Setup wandb (first time only)
python scripts/setup_wandb.py

# Training will show wandb dashboard link
# Example: https://wandb.ai/your-username/uniref50-sedd
```

## 🧪 Analysis and Evaluation

### Analyze Generated Sequences
```bash
# Analyze protein foldability and structural properties
python scripts/analyze_foldability.py --sequences generated_sequences.txt

# Compare generated sequences with training data
python scripts/analyze_generated_sequence.py --model_dir ./experiments/run_1

# Display generated protein sequences
python scripts/show_generated_sequences.py --checkpoint ./experiments/run_1/checkpoints/best.pt
```

### Model Evaluation
```bash
# Calculate model parameters and memory usage
python scripts/calculate_current_model_params.py --config configs/config_uniref50_optimized.yaml

# Test sampling methods
python scripts/simple_sampling_test.py --model_path ./experiments/run_1/checkpoints/best.pt
```

### Hyperparameter Sweeps
```bash
# Run hyperparameter optimization
python scripts/hyperparameter_sweep.py --config configs/config_uniref50_sweeps.yaml

# Analyze sweep results
python scripts/analyze_sweep_results.py --project uniref50_hyperparam_sweep
```

## 🎛️ Advanced Usage

### Custom Training Scripts
```bash
# Train with custom configuration
python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir ./my_experiment \
    --config ./my_config.yaml \
    --device cuda:0 \
    --wandb_project my_project

# Resume from checkpoint
python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir ./my_experiment \
    --resume ./my_experiment/checkpoints/latest.pt
```

### Data Preparation
```bash
# Create custom dataset
python scripts/create_test_protein_dataset.py --output custom_proteins.pt --num_sequences 1000

# Download full UniRef50 (large dataset)
python scripts/download_real_uniref50.py --output_dir ./data/uniref50_full
```

## 📁 Project Structure

```
protein_lig_sedd/
├── protlig_dd/              # Core codebase
│   ├── model/               # Model architectures
│   ├── training/            # Training scripts
│   ├── processing/          # Data processing
│   └── utils/               # Utilities
├── configs/                 # Configuration files
├── scripts/                 # Analysis and utility scripts
├── shell_scripts/           # Training and job scripts
├── READMEs/                 # Detailed documentation
└── requirements*.txt        # Dependencies
```

## ⚙️ Configuration

### Key Configuration Files
- `configs/config_uniref50_optimized.yaml` - Standard training
- `configs/config_uniref50_ddp.yaml` - Multi-GPU training  
- `configs/config_uniref50_stable.yaml` - Stable/conservative settings

### Important Parameters
```yaml
training:
  batch_size: 16              # Adjust based on GPU memory
  n_iters: 50000             # Training iterations
  eval_freq: 1000            # Evaluation frequency
  
model:
  hidden_size: 512           # Model dimension
  n_blocks: 8                # Transformer blocks
  n_heads: 8                 # Attention heads

noise:
  sigma_max: 0.9             # Maximum noise level
  type: "cosine"             # Noise schedule
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size in config
batch_size: 8  # or smaller

# Enable gradient checkpointing
memory:
  gradient_checkpointing: true
```

**Module Import Errors**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="$PWD:$PYTHONPATH"

# Or use module execution
python -m protlig_dd.training.run_train_uniref50_optimized
```

**CUDA Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU fallback
./shell_scripts/run_train_uniref50_optimized.sh --device cpu
```

## 📈 Performance & Benchmarks

### Model Sizes
- **Standard Model**: ~253M parameters
- **Memory Usage**: ~4GB GPU memory (batch_size=16)
- **Training Speed**: ~2-3 sequences/second (single V100)

### Scaling Performance
- **Single GPU**: V100, A100, RTX 3090/4090
- **Multi-GPU**: Linear scaling up to 16 GPUs tested
- **Aurora XPU**: Optimized with Intel IPEX

### Generated Sequence Quality
- **Average Length**: 200-400 amino acids
- **Amino Acid Distribution**: Matches natural protein statistics
- **Structural Foldability**: Analyzed with AlphaFold confidence prediction

## 📚 Documentation

Detailed guides available in `READMEs/`:
- [Training Guide](READMEs/FRESH_TRAINING_GUIDE.md)
- [DDP Training](READMEs/DDP_TRAINING_GUIDE.md)
- [Hyperparameter Sweeps](READMEs/HYPERPARAMETER_SWEEP_GUIDE.md)
- [Cross-Platform Support](READMEs/CROSS_PLATFORM_SUPPORT_SUMMARY.md)
- [Model Architecture](READMEs/SEDD_Architecture_Analysis.md)
- [Evaluation System](READMEs/COMPREHENSIVE_EVALUATION_SYSTEM.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the SEDD paper and implementation
- UniRef50 dataset from UniProt
- Built with PyTorch and Weights & Biases

---

**Need help?** Check the [troubleshooting section](#-troubleshooting) or open an issue!
