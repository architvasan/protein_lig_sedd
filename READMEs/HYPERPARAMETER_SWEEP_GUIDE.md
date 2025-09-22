# 🧬 UniRef50 Hyperparameter Optimization Guide

## Overview

This guide provides a comprehensive hyperparameter optimization system for UniRef50 SEDD training. The system includes predefined configurations, random search capabilities, and analysis tools to help you find the best model configurations.

## 🚀 **Quick Start**

### **1. Prerequisites**

```bash
# Install required packages
pip install wandb pandas matplotlib seaborn jupyter

# Login to Wandb
wandb login

# Ensure you have processed data
ls -la ./input_data/processed_uniref50.pt
```

### **2. Run a Simple Sweep**

```bash
# Run predefined configurations (recommended for first time)
./run_hyperparam_sweep.sh

# Or run random search
./run_hyperparam_sweep.sh --type random --num-random 15
```

### **3. Monitor Progress**

- Check Wandb dashboard: https://wandb.ai
- Monitor local logs in `./hyperparam_experiments/sweep_*/`

### **4. Analyze Results**

```bash
python analyze_sweep_results.py
```

## 📊 **Hyperparameter Search Space**

### **Model Architecture**
- **hidden_size**: [512, 768, 1024] - Model hidden dimension
- **n_heads**: [8, 12, 16] - Number of attention heads
- **n_blocks_prot**: [6, 8, 12] - Number of transformer blocks
- **dropout**: [0.1, 0.15, 0.2] - Dropout rate
- **cond_dim**: [128, 256, 512] - Conditioning dimension

### **Training Parameters**
- **batch_size**: [16, 32, 64] - Training batch size
- **accum**: [2, 4, 8] - Gradient accumulation steps
- **ema**: [0.999, 0.9995, 0.9999] - EMA decay rate

### **Optimizer Settings**
- **lr**: [1e-5, 5e-5, 1e-4, 2e-4] - Learning rate
- **weight_decay**: [0.01, 0.05, 0.1] - Weight decay
- **warmup**: [1000, 5000, 10000] - Warmup steps
- **beta2**: [0.95, 0.99, 0.999] - Adam beta2 parameter
- **grad_clip**: [0.5, 1.0, 2.0] - Gradient clipping

### **Diffusion Parameters**
- **noise.type**: ['cosine', 'linear'] - Noise schedule type
- **noise.sigma_max**: [0.5, 0.8, 1.0] - Maximum noise level
- **noise.eps**: [0.02, 0.05, 0.1] - Noise epsilon
- **sampling.steps**: [50, 100, 200] - Sampling steps

### **Data & System**
- **max_protein_len**: [256, 512, 1024] - Maximum sequence length
- **sampling_method**: ['rigorous', 'simple'] - Sampling method
- **curriculum.enabled**: [True, False] - Curriculum learning

## 🎯 **Predefined Configurations**

### **1. small_fast**
- **Purpose**: Quick iteration and debugging
- **Features**: 512 hidden, 8 heads, simple sampling
- **Training time**: ~2-3 hours
- **Use case**: Initial experiments, debugging

### **2. medium_rigorous**
- **Purpose**: Balanced performance and quality
- **Features**: 768 hidden, 12 heads, rigorous sampling
- **Training time**: ~4-6 hours
- **Use case**: Standard experiments

### **3. large_quality**
- **Purpose**: Maximum quality generation
- **Features**: 1024 hidden, 16 heads, rigorous sampling
- **Training time**: ~8-12 hours
- **Use case**: Final model training

### **4. high_lr_experiment**
- **Purpose**: Fast convergence experiment
- **Features**: High learning rate, large batch size
- **Training time**: ~3-4 hours
- **Use case**: Exploring fast training regimes

### **5. curriculum_focus**
- **Purpose**: Curriculum learning evaluation
- **Features**: Extended curriculum, rigorous sampling
- **Training time**: ~6-8 hours
- **Use case**: Curriculum learning research

## 🛠️ **Usage Examples**

### **Basic Usage**

```bash
# Run all predefined configurations
./run_hyperparam_sweep.sh

# Run with custom config
./run_hyperparam_sweep.sh --config configs/my_config.yaml

# Run with custom data file
./run_hyperparam_sweep.sh --datafile ./my_data/processed_data.pt
```

### **Advanced Usage**

```bash
# Random search with 20 configurations
./run_hyperparam_sweep.sh --type random --num-random 20

# Run with more concurrent jobs (if you have multiple GPUs)
./run_hyperparam_sweep.sh --jobs 4

# Dry run to see what would be executed
./run_hyperparam_sweep.sh --dry-run

# Custom work directory
./run_hyperparam_sweep.sh --work-dir ./my_experiments
```

### **Python API Usage**

```python
from hyperparameter_sweep import HyperparameterSweep

# Create sweep manager
sweep = HyperparameterSweep(
    base_config_path="configs/config_uniref50_optimized.yaml",
    work_dir="./experiments",
    datafile="./input_data/processed_uniref50.pt"
)

# Run predefined sweep
sweep.run_predefined_sweep(dry_run=False, max_concurrent=2)

# Run random sweep
sweep.run_random_sweep(num_configs=15, dry_run=False, max_concurrent=2)
```

## 📈 **Monitoring and Analysis**

### **Real-time Monitoring**

1. **Wandb Dashboard**: https://wandb.ai
   - Project: `uniref50_hyperparam_sweep`
   - Monitor training loss, validation loss, generation metrics
   - Compare runs side-by-side

2. **Local Logs**:
   ```bash
   # Check sweep directory
   ls -la ./hyperparam_experiments/sweep_*/
   
   # Monitor specific job
   tail -f ./hyperparam_experiments/sweep_*/log_*.txt
   ```

### **Post-Sweep Analysis**

```bash
# Generate analysis templates
python analyze_sweep_results.py

# Interactive analysis
cd sweep_analysis
jupyter notebook sweep_analysis.ipynb

# Automated analysis
python sweep_analysis/analyze_results.py
```

### **Key Metrics to Monitor**

- **val_loss**: Primary optimization target (lower is better)
- **train_loss**: Training progress indicator
- **quick_gen/success_rate**: Generation quality (higher is better)
- **quick_gen/avg_length**: Average sequence length
- **quick_gen/avg_unique_aa**: Amino acid diversity

## 🎛️ **Configuration Management**

### **File Structure**
```
hyperparam_experiments/
├── sweep_20241217_143022/          # Sweep directory
│   ├── config_small_fast.yaml      # Generated configs
│   ├── config_medium_rigorous.yaml
│   ├── log_small_fast.txt          # Training logs
│   ├── log_medium_rigorous.txt
│   └── sweep_summary.json          # Sweep metadata
└── sweep_analysis/                 # Analysis results
    ├── sweep_results.csv
    ├── sweep_analysis.ipynb
    └── analyze_results.py
```

### **Custom Configuration**

Create your own parameter combinations:

```python
custom_configs = [
    {
        'name': 'my_experiment',
        'model.hidden_size': 896,
        'model.n_heads': 14,
        'optim.lr': 7e-5,
        'training.batch_size': 48,
        'sampling_method': 'rigorous',
    }
]
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **GPU Memory Errors**:
   ```bash
   # Reduce batch size or model size
   # Check GPU memory: nvidia-smi
   ```

2. **Wandb Login Issues**:
   ```bash
   wandb login
   wandb status
   ```

3. **Data File Not Found**:
   ```bash
   # Download and process data
   ./download_uniref50_data.sh
   ```

4. **Permission Errors**:
   ```bash
   chmod +x run_hyperparam_sweep.sh
   ```

### **Performance Optimization**

1. **Multiple GPUs**: Increase `--jobs` parameter
2. **Faster Storage**: Use SSD for data and work directories
3. **Memory**: Ensure sufficient RAM for data loading
4. **Network**: Stable internet for Wandb logging

## 📋 **Best Practices**

### **Sweep Strategy**

1. **Start Small**: Begin with predefined configs
2. **Iterate**: Use results to guide next experiments
3. **Document**: Keep notes on promising configurations
4. **Validate**: Test best configs on held-out data

### **Resource Management**

1. **Monitor Usage**: Check GPU utilization
2. **Concurrent Jobs**: Balance speed vs resource usage
3. **Storage**: Clean up old experiments periodically
4. **Logging**: Keep detailed logs for debugging

### **Analysis Workflow**

1. **Real-time**: Monitor key metrics during training
2. **Post-sweep**: Comprehensive analysis of all runs
3. **Comparison**: Compare different sweep strategies
4. **Selection**: Choose best configs for production

## 🎉 **Expected Results**

### **Typical Performance Ranges**

- **Validation Loss**: 2.0 - 4.0 (lower is better)
- **Generation Success Rate**: 80% - 95%
- **Training Time**: 2-12 hours per config
- **Memory Usage**: 8-16 GB GPU memory

### **Success Indicators**

- ✅ Decreasing validation loss over time
- ✅ High generation success rate (>90%)
- ✅ Diverse amino acid usage in generated sequences
- ✅ Stable training without divergence

This hyperparameter optimization system provides a comprehensive framework for finding the best UniRef50 SEDD model configurations. Start with the predefined configs and use the analysis tools to guide your optimization strategy!
