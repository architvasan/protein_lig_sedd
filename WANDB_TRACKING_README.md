# 📊 Wandb Experiment Tracking for SEDD Training

This guide explains how to use Weights & Biases (Wandb) to track your SEDD protein model training experiments.

## 🚀 Quick Setup

### Option 1: Automated Setup
```bash
# Run the setup script
python setup_wandb.py
```

### Option 2: Manual Setup
```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login
```

## 📈 What Gets Tracked

### Training Metrics
- **Loss curves**: Training and validation loss over time
- **Learning rate**: LR schedule progression
- **Gradient norms**: Monitor gradient flow
- **Batch timing**: Training speed metrics
- **GPU memory**: Memory usage tracking

### Model Metrics
- **Model parameters**: Architecture details
- **EMA updates**: Exponential moving average tracking
- **Curriculum progress**: Difficulty ramp progression
- **Noise schedule**: Diffusion noise parameters

### Generated Samples
- **Protein sequences**: Generated amino acid sequences
- **Sample quality**: Length distributions and statistics
- **Sample diversity**: Sequence variation analysis

### System Metrics
- **GPU utilization**: Memory allocation and usage
- **Training speed**: Samples per second
- **Checkpoint info**: Best model tracking

## 🎯 Key Features

### 1. **Comprehensive Configuration Logging**
All hyperparameters are automatically logged:
```python
config = {
    'model_name': 'transformer',
    'hidden_size': 512,
    'learning_rate': 5e-5,
    'batch_size': 32,
    'curriculum_enabled': True,
    # ... and many more
}
```

### 2. **Real-time Loss Tracking**
Monitor training progress with detailed loss curves:
- Smoothed training loss
- Validation loss at regular intervals
- Best loss tracking
- Loss plateau detection

### 3. **Model Sample Generation**
Automatically generates and logs protein samples:
- Sample sequences at regular intervals
- Length distribution analysis
- Sequence diversity metrics
- Visual sample quality assessment

### 4. **System Performance Monitoring**
Track training efficiency:
- GPU memory usage
- Training speed (samples/sec)
- Batch processing time
- Memory optimization effectiveness

### 5. **Checkpoint Management**
Intelligent checkpoint tracking:
- Best model identification
- Checkpoint metadata
- Training resumption support
- Model versioning

## 📋 Usage Examples

### Basic Training with Wandb
```bash
# Start training with wandb tracking
./run_train_uniref50_optimized.sh
```

### Custom Wandb Configuration
```bash
python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir . \
    --config configs/config_uniref50_optimized.yaml \
    --datafile input_data/processed_uniref50.pt \
    --wandb_project "my-sedd-project" \
    --wandb_name "experiment-v1"
```

### Offline Mode (No Internet)
```bash
export WANDB_MODE=offline
./run_train_uniref50_optimized.sh
```

## 📊 Dashboard Views

### 1. **Training Overview**
- Loss curves (train/val)
- Learning rate schedule
- Training speed metrics
- GPU utilization

### 2. **Model Performance**
- Sample quality over time
- Generated sequence examples
- Length distribution evolution
- Curriculum learning progress

### 3. **System Metrics**
- Memory usage patterns
- Training efficiency
- Batch processing speed
- Hardware utilization

### 4. **Hyperparameter Analysis**
- Configuration comparison
- Parameter sensitivity
- Ablation study support
- Experiment comparison

## 🔧 Configuration Options

### Environment Variables
```bash
# Wandb project name
export WANDB_PROJECT="uniref50-sedd"

# Wandb entity (username/team)
export WANDB_ENTITY="your-username"

# Offline mode
export WANDB_MODE="offline"

# Disable wandb
export WANDB_DISABLED="true"
```

### Config File Settings
Edit `wandb_config.yaml`:
```yaml
project: uniref50-sedd
entity: your-wandb-username
tags:
  - uniref50
  - sedd
  - protein
  - diffusion
notes: |
  Optimized training with improved attention
```

## 🎨 Custom Logging

### Adding Custom Metrics
```python
# In your training code
wandb.log({
    'custom/my_metric': value,
    'custom/another_metric': another_value
}, step=step)
```

### Logging Images/Plots
```python
# Log matplotlib figures
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# ... create your plot
wandb.log({"plots/my_plot": wandb.Image(fig)})
```

### Logging Tables
```python
# Log structured data
table = wandb.Table(
    columns=['sequence', 'length', 'quality'],
    data=[['MKLLF...', 150, 0.95], ...]
)
wandb.log({"samples/sequences": table})
```

## 🔍 Monitoring Best Practices

### 1. **Regular Checkpoints**
- Monitor validation loss trends
- Watch for overfitting signs
- Track sample quality improvements

### 2. **Resource Monitoring**
- Keep GPU memory usage reasonable
- Monitor training speed consistency
- Watch for memory leaks

### 3. **Sample Quality**
- Check generated sequences make sense
- Monitor sequence length distributions
- Verify amino acid composition

### 4. **Hyperparameter Tuning**
- Compare different learning rates
- Test various batch sizes
- Experiment with curriculum schedules

## 🚨 Troubleshooting

### Common Issues

1. **"wandb not logged in"**
   ```bash
   wandb login
   # Or set API key
   export WANDB_API_KEY="your-api-key"
   ```

2. **"wandb offline mode"**
   ```bash
   export WANDB_MODE=online
   wandb sync wandb/offline-run-*
   ```

3. **"too many logs"**
   - Reduce `log_freq` in config
   - Use `wandb.log(..., commit=False)` for batching

4. **"memory issues"**
   - Disable model watching: `wandb.watch(model, log=None)`
   - Reduce sample generation frequency

### Performance Tips

1. **Batch Logging**
   ```python
   # Batch multiple metrics
   metrics = {
       'loss': loss,
       'lr': lr,
       'grad_norm': grad_norm
   }
   wandb.log(metrics, step=step)
   ```

2. **Conditional Logging**
   ```python
   # Only log expensive metrics occasionally
   if step % 100 == 0:
       wandb.log({'expensive_metric': compute_expensive()})
   ```

## 📚 Advanced Features

### Hyperparameter Sweeps
```yaml
# sweep.yaml
program: protlig_dd/training/run_train_uniref50_optimized.py
method: bayes
parameters:
  learning_rate:
    min: 1e-6
    max: 1e-3
  batch_size:
    values: [16, 32, 64]
```

### Model Artifacts
```python
# Save model as wandb artifact
artifact = wandb.Artifact('sedd-model', type='model')
artifact.add_file('checkpoint.pth')
wandb.log_artifact(artifact)
```

## 🎉 Success Metrics

Your Wandb integration is working well when you see:
- ✅ Smooth loss curves without gaps
- ✅ Regular sample generation logs
- ✅ Consistent system metrics
- ✅ Proper checkpoint tracking
- ✅ Meaningful hyperparameter logging

## 🔗 Useful Links

- [Wandb Documentation](https://docs.wandb.ai/)
- [PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)
- [Hyperparameter Sweeps](https://docs.wandb.ai/guides/sweeps)
- [Model Registry](https://docs.wandb.ai/guides/models)

Happy tracking! 🚀
