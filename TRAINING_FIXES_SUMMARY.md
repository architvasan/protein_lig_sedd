# 🔧 Training Script Fixes & Wandb Integration

## ✅ **Issues Fixed**

### 1. **Configuration Loading Error**
**Problem**: `AttributeError: module 'protlig_dd.utils.utils' has no attribute 'dict_to_namespace'`

**Solution**: 
- Replaced `utils.dict_to_namespace()` with `utils.Config(dictionary=cfg_dict)`
- Added `safe_getattr()` function for robust configuration access
- Added configuration validation and default fallbacks

### 2. **Wandb Web Interface Link**
**Problem**: No clear indication of where to find the Wandb dashboard

**Solution**:
- Added prominent banner displaying Wandb links when training starts
- Shows both run-specific URL and project dashboard
- Added startup banner with clear instructions

## 🚀 **Enhanced Features**

### **Comprehensive Wandb Integration**
- **Real-time metrics**: Loss, learning rate, gradient norms, GPU memory
- **Model samples**: Generated protein sequences logged as tables
- **System monitoring**: Performance and resource usage tracking
- **Configuration logging**: All hyperparameters automatically tracked
- **Model watching**: Gradient and parameter tracking (optional)

### **Robust Error Handling**
- Graceful fallbacks for missing configuration sections
- Safe attribute access with defaults
- Clear error messages and troubleshooting hints
- Offline mode support for Wandb

### **User Experience Improvements**
- Startup banner with key information
- Progress indicators and status updates
- Prominent display of Wandb dashboard links
- Comprehensive logging to files

## 📊 **What You'll See When Training Starts**

```
================================================================================
🧬 OPTIMIZED UNIREF50 SEDD TRAINING
================================================================================
🚀 Enhanced with V100-compatible attention & curriculum learning
📊 Full Wandb experiment tracking enabled
================================================================================

📁 Work directory: /path/to/your/project
⚙️  Config file: configs/config_uniref50_optimized.yaml
💾 Data file: input_data/processed_uniref50.pt
🏷️  Wandb project: uniref50_sedd_optimized
🏷️  Wandb run: uniref50_optimized_20250915_160619
🖥️  Device: cuda:0
🎲 Seed: 42

🔧 Initializing trainer...
✅ Trainer initialized successfully!

🚀 Setting up Wandb logging...

================================================================================
🌐 WANDB EXPERIMENT TRACKING
================================================================================
📊 Project: uniref50_sedd_optimized
🏷️  Run Name: uniref50_optimized_20250915_160619
🔗 Web Interface: https://wandb.ai/your-username/uniref50_sedd_optimized/runs/abc123
📈 Dashboard: https://wandb.ai/your-username/uniref50_sedd_optimized
================================================================================
💡 Open the link above to monitor your training in real-time!
================================================================================

✅ Wandb setup complete - tracking enabled!
✅ Wandb model watching enabled
```

## 🎯 **Key Wandb Metrics Tracked**

### **Training Metrics**
- `train/loss` - Training loss over time
- `train/learning_rate` - Learning rate schedule
- `train/grad_norm` - Gradient norms for monitoring
- `train/batch_time` - Training speed
- `train/samples_per_second` - Throughput

### **Validation Metrics**
- `val/loss` - Validation loss
- `val/perplexity` - Model perplexity (if available)

### **Generated Samples**
- `samples/generated_proteins` - Table of generated sequences
- `samples/avg_length` - Average sequence length
- Sample quality evolution over time

### **System Metrics**
- `system/gpu_memory_allocated_gb` - GPU memory usage
- `system/gpu_memory_reserved_gb` - Reserved GPU memory

### **Model Checkpoints**
- `checkpoint/best_loss` - Best model performance
- `checkpoint/is_best` - Best checkpoint indicator
- Automatic model versioning

## 🛠️ **Usage Instructions**

### **Quick Start**
```bash
# Download data (if not already done)
./download_uniref50_data.sh

# Start training with Wandb tracking
./run_train_uniref50_optimized.sh
```

### **Custom Training**
```bash
python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir . \
    --config configs/config_uniref50_optimized.yaml \
    --datafile input_data/processed_uniref50.pt \
    --wandb_project "my-sedd-experiments" \
    --wandb_name "attention-fix-test" \
    --device cuda:0
```

### **Offline Mode** (No Internet)
```bash
export WANDB_MODE=offline
./run_train_uniref50_optimized.sh
```

## 🔍 **Troubleshooting**

### **Common Issues & Solutions**

1. **"Config file not found"**
   - Ensure you're in the project root directory
   - Check that `configs/config_uniref50_optimized.yaml` exists

2. **"Data file not found"**
   - Run `./download_uniref50_data.sh` to create test data
   - Check the `--datafile` parameter path

3. **"Wandb not logged in"**
   - Run `wandb login` and follow instructions
   - Or use offline mode: `export WANDB_MODE=offline`

4. **"CUDA out of memory"**
   - Reduce batch size in config file
   - Enable gradient checkpointing
   - Use gradient accumulation

5. **"Import errors"**
   - Install missing packages: `pip install wandb matplotlib seaborn`
   - Check virtual environment activation

## 📈 **Monitoring Your Training**

### **Real-time Dashboard**
1. Click the Wandb link shown at training start
2. Monitor loss curves for convergence
3. Check generated samples for quality
4. Watch system metrics for efficiency

### **Key Things to Watch**
- **Loss curves**: Should decrease smoothly
- **Sample quality**: Generated sequences should look realistic
- **GPU memory**: Should be stable, not increasing
- **Training speed**: Should be consistent

### **Success Indicators**
- ✅ Smooth loss decrease without plateaus
- ✅ Realistic protein sequences generated
- ✅ Stable GPU memory usage
- ✅ Consistent training speed

## 🎉 **Benefits**

1. **Fixed Configuration Issues**: No more `dict_to_namespace` errors
2. **Clear Wandb Integration**: Prominent links and comprehensive tracking
3. **Robust Error Handling**: Graceful fallbacks and clear error messages
4. **Enhanced Monitoring**: Real-time insights into training progress
5. **Better User Experience**: Clear status updates and progress indicators

## 📚 **Next Steps**

1. **Start Training**: Use the fixed scripts to begin training
2. **Monitor Progress**: Watch the Wandb dashboard for insights
3. **Compare Results**: Test different configurations and attention mechanisms
4. **Analyze Samples**: Evaluate generated protein quality over time
5. **Optimize Further**: Use insights to improve training parameters

The enhanced training script now provides a robust, well-monitored training experience with comprehensive Wandb integration and clear visibility into your SEDD model's performance!
