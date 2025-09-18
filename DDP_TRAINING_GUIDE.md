# 🚀 UniRef50 Distributed Data Parallel (DDP) Training Guide

## Overview

This guide provides comprehensive instructions for distributed training of UniRef50 SEDD models across multiple GPUs using PyTorch's DistributedDataParallel (DDP).

## 🎯 **Quick Start**

### **Prerequisites**

```bash
# Ensure CUDA and NCCL are available
nvidia-smi
python -c "import torch; print(torch.distributed.is_nccl_available())"

# Login to Wandb
wandb login

# Prepare data
./download_uniref50_data.sh
```

### **Basic DDP Training**

```bash
# Run DDP training with default settings
./run_train_uniref50_ddp.sh

# Custom configuration
./run_train_uniref50_ddp.sh --config configs/my_ddp_config.yaml
```

## 📊 **Hyperparameter Scaling for DDP**

### **🔢 Key Scaling Rules**

#### **1. Learning Rate Scaling**
```yaml
# Single GPU: lr = 5e-5
# 2 GPUs: lr = 1e-4  (5e-5 × 2)
# 4 GPUs: lr = 2e-4  (5e-5 × 4)
# 8 GPUs: lr = 4e-4  (5e-5 × 8)
```

**Why:** Linear scaling maintains the same effective learning rate per sample when batch size increases.

#### **2. Batch Size Scaling**
```yaml
# Effective batch size = batch_size × accum × num_gpus
# Single GPU: 16 × 4 × 1 = 64
# 2 GPUs: 16 × 4 × 2 = 128
# 4 GPUs: 16 × 4 × 4 = 256
# 8 GPUs: 16 × 4 × 8 = 512
```

**Options:**
- **Keep per-GPU batch size constant** (recommended): Memory usage per GPU stays the same
- **Reduce per-GPU batch size**: If memory becomes an issue with larger models

#### **3. Warmup Steps Scaling**
```yaml
# Base warmup: 5000 steps
# 2 GPUs: 10000 steps (5000 × 2)
# 4 GPUs: 20000 steps (5000 × 4)
# 8 GPUs: 40000 steps (5000 × 8)
```

**Why:** Maintains the same effective warmup duration in terms of data seen.

### **📋 Recommended Configurations**

#### **2 GPUs Configuration**
```yaml
training:
  batch_size: 16        # Per-GPU
  accum: 4              # Effective batch: 128
optim:
  lr: 1e-4              # 5e-5 × 2
  warmup: 10000         # 5000 × 2
```

#### **4 GPUs Configuration**
```yaml
training:
  batch_size: 16        # Per-GPU
  accum: 4              # Effective batch: 256
optim:
  lr: 2e-4              # 5e-5 × 4
  warmup: 20000         # 5000 × 4
```

#### **8 GPUs Configuration**
```yaml
training:
  batch_size: 12        # Reduced per-GPU for memory
  accum: 4              # Effective batch: 384
optim:
  lr: 3e-4              # Slightly less aggressive scaling
  warmup: 30000         # 5000 × 6
```

## 🛠️ **Advanced DDP Configuration**

### **Memory Optimization**

```yaml
memory:
  gradient_checkpointing: False  # Better DDP performance
  mixed_precision: True          # Essential for large batches
  max_memory_per_gpu: 0.85      # Leave headroom

ddp:
  find_unused_parameters: False  # Better performance
  gradient_as_bucket_view: True  # Memory optimization
```

### **Performance Tuning**

```yaml
training:
  num_workers: 4        # Per-GPU workers
  
# Disable expensive logging in DDP
monitoring:
  log_gradients: False
  log_weights: False
```

### **Network Configuration**

```bash
# Environment variables for better NCCL performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## 🎛️ **Hyperparameter Scaling Strategies**

### **Linear Scaling (Recommended)**

**When to use:** Most cases, especially for batch sizes up to 512
```python
scaled_lr = base_lr * num_gpus
scaled_warmup = base_warmup * num_gpus
```

### **Square Root Scaling**

**When to use:** Very large batch sizes (>1024)
```python
scaled_lr = base_lr * sqrt(num_gpus)
scaled_warmup = base_warmup * sqrt(num_gpus)
```

### **Custom Scaling**

**When to use:** Specific model architectures or datasets
```python
# Example: Conservative scaling for stability
scaled_lr = base_lr * min(num_gpus, 4)  # Cap at 4x scaling
```

## 📈 **Performance Expectations**

### **Scaling Efficiency**

| GPUs | Effective Batch | Expected Speedup | Memory per GPU |
|------|----------------|------------------|----------------|
| 1    | 64             | 1.0x             | ~12GB          |
| 2    | 128            | 1.8x             | ~12GB          |
| 4    | 256            | 3.2x             | ~12GB          |
| 8    | 512            | 5.5x             | ~10GB          |

### **Training Time Estimates**

| Configuration | Time to 100k steps | Total Training Time |
|---------------|-------------------|-------------------|
| 1 GPU         | ~24 hours         | ~5 days           |
| 2 GPUs        | ~13 hours         | ~2.8 days         |
| 4 GPUs        | ~7.5 hours        | ~1.6 days         |
| 8 GPUs        | ~4.5 hours        | ~1 day            |

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. NCCL Initialization Errors**
```bash
# Check NCCL availability
python -c "import torch; print(torch.distributed.is_nccl_available())"

# Set debug mode
export NCCL_DEBUG=INFO
```

#### **2. Out of Memory Errors**
```yaml
# Reduce per-GPU batch size
training:
  batch_size: 8         # Reduced from 16

# Or reduce model size
model:
  hidden_size: 512      # Reduced from 768
```

#### **3. Slow Training**
```yaml
# Increase workers per GPU
training:
  num_workers: 6        # Increased from 4

# Disable expensive operations
monitoring:
  log_gradients: False
  sample_frequency: 5000  # Less frequent
```

#### **4. Convergence Issues**
```yaml
# Reduce learning rate scaling
optim:
  lr: !!float "1e-4"    # Instead of 2e-4 for 4 GPUs

# Increase warmup
optim:
  warmup: 25000         # Longer warmup
```

### **Debugging Commands**

```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor NCCL communication
export NCCL_DEBUG=INFO

# Check process synchronization
ps aux | grep python
```

## 📊 **Monitoring DDP Training**

### **Key Metrics to Watch**

1. **GPU Utilization**: Should be >90% on all GPUs
2. **Memory Usage**: Should be balanced across GPUs
3. **Loss Synchronization**: All processes should report similar losses
4. **Communication Overhead**: Should be <10% of total time

### **Wandb Integration**

```python
# DDP-specific metrics logged automatically
wandb.config.update({
    'ddp/world_size': world_size,
    'ddp/effective_batch_size': effective_batch_size,
    'ddp/scaled_lr': scaled_lr,
})
```

## 🎯 **Best Practices**

### **Configuration**

1. **Start with 2 GPUs** to validate DDP setup
2. **Use linear LR scaling** for most cases
3. **Keep per-GPU batch size constant** for consistent memory usage
4. **Scale warmup steps** to maintain effective warmup duration

### **Performance**

1. **Use mixed precision** (AMP) for memory efficiency
2. **Disable gradient checkpointing** for better DDP performance
3. **Set appropriate num_workers** (typically 4-8 per GPU)
4. **Use fast storage** (NVMe SSD) for data loading

### **Debugging**

1. **Test on small dataset first** to validate setup
2. **Monitor all GPUs** during training
3. **Check loss synchronization** across processes
4. **Use NCCL debug mode** for communication issues

## 🚀 **Production Deployment**

### **SLURM Integration**

```bash
#!/bin/bash
#SBATCH --job-name=uniref50_ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Set up environment
module load cuda/11.8
module load nccl

# Run DDP training
srun python -m protlig_dd.training.run_train_uniref50_ddp \
    --work_dir $SCRATCH/experiments \
    --config configs/config_uniref50_ddp.yaml \
    --datafile $SCRATCH/data/processed_uniref50.pt
```

### **Multi-Node Setup**

```bash
# Node 0 (master)
python -m protlig_dd.training.run_train_uniref50_ddp \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    --node_rank 0 \
    --nnodes 2

# Node 1
python -m protlig_dd.training.run_train_uniref50_ddp \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    --node_rank 1 \
    --nnodes 2
```

This DDP implementation provides efficient multi-GPU training with proper hyperparameter scaling and comprehensive monitoring capabilities!
