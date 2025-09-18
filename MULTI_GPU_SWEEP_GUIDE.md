# 🚀 Multi-GPU Hyperparameter Sweep Guide

## Overview

The hyperparameter sweep system has been enhanced to utilize all 4 available GPUs (CUDA_VISIBLE_DEVICES 0, 1, 2, 3) for concurrent training jobs with comprehensive logging.

## 🎯 **Key Enhancements**

### **✅ Multi-GPU Job Distribution**
- **Automatic GPU assignment** across available GPUs (0, 1, 2, 3)
- **Concurrent job execution** - up to 4 jobs running simultaneously
- **Smart GPU scheduling** - jobs automatically assigned to free GPUs
- **Process isolation** using CUDA_VISIBLE_DEVICES

### **✅ Enhanced Logging**
- **Separate stdout/stderr logs** for each job and GPU
- **Structured log naming**: `log_{job_name}_gpu{gpu_id}_stdout.txt`
- **Real-time progress tracking** with GPU assignment info
- **Comprehensive job completion monitoring**

### **✅ Improved Resource Management**
- **GPU availability validation** before starting sweep
- **Dynamic process management** - jobs start as GPUs become available
- **Memory isolation** - each job only sees its assigned GPU
- **Graceful error handling** and cleanup

## 🖥️ **GPU Configuration**

### **Default Setup (4 GPUs)**
```bash
# Uses all 4 GPUs by default
./run_hyperparam_sweep.sh
```

### **Custom GPU Selection**
```bash
# Use specific GPUs
./run_hyperparam_sweep.sh --gpus 0,1,2,3

# Use only 2 GPUs
./run_hyperparam_sweep.sh --gpus 0,1

# Use single GPU
./run_hyperparam_sweep.sh --gpus 2
```

## 📊 **Performance Benefits**

### **Throughput Improvements**
| GPUs | Concurrent Jobs | Expected Speedup | Total Time Reduction |
|------|----------------|------------------|---------------------|
| 1    | 1              | 1.0x             | Baseline            |
| 2    | 2              | 1.9x             | ~47% faster         |
| 4    | 4              | 3.6x             | ~72% faster         |

### **Example Timing**
```
Single GPU (sequential):  5 configs × 2 hours = 10 hours total
4 GPUs (parallel):       5 configs ÷ 4 GPUs = ~2.5 hours total
```

## 🚀 **Usage Examples**

### **Basic Multi-GPU Sweep**
```bash
# Run predefined sweep on all 4 GPUs
./run_hyperparam_sweep.sh

# Run random sweep with 12 configs on 4 GPUs
./run_hyperparam_sweep.sh --type random --num-random 12
```

### **Testing and Validation**
```bash
# Test multi-GPU setup
python test_multi_gpu_sweep.py

# Dry run to see GPU assignments
./run_hyperparam_sweep.sh --dry-run

# Test with specific GPUs
./run_hyperparam_sweep.sh --gpus 0,1 --dry-run
```

## 📋 **Log File Structure**

### **Log File Naming**
```
hyperparam_experiments/sweep_YYYYMMDD_HHMMSS/
├── sweep_summary.json                    # Sweep configuration
├── config_small_fast.yaml               # Generated configs
├── config_balanced.yaml
├── log_small_fast_gpu0_stdout.txt       # Job stdout logs
├── log_small_fast_gpu0_stderr.txt       # Job stderr logs
├── log_balanced_gpu1_stdout.txt
├── log_balanced_gpu1_stderr.txt
└── ...
```

### **Log Content Examples**

#### **Stdout Log** (`log_small_fast_gpu0_stdout.txt`)
```
🚀 Starting UniRef50 SEDD Training
📁 Work directory: ./hyperparam_experiments/sweep_20241218_143022
🎯 Configuration: small_fast
🖥️  Device: cuda:0
📊 Wandb project: uniref50_hyperparam_sweep

Epoch 1/50:   0%|          | 0/2968 [00:00<?, ?it/s]
Epoch 1/50:   1%|▏         | 30/2968 [00:15<24:32,  2.00it/s, loss=2.45]
...
```

#### **Stderr Log** (`log_small_fast_gpu0_stderr.txt`)
```
/opt/conda/lib/python3.11/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
```

## 🔧 **Implementation Details**

### **GPU Process Management**
```python
# Each job gets isolated GPU environment
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

process = subprocess.Popen(
    cmd, 
    stdout=stdout_file, 
    stderr=stderr_file,
    env=env  # Isolated GPU environment
)
```

### **Dynamic Job Scheduling**
```python
# Track active processes per GPU
gpu_processes = {gpu_id: None for gpu_id in available_gpus}

# Wait for available GPU
while True:
    # Check for completed processes
    for gpu_id in available_gpus:
        if gpu_processes[gpu_id] is not None:
            if gpu_processes[gpu_id].poll() is not None:
                gpu_processes[gpu_id] = None  # GPU now free
    
    # Find available GPU and start job
    available_gpu = find_free_gpu()
    if available_gpu is not None:
        start_job_on_gpu(available_gpu)
        break
```

## 📈 **Monitoring Multi-GPU Sweeps**

### **Real-Time Monitoring**
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Monitor active processes
ps aux | grep python

# Check log files in real-time
tail -f hyperparam_experiments/sweep_*/log_*_stdout.txt
```

### **Wandb Dashboard**
- **Project**: `uniref50_hyperparam_sweep`
- **Runs**: Each job creates separate Wandb run
- **Metrics**: All jobs log to same project for easy comparison
- **Tags**: Jobs automatically tagged with GPU ID and config name

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **GPU Not Available Error**
```bash
# Check GPU availability
nvidia-smi

# Validate specific GPUs
./run_hyperparam_sweep.sh --gpus 0,1 --dry-run
```

#### **Memory Issues**
```bash
# Reduce batch size in configs
# Or use fewer concurrent jobs
./run_hyperparam_sweep.sh --gpus 0,1  # Use only 2 GPUs
```

#### **Process Hanging**
```bash
# Check for zombie processes
ps aux | grep python

# Kill hanging processes
pkill -f "hyperparameter_sweep"
```

### **Log Analysis**
```bash
# Check for errors across all jobs
grep -r "ERROR\|Exception\|Traceback" hyperparam_experiments/sweep_*/

# Check job completion status
grep -r "Training completed\|🎉" hyperparam_experiments/sweep_*/log_*_stdout.txt

# Monitor GPU memory usage in logs
grep -r "GPU memory" hyperparam_experiments/sweep_*/log_*_stdout.txt
```

## 🎯 **Best Practices**

### **Configuration**
1. **Start with 2 GPUs** to validate multi-GPU setup
2. **Use dry-run first** to check GPU assignments
3. **Monitor first few jobs** to ensure proper GPU utilization
4. **Check log files** for any GPU-specific issues

### **Resource Management**
1. **Don't oversubscribe GPUs** - stick to 1 job per GPU
2. **Monitor memory usage** - reduce batch size if needed
3. **Use appropriate configs** - balance speed vs. thoroughness
4. **Clean up old sweep directories** to save disk space

### **Monitoring**
1. **Use nvidia-smi** to monitor GPU utilization
2. **Check Wandb dashboard** for training progress
3. **Monitor log files** for errors or warnings
4. **Validate results** by comparing across different GPUs

## 🎉 **Ready to Use**

The multi-GPU hyperparameter sweep system is ready for production use:

### **Quick Start**
```bash
# Test setup
python test_multi_gpu_sweep.py

# Run sweep
./run_hyperparam_sweep.sh --gpus 0,1,2,3

# Monitor progress
nvidia-smi -l 1
```

### **Expected Output**
```
🧬 UNIREF50 HYPERPARAMETER SWEEP
==================================

🎯 Running predefined hyperparameter sweep
📁 Sweep directory: ./hyperparam_experiments/sweep_20241218_143022
🔧 Number of configurations: 5
🖥️  Available GPUs: [0, 1, 2, 3]
⚡ Max concurrent jobs: 4

🚀 Job 1/5 started: small_fast on GPU 0
🚀 Job 2/5 started: balanced on GPU 1
🚀 Job 3/5 started: large_thorough on GPU 2
🚀 Job 4/5 started: memory_efficient on GPU 3
✅ Job completed on GPU 0 (1/5 total)
🚀 Job 5/5 started: experimental on GPU 0
...
```

This implementation provides efficient multi-GPU hyperparameter optimization with comprehensive logging and monitoring! 🚀
