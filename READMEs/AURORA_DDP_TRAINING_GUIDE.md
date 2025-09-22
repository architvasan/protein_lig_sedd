# 🚀 Aurora DDP Training Guide

This guide explains how to use the `start_fresh_train_ddp_aurora.sh` script for distributed training on Aurora supercomputer.

**🎯 Key Features:**
- ✅ **Dual Execution Modes** - Interactive testing or queue system submission
- ✅ **No Interactive Prompts** - Fully automated for queue systems
- ✅ **Hardcoded Configuration** - All parameters at the top for easy editing
- ✅ **Automatic Job Creation** - Generates and submits PBS job scripts
- ✅ **Checkpoint Management** - Configurable handling of existing checkpoints

## 🎯 Quick Start

```bash
# Simply run the script - all configuration is hardcoded at the top
./shell_scripts/start_fresh_train_ddp_aurora.sh
```

## 🔧 Configuration Parameters

All parameters are hardcoded at the top of the script for easy modification:

### **Training Configuration**
```bash
WORK_DIR="./experiments/aurora_ddp_$(date +%Y%m%d_%H%M%S)"  # Unique experiment directory
CONFIG_FILE="configs/config_uniref50_ddp.yaml"              # Training config file
DATAFILE="./input_data/subset_uniref50.pt"                  # Training data
WANDB_PROJECT="uniref50_aurora_ddp"                          # Wandb project name
DEVICE="xpu"                                                 # Intel XPU device
SEED=42                                                      # Random seed
```

### **DDP Configuration**
```bash
NUM_GPUS=4                    # Number of XPU devices (1-4 per node)
MASTER_PORT=29500            # Port for DDP communication
BACKEND="ccl"                # CCL backend for Intel XPU
```

### **Job Configuration**
```bash
JOB_NAME="aurora_sedd_ddp"   # PBS job name
QUEUE="workq"                # Aurora queue
WALLTIME="02:00:00"          # Job duration (2 hours)
NODES=1                      # Number of nodes
PPN=4                        # Processes per node (= NUM_GPUS)
```

### **Environment**
```bash
VENV_PATH="pldd_venv"        # Virtual environment path
LOG_DIR="logs"               # Log directory
```

### **Execution Mode**
```bash
EXECUTION_MODE="queue"       # Options: "interactive", "queue"
                            # interactive: Run training directly (for testing/debugging)
                            # queue: Submit PBS job to Aurora queue system
```

### **Checkpoint Handling**
```bash
CHECKPOINT_ACTION="remove"   # Options: "remove", "backup", "ignore", "cancel"
                            # remove: Delete existing checkpoints
                            # backup: Move checkpoints to timestamped backup
                            # ignore: Use --fresh flag to ignore checkpoints
                            # cancel: Exit if checkpoints exist
```

## 📝 How to Customize

### **1. Edit Configuration Parameters**
Open `shell_scripts/start_fresh_train_ddp_aurora.sh` and modify the parameters in the configuration section:

```bash
# Example: Use 8 GPUs across 2 nodes
NUM_GPUS=8
NODES=2
PPN=4
WALLTIME="04:00:00"  # 4 hours for larger job

# Example: Different config and project
CONFIG_FILE="configs/config_uniref50_large.yaml"
WANDB_PROJECT="uniref50_large_scale"
```

### **2. Run the Script**
```bash
./shell_scripts/start_fresh_train_ddp_aurora.sh
```

### **3. Monitor the Job**
```bash
# Check job status
qstat <JOB_ID>

# View logs
tail -f logs/aurora_ddp_<JOB_ID>.out
tail -f logs/aurora_ddp_<JOB_ID>.err
```

## 🎯 Execution Modes

### **Interactive Mode** (`EXECUTION_MODE="interactive"`)
- **Use for**: Testing, debugging, small experiments
- **Runs**: Training directly on current session
- **Logs**: Real-time output + saved to log file
- **Best for**: Development and troubleshooting

```bash
# Set in script
EXECUTION_MODE="interactive"
NUM_GPUS=2  # Use fewer GPUs for testing
WALLTIME="00:30:00"  # Shorter time for testing
```

### **Queue Mode** (`EXECUTION_MODE="queue"`)
- **Use for**: Production training, long runs
- **Runs**: Submits PBS job to Aurora scheduler
- **Logs**: Saved to timestamped files
- **Best for**: Full-scale training experiments

```bash
# Set in script
EXECUTION_MODE="queue"
NUM_GPUS=4  # Full GPU allocation
WALLTIME="04:00:00"  # Longer training time
```

## 🎛️ Common Configurations

### **Small Scale (4 XPUs, 1 Node)**
```bash
NUM_GPUS=4
NODES=1
PPN=4
WALLTIME="02:00:00"
```

### **Medium Scale (8 XPUs, 2 Nodes)**
```bash
NUM_GPUS=8
NODES=2
PPN=4
WALLTIME="04:00:00"
```

### **Large Scale (16 XPUs, 4 Nodes)**
```bash
NUM_GPUS=16
NODES=4
PPN=4
WALLTIME="08:00:00"
```

## 🔍 Script Features

### **Automatic Setup**
- ✅ Creates unique experiment directories
- ✅ Handles checkpoint management (remove/backup/ignore)
- ✅ Generates PBS job script automatically
- ✅ Sets up proper Aurora environment

### **Checkpoint Management**
The script automatically detects existing checkpoints and handles them based on the `CHECKPOINT_ACTION` flag:
- **"remove"** - Delete existing checkpoints and start fresh
- **"backup"** - Move checkpoints to timestamped backup folder
- **"ignore"** - Use --fresh flag to ignore existing checkpoints
- **"cancel"** - Exit if checkpoints exist (useful for safety)

### **Job Submission**
- Creates PBS job script with proper Aurora settings
- Submits job to Aurora queue
- Provides job monitoring commands
- Logs output to timestamped files

## 🐛 Troubleshooting

### **Job Fails to Submit**
```bash
# Check queue status
qstat -Q

# Check account allocation
qstat -u $USER
```

### **Training Fails**
```bash
# Check error logs
cat logs/aurora_ddp_<JOB_ID>.err

# Check output logs
cat logs/aurora_ddp_<JOB_ID>.out
```

### **Module Loading Issues**
```bash
# Verify frameworks module
module avail frameworks
module show frameworks
```

## 📊 Monitoring Training

### **Job Status**
```bash
qstat <JOB_ID>           # Basic status
qstat -f <JOB_ID>        # Detailed status
```

### **Wandb Dashboard**
- Check the Wandb project specified in `WANDB_PROJECT`
- Look for run name with timestamp: `aurora_ddp_YYYYMMDD_HHMMSS`

### **Log Files**
```bash
# Real-time monitoring
tail -f logs/aurora_ddp_<JOB_ID>.out

# Search for errors
grep -i error logs/aurora_ddp_<JOB_ID>.err
```

## 🎯 Best Practices

1. **Start Small**: Begin with 4 XPUs to test your configuration
2. **Monitor Resources**: Check GPU utilization and memory usage
3. **Use Appropriate Walltime**: Don't request more time than needed
4. **Backup Important Runs**: Use the backup option for valuable checkpoints
5. **Check Logs Early**: Monitor the first few minutes to catch setup issues

## 🚀 Ready to Train!

Your Aurora DDP training script is now configured and ready to use. Simply modify the parameters at the top of the script and run it to submit distributed training jobs to Aurora!
