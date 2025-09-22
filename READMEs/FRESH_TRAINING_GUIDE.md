# 🆕 How to Start Training from Scratch

## 🎯 **Quick Answer**

To start training from scratch (ignoring existing checkpoints), you have **4 easy options**:

### **Option 1: Remove Checkpoints (Simplest)**
```bash
# Remove existing checkpoints
rm -rf checkpoints/

# Start fresh training
./run_train_uniref50_optimized.sh
```

### **Option 2: Use the Fresh Training Script (Recommended)**
```bash
# Interactive script with multiple options
./start_fresh_training.sh
```

### **Option 3: Use the --fresh Flag**
```bash
# Ignore existing checkpoints without deleting them
python protlig_dd/training/run_train_uniref50_optimized.py \
    --work_dir . \
    --config configs/config_uniref50_stable.yaml \
    --datafile ./input_data/processed_uniref50.pt \
    --wandb_project uniref50_sedd_optimized \
    --device auto \
    --fresh
```

### **Option 4: Backup and Start Fresh**
```bash
# Backup existing checkpoints
mv checkpoints/ checkpoints_backup_$(date +%Y%m%d_%H%M%S)/

# Start fresh training
./run_train_uniref50_optimized.sh
```

## 🔧 **Detailed Options**

### **🗑️ Option 1: Remove Checkpoints**

**When to use**: You don't need the existing checkpoint anymore

```bash
# Remove all checkpoints
rm -rf checkpoints/

# Start training (will create new checkpoints)
./run_train_uniref50_optimized.sh
```

**What happens**:
- ✅ No checkpoint found → starts from step 0
- ✅ Creates fresh checkpoint directory
- ✅ Begins with randomly initialized model
- ✅ Uses enhanced evaluation system from beginning

### **📦 Option 2: Backup Checkpoints**

**When to use**: You want to keep existing checkpoints as backup

```bash
# Create timestamped backup
backup_dir="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
mv checkpoints/ "$backup_dir"
echo "Checkpoints backed up to: $backup_dir"

# Start fresh training
./run_train_uniref50_optimized.sh
```

**Benefits**:
- ✅ Preserves your existing training progress
- ✅ Can restore later if needed
- ✅ Organized with timestamps

### **🚀 Option 3: Use --fresh Flag**

**When to use**: You want to keep checkpoints but ignore them for this run

```bash
python protlig_dd/training/run_train_uniref50_optimized.py \
    --work_dir . \
    --config configs/config_uniref50_stable.yaml \
    --datafile ./input_data/processed_uniref50.pt \
    --wandb_project uniref50_sedd_optimized \
    --wandb_name "fresh_start_$(date +%Y%m%d_%H%M%S)" \
    --device auto \
    --seed 42 \
    --fresh
```

**What the --fresh flag does**:
- ✅ Ignores existing checkpoints completely
- ✅ Starts from step 0 with random initialization
- ✅ Keeps existing checkpoints untouched
- ✅ Creates new checkpoints alongside old ones

### **🎭 Option 4: Interactive Script**

**When to use**: You want a guided experience with multiple options

```bash
./start_fresh_training.sh
```

**Features**:
- 🔍 Detects existing checkpoints
- 🤔 Asks what you want to do
- 📦 Offers backup options
- ⚙️ Shows configuration before starting
- 🎯 Handles everything automatically

## 📊 **What to Expect with Fresh Training**

### **Training Progress**
```
Step 0-1000:     Random → Basic patterns
Step 1000-5000:  Short sequences (10-50 AAs)
Step 5000-15000: Medium sequences (50-150 AAs)
Step 15000+:     Long, realistic proteins (150-400 AAs)
```

### **Enhanced Evaluation from Day 1**
- 🧬 **Real generation** (not fake random sequences)
- 📊 **Comprehensive metrics** (length, composition, diversity)
- 🔬 **Biochemical analysis** (hydrophobic, polar, charged %)
- 📈 **Training comparison** (generated vs training data)
- 🎯 **Rich Wandb logging** with sequence tables

### **Expected Timeline**
- **V100 GPU**: ~3-5 days to reach high quality (50K steps)
- **Apple Silicon**: ~5-7 days with MPS acceleration
- **CPU**: ~2-3 weeks (much slower but works)

## 🎯 **Recommended Approach**

### **For Experimentation**
```bash
# Use --fresh flag to keep existing checkpoints
python protlig_dd/training/run_train_uniref50_optimized.py \
    --work_dir . \
    --config configs/config_uniref50_stable.yaml \
    --fresh
```

### **For Production Training**
```bash
# Backup existing and start fresh
./start_fresh_training.sh
```

### **For Quick Testing**
```bash
# Just remove and restart
rm -rf checkpoints/
./run_train_uniref50_optimized.sh
```

## 🔧 **Configuration Options**

### **Key Parameters for Fresh Training**
```yaml
# In configs/config_uniref50_stable.yaml
training:
  n_iters: 500000      # Total training steps
  batch_size: 32       # Batch size
  eval_freq: 1000      # Evaluation frequency
  
optim:
  lr: 5e-5            # Learning rate
  warmup: 10000       # Warmup steps
  
curriculum:
  enabled: True       # Curriculum learning
  preschool_time: 20000  # Gradual difficulty ramp
```

### **Device Selection**
```bash
# Auto-detect best device
--device auto

# Specific devices
--device cuda:0      # NVIDIA GPU
--device mps         # Apple Silicon
--device cpu         # CPU only
```

## 🎉 **Benefits of Fresh Training**

### **Clean Slate**
- ✅ No residual effects from previous training
- ✅ Consistent initialization across runs
- ✅ Reproducible results with same seed

### **Enhanced Monitoring**
- ✅ Track progress from step 0
- ✅ See learning curve development
- ✅ Compare different training runs

### **Better Evaluation**
- ✅ Real diffusion-based generation from start
- ✅ Comprehensive quality metrics
- ✅ Rich Wandb visualizations

## 🚀 **Ready to Start Fresh!**

Choose your preferred method and start training:

1. **Quick & Simple**: `rm -rf checkpoints/ && ./run_train_uniref50_optimized.sh`
2. **Interactive**: `./start_fresh_training.sh`
3. **Keep Backups**: Use the backup approach
4. **Ignore Existing**: Use `--fresh` flag

**Your SEDD model will start learning from scratch with world-class evaluation and monitoring!** 🎯✨
