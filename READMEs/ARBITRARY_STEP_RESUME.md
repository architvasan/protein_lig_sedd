# Arbitrary Step Resume Feature

## 🎯 **Overview**

The Aurora DDP training script now supports starting training from any arbitrary step, allowing you to resume training in the middle of an epoch. This is useful for:

- **Fine-grained control** over training resumption
- **Debugging specific training steps** 
- **Recovering from mid-epoch failures**
- **Hyperparameter experiments** starting from specific points

## 🔧 **How It Works**

### **1. New Parameter: `start_step`**

Added a new optional parameter to the trainer:
```python
start_step: Optional[int] = None  # Start training from arbitrary step
```

### **2. Command Line Argument**

```bash
--start_step STEP_NUMBER
```

### **3. Training Logic**

The trainer now:
1. **Loads checkpoint** (if available) to get model/optimizer state
2. **Overrides step number** with `start_step` if specified
3. **Calculates target epoch** based on steps per epoch
4. **Skips batches** in the target epoch until reaching the desired step

## 📋 **Usage Examples**

### **Normal Training (No Change)**
```bash
python protlig_dd/training/run_train_uniref_ddp_aurora.py \
    --work_dir ./output \
    --config config.yaml \
    --datafile data.pt
```

### **Start from Arbitrary Step (No Checkpoint)**
```bash
python protlig_dd/training/run_train_uniref_ddp_aurora.py \
    --work_dir ./output \
    --config config.yaml \
    --datafile data.pt \
    --start_step 1500
```
*Starts training from step 1500, calculating the appropriate epoch*

### **Resume from Checkpoint + Override Step**
```bash
python protlig_dd/training/run_train_uniref_ddp_aurora.py \
    --work_dir ./output \
    --config config.yaml \
    --datafile data.pt \
    --resume_checkpoint ./checkpoints/checkpoint_step_1000.pth \
    --start_step 2000
```
*Loads model/optimizer state from checkpoint at step 1000, but starts training from step 2000*

### **Aurora DDP Example**
```bash
mpirun -n 4 python protlig_dd/training/run_train_uniref_ddp_aurora.py \
    --work_dir ./output \
    --config config.yaml \
    --datafile data.pt \
    --start_step 5000 \
    --device xpu:0
```

## 🔍 **Implementation Details**

### **Step Calculation Logic**
```python
# Calculate which epoch this step corresponds to
steps_per_epoch = len(train_loader)
start_epoch = step // steps_per_epoch
steps_to_skip = step - (start_epoch * steps_per_epoch)
```

### **Batch Skipping**
```python
for batch_idx, batch in enumerate(data_iterator):
    # Skip batches if resuming mid-epoch
    if steps_to_skip > 0:
        steps_to_skip -= 1
        continue
    
    # Normal training step
    loss = train_step(batch)
```

### **Checkpoint Override Priority**
1. **`start_step` parameter** (highest priority)
2. **Checkpoint step** (if no start_step specified)
3. **Step 0** (if no checkpoint and no start_step)

## ⚠️ **Important Considerations**

### **1. Data Shuffling**
- The distributed sampler is set to the calculated epoch
- Data order will match what it would be at that epoch/step
- Ensures reproducible training continuation

### **2. Learning Rate Schedule**
- The scheduler state should be loaded from checkpoint for proper LR
- If no checkpoint, LR schedule starts from the beginning (may not be ideal)

### **3. Step Counting**
- Steps are counted globally across all epochs
- Step 1500 might be epoch 2, batch 300 (depending on batch size)

### **4. Validation Timing**
- Evaluation schedules are based on global step count
- May trigger evaluations immediately if step aligns with eval_freq

## 🧪 **Testing**

Run the test script to verify functionality:
```bash
python tests/test_arbitrary_step_resume.py
```

The test covers:
- ✅ Normal training initialization
- ✅ Arbitrary step without checkpoint
- ✅ Checkpoint loading with step override

## 🚀 **Use Cases**

### **1. Mid-Epoch Recovery**
If training crashes at step 1337:
```bash
--start_step 1337 --resume_checkpoint last_checkpoint.pth
```

### **2. Hyperparameter Experiments**
Test different learning rates from step 5000:
```bash
--start_step 5000 --resume_checkpoint checkpoint_step_5000.pth
```

### **3. Debugging Specific Steps**
Investigate issues around step 2500:
```bash
--start_step 2490 --resume_checkpoint checkpoint_step_2000.pth
```

### **4. Curriculum Learning Adjustments**
Change curriculum parameters mid-training:
```bash
--start_step 10000 --resume_checkpoint checkpoint_step_10000.pth
```

## 📊 **Output Examples**

### **Normal Resume**
```
✅ Resumed from checkpoint:
   Checkpoint step: 1000
   Starting step: 1000
   Starting epoch: 2
   Best loss: 0.4521
```

### **Step Override**
```
🔄 Overriding checkpoint step 1000 with start_step 2000
📊 Calculated start_epoch 4 based on 500 steps per epoch
✅ Resumed from checkpoint:
   Checkpoint step: 1000
   Starting step: 2000
   Starting epoch: 4
   Best loss: 0.4521
```

### **Mid-Epoch Resume**
```
🔄 Resuming mid-epoch: skipping 123 batches in epoch 5
```

This feature provides fine-grained control over training resumption while maintaining compatibility with existing checkpoint and DDP functionality.
