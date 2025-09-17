# 🧬 Quick Generation Test Feature

## Overview

The UniRef50 optimized trainer now includes a **Quick Generation Test** feature that runs during training to monitor protein sequence generation quality in real-time. This provides continuous feedback on model performance without the overhead of full comprehensive evaluations.

## 🎯 **Key Features**

### **Real-Time Monitoring**
- ✅ **Lightweight testing**: Generates only 3 small sequences (80 AA max)
- ✅ **Fast execution**: Completes in ~0.01-0.08 seconds
- ✅ **Frequent checks**: Runs every 2x log frequency (configurable)
- ✅ **Immediate feedback**: Shows generation quality during training

### **Comprehensive Metrics**
- ✅ **Success rate**: Percentage of valid sequences generated
- ✅ **Average length**: Mean sequence length
- ✅ **Amino acid diversity**: Average unique amino acids per sequence
- ✅ **Generation time**: Performance monitoring
- ✅ **Example sequences**: Visual inspection of generated proteins

### **Smart Integration**
- ✅ **Initial test**: Runs at step 0 to verify setup
- ✅ **Training integration**: Seamlessly integrated into training loop
- ✅ **Wandb logging**: All metrics logged for visualization
- ✅ **Error handling**: Robust error handling with fallback

## 🚀 **How It Works**

### **Training Integration**
The quick generation test is automatically integrated into the training loop:

```python
# Initial test at start of training
if step == 0:
    print("\n🧪 Running initial generation test to verify setup...")
    initial_test_success = self.quick_generation_test(step, 0, num_samples=2, max_length=50)

# Regular tests during training
quick_gen_freq = getattr(self.cfg.training, 'quick_gen_freq', self.cfg.training.log_freq * 2)
if step % quick_gen_freq == 0 and step > 0:
    self.quick_generation_test(step, epoch)
```

### **Test Execution**
Each quick generation test:
1. **Generates 3 short sequences** (80 amino acids max)
2. **Uses the configured sampling method** (rigorous or simple)
3. **Analyzes sequence quality** (length, diversity, validity)
4. **Logs metrics to Wandb** for monitoring
5. **Displays example sequences** for visual inspection

### **Sample Output**
```
🧬 Quick generation test - Step 100
   ✅ Generated 3/3 valid sequences
   📊 Avg length: 30.0, Avg unique AAs: 15.7
   🧬 Example: HHRESIEMDCPELRAEPGAVGWHGGPQ...
```

## ⚙️ **Configuration**

### **Frequency Control**
The test frequency is configurable via the training config:

```yaml
training:
  log_freq: 10              # Base logging frequency
  quick_gen_freq: 20        # Quick generation frequency (optional)
  # If quick_gen_freq not specified, defaults to log_freq * 2
```

### **Default Behavior**
- **Initial test**: Always runs at step 0
- **Regular tests**: Every `log_freq * 2` steps by default
- **Example**: If `log_freq = 10`, quick tests run at steps 20, 40, 60, 80, 100...

### **Customization**
You can customize the test parameters:

```python
# In the trainer class
def quick_generation_test(self, step: int, epoch: int, 
                         num_samples: int = 3,      # Number of sequences
                         max_length: int = 80):     # Max sequence length
```

## 📊 **Metrics Logged**

### **Wandb Metrics**
All metrics are automatically logged to Wandb under the `quick_gen/` namespace:

```python
wandb.log({
    'quick_gen/valid_sequences': valid_count,           # Number of valid sequences
    'quick_gen/total_sequences': num_samples,           # Total sequences attempted
    'quick_gen/success_rate': valid_count / num_samples, # Success percentage
    'quick_gen/avg_length': avg_length,                 # Average sequence length
    'quick_gen/avg_unique_aa': avg_unique_aa,          # Average amino acid diversity
    'quick_gen/generation_time': generation_time,       # Time taken (seconds)
    'quick_gen/sampling_method': self.sampling_method   # Method used
}, step=step)
```

### **Monitoring Dashboard**
Create Wandb dashboard panels to monitor:
- **Success rate over time**: Track generation reliability
- **Average sequence length**: Monitor sequence quality
- **Amino acid diversity**: Ensure diverse generation
- **Generation time**: Performance monitoring
- **Method comparison**: Compare rigorous vs simple sampling

## 🧪 **Testing and Validation**

### **Test Suite**
Run the test suite to verify functionality:

```bash
python test_quick_generation.py
```

This validates:
- ✅ Quick generation workflow
- ✅ Metrics collection and logging
- ✅ Frequency logic
- ✅ Error handling

### **Expected Output**
The test generates sequences like:
```
Step 0:  HVGHYLWTQYWFCQSHEGRGNQLVIAYPGKRGSVSMY...
Step 10: GTQMAKIGGGAKRWHIWCLWIAGCGARLAHHWFKT...
Step 20: ERTRCMRFFDNVEFHGKGWLSEHVRDHELCCA...
```

## 🎯 **Benefits**

### **Training Monitoring**
- ✅ **Early detection**: Spot generation issues early in training
- ✅ **Quality tracking**: Monitor improvement over training steps
- ✅ **Method comparison**: Compare rigorous vs simple sampling performance
- ✅ **Debugging aid**: Quick feedback for troubleshooting

### **Performance Insights**
- ✅ **Success rate trends**: Track generation reliability over time
- ✅ **Sequence quality**: Monitor length and diversity metrics
- ✅ **Speed monitoring**: Track generation performance
- ✅ **Method effectiveness**: Compare sampling method performance

### **Research Value**
- ✅ **Training dynamics**: Understand how generation improves during training
- ✅ **Method analysis**: Compare different sampling approaches
- ✅ **Quality metrics**: Quantitative assessment of generation quality
- ✅ **Reproducibility**: Consistent monitoring across experiments

## 🔧 **Implementation Details**

### **Lightweight Design**
- **Small samples**: Only 3 sequences per test
- **Short sequences**: Maximum 80 amino acids
- **Fast sampling**: Uses efficient parameters for speed
- **Minimal overhead**: ~0.01-0.08 seconds per test

### **Error Handling**
```python
try:
    # Generation and analysis
    sequences = self.generate_protein_sequences(...)
    # ... metrics calculation ...
    return True
except Exception as e:
    print(f"❌ Quick generation test failed: {e}")
    wandb.log({'quick_gen/error': str(e)}, step=step)
    return False
```

### **Integration Points**
1. **Trainer initialization**: Configures test frequency
2. **Training loop**: Runs tests at specified intervals
3. **Initial setup**: Verifies generation works before training
4. **Metrics logging**: Integrates with Wandb logging system

## 📈 **Usage Examples**

### **During Training**
```bash
python protlig_dd/training/run_train_uniref50_optimized.py \
    --work_dir ./work \
    --config ./configs/config.yaml \
    --sampling_method rigorous
```

You'll see output like:
```
🧪 Running initial generation test to verify setup...
🧬 Quick generation test - Step 0
   ✅ Generated 2/2 valid sequences
   📊 Avg length: 35.0, Avg unique AAs: 16.0
   🧬 Example: MKTLVFGCQWHPERDNYLSAIGVEK...
✅ Initial generation test passed - training can proceed

... training continues ...

🧬 Quick generation test - Step 20
   ✅ Generated 3/3 valid sequences
   📊 Avg length: 42.3, Avg unique AAs: 17.7
   🧬 Example: AERDCWLGGYWGADEANYGIEWPQMTLNT...
```

### **Monitoring in Wandb**
- Navigate to your Wandb project
- Look for metrics under `quick_gen/` namespace
- Create custom dashboard panels for visualization
- Track trends over training steps

## 🎉 **Summary**

The Quick Generation Test feature provides:

- ✅ **Real-time monitoring** of generation quality during training
- ✅ **Lightweight testing** with minimal computational overhead
- ✅ **Comprehensive metrics** for detailed analysis
- ✅ **Seamless integration** with existing training pipeline
- ✅ **Flexible configuration** for different use cases
- ✅ **Robust error handling** for reliable operation

This feature enhances the training experience by providing continuous feedback on model performance, enabling early detection of issues and better understanding of training dynamics.
