# 🧬 Dual Sampling Methods in UniRef50 Training

## Overview

The UniRef50 optimized trainer now supports **two distinct sampling methods** for protein sequence generation:

1. **Rigorous CTMC Sampling** (Default) - Based on continuous-time Markov chain theory
2. **Simple Heuristic Sampling** - Direct temperature-based token replacement

Both methods are designed for **single modality** (protein-only) generation while maintaining compatibility with the existing training pipeline.

## 🔬 **Rigorous CTMC Sampling** (Default)

### **Mathematical Foundation**
- Based on **continuous-time Markov chain (CTMC)** theory
- Uses formal **Euler predictor** with proper rate matrices
- Implements **graph-theoretic sampling** with detailed balance
- Includes explicit **denoising step** for quality improvement

### **Key Features**
- ✅ **Mathematically rigorous**: Follows formal diffusion theory
- ✅ **Proper noise integration**: Uses noise derivatives and rate matrices
- ✅ **Categorical sampling**: Gumbel-max trick for token selection
- ✅ **Denoising**: Explicit final denoising step
- ✅ **EMA integration**: Applies exponential moving average weights

### **Implementation Details**
```python
# Uses the formal sampling framework
sampling_fn = sampling.get_sampling_fn(
    config=self.cfg,
    graph=self.graph,
    noise=self.noise,
    batch_dims=(batch_size, max_length),
    eps=1e-5,
    device=self.device
)

# Rigorous model wrapper
class ModelWrapper:
    def __call__(self, x, sigma, **kwargs):
        timesteps = sigma * torch.ones(x.shape[0], device=x.device)
        return self.model(x, timesteps)

samples = sampling_fn(model_wrapper, task="protein_only")
```

### **Configuration**
Uses config parameters from `sampling` section:
```yaml
sampling:
  predictor: euler      # Predictor type
  steps: 100           # Number of sampling steps
  noise_removal: true  # Enable denoising
```

## 🎲 **Simple Heuristic Sampling**

### **Mathematical Foundation**
- **Heuristic approach** with gradual token replacement
- **Temperature-controlled** multinomial sampling
- **Progressive replacement** strategy
- Direct model logit interpretation

### **Key Features**
- ✅ **Simplicity**: Easy to understand and debug
- ✅ **Temperature control**: Fine-grained sampling randomness
- ✅ **Computational efficiency**: Direct model calls
- ✅ **Interpretability**: Clear step-by-step process

### **Implementation Details**
```python
# Manual diffusion process
for step in range(num_diffusion_steps):
    t = torch.tensor([1.0 - step / num_diffusion_steps], device=self.device)
    logits = self.model(sample, t)
    
    # Temperature sampling
    probs = torch.softmax(logits / temperature, dim=-1)
    new_tokens = torch.multinomial(probs_flat, 1).view(batch_size, seq_len)
    
    # Gradual replacement
    replace_prob = (step + 1) / num_diffusion_steps
    mask = torch.rand(batch_size, seq_len, device=self.device) < replace_prob
    sample = torch.where(mask, new_tokens, sample)
```

### **Parameters**
- `num_diffusion_steps`: Number of denoising steps (default: 50)
- `temperature`: Sampling temperature (default: 1.0)

## 🚀 **Usage**

### **Command Line**
```bash
# Use rigorous CTMC sampling (default)
python protlig_dd/training/run_train_uniref50_optimized.py \
    --work_dir ./work \
    --config ./configs/config_uniref50_optimized.yaml \
    --sampling_method rigorous

# Use simple heuristic sampling
python protlig_dd/training/run_train_uniref50_optimized.py \
    --work_dir ./work \
    --config ./configs/config_uniref50_optimized.yaml \
    --sampling_method simple
```

### **Python API**
```python
from protlig_dd.training.run_train_uniref50_optimized import OptimizedUniRef50Trainer

# Initialize trainer with sampling method
trainer = OptimizedUniRef50Trainer(
    work_dir="./work",
    config_file="./configs/config.yaml",
    sampling_method="rigorous"  # or "simple"
)

# Generate sequences with specific method
sequences = trainer.generate_protein_sequences(
    num_samples=10,
    max_length=200,
    sampling_method="rigorous"  # Override trainer default
)

# Or use simple method with custom parameters
sequences = trainer.generate_protein_sequences(
    num_samples=10,
    max_length=200,
    sampling_method="simple",
    num_diffusion_steps=30,
    temperature=0.9
)
```

## 📊 **Comparison**

| Aspect | Rigorous CTMC | Simple Heuristic |
|--------|---------------|------------------|
| **Mathematical Basis** | Formal CTMC theory | Heuristic approach |
| **Sampling Strategy** | Gumbel-max categorical | Temperature multinomial |
| **Token Updates** | Rate matrix sampling | Gradual replacement |
| **Denoising** | Explicit denoising step | No explicit denoising |
| **Complexity** | Higher computational cost | Lower computational cost |
| **Quality** | Theoretically superior | Good practical results |
| **Interpretability** | Complex (graph theory) | Simple (direct process) |
| **Parameters** | Config-driven | Method-specific |

## 🧪 **Testing**

Run the test suite to verify both methods work correctly:

```bash
python test_sampling_methods.py
```

This will:
1. Create minimal test configuration
2. Generate dummy protein data
3. Test both sampling methods
4. Compare outputs and performance
5. Verify unified interface

## ⚙️ **Configuration**

### **For Rigorous Sampling**
Add to your config file:
```yaml
sampling:
  predictor: euler
  steps: 100
  noise_removal: true

noise:
  type: cosine
  sigma_min: !!float "1e-4"
  sigma_max: 0.5
  eps: !!float "0.02"
```

### **For Simple Sampling**
No special configuration needed. Parameters are passed directly:
- `num_diffusion_steps`: 20-50 (more steps = better quality)
- `temperature`: 0.8-1.2 (lower = more focused)

## 🎯 **When to Use Each Method**

### **Use Rigorous CTMC When:**
- ✅ Want mathematically sound generation
- ✅ Need highest quality sequences
- ✅ Have sufficient computational resources
- ✅ Working with research/publication
- ✅ Want to leverage formal diffusion theory

### **Use Simple Heuristic When:**
- ✅ Need fast generation for debugging
- ✅ Want interpretable sampling process
- ✅ Have limited computational resources
- ✅ Need fine control over temperature
- ✅ Prefer simplicity over theoretical rigor

## 🔧 **Implementation Notes**

### **Model Interface Compatibility**
Both methods work with the same V100-compatible SEDD model:
```python
# Model interface: model(tokens, timesteps) -> logits
logits = model(protein_tokens, timesteps)
```

### **EMA Integration**
Both methods properly apply EMA weights during generation:
```python
self.ema.store(self.model.parameters())
self.ema.copy_to(self.model.parameters())
# ... generation ...
self.ema.restore(self.model.parameters())
```

### **Error Handling**
Rigorous sampling includes fallback to simple method if CTMC sampling fails.

## 🎉 **Benefits**

- ✅ **Flexibility**: Choose method based on needs
- ✅ **Backward Compatibility**: Existing code works unchanged
- ✅ **Research Value**: Compare theoretical vs heuristic approaches
- ✅ **Debugging**: Simple method for quick testing
- ✅ **Production Ready**: Rigorous method for final results

The dual sampling approach provides the best of both worlds: theoretical rigor when needed, and practical simplicity for development and debugging.
