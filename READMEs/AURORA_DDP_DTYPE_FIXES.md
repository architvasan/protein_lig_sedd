# Aurora DDP Dtype Consistency Fixes

## 🐛 **Problem Identified**

The Aurora DDP training was failing with the error:
```
❌ Training failed with error: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

This error occurs when IPEX optimization converts the model to bfloat16, but some tensors remain in float32, causing dtype mismatches during matrix operations.

## 🔧 **Fixes Applied**

### 1. **IPEX Optimization Fix** (Lines 795-817)
- **Problem**: IPEX was forcing `dtype=torch.bfloat16` conversion, causing dtype mismatches
- **Solution**: Removed explicit dtype forcing, let IPEX handle mixed precision automatically
- **Change**: Removed `dtype=torch.bfloat16` parameter from `ipex.optimize()`

### 2. **Compute Loss Dtype Consistency** (Lines 1959-1970)
- **Problem**: `batch`, `sigma`, and `dsigma` tensors had inconsistent dtypes
- **Solution**: Added dtype consistency checks and conversions
- **Code Added**:
```python
# Ensure consistent dtypes for all tensors
model_dtype = next(self.model.parameters()).dtype
if batch.dtype != model_dtype:
    batch = batch.to(dtype=model_dtype)
if sigma.dtype != model_dtype:
    sigma = sigma.to(dtype=model_dtype)
if dsigma.dtype != model_dtype:
    dsigma = dsigma.to(dtype=model_dtype)
```

### 3. **Training Step Dtype Consistency** (Lines 2050-2063)
- **Problem**: Input batches might have different dtypes than model parameters
- **Solution**: Added dtype conversion in `train_step()`
- **Code Added**:
```python
# Ensure consistent dtype - convert to model's dtype if needed
model_dtype = next(self.model.parameters()).dtype
if batch.dtype != model_dtype:
    print(f"🔧 Converting batch dtype from {batch.dtype} to {model_dtype}")
    batch = batch.to(dtype=model_dtype)
```

### 4. **Simple Generation Dtype Fix** (Lines 1221-1252)
- **Problem**: Generation method was passing raw timesteps instead of sigma values
- **Solution**: 
  - Use noise scheduler to convert timesteps to sigma values
  - Ensure all tensors have consistent dtypes
  - Pass `sigma` to model instead of raw timestep `t`

### 5. **DDP Barrier Timeout Fix** (Lines 2324-2329)
- **Problem**: `torch.distributed.barrier(timeout=30)` not supported in all PyTorch versions
- **Solution**: Removed timeout parameter from barrier call

### 6. **Optional Dependencies** (Lines 1-22)
- **Problem**: Script failed when MPI or Intel extensions weren't available
- **Solution**: Made MPI and Intel extensions optional imports with graceful fallbacks

## 🧪 **Testing**

Created comprehensive dtype consistency tests (`test_dtype_simple.py`) that verify:
- ✅ Dtype conversion logic works correctly
- ✅ Matrix multiplication with mixed dtypes can be fixed
- ✅ Model forward passes work with dtype consistency

All tests pass, confirming the fixes should resolve the dtype mismatch errors.

## 🚀 **Expected Results**

After these fixes, the Aurora DDP training should:
1. **No longer fail** with "mat1 and mat2 must have the same dtype" errors
2. **Handle mixed precision** correctly with IPEX optimization
3. **Generate sequences** successfully during training
4. **Complete training loops** without dtype-related crashes

## 📋 **Usage**

The fixed training script can now be run on Aurora with:
```bash
mpirun -n 4 python protlig_dd/training/run_train_uniref_ddp_aurora.py \
    --config config.yaml \
    --data data.pt \
    --work_dir ./output
```

The script will automatically:
- Detect available hardware (XPU/CUDA/CPU)
- Apply appropriate optimizations (IPEX for XPU)
- Handle dtype consistency throughout training
- Gracefully fallback when dependencies are missing

## 🔍 **Key Changes Summary**

| Component | Issue | Fix |
|-----------|-------|-----|
| IPEX Optimization | Forced bfloat16 causing mismatches | Let IPEX handle mixed precision automatically |
| Compute Loss | Inconsistent tensor dtypes | Added dtype consistency checks |
| Training Step | Input/model dtype mismatch | Convert inputs to model dtype |
| Generation | Wrong tensor types to model | Use sigma values with correct dtypes |
| DDP Cleanup | Barrier timeout not supported | Removed timeout parameter |
| Dependencies | Hard MPI/Intel requirements | Made imports optional with fallbacks |

These fixes ensure robust dtype handling across all training components while maintaining compatibility with different hardware configurations.
