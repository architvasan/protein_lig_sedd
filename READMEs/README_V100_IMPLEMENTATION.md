# 🧬 SEDD V100 Cross-Platform Implementation

## 🎯 **Overview**

This branch contains a **V100-compatible, cross-platform implementation** of the SEDD (Score Entropy Discrete Diffusion) model for protein-ligand generation. The implementation resolves multiple compatibility issues and provides stable training across different hardware platforms.

## ✨ **Key Features**

### 🔧 **Cross-Platform Compatibility**
- **✅ V100 GPUs**: No flash attention dependencies
- **✅ Apple Silicon (M1/M2/M3)**: MPS support
- **✅ CPU Training**: Full compatibility for development
- **✅ Auto-Detection**: Automatically selects best available device

### 🏗️ **Architecture Improvements**
- **Dual-Track Transformer**: Separate processing for protein and ligand
- **Cross-Attention Mechanism**: Bidirectional protein-ligand interactions
- **Device-Compatible Operations**: Fallback systems for different hardware
- **Stable Training**: Fixed curriculum learning and loss computation

### 📊 **Model Specifications**
- **Parameters**: 66,724,096 (66M+)
- **Hidden Size**: 768 dimensions
- **Attention Heads**: 12 multi-head attention
- **Transformer Layers**: 8 blocks
- **Vocabularies**: 37 protein tokens, 2,365 ligand tokens

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Clone and navigate to the repository
git clone <repository-url>
cd protein_lig_sedd
git checkout sedd-v100-cross-platform-implementation

# Install dependencies (if needed)
pip install torch torchvision torchaudio
pip install wandb einops omegaconf pyyaml
```

### **2. Download Data**
```bash
# Download UniRef50 dataset (10k sequences for testing)
./download_uniref50_data.sh
```

### **3. Setup Wandb (Optional)**
```bash
# Configure experiment tracking
python setup_wandb.py
```

### **4. Start Training**
```bash
# Auto-detect best device
./run_train_uniref50_optimized.sh

# Or specify device explicitly
./run_train_uniref50_optimized.sh --cpu    # CPU training
./run_train_uniref50_optimized.sh --cuda   # CUDA training
./run_train_uniref50_optimized.sh --mps    # Apple Silicon
```

## 📁 **Key Files**

### **Core Implementation**
- `protlig_dd/model/transformer_v100.py` - V100-compatible SEDD model
- `protlig_dd/training/run_train_uniref50_optimized.py` - Cross-platform trainer
- `protlig_dd/processing/graph_lib.py` - Fixed curriculum learning

### **Configuration**
- `configs/config_uniref50_stable.yaml` - Stable training configuration
- `configs/config_uniref50_optimized.yaml` - Original configuration
- `wandb_config.yaml` - Experiment tracking setup

### **Scripts**
- `run_train_uniref50_optimized.sh` - Training launcher
- `download_uniref50_data.sh` - Data download script
- `setup_wandb.py` - Wandb configuration

### **Documentation**
- `SEDD_Architecture_Analysis.md` - Detailed architecture explanation
- `DEVICE_COMPATIBILITY_FIX_SUMMARY.md` - Device compatibility fixes
- `TRAINING_LOSS_ANALYSIS_SUMMARY.md` - Training stability improvements

## 🔧 **Technical Improvements**

### **1. Device Compatibility**
- **Graph Fuser Fix**: Resolved JIT compilation issues on non-CUDA devices
- **Fallback Operations**: Device-compatible alternatives to fused operations
- **Memory Management**: Platform-specific optimizations

### **2. Training Stability**
- **Curriculum Learning**: Fixed tensor type errors in exponential curriculum
- **Loss Computation**: Stable loss weighting with dsigma
- **Learning Rate**: Optimized scheduling with warmup and cosine decay

### **3. Cross-Attention Architecture**
- **Bidirectional Flow**: Protein ↔ Ligand information exchange
- **Multi-Head Attention**: 12 heads for rich representations
- **Timestep Conditioning**: AdaLN modulation for diffusion control

## 📊 **Training Configurations**

### **Stable Configuration** (`config_uniref50_stable.yaml`)
```yaml
# Optimized for stability and convergence
optim:
  lr: 5e-5              # Reduced learning rate
  warmup: 10000         # Longer warmup
  grad_clip: 0.5        # Gentler clipping

noise:
  sigma_max: 0.5        # Reduced noise range

curriculum:
  preschool_time: 20000 # Slower ramp-up
  difficulty_ramp: linear
```

### **Expected Training Behavior**
```
Phase 1 (0-20k steps):   Curriculum ramp-up, loss increases gradually
Phase 2 (20k-50k steps): Full difficulty, loss stabilizes
Phase 3 (50k+ steps):    Model improvement, loss decreases
```

## 🎯 **Use Cases**

### **Research Applications**
- **Drug Discovery**: Generate ligands for target proteins
- **Protein Design**: Design proteins for specific ligands
- **Binding Analysis**: Study protein-ligand interactions
- **Molecular Optimization**: Improve binding affinity

### **Development Scenarios**
- **Local Development**: CPU training on laptops
- **Prototyping**: Quick experiments with small datasets
- **Cross-Platform**: Consistent results across different hardware
- **Educational**: Learning diffusion models for biology

## 🔍 **Architecture Highlights**

### **Cross-Attention Mechanism**
```python
# Protein → Ligand Attention
Query: Protein hidden states [B × L_p × 768]
Key/Value: Ligand hidden states [B × L_l × 768]
Output: Updated protein representations

# Ligand → Protein Attention  
Query: Ligand hidden states [B × L_l × 768]
Key/Value: Protein hidden states [B × L_p × 768]
Output: Updated ligand representations
```

### **Training Modes**
- **Joint Training**: Both protein and ligand with cross-attention
- **Protein-Only**: Standard protein sequence modeling
- **Ligand-Only**: Standard ligand generation
- **Conditional**: Generate one modality given the other

## 🐛 **Issues Resolved**

### **✅ Fixed Issues**
1. **Flash Attention Dependency** → V100-compatible model
2. **Configuration Errors** → Robust config handling
3. **Graph Fuser Errors** → Device-compatible operations
4. **Curriculum Learning** → Fixed tensor type issues
5. **Training Loss Increase** → Stable curriculum progression
6. **Cross-Platform Support** → Universal device compatibility

### **🔧 Improvements Made**
- **Memory Optimization**: Efficient attention computation
- **Error Handling**: Graceful fallbacks and clear messages
- **Documentation**: Comprehensive guides and analysis
- **Testing**: Extensive compatibility testing

## 📈 **Performance**

### **Model Capacity**
- **66M+ Parameters**: Large-scale protein-ligand modeling
- **Cross-Modal Learning**: Rich protein-ligand interactions
- **Scalable Architecture**: Efficient attention mechanisms

### **Training Efficiency**
- **Stable Convergence**: Reliable training dynamics
- **Cross-Platform**: Consistent performance across devices
- **Memory Efficient**: Optimized for various hardware constraints

## 🤝 **Contributing**

This implementation provides a solid foundation for:
- **Research Extensions**: Adding new features or architectures
- **Performance Optimization**: Further speed/memory improvements
- **Application Development**: Building on the stable base
- **Educational Use**: Learning advanced ML concepts

## 📚 **References**

- **SEDD Paper**: Score Entropy Discrete Diffusion for sequence generation
- **Cross-Attention**: Transformer architectures for multi-modal learning
- **Protein-Ligand Modeling**: Computational drug discovery methods

---

**This implementation represents a significant advancement in making SEDD models accessible across different hardware platforms while maintaining research-grade performance and stability.** 🚀✨
