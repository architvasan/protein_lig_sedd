#!/usr/bin/env python3
"""
Calculate parameters for the current SEDD model and suggest scaling strategies.
"""

def calculate_sedd_params(hidden_size, n_blocks, n_heads, cond_dim, max_len=512):
    """Calculate parameters for SEDD transformer model."""
    
    vocab_size = 25  # 20 AAs + special tokens
    
    print(f"🧬 SEDD MODEL PARAMETER CALCULATION")
    print("=" * 50)
    print(f"Hidden size: {hidden_size}")
    print(f"Blocks: {n_blocks}")
    print(f"Heads: {n_heads}")
    print(f"Conditional dim: {cond_dim}")
    print(f"Max length: {max_len}")
    print()
    
    total_params = 0
    
    # 1. Embeddings
    print("📊 EMBEDDINGS")
    print("-" * 15)
    
    # Token embedding
    token_embed = vocab_size * hidden_size
    total_params += token_embed
    print(f"Token embedding: {token_embed:,}")
    
    # Position embedding
    pos_embed = max_len * hidden_size
    total_params += pos_embed
    print(f"Position embedding: {pos_embed:,}")
    
    # 2. Transformer blocks
    print(f"\n🔄 TRANSFORMER BLOCKS (×{n_blocks})")
    print("-" * 30)
    
    # Per block calculation
    block_params = 0
    
    # Multi-head attention
    qkv_params = 3 * hidden_size * hidden_size  # Q, K, V projections
    attn_out_params = hidden_size * hidden_size  # Output projection
    attn_total = qkv_params + attn_out_params
    block_params += attn_total
    
    # Feed-forward (typically 4x hidden size)
    ff_intermediate = 4 * hidden_size
    ff_up = hidden_size * ff_intermediate
    ff_down = ff_intermediate * hidden_size
    ff_total = ff_up + ff_down
    block_params += ff_total
    
    # Layer norms (2 per block)
    ln_params = 4 * hidden_size  # 2 layer norms × 2 params each (weight + bias)
    block_params += ln_params
    
    print(f"Per block:")
    print(f"  • Attention: {attn_total:,}")
    print(f"  • Feed-forward: {ff_total:,}")
    print(f"  • Layer norms: {ln_params:,}")
    print(f"  • Total per block: {block_params:,}")
    
    total_transformer = block_params * n_blocks
    total_params += total_transformer
    
    # 3. SEDD-specific conditioning
    print(f"\n🎯 SEDD CONDITIONING")
    print("-" * 20)
    
    # Time embedding for diffusion
    time_embed = cond_dim * 2  # Sinusoidal + projection
    total_params += time_embed
    
    # Condition projection
    cond_proj = cond_dim * hidden_size
    total_params += cond_proj
    
    print(f"Time embedding: {time_embed:,}")
    print(f"Condition projection: {cond_proj:,}")
    
    # 4. Output layers
    print(f"\n📤 OUTPUT")
    print("-" * 10)
    
    # Final layer norm
    final_ln = 2 * hidden_size
    total_params += final_ln
    
    # Output projection
    output_proj = hidden_size * vocab_size
    total_params += output_proj
    
    print(f"Final layer norm: {final_ln:,}")
    print(f"Output projection: {output_proj:,}")
    
    # Summary
    print(f"\n🎯 TOTAL PARAMETERS")
    print("=" * 25)
    print(f"Embeddings: {token_embed + pos_embed:,}")
    print(f"Transformer: {total_transformer:,}")
    print(f"Conditioning: {time_embed + cond_proj:,}")
    print(f"Output: {final_ln + output_proj:,}")
    print("-" * 25)
    print(f"TOTAL: {total_params:,}")
    
    # Readable format
    if total_params >= 1_000_000_000:
        readable = f"{total_params / 1_000_000_000:.2f}B"
    elif total_params >= 1_000_000:
        readable = f"{total_params / 1_000_000:.1f}M"
    else:
        readable = f"{total_params / 1_000:.1f}K"
    
    print(f"TOTAL: ~{readable}")
    
    return total_params

def suggest_scaling_strategies(current_params):
    """Suggest ways to scale up the model."""
    
    print(f"\n🚀 SCALING STRATEGIES FOR 36M DATASET")
    print("=" * 40)
    
    print("📈 CURRENT MODEL ANALYSIS:")
    print(f"• Your model: ~{current_params/1_000_000:.0f}M parameters")
    print("• Dataset: 36M samples (360x larger than current)")
    print("• Likely underparameterized for this dataset size")
    print()
    
    print("🎯 SCALING OPTIONS:")
    print()
    
    # Option 1: Deeper
    print("1️⃣ GO DEEPER (More Blocks)")
    print("   Current: 12 blocks → Try: 16-20 blocks")
    print("   Pros: Better representation, more parameters")
    print("   Cons: Slower training, more memory")
    print("   Param increase: ~33-67%")
    print()
    
    # Option 2: Wider
    print("2️⃣ GO WIDER (Larger Hidden Size)")
    print("   Current: 768 → Try: 1024 or 1280")
    print("   Pros: More capacity per layer")
    print("   Cons: You said 1024 worked worse(?)")
    print("   Param increase: ~78% (1024) or ~180% (1280)")
    print()
    
    # Option 3: More heads
    print("3️⃣ MORE ATTENTION HEADS")
    print("   Current: 16 heads → Try: 20-24 heads")
    print("   Pros: More diverse attention patterns")
    print("   Cons: Diminishing returns, memory")
    print("   Param increase: Minimal")
    print()
    
    # Option 4: Hybrid
    print("4️⃣ HYBRID SCALING")
    print("   Try: 16 blocks × 768 hidden × 20 heads")
    print("   Or: 14 blocks × 896 hidden × 16 heads")
    print("   Pros: Balanced scaling")
    print("   Cons: Need to tune carefully")
    print()
    
    print("💡 RECOMMENDATIONS:")
    print("• Start with deeper (16 blocks) - usually most effective")
    print("• If 1024 hidden didn't work, try 896 or 960 (between 768-1024)")
    print("• Consider why 1024 failed - overfitting? optimization issues?")
    print("• With 36M samples, you can likely support 200-400M parameters")

if __name__ == "__main__":
    # Your ACTUAL model parameters
    print("🔍 ANALYZING YOUR ACTUAL MODEL")
    print("From your config:")
    print("  hidden_size: 1024")
    print("  n_blocks_prot: 20")
    print("  n_heads: 16")
    print("  cond_dim: 512")
    print("  GPU memory: 7GB")
    print()

    current_params = calculate_sedd_params(
        hidden_size=1024,
        n_blocks=20,
        n_heads=16,
        cond_dim=512
    )
    
    # Scaling suggestions
    suggest_scaling_strategies(current_params)
    
    print(f"\n📊 LARGE MODEL CONFIGURATIONS (~500M PARAMS)")
    print("=" * 50)

    large_configs = [
        ("Current", 768, 12, 16, 256),
        ("Large-1", 1536, 20, 24, 512),
        ("Large-2", 1280, 24, 20, 512),
        ("Large-3", 1792, 18, 28, 512),
        ("Large-4", 1408, 22, 22, 512),
        ("Mega", 2048, 16, 32, 512)
    ]

    print(f"{'Config':<12} {'Hidden':<8} {'Blocks':<8} {'Heads':<8} {'Cond':<6} {'Params':<12} {'Memory':<10}")
    print("-" * 70)

    for name, hidden, blocks, heads, cond in large_configs:
        params = calculate_sedd_params(hidden, blocks, heads, cond, max_len=512)
        readable = f"{params/1_000_000:.0f}M"
        memory_gb = f"{(params * 2) / (1024**3):.1f}GB"  # FP16 weights
        print(f"{name:<12} {hidden:<8} {blocks:<8} {heads:<8} {cond:<6} {readable:<12} {memory_gb:<10}")
        print()  # Space between calculations

    print(f"\n🎯 RECOMMENDED 500M CONFIGURATIONS")
    print("=" * 35)
    print("For 36M samples, these ratios work well:")
    print("• Large-1: 1536×20×24 = ~520M params (14:1 sample:param ratio)")
    print("• Large-2: 1280×24×20 = ~480M params (15:1 sample:param ratio)")
    print("• Large-4: 1408×22×22 = ~500M params (14.4:1 sample:param ratio)")
    print()
    print("💡 TRAINING CONSIDERATIONS:")
    print("• Need 8+ GPUs for efficient training")
    print("• Reduce learning rate to ~5e-6 to 1e-5")
    print("• Increase warmup to 15K-20K steps")
    print("• Use gradient checkpointing to save memory")
    print("• Consider mixed precision (FP16) training")
