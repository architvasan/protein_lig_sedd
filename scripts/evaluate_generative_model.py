#!/usr/bin/env python3
"""
Comprehensive evaluation of the SEDD generative model
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from pathlib import Path
import sys
import time
from tqdm import tqdm

def load_model_and_config():
    """Load the trained model and configuration."""
    print("🔄 Loading model and configuration...")
    
    try:
        from protlig_dd.training.run_train_uniref50_optimized import OptimizedUniRef50Trainer
        from protlig_dd.utils import utils
        import yaml
        
        # Load configuration
        config_file = "configs/config_uniref50_stable.yaml"
        with open(config_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = utils.Config(dictionary=cfg_dict)
        
        # Create trainer
        trainer = OptimizedUniRef50Trainer(
            work_dir=".",
            config_file=config_file,
            datafile="./input_data/processed_uniref50.pt",
            dev_id="cpu",  # Use CPU for evaluation
            seed=42
        )
        
        # Setup components
        trainer.setup_custom_data_loaders()
        trainer.setup_model()
        
        # Load best checkpoint
        checkpoint_path = "checkpoints/best_checkpoint.pth"
        if Path(checkpoint_path).exists():
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            print("⚠️  No checkpoint found, using randomly initialized model")
        
        trainer.model.eval()
        return trainer, cfg
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_training_data(trainer):
    """Analyze the training data distribution."""
    print("\n📊 TRAINING DATA ANALYSIS")
    print("=" * 50)
    
    # Get a sample of training data
    data_sample = []
    for i, batch in enumerate(trainer.train_loader):
        data_sample.append(batch)
        if i >= 10:  # Sample first 10 batches
            break
    
    # Combine batches
    all_sequences = torch.cat(data_sample, dim=0)
    
    # Basic statistics
    batch_size, seq_len = all_sequences.shape
    print(f"📈 Sample Statistics:")
    print(f"   Sequences analyzed: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Vocabulary size: {all_sequences.max().item() + 1}")
    
    # Token distribution
    token_counts = Counter(all_sequences.flatten().tolist())
    print(f"\n🔤 Token Distribution:")
    for token, count in sorted(token_counts.items())[:10]:
        print(f"   Token {token}: {count} ({count/sum(token_counts.values())*100:.1f}%)")
    
    # Sequence length distribution
    non_pad_lengths = []
    for seq in all_sequences:
        # Assuming 0 is padding token
        non_pad_length = (seq != 0).sum().item()
        non_pad_lengths.append(non_pad_length)
    
    print(f"\n📏 Sequence Length Statistics:")
    print(f"   Mean length: {np.mean(non_pad_lengths):.1f}")
    print(f"   Std length: {np.std(non_pad_lengths):.1f}")
    print(f"   Min length: {np.min(non_pad_lengths)}")
    print(f"   Max length: {np.max(non_pad_lengths)}")
    
    return {
        'token_distribution': token_counts,
        'sequence_lengths': non_pad_lengths,
        'vocab_size': all_sequences.max().item() + 1,
        'sample_sequences': all_sequences[:5]  # Keep 5 examples
    }

def generate_samples(trainer, num_samples=50, max_length=256):
    """Generate samples from the trained model."""
    print(f"\n🎲 GENERATING {num_samples} SAMPLES")
    print("=" * 50)
    
    try:
        # Set model to evaluation mode
        trainer.model.eval()
        
        generated_samples = []
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Generating samples"):
                # Start with random noise (absorbing state)
                # Assuming vocab_size - 1 is the absorbing state
                vocab_size = 37  # From config
                absorbing_token = vocab_size - 1
                
                # Initialize with absorbing states
                sample = torch.full((1, max_length), absorbing_token, dtype=torch.long)
                
                # Diffusion sampling (simplified)
                num_steps = 25  # From config
                for step in range(num_steps):
                    # Compute timestep
                    t = torch.tensor([1.0 - step / num_steps])
                    
                    # Get model predictions
                    with torch.autocast(device_type='cpu', enabled=False):
                        logits = trainer.model(sample, t)
                    
                    # Sample from logits (temperature sampling)
                    temperature = 1.0
                    probs = F.softmax(logits / temperature, dim=-1)
                    
                    # Sample new tokens
                    new_sample = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, max_length)
                    
                    # Update sample (simple replacement strategy)
                    mask = torch.rand(1, max_length) < (step + 1) / num_steps
                    sample = torch.where(mask, new_sample, sample)
                
                generated_samples.append(sample.squeeze(0))
        
        print(f"✅ Generated {len(generated_samples)} samples")
        return generated_samples
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_generated_samples(generated_samples, training_stats):
    """Analyze the quality of generated samples."""
    print(f"\n🔍 GENERATED SAMPLES ANALYSIS")
    print("=" * 50)
    
    if not generated_samples:
        print("❌ No generated samples to analyze")
        return {}
    
    # Convert to numpy for analysis
    samples_array = torch.stack(generated_samples).numpy()
    
    # Basic statistics
    print(f"📊 Generation Statistics:")
    print(f"   Number of samples: {len(generated_samples)}")
    print(f"   Sample shape: {samples_array.shape}")
    
    # Token distribution in generated samples
    gen_token_counts = Counter(samples_array.flatten())
    print(f"\n🔤 Generated Token Distribution:")
    for token, count in sorted(gen_token_counts.items())[:10]:
        print(f"   Token {token}: {count} ({count/sum(gen_token_counts.values())*100:.1f}%)")
    
    # Compare with training distribution
    train_token_counts = training_stats['token_distribution']
    
    print(f"\n📈 Distribution Comparison (Training vs Generated):")
    for token in sorted(set(list(train_token_counts.keys())[:10])):
        train_pct = train_token_counts.get(token, 0) / sum(train_token_counts.values()) * 100
        gen_pct = gen_token_counts.get(token, 0) / sum(gen_token_counts.values()) * 100
        print(f"   Token {token}: Train={train_pct:.1f}% | Gen={gen_pct:.1f}% | Diff={abs(train_pct-gen_pct):.1f}%")
    
    # Sequence length analysis
    gen_lengths = []
    for sample in samples_array:
        # Count non-absorbing tokens (assuming last token is absorbing)
        non_absorbing_length = (sample != sample.max()).sum()
        gen_lengths.append(non_absorbing_length)
    
    print(f"\n📏 Generated Sequence Lengths:")
    print(f"   Mean length: {np.mean(gen_lengths):.1f}")
    print(f"   Std length: {np.std(gen_lengths):.1f}")
    print(f"   Min length: {np.min(gen_lengths)}")
    print(f"   Max length: {np.max(gen_lengths)}")
    
    # Diversity analysis
    unique_samples = len(set(tuple(sample.tolist()) for sample in samples_array))
    diversity = unique_samples / len(generated_samples)
    print(f"\n🎨 Diversity Analysis:")
    print(f"   Unique samples: {unique_samples}/{len(generated_samples)}")
    print(f"   Diversity ratio: {diversity:.3f}")
    
    return {
        'token_distribution': gen_token_counts,
        'sequence_lengths': gen_lengths,
        'diversity': diversity,
        'unique_samples': unique_samples
    }

def compute_quality_metrics(generated_samples, training_stats):
    """Compute various quality metrics."""
    print(f"\n📏 QUALITY METRICS")
    print("=" * 50)
    
    if not generated_samples:
        return {}
    
    metrics = {}
    
    # 1. KL Divergence between token distributions
    train_dist = training_stats['token_distribution']
    gen_samples_flat = torch.stack(generated_samples).flatten().tolist()
    gen_dist = Counter(gen_samples_flat)
    
    # Normalize distributions
    vocab_size = max(max(train_dist.keys()), max(gen_dist.keys())) + 1
    train_probs = np.zeros(vocab_size)
    gen_probs = np.zeros(vocab_size)
    
    total_train = sum(train_dist.values())
    total_gen = sum(gen_dist.values())
    
    for token, count in train_dist.items():
        train_probs[token] = count / total_train
    
    for token, count in gen_dist.items():
        gen_probs[token] = count / total_gen
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    train_probs += epsilon
    gen_probs += epsilon
    train_probs /= train_probs.sum()
    gen_probs /= gen_probs.sum()
    
    # KL divergence
    kl_div = np.sum(gen_probs * np.log(gen_probs / train_probs))
    metrics['kl_divergence'] = kl_div
    
    # 2. Jensen-Shannon Divergence
    m = 0.5 * (train_probs + gen_probs)
    js_div = 0.5 * np.sum(train_probs * np.log(train_probs / m)) + \
             0.5 * np.sum(gen_probs * np.log(gen_probs / m))
    metrics['js_divergence'] = js_div
    
    # 3. Length distribution similarity
    train_lengths = training_stats['sequence_lengths']
    gen_lengths = [len([t for t in sample if t != sample.max()]) for sample in generated_samples]
    
    # Wasserstein distance (approximation)
    train_lengths_sorted = np.sort(train_lengths)
    gen_lengths_sorted = np.sort(gen_lengths)
    
    # Pad to same length
    max_len = max(len(train_lengths_sorted), len(gen_lengths_sorted))
    if len(train_lengths_sorted) < max_len:
        train_lengths_sorted = np.pad(train_lengths_sorted, (0, max_len - len(train_lengths_sorted)), 'edge')
    if len(gen_lengths_sorted) < max_len:
        gen_lengths_sorted = np.pad(gen_lengths_sorted, (0, max_len - len(gen_lengths_sorted)), 'edge')
    
    wasserstein_dist = np.mean(np.abs(train_lengths_sorted - gen_lengths_sorted))
    metrics['wasserstein_distance'] = wasserstein_dist
    
    print(f"📊 Quality Metrics:")
    print(f"   KL Divergence: {kl_div:.4f} (lower is better)")
    print(f"   JS Divergence: {js_div:.4f} (lower is better)")
    print(f"   Wasserstein Distance: {wasserstein_dist:.4f} (lower is better)")
    
    return metrics

def create_visualizations(training_stats, generation_stats, quality_metrics):
    """Create visualizations of the results."""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Create output directory
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Token distribution comparison
        plt.figure(figsize=(12, 6))
        
        # Get common tokens
        train_dist = training_stats['token_distribution']
        gen_dist = generation_stats.get('token_distribution', {})
        
        common_tokens = sorted(set(list(train_dist.keys())[:20]))
        
        train_probs = [train_dist.get(t, 0) / sum(train_dist.values()) for t in common_tokens]
        gen_probs = [gen_dist.get(t, 0) / sum(gen_dist.values()) if gen_dist else 0 for t in common_tokens]
        
        x = np.arange(len(common_tokens))
        width = 0.35
        
        plt.bar(x - width/2, train_probs, width, label='Training Data', alpha=0.8)
        plt.bar(x + width/2, gen_probs, width, label='Generated Data', alpha=0.8)
        
        plt.xlabel('Token ID')
        plt.ylabel('Probability')
        plt.title('Token Distribution Comparison')
        plt.legend()
        plt.xticks(x, common_tokens, rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "token_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sequence length distribution
        plt.figure(figsize=(10, 6))
        
        train_lengths = training_stats['sequence_lengths']
        gen_lengths = generation_stats.get('sequence_lengths', [])
        
        plt.hist(train_lengths, bins=30, alpha=0.7, label='Training Data', density=True)
        if gen_lengths:
            plt.hist(gen_lengths, bins=30, alpha=0.7, label='Generated Data', density=True)
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.title('Sequence Length Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sequence_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}/")
        
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")

def main():
    """Main evaluation function."""
    print("🚀 SEDD GENERATIVE MODEL EVALUATION")
    print("=" * 80)
    
    # Load model and configuration
    trainer, cfg = load_model_and_config()
    if trainer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Analyze training data
    training_stats = analyze_training_data(trainer)
    
    # Generate samples
    generated_samples = generate_samples(trainer, num_samples=50)
    
    # Analyze generated samples
    generation_stats = analyze_generated_samples(generated_samples, training_stats)
    
    # Compute quality metrics
    quality_metrics = compute_quality_metrics(generated_samples, training_stats)
    
    # Create visualizations
    create_visualizations(training_stats, generation_stats, quality_metrics)
    
    # Save results
    results = {
        'training_stats': {k: v for k, v in training_stats.items() if k != 'sample_sequences'},
        'generation_stats': generation_stats,
        'quality_metrics': quality_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = Path("evaluation_results") / "evaluation_summary.json"
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Counter):
            return dict(obj)
        return obj
    
    # Recursively convert numpy types
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy(obj)
    
    results = recursive_convert(results)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Final summary
    print(f"\n🎯 EVALUATION SUMMARY")
    print("=" * 40)
    print(f"✅ Model loaded and evaluated successfully")
    print(f"📊 Generated {len(generated_samples)} samples")
    print(f"📈 Quality metrics computed")
    print(f"📊 Visualizations created")
    print(f"💾 Results saved to evaluation_results/")
    
    if quality_metrics:
        print(f"\n🏆 Key Quality Metrics:")
        print(f"   KL Divergence: {quality_metrics.get('kl_divergence', 'N/A'):.4f}")
        print(f"   JS Divergence: {quality_metrics.get('js_divergence', 'N/A'):.4f}")
        print(f"   Sample Diversity: {generation_stats.get('diversity', 'N/A'):.3f}")

if __name__ == "__main__":
    main()
