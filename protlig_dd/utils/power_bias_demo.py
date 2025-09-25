"""
Demonstration of how power transformation biases toward lower values.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_power_bias():
    """Show how different power values bias uniform random samples toward 0."""
    
    # Generate uniform samples
    torch.manual_seed(42)
    uniform_samples = torch.rand(10000)
    
    # Different power values
    powers = [1.0, 1.5, 2.0, 3.0, 4.0]
    
    print("POWER TRANSFORMATION BIAS DEMONSTRATION")
    print("="*60)
    print("Original uniform samples: mean ≈ 0.5, evenly distributed")
    print()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, power in enumerate(powers):
        # Apply power transformation
        biased_samples = torch.pow(uniform_samples, power)
        
        # Calculate statistics
        mean_val = biased_samples.mean().item()
        std_val = biased_samples.std().item()
        low_frac = (biased_samples < 0.3).float().mean().item()
        high_frac = (biased_samples > 0.7).float().mean().item()
        
        print(f"Power = {power:.1f}:")
        print(f"  Mean: {mean_val:.3f} (lower = more bias toward 0)")
        print(f"  Std:  {std_val:.3f}")
        print(f"  < 0.3: {low_frac*100:5.1f}% (higher = more low values)")
        print(f"  > 0.7: {high_frac*100:5.1f}% (lower = fewer high values)")
        print()
        
        # Plot histogram
        if i < len(axes):
            axes[i].hist(biased_samples.numpy(), bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'Power = {power:.1f}\nMean = {mean_val:.3f}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(powers) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('power_bias_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


def curriculum_progression_example():
    """Show how the curriculum progresses over training steps."""
    
    print("\nCURRICULUM PROGRESSION EXAMPLE")
    print("="*60)
    print("bias_strength = 2.0, preschool_time = 3000")
    print()
    
    preschool_time = 3000
    bias_strength = 2.0
    training_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000]
    
    torch.manual_seed(42)
    
    for step in training_steps:
        # Calculate curriculum progress
        progress = float(step) / float(preschool_time)
        
        # Calculate bias factor (exponential decay)
        bias_factor = torch.exp(torch.tensor(-bias_strength * progress)).item()
        
        # Calculate power
        power = 1.0 + bias_factor * bias_strength
        
        # Sample some timesteps
        uniform_samples = torch.rand(1000)
        biased_samples = torch.pow(uniform_samples, power)
        
        # Statistics
        mean_t = biased_samples.mean().item()
        low_frac = (biased_samples < 0.3).float().mean().item()
        high_frac = (biased_samples > 0.7).float().mean().item()
        
        print(f"Step {step:4d}: progress={progress:.2f}, bias_factor={bias_factor:.3f}, "
              f"power={power:.2f} → mean_t={mean_t:.3f}, "
              f"low%={low_frac*100:4.1f}, high%={high_frac*100:4.1f}")


def compare_with_curriculum():
    """Compare uniform vs curriculum sampling at different training stages."""
    
    print("\nCOMPARISON: UNIFORM vs CURRICULUM SAMPLING")
    print("="*60)
    
    from protlig_dd.processing.noise_lib import sample_timesteps_curriculum
    
    torch.manual_seed(42)
    batch_size = 5000
    device = 'cpu'
    preschool_time = 3000
    
    training_steps = [0, 1000, 2000, 3000]
    
    for step in training_steps:
        print(f"\nTraining Step {step}:")
        print("-" * 30)
        
        # Uniform sampling
        t_uniform = torch.rand(batch_size, device=device) * (1 - 1e-3) + 1e-3
        
        # Curriculum sampling
        t_curriculum = sample_timesteps_curriculum(
            batch_size, device, step, preschool_time, 
            curriculum_type="exponential", bias_strength=2.0
        )
        
        # Compare statistics
        print(f"Uniform:    mean={t_uniform.mean():.3f}, "
              f"low%={(t_uniform < 0.3).float().mean()*100:4.1f}, "
              f"high%={(t_uniform > 0.7).float().mean()*100:4.1f}")
        
        print(f"Curriculum: mean={t_curriculum.mean():.3f}, "
              f"low%={(t_curriculum < 0.3).float().mean()*100:4.1f}, "
              f"high%={(t_curriculum > 0.7).float().mean()*100:4.1f}")
        
        improvement = (t_curriculum < 0.3).float().mean() / (t_uniform < 0.3).float().mean()
        print(f"→ {improvement:.1f}x more low-noise samples with curriculum")


if __name__ == "__main__":
    print("🎯 Understanding Power Transformation Bias\n")
    
    demonstrate_power_bias()
    curriculum_progression_example()
    compare_with_curriculum()
    
    print("\n✅ Demo complete! Check 'power_bias_demo.png' for visualization.")
    print("\n💡 Key Takeaway: Higher power → values pushed toward 0 → more low-noise training")
