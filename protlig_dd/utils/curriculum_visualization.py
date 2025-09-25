"""
Visualization utilities for curriculum learning approaches.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from protlig_dd.processing.noise_lib import sample_timesteps_curriculum


def compare_curriculum_approaches(preschool_time=5000, batch_size=1000, device='cpu'):
    """
    Compare different curriculum approaches and visualize the differences.
    
    Args:
        preschool_time: Steps for curriculum
        batch_size: Number of samples for visualization
        device: Device to run on
    
    Returns:
        dict: Results for plotting
    """
    training_steps = [0, preschool_time//4, preschool_time//2, 3*preschool_time//4, preschool_time]
    results = {}
    
    for step in training_steps:
        # Uniform sampling (no curriculum)
        t_uniform = torch.rand(batch_size, device=device) * (1 - 1e-3) + 1e-3
        
        # Probabilistic curriculum - exponential
        t_prob_exp = sample_timesteps_curriculum(
            batch_size, device, step, preschool_time, 
            curriculum_type="exponential", bias_strength=2.0
        )
        
        # Probabilistic curriculum - cosine
        t_prob_cos = sample_timesteps_curriculum(
            batch_size, device, step, preschool_time, 
            curriculum_type="cosine", bias_strength=2.0
        )
        
        # Probabilistic curriculum - linear
        t_prob_lin = sample_timesteps_curriculum(
            batch_size, device, step, preschool_time, 
            curriculum_type="linear", bias_strength=2.0
        )
        
        results[step] = {
            'uniform': t_uniform.numpy(),
            'prob_exponential': t_prob_exp.numpy(),
            'prob_cosine': t_prob_cos.numpy(),
            'prob_linear': t_prob_lin.numpy(),
        }
    
    return results


def plot_curriculum_comparison(results, save_path=None):
    """
    Plot histograms comparing different curriculum approaches.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    training_steps = sorted(results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    labels = ['Uniform', 'Prob Exponential', 'Prob Cosine', 'Prob Linear']
    
    for i, step in enumerate(training_steps[:6]):  # Show first 6 steps
        ax = axes[i]
        
        for j, (method, data) in enumerate(results[step].items()):
            ax.hist(data, bins=50, alpha=0.6, color=colors[j], 
                   label=labels[j], density=True)
        
        ax.set_title(f'Step {step} (Progress: {step/max(training_steps):.1%})')
        ax.set_xlabel('Timestep t')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_noise_exposure(results, noise_scheduler_type='cosine', sigma_min=0.01, sigma_max=0.95):
    """
    Analyze how much exposure each approach gives to different noise levels.
    """
    from protlig_dd.processing.noise_lib import CosineNoise
    
    # Create noise scheduler
    if noise_scheduler_type == 'cosine':
        noise = CosineNoise(sigma_min=sigma_min, sigma_max=sigma_max)
    else:
        raise NotImplementedError(f"Noise type {noise_scheduler_type} not implemented")
    
    analysis = {}
    
    for step, data in results.items():
        step_analysis = {}
        
        for method, t_samples in data.items():
            t_tensor = torch.tensor(t_samples)
            sigma_samples = noise.total_noise(t_tensor)
            
            step_analysis[method] = {
                'mean_sigma': sigma_samples.mean().item(),
                'std_sigma': sigma_samples.std().item(),
                'low_noise_frac': (sigma_samples < 0.1).float().mean().item(),
                'med_noise_frac': ((sigma_samples >= 0.1) & (sigma_samples < 0.5)).float().mean().item(),
                'high_noise_frac': (sigma_samples >= 0.5).float().mean().item(),
            }
        
        analysis[step] = step_analysis
    
    return analysis


def print_curriculum_summary(analysis):
    """
    Print a summary of curriculum analysis.
    """
    print("\n" + "="*80)
    print("CURRICULUM LEARNING ANALYSIS")
    print("="*80)
    
    methods = ['uniform', 'prob_exponential', 'prob_cosine', 'prob_linear']
    method_names = ['Uniform (No Curriculum)', 'Probabilistic Exponential', 
                   'Probabilistic Cosine', 'Probabilistic Linear']
    
    for step, step_data in analysis.items():
        progress = step / max(analysis.keys()) if max(analysis.keys()) > 0 else 0
        print(f"\nStep {step} (Progress: {progress:.1%})")
        print("-" * 50)
        
        for method, name in zip(methods, method_names):
            if method in step_data:
                data = step_data[method]
                print(f"{name:25s}: "
                      f"σ_avg={data['mean_sigma']:.3f}, "
                      f"low={data['low_noise_frac']*100:4.1f}%, "
                      f"med={data['med_noise_frac']*100:4.1f}%, "
                      f"high={data['high_noise_frac']*100:4.1f}%")


if __name__ == "__main__":
    # Example usage
    print("Comparing curriculum approaches...")
    
    results = compare_curriculum_approaches(preschool_time=3000, batch_size=5000)
    analysis = analyze_noise_exposure(results)
    
    print_curriculum_summary(analysis)
    plot_curriculum_comparison(results, save_path="curriculum_comparison.png")
    
    print("\n✅ Analysis complete! Check 'curriculum_comparison.png' for visualization.")
