import abc
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def get_noise(config):
    if config.noise.type == "geometric":
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    elif config.noise.type == "loglinear":
        return LogLinearNoise(eps=config.noise.eps)
    elif config.noise.type == "cosine":
        return CosineNoise(
                    sigma_min=config.noise.sigma_min,
                    sigma_max=config.noise.sigma_max)
    else:
        raise ValueError(f"{config.noise.type} is not a valid noise")


def sample_timesteps_curriculum(batch_size, device, training_step, preschool_time=5000,
                               curriculum_type="exponential", bias_strength=2.0):
    """
    Sample timesteps with curriculum learning that biases toward lower noise levels early in training.

    Instead of modifying sigma values, this modifies the probability distribution of timestep sampling.
    Early in training, we bias toward lower t values (which correspond to lower noise).

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensors on
        training_step: Current training step
        preschool_time: Steps to reach uniform sampling
        curriculum_type: "linear", "exponential", or "cosine"
        bias_strength: How strongly to bias toward low noise (higher = stronger bias)

    Returns:
        t: Sampled timesteps [batch_size] in range [1e-3, 1-1e-3]
    """
    # Calculate curriculum progress (0 = start, 1 = full curriculum)
    if training_step >= preschool_time:
        # After preschool, use uniform sampling
        return torch.rand(batch_size, device=device) * (1 - 1e-3) + 1e-3

    progress = float(training_step) / float(preschool_time)

    # Calculate bias factor based on curriculum type
    if curriculum_type == "exponential":
        # Exponential decay of bias: strong bias early, rapid transition to uniform
        bias_factor = torch.exp(torch.tensor(-bias_strength * progress))
    elif curriculum_type == "cosine":
        # Cosine decay: smooth transition from biased to uniform
        bias_factor = 0.5 * (1 + torch.cos(torch.tensor(math.pi * progress)))
    else:  # linear
        # Linear decay: steady transition from biased to uniform
        bias_factor = 1.0 - progress

    bias_factor = bias_factor.item()

    # Sample base timesteps uniformly
    t_uniform = torch.rand(batch_size, device=device)

    # Apply bias toward lower values using power transformation
    #
    # KEY INSIGHT: For x ∈ [0,1], x^p where p > 1 pushes values toward 0
    # Examples:
    #   - 0.5^1 = 0.5 (no change)
    #   - 0.5^2 = 0.25 (biased toward 0)
    #   - 0.5^3 = 0.125 (more bias toward 0)
    #
    # Our formula:
    #   - bias_factor=1 (early training) → power=1+1*2=3 → strong bias toward low t
    #   - bias_factor=0 (late training) → power=1+0*2=1 → uniform (no bias)
    power = 1.0 + bias_factor * bias_strength
    t_biased = torch.pow(t_uniform, power)

    # Scale to proper range [1e-3, 1-1e-3]
    t = t_biased * (1 - 2e-3) + 1e-3

    return t


def get_curriculum_stats(t_samples, training_step, preschool_time):
    """
    Get statistics about curriculum sampling for logging/debugging.

    Args:
        t_samples: Sampled timesteps [batch_size]
        training_step: Current training step
        preschool_time: Total preschool steps

    Returns:
        dict: Statistics about the sampling
    """
    progress = min(1.0, float(training_step) / float(preschool_time))

    return {
        'curriculum_progress': progress,
        'mean_timestep': t_samples.mean().item(),
        'std_timestep': t_samples.std().item(),
        'min_timestep': t_samples.min().item(),
        'max_timestep': t_samples.max().item(),
        'low_noise_fraction': (t_samples < 0.3).float().mean().item(),  # Fraction with t < 0.3
        'high_noise_fraction': (t_samples > 0.7).float().mean().item(),  # Fraction with t > 0.7
    }


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """
    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass


class GeometricNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):#learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)


class CosineNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1.0, T=1.0, learnable=False):
        super().__init__()
        sigmas = torch.tensor([sigma_min, sigma_max], dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas) if learnable else sigmas
        self.T = T

    def total_noise(self, t):
        # σ(t) = σ_max + 0.5(σ_min-σ_max)[1 + cos(pi t/T)]
        s0, s1 = self.sigmas[0], self.sigmas[1]  # min, max
        return s1 + 0.5*(s0 - s1)*(1 + torch.cos(math.pi * t / self.T))

    def rate_noise(self, t):
        # dσ/dt = (π/(2T))(σ_max-σ_min) sin(π t/T)
        s0, s1 = self.sigmas[0], self.sigmas[1]
        return (math.pi/(2*self.T))*(s1 - s0)*torch.sin(math.pi * t / self.T)

