import abc
import torch
import torch.nn as nn
import numpy as np
import math

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

