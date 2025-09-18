"""
SMILES-specific loss functions for diffusion training.
"""

import torch
import torch.nn.functional as F
from protlig_dd.processing import graph_lib


def get_smiles_loss_fn(noise, graph, train=True):
    """Get loss function for SMILES-only training."""
    
    def loss_fn(model, batch):
        """Compute loss for SMILES batch."""
        # batch is already tokenized SMILES sequences
        x = batch
        
        # Sample noise level
        batch_size = x.shape[0]
        t = torch.rand(batch_size, device=x.device)
        sigma = noise(t)
        
        # Add noise to the data
        noise_sample = torch.randn_like(x.float())
        perturbed_x = graph.sample_transition(x, sigma.unsqueeze(-1), noise_sample)
        
        # Predict the score
        with torch.amp.autocast('cuda', enabled=train):
            predicted_score = model(perturbed_x, sigma)
        
        # Compute loss (score matching)
        target_score = -noise_sample / sigma.unsqueeze(-1)
        loss = F.mse_loss(predicted_score, target_score)
        
        return loss
    
    return loss_fn