import os
import torch
import yaml
from dataclasses import dataclass
from protlig_dd.model.transformers_prot_lig import ProteinLigandSharedDiffusion
from protlig_dd.utils.utils import Config


def load_fresh_model(config_path: str, device: torch.device):
    """
    Load a brand new, untrained model instance according to the configuration file.

    Args:
        config_path (str): Path to the model YAML configuration file.
        device (torch.device): Device to load the model onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        torch.nn.Module: A brand new ProteinLigandSharedDiffusion model instance.
    """

    config = Config(yamlfile=config_path)
    model = ProteinLigandSharedDiffusion(config).to(device)
    
    return model, config


# ==============================================================================
#                                  Main Program
# ==============================================================================
if __name__ == '__main__':
    
    CONFIG_FILE_PATH = 'config.yaml' 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device in use: {DEVICE}")

    # We only care about the model itself, so use '_' to ignore the returned config object
    untrained_model, model_config = load_fresh_model(
        config_path=CONFIG_FILE_PATH,
        device=DEVICE
    )

    print(f"Model type: {type(untrained_model)}")
    print(f"Model loaded on device: {next(untrained_model.parameters()).device}")

    # Calculate and print the number of model parameters
    num_params = sum(p.numel() for p in untrained_model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {num_params / 1_000_000:.2f} M")