import os
import torch
from protlig_dd.model.transformers_prot_lig import ProteinLigandSharedDiffusion
from protlig_dd.utils.utils import Config


def load_fresh_model(config_path: str, device: torch.device):
    config = Config(yamlfile=config_path)
    model = ProteinLigandSharedDiffusion(config).to(device)
    
    return model, config


if __name__ == '__main__':
    
    CONFIG_FILE_PATH = 'config.yaml' 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    untrained_model, model_config = load_fresh_model(
        config_path=CONFIG_FILE_PATH,
        device=DEVICE
    )

    num_params = sum(p.numel() for p in untrained_model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {num_params / 1_000_000:.2f} M")