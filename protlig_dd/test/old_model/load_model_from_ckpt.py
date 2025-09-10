# load_model_from_ckpt.py

import os
import torch
from collections import OrderedDict

from protlig_dd.model import SEDD
from protlig_dd.model.ema import ExponentialMovingAverage
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
from protlig_dd.training.run_train_plinder import Config


def load_model_from_ckpt(cfg_path: str, ckpt_path: str, device: torch.device):
    """
    Loads a trained SEDD model and its essential components from a checkpoint.

    This function is designed to exactly replicate the logic from a procedural
    loading script. It correctly handles checkpoints saved using DistributedDataParallel (DDP)
    by stripping the 'module.' prefix from the state dictionary keys.

    Args:
        cfg_path (str): Path to the training configuration YAML file. This is
                        required to build the model with the correct architecture.
        ckpt_path (str): Path to the model checkpoint (.pth) file.
        device (torch.device): The device to load the model onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        tuple: A tuple containing all the critical components for inference or further analysis:
            - model (SEDD): The main neural network, with weights loaded and set to eval mode.
            - ema_handler (ExponentialMovingAverage): The EMA object, with its averaged weights loaded.
            - noise_scheduler (torch.nn.Module): The noise scheduler used during training.
            - token_graph (object): The token graph defining token relationships.
            - config (Config): The full configuration object used for training.
            - checkpoint_state (dict): The raw, unprocessed state dictionary loaded from the file.
    """
    # --- Step 1: Validate paths and load configuration ---
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    config = Config(yamlfile=cfg_path)

    # --- Step 2: Instantiate all model components ---
    # Create empty shells with the correct architecture defined by the config
    model = SEDD(config).to(device)
    ema_handler = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
    noise_scheduler = noise_lib.get_noise(config).to(device)
    token_graph = graph_lib.get_graph(config, device)

    # --- Step 3: Load the checkpoint file from disk ---
    checkpoint_state = torch.load(ckpt_path, map_location=device)

    # --- Step 4: Prepare and load the weights (handle DDP) ---
    # Helper function to remove the 'module.' prefix from DDP-saved models
    def unwrap_state_dict(state_dict):
        unwrapped = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            unwrapped[name] = v
        return unwrapped

    # Load the unwrapped weights into the model
    model.load_state_dict(unwrap_state_dict(checkpoint_state['model']))

    # Load the unwrapped weights into the EMA handler
    if 'ema' in checkpoint_state and checkpoint_state['ema'] is not None:
        ema_handler.load_state_dict(unwrap_state_dict(checkpoint_state['ema']))

    # Load the state for the noise scheduler (if it's learnable)
    if 'noise' in checkpoint_state and checkpoint_state['noise'] is not None:
        noise_scheduler.load_state_dict(checkpoint_state['noise'])

    # --- Step 5: Finalize model for inference ---
    model.eval()

    return model, ema_handler, noise_scheduler, token_graph, config, checkpoint_state


# ==============================================================================
#  Example Usage (this part runs if you execute the script directly)
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    CFG_PATH = '/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/configs/config.yaml'
    CKPT_PATH = '/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/checkpoints/checkpoint_9.pth'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("--- Starting Model Loading Process ---")
    print(f"Using device: {DEVICE}")

    # Call the function to load everything, giving each returned item a meaningful name
    model, \
    ema_handler, \
    noise_scheduler, \
    token_graph, \
    config, \
    checkpoint_state = load_model_from_ckpt(
        cfg_path=CFG_PATH,
        ckpt_path=CKPT_PATH,
        device=DEVICE
    )

    # --- Verification ---
    loaded_step = checkpoint_state.get('step', 'N/A')
    print("\n--- ✅ SUCCESS! ---")
    print(f"Model and components loaded successfully from training step: {loaded_step}")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    
    # Example of what to do next for inference:
    # For the best results, copy the averaged EMA weights into the model
    ema_handler.copy_to(model.parameters())
    print("\nNote: EMA weights have been copied to the model for optimal inference quality.")