import argparse
import os
import torch

from protlig_dd.model import SEDD
from protlig_dd.model.ema import ExponentialMovingAverage
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
from protlig_dd.training.run_train_plinder import Config

cfg_path = '/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/configs/config.yaml'
ckpt_path = '/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/checkpoints/checkpoint_9.pth'


# --- Step 1: Set the device ---
print("\n[Step 1] Setting device...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"-> Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, using CPU. This will be slow.")


# --- Step 2: Load the configuration from the YAML file ---
# The model's architecture is defined here. We need this to build the model object
# *before* we can load the saved weights into it.
print(f"\n[Step 2] Loading training configuration from: {cfg_path}")
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"FATAL: Config file not found at {cfg_path}")
cfg = Config(yamlfile=cfg_path)
print("-> Config loaded successfully.")
print(f"-> Sanity check: Model type='{cfg.model.type}', Hidden size={cfg.model.hidden_size}")


# --- Step 3: Create instances of the model and its components ---
# These are empty shells with the correct architecture, ready to be filled with weights.
print("\n[Step 3] Instantiating model and components based on config...")

# A. The main score model (SEDD)
score_model = SEDD(cfg).to(device)
print("  - SEDD model instance created.")

# B. The Exponential Moving Average (EMA) handler
ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
print("  - EMA handler instance created.")

# C. The noise scheduler
noise = noise_lib.get_noise(cfg).to(device)
print("  - Noise scheduler instance created.")

# D. The token graph
graph = graph_lib.get_graph(cfg, device)
print("  - Token graph instance created.")

print("-> All components instantiated successfully.")


# --- Step 4: Load the checkpoint file from disk ---
print(f"\n[Step 4] Loading checkpoint from: {ckpt_path}")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"FATAL: Checkpoint file not found at {ckpt_path}")

# map_location=device ensures that the tensors are loaded onto the correct device
state = torch.load(ckpt_path, map_location=device)
print("-> Checkpoint file loaded into memory.")

# A very useful debugging step: check what's inside the checkpoint file
print(f"  - Checkpoint contains keys: {list(state.keys())}")


# --- Step 5: Populate the model and EMA with the loaded weights ---
print("\n[Step 5] Applying weights from checkpoint to model instances...")

from collections import OrderedDict
unwrapped_state_dict = OrderedDict()
for k, v in state['model'].items():
    if k.startswith('module.'):
        # Remove the 'module.' prefix
        name = k[7:] 
        unwrapped_state_dict[name] = v
    else:
        # If for some reason a key doesn't have the prefix, keep it as is
        unwrapped_state_dict[k] = v
# --- NEW CODE END ---

# A. Load the MODIFIED state dictionary into the main model object
score_model.load_state_dict(unwrapped_state_dict) # <--- Use the unwrapped state_dict
print("  - Weights loaded into the main model (score_model).")


# B. Load the state dictionary for the EMA weights (EMA state might also have the prefix)
if 'ema' in state and state['ema'] is not None:
    unwrapped_ema_dict = OrderedDict()
    for k, v in state['ema'].items():
        if k.startswith('module.'):
            name = k[7:]
            unwrapped_ema_dict[name] = v
        else:
            unwrapped_ema_dict[k] = v
    ema.load_state_dict(unwrapped_ema_dict)
    print("  - Weights loaded into the EMA handler.")
else:
    print("  - EMA state not found in checkpoint.")

# C. Good practice: also load the noise state if it's present and learnable
if 'noise' in state and state['noise'] is not None:
    noise.load_state_dict(state['noise'])
    print("  - State loaded into the noise scheduler.")
else:
    print("  - Noise scheduler state not found in checkpoint (this is usually fine).")


# --- Step 6: Finalize model for inference ---
print("\n[Step 6] Setting model to evaluation mode...")
score_model.eval()
print("-> Model is now in 'eval' mode (e.g., dropout is disabled).")
