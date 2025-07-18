import yaml
import wandb 

cfg_fil = "./configs/config.yaml"
with open(cfg_fil, "r") as f:
        cfg = yaml.safe_load(f)

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="noor-harwell-howard-university",
    # Set the wandb project where this run will be logged.
    project="protein-lig-sedd",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{run_1}",
    # Track hyperparameters and run metadata.
    config=cfg
)

# wandb.log_artifact('protein_lig_sedd')
