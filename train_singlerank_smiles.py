import run_train_smiles
import yaml

with open('configs/config.yaml', "r") as f:
    config = yaml.safe_load(f)
print(config)

run_train_smiles._run(rank=0, world_size=0, cfg=config)
