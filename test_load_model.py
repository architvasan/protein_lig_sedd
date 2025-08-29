from protlig_dd.training.run_train_protlig_special import Config
from protlig_dd.model.transformers_prot_lig import ProteinLigandSharedDiffusion
config = Config(yamlfile = 'configs/config_ddp.yaml')
model = ProteinLigandSharedDiffusion(config).to('cuda')
print(model)