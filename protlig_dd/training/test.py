import datetime
import os
import os.path
import gc
from itertools import chain
import wandb
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import protlig_dd.data as data
import protlig_dd.processing.losses as losses
import protlig_dd.sampling.sampling as sampling
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
import protlig_dd.utils.utils as utils
from protlig_dd.model import SEDD
from protlig_dd.model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from SmilesPE.tokenizer import *
#from smallmolec_campaign.utils.smiles_pair_encoders_functions import *
from protlig_dd.data.process_full_plinder import ddp_data_loaders
from dataclasses import dataclass
import yaml
import inspect
import protlig_dd.data.get_embeddings as get_embeddings
from collections import OrderedDict

@dataclass
class Config:
    dictionary: object | None = None
    yamlfile: str | None = None

    def __post_init__(self):
        if self.dictionary==None and self.yamlfile!=None:
            with open(self.yamlfile, "r") as f:
                self.dictionary = yaml.safe_load(f)
               
        for key, value in self.dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)



@dataclass
class TestModel:
    cfg_fil: str | None = None
    cfg_dict: dict | None = None
    #graph: graph_lib.Graph
    #noise: noise_lib.Noise
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_eps: float = 1e-5
    checkpoint_path: str | None = None
    ddp:bool = False
    def __post_init__(self):
        self.cfg = Config(
                    dictionary=self.cfg_dict,
                    yamlfile=self.cfg_fil   
                    )
        print(self.cfg)
        self.score_model = SEDD(self.cfg).to(self.device)
        self.graph = graph_lib.get_graph(self.cfg, self.device)
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        self.ema = ExponentialMovingAverage(
                        self.score_model.parameters(),
                        decay=self.cfg.training.ema)
        self.sampling_fn = sampling.get_sampling_fn(
                                        self.cfg,
                                        self.graph, 
                                        self.noise, 
                                        (2, self.cfg.model.length),#(self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length), 
                                        self.sampling_eps, 
                                        self.device)

        if self.checkpoint_path is not None:
            self.load_state_dict()
    def load_state_dict(self):  
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        if self.ddp==True:
            new_state_dict = OrderedDict()
            for k, v in state_dict["model"].items():
                new_key = k.replace("module.", "")  # remove "module." prefix
                new_state_dict[new_key] = v
            state_dict["model"] = new_state_dict
        self.score_model.load_state_dict(state_dict['model'])#, strict=False)
        self.ema.store(self.score_model.parameters())
        self.ema.copy_to(self.score_model.parameters())
    
    def sample(self):
        self.ema.restore(self.score_model.parameters())
        sample = self.sampling_fn(self.score_model)
        self.ema.store(self.score_model.parameters())
        return sample
    
tester = TestModel(cfg_dict=yaml.safe_load(open("/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/configs/config.yaml")),
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    checkpoint_path='/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/checkpoints/checkpoint_9.pth',
                    ddp=True)
samples = tester.sample()
print(samples)
for i, sample in enumerate(samples):
    print(f"Sample {i}: {sample.tolist()}")

'''
Below is my attempt to decode the samples into SMILES and protein sequences... However, it gives something weird like:
Generated Ligand (SMILES):
2111c11c[GaH4-]c1[Co+2][123Sn]111n211[Fm]([TiH+3][86Sr][B-][64Zn][Nd]1c[Ti+2][82Kr]1[207Pb]1[RuH2+2]1[68Ge]c1111[CeH3]1231141[C@@H]1111(2121c[98Tc+4]2(1n[Na+][27Al]1[n+]1c((111111111c2(2c11(111[34SH]111[SnH]1[139La]cc1nNo[144Ce]11[235U+2].1[SnH4+2][Rh+2][150Pm]12n[12C][P-3][238Th]

Generated Protein Sequence:
- X O

I think it has something to do with the tokenizers? Because I think tester.cfg.data.max_ligand_len and tester.cfg.data.max_protein_len are not set correctly.
In that case, it would be the tokenizers are not configured properly to decodes SMILES and protein sequences.
'''

from transformers import AutoTokenizer

MOL_MODEL_ID = "ibm/MoLFormer-XL-both-10pct"
PROT_MODEL_ID = "facebook/esm2_t30_150M_UR50D"
PROTEIN_TOKEN_OFFSET = 2363

print("\n[Step 1] Loading MoLFormer and ESM2 tokenizers...")
mol_tokenizer = AutoTokenizer.from_pretrained(MOL_MODEL_ID, trust_remote_code=True)
protein_tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL_ID)

# --- 2. Separate ligand and protein parts from the 'samples' tensor ---
# These length values must exactly match the settings in your config.yaml
max_ligand_len = tester.cfg.data.max_ligand_len   # e.g., 128
max_protein_len = tester.cfg.data.max_protein_len # e.g., 1024
total_len = tester.cfg.model.length             # e.g., 1088

print(f"\n[Step 2] Separating Ligand and Protein Token IDs...")
print(f"-> Total length: {total_len}, Ligand length: {max_ligand_len}, Protein length: {max_protein_len}")

for i in range(len(samples)):
    sample_ids = samples[i]

    ligand_ids = sample_ids[:max_ligand_len]
    protein_ids_with_offset = sample_ids[max_ligand_len:total_len]

    # --- 3. Decode the protein sequence ---
    # This is the most crucial step: subtract the offset to restore the original ESM2 Token IDs.
    protein_ids = protein_ids_with_offset - PROTEIN_TOKEN_OFFSET

    # The tokenizer's decode method with skip_special_tokens=True will automatically clean up [PAD], [CLS], etc.
    # clamp(min=0) is a safety measure to prevent negative IDs from causing an error after subtraction.
    protein_sequence = protein_tokenizer.decode(
        protein_ids.clamp(min=0), 
        skip_special_tokens=True
    )
    # --- 4. Decode the ligand SMILES ---
    # The ligand part does not require any offset adjustment.
    ligand_smiles = mol_tokenizer.decode(
        ligand_ids, 
        skip_special_tokens=True
    )
    ligand_smiles_cleaned = ligand_smiles.replace(" ", "")

    # --- 5. Display the final results ---
    print(f"\nGenerated Ligand (SMILES):\n{ligand_smiles_cleaned}")
    print(f"\nGenerated Protein Sequence:\n{protein_sequence}")

if False:
    #model = SEDD.from_pretrained(model_path=checkpoint["model"], device=device, cfg=cfg)
    #print(model)
    if False:
        model.load_state_dict(checkpoint["model"]['module'])  # if saved with optimizer, epoch, etc.
        #model = SEDD.from_pretrained(model_path, device=device)
        model.eval()
    #return model

    mod = load_model(
        cfg=yaml.safe_load(open("/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/configs/config_ddp.yaml")),
                    checkpoint_path='/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/checkpoints/checkpoint_9.pth',
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(mod)
if False:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    sampling_shape = (self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length)
    self.sampling_fn = sampling.get_sampling_fn(self.cfg, self.graph, self.noise, sampling_shape, self.sampling_eps, self.device)
    
    self.ema.store(self.score_model.parameters())
    self.ema.copy_to(self.score_model.parameters())
    sample = self.sampling_fn(self.score_model)
    self.ema.restore(self.score_model.parameters())
    
    vocab_tok_smiles = list("CNOSPFBrClI()[]+=\\#-@:123456789%/c.nsop")
    sentences = [vocab_tok_smiles[i_v] for i_v in sample[0]]
    print(''.join(sentences))
    
    file_name = os.path.join(this_sample_dir, "sample_.txt")
    with open(file_name, 'w') as file:
        for sentence in sentences:
            file.write(sentence + "\n")
            file.write("=" * 92 + "\n")