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
                                        (self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length), 
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