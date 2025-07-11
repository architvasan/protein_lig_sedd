import datetime
import os
import os.path
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from SmilesPE.tokenizer import *
#from smallmolec_campaign.utils.smiles_pair_encoders_functions import *
from updated_data_pipeline import create_improved_data_loaders
from dataclasses import dataclass
import yaml

@dataclass
class Train_pl_sedd:
    work_dir: str
    cfg_fil: str
    plinder_output_dir: str='./plinder',
    plinder_data_dir:str ='./plinder',

    max_samples:int =200,
    batch_size: int =2,
    num_workers: int=1,
    train_ratio: int=0.8,
    val_ratio: int=0.1,#args.val_ratio,
    max_protein_len: int=1024,#args.protein_max_len,
    max_ligand_len:int =128,#args.mol_max_len,
    use_structure:bool =False,#args.use_structure,
    seed=42,#args.seed,
    force_reprocess=False,#args.force_reprocess

    def __post__init(self):
        with open(cfg_fil, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.sample_dir = os.path.join(self.work_dir, "samples")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.checkpoint_meta_dir = os.path.join(self.work_dir, "checkpoints-meta", "checkpoint.pth")
        os.makedirs(self.sample_dir, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        os.makedirs(self.checkpoint_meta_dir, exist_ok = True)
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def smilespetok(
            vocab_file = '../../VocabFiles/vocab_spe.txt',
            spe_file = '../../VocabFiles/SPE_ChEMBL.txt'):
        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
        return tokenizer
    
    def setup_loaders(self):
        self.train_ds, self.eval_ds, test_loader = create_improved_data_loaders(
            plinder_output_dir=self.plinder_output_dir,#'./plinder',
            plinder_data_dir=self.plinder_data_dir,#'./plinder',
            max_samples=200,
            batch_size=2,
            num_workers=1,
            train_ratio=0.8,
            val_ratio=0.1,#args.val_ratio,
            max_protein_len=1024,#args.protein_max_len,
            max_ligand_len=128,#args.mol_max_len,
            use_structure=False,#args.use_structure,
            seed=42,#args.seed,
            force_reprocess=False,#args.force_reprocess
        )

    def load_model(
            self,
            ):
        # build token graph
        graph = graph_lib.get_graph(self.cfg, self.device)
        
        # build score model
        self.score_model = SEDD(self.cfg).to(self.device)
        #score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), decay=self.cfg.training.ema)
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
         
    def optim_state(
            self):
        self.sampling_eps = 1e-5
    
        # build optimization state
        self.optimizer = losses.get_optimizer(self.cfg, chain(self.score_model.parameters(), self.noise.parameters()))
        print(f"Optimizer: {self.optimizer}")
        self.scaler = torch.cuda.amp.GradScaler()
        print(f"Scaler: {scaler}")
        state = dict(optimizer=self.optimizer,
                        scaler=self.scaler,
                        model=self.score_model,
                        noise=self.noise,
                        ema=self.ema,
                        step=0) 
    
    
        # load in state
        self.state = utils.restore_checkpoint(self.checkpoint_meta_dir, state, self.device)
        self.initial_step = int(state['step'])

    def train(self): 
        train_iter = iter(self.train_ds)
        eval_iter = iter(self.eval_ds)






         








def setup_stuff(work_dir):
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(sample_dir, exist_ok = True)
    os.makedirs(checkpoint_dir, exist_ok = True)
    os.makedirs(checkpoint_meta_dir, exist_ok = True)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return sample_dir, checkpoint_dir, checkpoint_meta_dir, device

#def training()

