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
from protlig_dd.data.process_full_plinder import create_improved_data_loaders
from dataclasses import dataclass
import yaml
import inspect
import get_embeddings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
class Train_pl_sedd:
    """
    config params:
    model:
        name: medium           
        type: ddit
        hidden_size: 1024
        cond_dim: 1024
        length: 1088
        n_blocks: 8
        n_heads: 8
        scale_by_sigma: True
        dropout: 0.1
        esm_dim: 640
        molformer_dim: 768         

    tokens: 2396

    training:
        batch_size: 32
        accum: 1
        epochs: 100
        max_samples: 10000
        num_workers: 1
        seed: 42
        force_reprocess: False
        n_iters: 1000000
        snapshot_freq: 200
        log_freq: 50
        eval_freq: 200
        snapshot_freq_for_preemption: 200
        weight: standard
        snapshot_sampling: True
        ema: 0.9999

    data:
        train: acyp
        valid: acyp
        cache_dir: data
        train_ratio: 0.8
        val_ratio: 0.8
        max_protein_len: 1024
        max_ligand_len: 128
        use_structure: False

    graph:
        type: absorb
        file: data
        report_all: False
    
    noise:
        type: loglinear #geometric #
        sigma_min: !!float "1e-5"
        sigma_max: 20
    
    sampling:
        predictor: euler
        steps: 25
        noise_removal: True
    
    eval:
        batch_size: 25
        perplexity: True
        perplexity_batch_size: 4
    
    optim:
        weight_decay: 0
        optimizer: AdamW
        lr: !!float "3e-5"
        beta1: 0.9
        beta2: 0.999
        eps: !!float "1e-8"
        warmup: 1_000
        grad_clip: 1.
    """
    work_dir: str
    datafile: str
    cfg_fil: str | None = None # either yaml file or dict
    cfg_dict: object | None = None
    mol_emb_id: str = "ibm/MoLFormer-XL-both-10pct"
    prot_emb_id: str = "facebook/esm2_t30_150M_UR50D"
    dev_id: str = 'cuda:0'
    seed: int = 42

    def __post_init__(self):
        """
        Load config file/dictionary
        Create embedding object for mol, prot
        Mkdir for checkpoint/sample dirs
        Set device
        """

        wandb.login()
        self.cfg = Config(
                    yamlfile=self.cfg_fil,
                    dictionary=self.cfg_dict)
        print(self.cfg.dictionary)
        self.embedding_mol_prot = get_embeddings.Embed_Mol_Prot(self.mol_emb_id, self.prot_emb_id)

        self.sample_dir = os.path.join(self.work_dir, "samples")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.checkpoint_meta_dir = os.path.join(self.work_dir, "checkpoints-meta", "checkpoint.pth")
        os.makedirs(self.sample_dir, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        os.makedirs(self.checkpoint_meta_dir, exist_ok = True)
        self.device = torch.device(self.dev_id)
        #f"cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def smilespetok(
            vocab_file = '../../VocabFiles/vocab_spe.txt',
            spe_file = '../../VocabFiles/SPE_ChEMBL.txt'):
        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
        return tokenizer

    def setup_loaders(self):
        self.train_ds, self.eval_ds, test_loader = create_improved_data_loaders(
                                                            data_file=self.datafile,
                                                            max_samples=self.cfg.training.max_samples,
                                                            batch_size=self.cfg.training.batch_size,
                                                            num_workers=1,
                                                            train_ratio=self.cfg.data.train_ratio,
                                                            val_ratio=self.cfg.data.val_ratio,
                                                            max_protein_len=self.cfg.data.max_protein_len,
                                                            max_ligand_len=self.cfg.data.max_ligand_len,
                                                            use_structure=self.cfg.data.use_structure,
                                                            seed=self.seed,
                                                            force_reprocess=False,
                                                            )

    def load_model(
            self,
            ):
        # build token graph
        self.graph = graph_lib.get_graph(self.cfg, self.device)
        
        # build score model
        self.score_model = SEDD(self.cfg).to(self.device)
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), decay=self.cfg.training.ema)
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
         
    def optim_state(
            self):
        self.sampling_eps = 1e-3
    
        # build optimization state
        self.optimizer = losses.get_optimizer(self.cfg, chain(self.score_model.parameters(), self.noise.parameters()))
        self.scaler = torch.cuda.amp.GradScaler()
        self.state = dict(optimizer=self.optimizer,
                        scaler=self.scaler,
                        model=self.score_model,
                        noise=self.noise,
                        ema=self.ema,
                        step=0) 
    
        # load in state
        self.initial_step = int(self.state['step'])
                        
    def train(self, wandbproj, wandbname): 
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avasan",
            # Set the wandb project where this run will be logged.
            project=wandbproj, #"protein-lig-sedd",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandbname, #f"experiment_run_1",
            # Track hyperparameters and run metadata.
            config=self.cfg.dictionary
        )

        self.cfg.optim.lr = float(self.cfg.optim.lr)
        self.setup_loaders()
        self.load_model()
        self.optim_state()

        optimize_fn = losses.optimization_manager(self.cfg)
        train_step_fn = losses.get_step_fn(self.noise, self.graph, True, optimize_fn, self.cfg.training.accum)
        eval_step_fn = losses.get_step_fn(self.noise, self.graph, False, optimize_fn, self.cfg.training.accum)

        if self.cfg.training.snapshot_sampling:
            sampling_shape = (self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length)
            self.sampling_fn = sampling.get_sampling_fn(self.cfg, self.graph, self.noise, sampling_shape, self.sampling_eps, self.device)

        num_train_steps = self.cfg.training.n_iters
        print(f"Starting training loop at step {self.initial_step}.")
        print(f"Training for {self.cfg.training.epochs} epochs or {num_train_steps} steps.")

        step = self.state['step']
        for ep in range(self.cfg.training.epochs):
            train_iter = iter(self.train_ds)  # Reset iterator at start of epoch
            while True:
                if step >= num_train_steps:
                    print(f"Reached max training steps: {num_train_steps}. Ending training.")
                    return  # Exit training early

                try:
                    if self.cfg.data.train != "text8":
                        batch_tot = next(train_iter)
                        batch_lig_seq = batch_tot['ligand_smiles']
                        batch_prot_seq = batch_tot['protein_seq']
                        #print(batch_prot_seq)
                        batch_lig, batch_prot, mol_cond, esm_cond = self.embedding_mol_prot.process_embeddings(
                            batch_prot_seq,
                            batch_lig_seq)
                        batch_lig = batch_lig.to(self.device)
                        #print(batch_lig)
                        print(f"{batch_lig['input_ids'].shape=}")
    
                        batch_prot = batch_prot.to(self.device)
                        print(f"{batch_prot['input_ids'].shape=}")

                        batch = torch.concat([batch_lig['input_ids'], batch_prot['input_ids']+2363], axis=1)
                        print(f"{batch.shape=}")
                    else:
                        batch = next(train_iter).to(self.device)
                except StopIteration:
                    # End of dataset for this epoch
                    break

                loss = train_step_fn(self.state, batch, esm_cond=esm_cond, mol_cond=mol_cond)
                step = self.state['step']

                # Logging and checkpointing
                wandb.log({"training_loss": loss.item()})
                if step % self.cfg.training.log_freq == 0:
                    print(f"epoch: {ep}, step: {step}, training_loss: {loss.item():.5e}")

                if step % self.cfg.training.snapshot_freq_for_preemption == 0:
                    utils.save_checkpoint(f'{self.checkpoint_meta_dir}/check.pth', self.state)

                if step % self.cfg.training.eval_freq == 0:
                    eval_iter = iter(self.eval_ds)
                    try:
                        if self.cfg.data.valid != "text8":
                            eval_batch_lig_seq = batch_tot['ligand_smiles']
                            eval_batch_prot_seq = batch_tot['protein_seq']
                            eval_batch_lig, eval_batch_prot, eval_mol_cond, eval_esm_cond = self.embedding_mol_prot.process_embeddings(
                                eval_batch_prot_seq,
                                eval_batch_lig_seq)
                            eval_batch_lig = eval_batch_lig.to(self.device)
                            eval_batch_prot = eval_batch_prot.to(self.device)
                            eval_batch = torch.concat([eval_batch_lig['input_ids'], eval_batch_prot['input_ids']+2363], axis=1)
                        else:
                            eval_batch = next(eval_iter).to(self.device)
                    except StopIteration:
                        # Handle empty eval set or re-init if needed
                        eval_batch = None

                    if eval_batch is not None:
                        eval_loss = eval_step_fn(self.state, eval_batch, esm_cond=esm_cond, mol_cond=mol_cond)
                        print(f"epoch: {ep}, step: {step}, evaluation_loss: {eval_loss.item():.5e}")
                        wandb.log({"evaluation_loss": eval_loss.item()})


                if (step > 0 and step % self.cfg.training.snapshot_freq == 0) or step == num_train_steps:
                    save_step = step // self.cfg.training.snapshot_freq
                    utils.save_checkpoint(os.path.join(self.checkpoint_dir, f'checkpoint_{save_step}.pth'), self.state)


        wandb.finish()

def run_train(
        work_dir,
        wandbproj,
        wandbname,
        cfg_fil=None,
        cfg_dict=None,
        datafile = './input_data/merged_plinder.pt',
        mol_emb_id = "ibm/MoLFormer-XL-both-10pct",
        prot_emb_id = "facebook/esm2_t30_150M_UR50D",
        dev_id = "cuda:0",
        seed=42):
        #epochs=10, max_samples=1000000, batch_size=32, num_workers=1, train_ratio=0.8, val_ratio=0.1, max_protein_len=1024, max_ligand_len=128, use_structure=False, seed=42, force_reprocess=False):
    trainer_object= Train_pl_sedd(
                        work_dir = work_dir,
                        cfg_fil = cfg_fil,
                        cfg_dict = cfg_dict,
                        datafile = datafile,
                        mol_emb_id = mol_emb_id,
                        prot_emb_id = prot_emb_id,
                        dev_id = dev_id,
                        seed = seed)

    trainer_object.train(wandbproj, wandbname)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run model training")

    parser.add_argument('-WD', '--work_dir', type=str, default='.', help='working dir')
    parser.add_argument('-cf', '--config_file', type=str, default='configs/config.yaml', help='Yaml file with arguments')
    parser.add_argument('-wp', '--wandbproj', type=str, default='protlig_sedd', help='WandB project')
    parser.add_argument('-wn', '--wandbname', type=str, default='run1', help='WandB name')
    parser.add_argument('-df', '--datafile', type=str, default='./input_data/merged_plinder.pt', help='input plinder data')
    parser.add_argument('-me', '--mol_emb_id', type=str, default="ibm/MoLFormer-XL-both-10pct", help='model for mol embedding')
    parser.add_argument('-pe', '--prot_emb_id', type=str, default="facebook/esm2_t30_150M_UR50D", help='model for protein embedding')
    parser.add_argument('-di', '--dev_id', type=str, default='cuda:0', help='device')
    parser.add_argument('-s', '--seed', type=int, default=42, help='seed')
    
    args = parser.parse_args()

    run_train(
        work_dir=args.work_dir,
        wandbproj=args.wandbproj,
        wandbname=args.wandbname,
        cfg_fil=args.config_file,
        datafile=args.datafile,
        mol_emb_id=args.mol_emb_id,
        prot_emb_id=args.prot_emb_id,
        dev_id=args.dev_id,
        seed=args.seed
        )

if __name__ == '__main__':
    main()




# def setup_stuff(work_dir):
#     sample_dir = os.path.join(work_dir, "samples")
#     checkpoint_dir = os.path.join(work_dir, "checkpoints")
#     checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
#     os.makedirs(sample_dir, exist_ok = True)
#     os.makedirs(checkpoint_dir, exist_ok = True)
#     os.makedirs(checkpoint_meta_dir, exist_ok = True)
#     device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
#     return sample_dir, checkpoint_dir, checkpoint_meta_dir, device

# #def training()

