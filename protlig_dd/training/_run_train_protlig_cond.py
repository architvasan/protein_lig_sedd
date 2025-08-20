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
from protlig_dd.data.updated_data_pipeline import create_improved_data_loaders
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
    cfg_fil: str | None = None # either yaml file or dict
    cfg_dict: object | None = None
    plinder_output_dir: str='./plinder'
    plinder_data_dir:str ='./plinder'
    mol_emb_id: str = "ibm/MoLFormer-XL-both-10pct"
    dev_id: str = 'cuda:0'
    
    if False:
        epochs: int=10
        max_samples:int =200
        batch_size: int =2
        num_workers: int=1
        train_ratio: int=0.8
        val_ratio: int=0.1#args.val_ratio,
        max_protein_len: int=1024#args.protein_max_len,
        max_ligand_len:int =128#args.mol_max_len,
        use_structure:bool =False#args.use_structure,
        seed:int=42#args.seed,
        force_reprocess:bool=False#args.force_reprocess


    def __post_init__(self):
        """
        Load config file/dictionary
        Create embedding object for mol, prot
        Mkdir for checkpoint/sample dirs
        Set device
        """

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
                            plinder_output_dir=self.plinder_output_dir,
                            plinder_data_dir=self.plinder_data_dir,
                            max_samples=self.cfg.training.max_samples,
                            batch_size=self.cfg.training.batch_size,
                            num_workers=1,
                            train_ratio=self.cfg.data.train_ratio,
                            val_ratio=self.cfg.data.val_ratio,
                            max_protein_len=self.cfg.data.max_protein_len,
                            max_ligand_len=self.cfg.data.max_ligand_len,
                            use_structure=self.cfg.data.use_structure,
                            seed=42,
                            force_reprocess=False,
                            )

    def load_model(
            self,
            ):
        # build token graph
        self.graph = graph_lib.get_graph(self.cfg, self.device)
        
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
        #print(f"Optimizer: {self.optimizer}")
        self.scaler = torch.cuda.amp.GradScaler()
        #print(f"Scaler: {self.scaler}")
        self.state = dict(optimizer=self.optimizer,
                        scaler=self.scaler,
                        model=self.score_model,
                        noise=self.noise,
                        ema=self.ema,
                        step=0) 
    
    
        # load in state
        #self.state = utils.restore_checkpoint(self.checkpoint_meta_dir, state, self.device)
        self.initial_step = int(self.state['step'])

    def _train(self): 

        #print(f"{self.cfg.optim.lr=}")
        self.cfg.optim.lr = float(self.cfg.optim.lr)
        self.setup_loaders()
        self.load_model()
        self.optim_state()
        train_iter = iter(self.train_ds)
        eval_iter = iter(self.eval_ds)
        #print(train_iter, eval_iter)
        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(self.cfg)
        train_step_fn = losses.get_step_fn(self.noise, self.graph, True, optimize_fn, self.cfg.training.accum)
        eval_step_fn = losses.get_step_fn(self.noise, self.graph, False, optimize_fn, self.cfg.training.accum)
        
        #print(inspect.signature(optimize_fn))
        #print(inspect.signature(train_step_fn))
        #print(inspect.signature(eval_step_fn))
#
        if self.cfg.training.snapshot_sampling:
            sampling_shape = (self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length)
            self.sampling_fn = sampling.get_sampling_fn(self.cfg, self.graph, self.noise, sampling_shape, self.sampling_eps, self.device)

        num_train_steps = self.cfg.training.n_iters
        print(f"Starting training loop at step {self.initial_step}.")
        print(self.epochs)
        for ep in range(self.epochs):
            for b in train_iter:
            #while self.state['step'] < num_train_steps + 1: ##OLD
                step = self.state['step'] ##OLD
                self.state['step']+=1 ## NEW

                if self.cfg.data.train != "text8":
                    #print(next(train_iter)['ligand_tokens'].to(device))
                    #batch = next(train_iter)['ligand_tokens'].to(self.device) ##OLD
                    batch = b['ligand_tokens'].to(self.device) ##NEW
                else:
                    #batch = next(train_iter).to(self.device) ##OLD
                    batch = b.to(self.device) ##NEW
                    #print(batch)
                loss = train_step_fn(self.state, batch)
                #print(f"{loss=}")
                # flag to see if there was movement ie a full batch got computed
                if step != self.state['step']:
                    if step % self.cfg.training.log_freq == 0:
                        #dist.all_reduce(loss)
                        #loss /= world_size

                        print("epoch: %d, step: %d, training_loss: %.5e" % (ep, step, loss.item()))

                    if step % self.cfg.training.snapshot_freq_for_preemption == 0 :
                        utils.save_checkpoint(f'{self.checkpoint_meta_dir}/check.pth', self.state)

                    
                    if step % self.cfg.training.eval_freq == 0:
                        if self.cfg.data.valid != "text8":
                            #print(next(eval_iter))
                            eval_batch = next(eval_iter)['ligand_tokens'].to(self.device)
                        else:
                            eval_batch = next(eval_iter).to(self.device)

                        eval_loss = eval_step_fn(self.state, eval_batch)

                        #dist.all_reduce(eval_loss)
                        #eval_loss /= world_size

                        print("epoch: %d, step: %d, evaluation_loss: %.5e" % (ep, step, eval_loss.item()))

                    if step > 0 and step % self.cfg.training.snapshot_freq == 0 or step == num_train_steps:
                        # Save the checkpoint.
                        save_step = step // self.cfg.training.snapshot_freq
                        if True:
                            utils.save_checkpoint(os.path.join(
                                self.checkpoint_dir, f'checkpoint_{save_step}.pth'), self.state)

                        # Generate and save samples
                        if self.cfg.training.snapshot_sampling:
                            print(f"Generating text at step: {step}")

                            this_sample_dir = os.path.join(self.sample_dir, "iter_{}".format(step))
                            utils.makedirs(this_sample_dir)

                            self.ema.store(self.score_model.parameters())
                            self.ema.copy_to(self.score_model.parameters())
                            sample = self.sampling_fn(self.score_model)
                            self.ema.restore(self.score_model.parameters())

                            vocab_tok_smiles = list("CNOSPFBrClI()[]+=\\#-@:123456789%/c.nsop")
                            sentences = [vocab_tok_smiles[i_v] for i_v in sample[0]]#self.tokenizer.batch_decode(sample)
                            print(''.join(sentences))
                            #print(sentences) 
                            file_name = os.path.join(this_sample_dir, f"sample_.txt")
                            with open(file_name, 'w') as file:
                                for sentence in sentences:
                                    file.write(sentence + "\n")
                                    file.write("============================================================================================\n")

                            if self.cfg.eval.perplexity:
                                with torch.no_grad():
                                    pass

                                #dist.barrier()
                    #if step>=num_train_steps# *(ep+1):
                    #    break
                        
    def train(self, wandbproj, wandbname): 
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avasan",
            # Set the wandb project where this run will be logged.
            project=wandbproj, #"protein-lig-sedd",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandbname #f"experiment_run_1",
            # Track hyperparameters and run metadata.
        )

        self.cfg.optim.lr = float(self.cfg.optim.lr)
        self.setup_loaders()
        self.load_model()
        self.optim_state()

        wandb.login()
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avasan",
            # Set the wandb project where this run will be logged.
            project="protein-lig-sedd",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_run_1",
            # Track hyperparameters and run metadata.
            config={                         # Track hyperparameters and metadata
                    "epochs": 10,
                    },
        )

        optimize_fn = losses.optimization_manager(self.cfg)
        train_step_fn = losses.get_step_fn(self.noise, self.graph, True, optimize_fn, self.cfg.training.accum)
        eval_step_fn = losses.get_step_fn(self.noise, self.graph, False, optimize_fn, self.cfg.training.accum)

        if self.cfg.training.snapshot_sampling:
            sampling_shape = (self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length)
            self.sampling_fn = sampling.get_sampling_fn(self.cfg, self.graph, self.noise, sampling_shape, self.sampling_eps, self.device)

        num_train_steps = self.cfg.training.n_iters
        print(f"Starting training loop at step {self.initial_step}.")
        print(f"Training for {self.epochs} epochs or {num_train_steps} steps.")

        step = self.state['step']
        for ep in range(self.epochs):
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
                        #print(batch_lig)
                        batch_lig = batch_lig.to(self.device)
                        batch_prot = batch_prot.to(self.device)
                        #batch_lig = batch_tot['ligand_tokens'].to(self.device)
                        #batch_prot = (batch_tot['protein_tokens']+self.cfg.tokens_lig).to(self.device)
                        #print(batch_prot['input_ids'])
                        batch = torch.concat([batch_lig['input_ids'], batch_prot['input_ids']+2363], axis=1)
                        #print(batch.shape)
                        #print(batch)
                        #print(batch.shape)
                        #import sys
                        #sys.exit()
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
                            #eval_batch = next(eval_iter)['ligand_tokens'].to(self.device)
                            eval_batch_lig_seq = batch_tot['ligand_smiles']
                            eval_batch_prot_seq = batch_tot['protein_seq']
                            #print(batch_prot_seq)
                            eval_batch_lig, eval_batch_prot, eval_mol_cond, eval_esm_cond = self.embedding_mol_prot.process_embeddings(
                                eval_batch_prot_seq,
                                eval_batch_lig_seq)
                            #print(batch_lig)
                            eval_batch_lig = eval_batch_lig.to(self.device)
                            eval_batch_prot = eval_batch_prot.to(self.device)
                            #batch_lig = batch_tot['ligand_tokens'].to(self.device)
                            #batch_prot = (batch_tot['protein_tokens']+self.cfg.tokens_lig).to(self.device)
                            #print(batch_prot['input_ids'])
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
        work_dir, cfg_fil, plinder_output_dir = './plinder_10k/processed_plinder_data', plinder_data_dir='./plinder_10k/processed_plinder_data', epochs=10, max_samples=1000000, batch_size=32, num_workers=1, train_ratio=0.8, val_ratio=0.1, max_protein_len=1024, max_ligand_len=128, use_structure=False, seed=42, force_reprocess=False):

    trainer_object= Train_pl_sedd(
                            work_dir,
                            cfg_fil,
                            plinder_output_dir,
                            plinder_data_dir,
                            epochs,
                            max_samples,
                            batch_size,
                            num_workers,
                            train_ratio,
                            val_ratio,
                            max_protein_len,
                            max_ligand_len,
                            use_structure,
                            seed,
                            force_reprocess,
    )

    trainer_object.train()


if __name__=="__main__":
    work_dir = "/eagle/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd" 
    cfg_fil = "./configs/config.yaml"

    run_train(work_dir, cfg_fil)

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

