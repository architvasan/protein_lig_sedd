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
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import protlig_dd.data as data
import protlig_dd.processing.losses_conditional as losses
import protlig_dd.sampling.sampling as sampling
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
import protlig_dd.utils.utils as utils
#from protlig_dd.model.transformers_prot_lig import ProteinLigandSharedDiffusion
from protlig_dd.model.transformers_protlig_cross import ProteinLigandDiffusionModel
from protlig_dd.model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
#from SmilesPE.tokenizer import *
#from smallmolec_campaign.utils.smiles_pair_encoders_functions import *
from protlig_dd.data.process_full_plinder import create_improved_data_loaders#ddp_data_loaders
from protlig_dd.data.tokenize import Tok_Mol, Tok_Prot
from protlig_dd.utils.lr_scheduler import WarmupCosineLR
from dataclasses import dataclass
import yaml
import inspect
import protlig_dd.data.get_embeddings as get_embeddings
import protlig_dd.sampling.sampling as sampling_dd
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import protlig_dd.utils.model_summ as model_summ

@dataclass
class Config:
    dictionary: object | None = None
    yamlfile: str | None = None

    def __post_init__(self):
        if self.dictionary is None and self.yamlfile is not None:
            with open(self.yamlfile, "r") as f:
                self.dictionary = yaml.safe_load(f)
               
        for key, value in self.dictionary.items():
            if isinstance(value, dict):
                value = Config(dictionary=value)
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass
class Train_pl_sedd:
    """
    #### Protein-Ligand Diffusion Model Training class
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
    model_weights: str | None = None

    check_dir: str = "checkpoints"
    dev_id: str = 'cuda:0'
    seed: int = 42
    sampling_eps: float = 1e-3
    use_pretrained_conditioning: bool = False
    mol_emb_id: str = "ibm/MoLFormer-XL-both-10pct" 
    prot_emb_id: str = "facebook/esm2_t30_150M_UR50D"

    def __post_init__(self):
        """
        Load config file/dictionary
        Mkdir for checkpoint/sample dirs
        Set device
        """

        self.cfg = Config(
                    yamlfile=self.cfg_fil,
                    dictionary=self.cfg_dict)

        self.task = self.cfg.training.task
        self.sample_dir = os.path.join(self.work_dir, "samples")
        self.checkpoint_dir = os.path.join(self.work_dir, self.check_dir)
        self.checkpoint_meta_dir = os.path.join(self.work_dir, f"{self.check_dir}-meta")

        self.tokenizer_mol = Tok_Mol(self.mol_emb_id)
        self.tokenizer_prot = Tok_Prot(self.prot_emb_id)

        os.makedirs(self.sample_dir, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        os.makedirs(self.checkpoint_meta_dir, exist_ok = True)
        self.device = 'cuda'

        # -----------------------
        # Setup DDP
        # -----------------------
        #self.local_rank = int(os.environ['LOCAL_RANK'])
        #torch.cuda.set_device(self.local_rank)
        #dist.init_process_group(backend='nccl')
        #self.device = torch.device(f'cuda:{self.local_rank}')
        #if dist.get_rank() == 0:
        
        """
        Set torch, numpy seeds
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        wandb.login()

    def setup_loaders(self):
        self.train_ds, self.eval_ds, test_loader = create_improved_data_loaders(
                                                            data_file=self.datafile,
                                                            max_samples=self.cfg.training.max_samples,
                                                            batch_size=self.cfg.training.batch_size,
                                                            num_workers=self.cfg.training.num_workers,
                                                            train_ratio=self.cfg.data.train_ratio,
                                                            val_ratio=self.cfg.data.val_ratio,
                                                            max_protein_len=self.cfg.data.max_protein_len,
                                                            max_ligand_len=self.cfg.data.max_ligand_len,
                                                            use_structure=self.cfg.data.use_structure,
                                                            seed=self.seed,
                                                            force_reprocess=self.cfg.training.force_reprocess,
                                                            )

    def freeze_model(self, model, freeze_what):
        """Freeze specific model parameters"""
        frozen_count = 0
        for name, param in model.named_parameters():
            if freeze_what in name:
                print(f"Freeza {name}")
                param.requires_grad = False
                frozen_count +=1

            elif self.task in ['protein_only', 'ligand_only'] and 'cross_attn' in name:
                print(f"Freeza {name}")
                param.requires_grad = False
                frozen_count +=1
        return model


    def load_model(
            self,
            ):
        # build token graph
        self.graph_prot = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.data.vocab_size_protein)
        self.graph_lig = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.data.vocab_size_ligand)
        # build score model
        self.score_model = ProteinLigandDiffusionModel(self.cfg).to(self.device)

        # Load pretrained weights if available
        if self.model_weights is not None:
            checkpoint = torch.load(self.model_weights, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.score_model.load_state_dict(checkpoint['model'])
            else:
                self.score_model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {self.model_weights}")        

        # Freeze ligand side for protein-only or prot|lig training
        # vice-versa for ligand training

        if self.task in ['protein_given_ligand', 'protein_only']:
            self.score_model = self.freeze_model(self.score_model, 'lig')        
        elif self.task in ['ligand_given_protein', 'ligand_only']:
            self.score_model = self.freeze_model(self.score_model, 'prot')

        self.score_model.train()

        model_summ.count_parameters(self.score_model)
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), decay=self.cfg.training.ema)
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
         
    def optim_state(
            self):
    
        # build optimization state
        self.optimizer = losses.get_optimizer(
                                self.cfg,
                                chain(self.score_model.parameters(),
                                        self.noise.parameters()))

        self.scaler = torch.cuda.amp.GradScaler()

        self.state = dict(optimizer=self.optimizer,
                        scaler=self.scaler,
                        model=self.score_model.state_dict(),
                        noise=self.noise.state_dict(),
                        ema=self.ema.state_dict(),
                        step=0) 
    
        # load in state
        self.initial_step = int(self.state['step'])

        # Setup learning rate scheduler
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.cfg.optim.warmup,
            max_steps=self.cfg.training.n_iters,
            base_lr=float(self.cfg.optim.lr) / 10,
            max_lr=float(self.cfg.optim.lr)
        )

    def calc_loss(self, batch, task, t = None, training=True):

        batch_len = batch['prot_tokens'].shape[0] if 'prot_tokens' in batch else batch['lig_tokens'].shape[0]

        if t is None:
            t = (1 - self.sampling_eps) * torch.rand(batch_len, device=self.device) + self.sampling_eps #batch.shape[0]
        
        sigma, dsigma = self.noise(t)
        sigma_reshaped = sigma.reshape(-1)
        # Handle different task types
        if task == "protein_only":
            protein_indices = self.graph_prot.sample_transition(
                batch['prot_tokens'].to(self.device), sigma[:, None]
            )
            log_score = self.score_model(
                protein_indices=protein_indices, 
                timesteps=sigma_reshaped, 
                mode=task
            )
            loss = self.graph_prot.score_entropy(
                log_score, sigma[:, None], protein_indices, batch['prot_tokens'].to(self.device)
            )
            
        elif task == "ligand_only":
            ligand_indices = self.graph_lig.sample_transition(
                batch['lig_tokens'].to(self.device), sigma[:, None]
            )
            log_score = self.score_model(
                ligand_indices=ligand_indices, 
                timesteps=sigma_reshaped, 
                mode=task
            )
            loss = self.graph_lig.score_entropy(
                log_score, sigma[:, None], ligand_indices, batch['lig_tokens'].to(self.device)
            )
            
        elif task == "joint":
            protein_indices = self.graph_prot.sample_transition(
                batch['prot_tokens'].to(self.device), sigma[:, None]
            )
            ligand_indices = self.graph_lig.sample_transition(
                batch['lig_tokens'].to(self.device), sigma[:, None]
            )
            
            log_score = self.score_model(
                protein_indices=protein_indices, 
                ligand_indices=ligand_indices, 
                timesteps=sigma_reshaped, 
                mode=task
            )
            
            # Calculate separate losses
            loss_prot = self.graph_prot.score_entropy(
                log_score[0], sigma[:, None], protein_indices, batch['prot_tokens'].to(self.device)
            )
            loss_lig = self.graph_lig.score_entropy(
                log_score[1], sigma[:, None], ligand_indices, batch['lig_tokens'].to(self.device)
            )
            
            # Weight losses with dsigma
            loss_prot = (dsigma[:, None] * loss_prot).mean()
            loss_lig = (dsigma[:, None] * loss_lig).mean()
            loss = (loss_prot + loss_lig) / 2
            
            return loss, loss_prot, loss_lig
            
        elif task == "ligand_given_protein":
            ligand_indices = self.graph_lig.sample_transition(
                batch['lig_tokens'].to(self.device), sigma[:, None]
            )
            log_score = self.score_model(
                ligand_indices=ligand_indices, 
                protein_seq_str=batch['protein_seq'], 
                timesteps=sigma_reshaped, 
                mode=task,
                use_pretrained_conditioning=self.use_pretrained_conditioning
            )
            loss = self.graph_lig.score_entropy(
                log_score, sigma[:, None], ligand_indices, batch['lig_tokens'].to(self.device)
            )
            
        elif task == "protein_given_ligand":
            protein_indices = self.graph_prot.sample_transition(
                batch['prot_tokens'].to(self.device), sigma[:, None]
            )
            log_score = self.score_model(
                protein_indices=protein_indices, 
                ligand_smiles_str=batch['ligand_smiles'], 
                timesteps=sigma_reshaped, 
                mode=task,
                use_pretrained_conditioning=self.use_pretrained_conditioning
            )
            loss = self.graph_prot.score_entropy(
                log_score, sigma[:, None], protein_indices, batch['prot_tokens'].to(self.device)
            )
        else:
            raise NotImplementedError(f"Task {task} not implemented")

        # Apply dsigma weighting for non-joint tasks
        if task != "joint":
            loss = (dsigma[:, None] * loss).mean()
            
        return loss

    def train_step(self, batch, task):

        if task == "all":
            task = np.random.choice(["ligand_given_protein", "protein_given_ligand", "joint"])
            #print(f"Training task: {task}")
        
        if task == "joint":
            loss, loss_prot, loss_lig = self.calc_loss(batch, task)
            loss = loss.mean()
            loss_prot = loss_prot.mean()
            loss_lig = loss_lig.mean()
        else:
            loss = self.calc_loss(batch, task).mean()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.score_model.parameters(),
            self.cfg.optim.grad_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Update EMA
        self.ema.update(self.score_model.parameters())

        # Update learning rate
        self.scheduler.step()
        
        # Update step counter
        self.state['step'] += 1
        
        if task == "joint":
            return loss.item(), loss_prot.item(), loss_lig.item()
        return loss.item()


    def eval_step(self, batch, task, cond=None):
        with torch.no_grad():
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())

            #print(batch)
            if task == "all":
                task = np.random.choice(["ligand_only", "protein_only", "ligand_given_protein", "protein_given_ligand", "joint"])
            if task == "joint":
                loss, loss_prot, loss_lig = self.calc_loss(batch, task, training=False)
                result = (loss.item(), loss_prot.item(), loss_lig.item())
            
            else:
                loss = self.calc_loss(batch, task, training=False)
                result = loss.item()
            self.ema.restore(self.score_model.parameters())
        return result

    def setupsampler(self, datatype, sampling_shape):
        if datatype == 'protein':
            graph_sampler = self.graph_prot
        elif datatype == 'ligand':
            graph_sampler = self.graph_lig
        else:
            raise ValueError(f"Unknown datatype: {datatype}")

        sampling_fn = sampling_dd.get_sampling_fn(
                                self.cfg,
                                graph_sampler,
                                self.noise,
                                sampling_shape,
                                self.sampling_eps,
                                self.device
                                )
        return sampling_fn

    def generation(self, sampling_fn, task, prot_indices=None, lig_indices=None):
        """Generate samples using the model"""
        with torch.no_grad():
            # Apply EMA weights for generation
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())


        samples = sampling_fn(self.score_model, task, prot_indices = prot_indices, lig_indices = lig_indices)

        # Decode samples
        if task in ['protein_given_ligand', 'protein_only']:
            real_samples = self.tokenizer_prot.protein_tokenizer.batch_decode(samples)
        elif task in ['ligand_given_protein', 'ligand_only']:
            real_samples = self.tokenizer_mol.mol_tokenizer.batch_decode(samples)
        else:
            real_samples = None

        # Restore original weights
            self.ema.restore(self.score_model.parameters())
        return real_samples

    def save_checkpoint(self, path, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'step': self.state['step'],
            'model': self.score_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'ema': self.ema.state_dict(),
            'noise': self.noise.state_dict(),
            'config': self.cfg.dictionary,
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.state['step'] = checkpoint['step']
        self.score_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.ema.load_state_dict(checkpoint['ema'])
        self.noise.load_state_dict(checkpoint['noise'])
        
        print(f"Loaded checkpoint from {path} at step {self.state['step']}")

    def train(self, wandbproj, wandbname): 

        run = wandb.init(
            entity="avasan",
            project=wandbproj,
            name=wandbname,
            config=self.cfg.dictionary
        )

        self.setup_loaders()
        self.load_model()
        self.optim_state()

        self.sampler_prot = self.setupsampler('protein', (self.cfg.training.batch_size, self.cfg.data.max_protein_len))
        self.sampler_lig = self.setupsampler('ligand', (self.cfg.training.batch_size, self.cfg.data.max_ligand_len))

        num_train_steps = self.cfg.training.n_iters
        print(f"Starting training loop at step {self.initial_step}.")
        print(f"Training for {self.cfg.training.epochs} epochs or {num_train_steps} steps.")

        step = self.state['step']
        best_eval_loss = float('inf')
        eval_loss_history = []

        for epoch in tqdm(range(self.cfg.training.epochs), desc="Epochs"):
            #self.train_sampler.set_epoch(ep)  # Ensures data is shuffled differently each epoch
            train_iter = iter(self.train_ds)  # Reset iterator at start of epoch            
            
            train_losses = []

            """
            Training loop
            1. Get train batch
            2. Tokenize train batch
            3. Get embeddings for train batch
            4. Train step
            5. Log train loss
            """

            for batch_idx in tqdm(range(len(self.train_ds)), desc=f"Epoch {epoch} - Training", leave=False):
                if step >= num_train_steps:
                    print(f"Reached max training steps: {num_train_steps}")
                    wandb.finish()
                    return
                
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                
                # Tokenize batch
                if batch.get('ligand_smiles'):
                    batch['lig_tokens'] = self.tokenizer_mol.tokenize(batch['ligand_smiles'])['input_ids']
                if batch.get('protein_seq'):
                    batch['prot_tokens'] = self.tokenizer_prot.tokenize(batch['protein_seq'])['input_ids']
                
                # Training step
                if self.task == "joint":
                    loss, loss_prot, loss_lig = self.train_step(batch, self.task)
                    train_losses.append(loss)
                    
                    # Log losses
                    wandb.log({
                        "train/loss": loss,
                        "train/loss_protein": loss_prot,
                        "train/loss_ligand": loss_lig,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "step": step
                    })
                else:
                    loss = self.train_step(batch, self.task)
                    train_losses.append(loss)
                    
                    wandb.log({
                        "train/loss": loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "step": step
                    })
                
                step = self.state['step']
                
                # Periodic logging
                if step % self.cfg.training.log_freq == 0:
                    avg_loss = np.mean(train_losses[-50:]) if train_losses else 0
                    print(f"Epoch {epoch}, Step {step}, Loss: {avg_loss:.5e}")
                
                # Periodic evaluation
                if step % self.cfg.training.eval_freq == 0 and step > 0:
                    self.evaluate_and_generate(epoch, step, best_eval_loss, wandbproj, wandbname)
                
                # Periodic checkpointing
                if step % self.cfg.training.snapshot_freq == 0 and step > 0:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, 
                        f"checkpoint_step_{step}.pth"
                    )
                    self.save_checkpoint(checkpoint_path)
            
            # End-of-epoch evaluation
            print(f"Running end-of-epoch evaluation for epoch {epoch}")
            best_eval_loss = self.evaluate_and_generate(
                epoch, step, best_eval_loss, wandbproj, wandbname, 
                is_end_of_epoch=True
            )
        
        wandb.finish()
        print("Training completed!")


    def evaluate_and_generate(self, epoch, step, best_eval_loss, wandbproj, wandbname, is_end_of_epoch=False):
        """Run evaluation and generation"""
        eval_losses = []
        eval_losses_prot = []
        eval_losses_lig = []
        
        # Evaluation loop
        eval_iter = iter(self.eval_ds)
        num_eval_batches = min(50, len(self.eval_ds))  # Limit evaluation batches
        
        for _ in tqdm(range(num_eval_batches), desc="Evaluating", leave=False):
            try:
                batch_eval = next(eval_iter)
            except StopIteration:
                break
            
            # Tokenize batch
            if batch_eval.get('ligand_smiles'):
                batch_eval['lig_tokens'] = self.tokenizer_mol.tokenize(batch_eval['ligand_smiles'])['input_ids']
            if batch_eval.get('protein_seq'):
                batch_eval['prot_tokens'] = self.tokenizer_prot.tokenize(batch_eval['protein_seq'])['input_ids']
            
            # Evaluation step
            if self.task == "joint":
                eval_loss, eval_loss_prot, eval_loss_lig = self.eval_step(batch_eval, self.task)
                eval_losses.append(eval_loss)
                eval_losses_prot.append(eval_loss_prot)
                eval_losses_lig.append(eval_loss_lig)
            else:
                eval_loss = self.eval_step(batch_eval, self.task)
                eval_losses.append(eval_loss)
        
        # Calculate average losses
        avg_eval_loss = np.mean(eval_losses) if eval_losses else float('inf')
        
        # Log evaluation metrics
        log_dict = {
            "eval/loss": avg_eval_loss,
            "epoch": epoch,
            "step": step
        }
        
        if self.task == "joint":
            avg_eval_loss_prot = np.mean(eval_losses_prot) if eval_losses_prot else float('inf')
            avg_eval_loss_lig = np.mean(eval_losses_lig) if eval_losses_lig else float('inf')
            log_dict.update({
                "eval/loss_protein": avg_eval_loss_prot,
                "eval/loss_ligand": avg_eval_loss_lig
            })
        
        wandb.log(log_dict)
        
        print(f"Epoch {epoch}, Step {step}, Eval Loss: {avg_eval_loss:.5e}")
        
        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            checkpoint_path = os.path.join(
                self.checkpoint_meta_dir,
                f"best_model_{wandbproj}_{wandbname}.pth"
            )
            self.save_checkpoint(checkpoint_path, is_best=True)
            print(f"Saved best model with eval loss: {avg_eval_loss:.5e}")
        
        # Generate samples for visualization
        if is_end_of_epoch or (step % (self.cfg.training.eval_freq * 5) == 0):
            self.generate_samples(batch_eval)
        
        return best_eval_loss

    def generate_samples(self, batch_eval):
        """Generate and log sample outputs"""
        try:
            # Generate proteins
            if self.task in ['protein_given_ligand', 'joint', 'protein_only']:
                if self.task != 'protein_only':
                    protein_gen = self.generation(
                        self.sampler_prot,
                        task='protein_given_ligand',
                        prot_indices=None,
                        lig_indices=batch_eval['lig_tokens'][:self.cfg.eval.batch_size]
                    )
                else:
                    protein_gen = self.generation(
                        self.sampler_prot,
                        task='protein_only',
                        prot_indices=None,
                        lig_indices=None
                    )
                
                if protein_gen:
                    print("\nGenerated Proteins:")
                    for i, pgen in enumerate(protein_gen[:3]):  # Show first 3
                        print(f"  {i+1}: {pgen[:100]}...")  # Show first 100 chars
                    
                    # Log to wandb
                    wandb.log({
                        "generated_proteins": wandb.Table(
                            columns=["index", "sequence"],
                            data=[[i, seq[:200]] for i, seq in enumerate(protein_gen[:5])]
                        )
                    })
            
            # Generate ligands
            if self.task in ['ligand_given_protein', 'joint', 'ligand_only']:
                if self.task != 'ligand_only':
                    ligand_gen = self.generation(
                        self.sampler_lig,
                        task='ligand_given_protein',
                        prot_indices=batch_eval['prot_tokens'][:self.cfg.eval.batch_size],
                        lig_indices=None
                    )
                else:
                    ligand_gen = self.generation(
                        self.sampler_lig,
                        task='ligand_only',
                        prot_indices=None,
                        lig_indices=None
                    )
                
                if ligand_gen:
                    print("\nGenerated Ligands:")
                    for i, lgen in enumerate(ligand_gen[:3]):  # Show first 3
                        print(f"  {i+1}: {lgen}")
                    
                    # Log to wandb
                    wandb.log({
                        "generated_ligands": wandb.Table(
                            columns=["index", "smiles"],
                            data=[[i, smiles] for i, smiles in enumerate(ligand_gen[:5])]
                        )
                    })
                    
        except Exception as e:
            print(f"Error during generation: {e}")

def run_train(
    work_dir,
    wandbproj,
    wandbname,
    cfg_fil=None,
    cfg_dict=None,
    datafile='./input_data/merged_plinder.pt',
    mol_emb_id="ibm/MoLFormer-XL-both-10pct",
    prot_emb_id="facebook/esm2_t30_150M_UR50D",
    dev_id="cuda:0",
    seed=42
):
    """Main training function"""
    trainer = Train_pl_sedd(
        work_dir=work_dir,
        cfg_fil=cfg_fil,
        cfg_dict=cfg_dict,
        datafile=datafile,
        mol_emb_id=mol_emb_id,
        prot_emb_id=prot_emb_id,
        dev_id=dev_id,
        seed=seed
    )
    
    trainer.train(wandbproj, wandbname)

import argparse


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Protein-Ligand Diffusion Model")
    
    parser.add_argument('-WD', '--work_dir', type=str, default='.', help='Working directory')
    parser.add_argument('-cf', '--config_file', type=str, default='configs/config.yaml', help='YAML config file')
    parser.add_argument('-wp', '--wandbproj', type=str, default='protlig_sedd', help='WandB project name')
    parser.add_argument('-wn', '--wandbname', type=str, default='run1', help='WandB run name')
    parser.add_argument('-df', '--datafile', type=str, default='./input_data/merged_plinder.pt', help='Input plinder data')
    parser.add_argument('-me', '--mol_emb_id', type=str, default="ibm/MoLFormer-XL-both-10pct", help='Model for mol embedding')
    parser.add_argument('-pe', '--prot_emb_id', type=str, default="facebook/esm2_t30_150M_UR50D", help='Model for protein embedding')
    parser.add_argument('-di', '--dev_id', type=str, default='cuda:0', help='Device')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-mw', '--model_weights', type=str, default=None, help='Path to pretrained model weights')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer with resume capability
    trainer = Train_pl_sedd(
        work_dir=args.work_dir,
        cfg_fil=args.config_file,
        datafile=args.datafile,
        model_weights=args.model_weights,
        mol_emb_id=args.mol_emb_id,
        prot_emb_id=args.prot_emb_id,
        dev_id=args.dev_id,
        seed=args.seed
    )
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_model()
        trainer.optim_state()
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.train(args.wandbproj, args.wandbname)

if __name__ == '__main__':
    main()

