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
    model_weights: str | None = None
    dev_id: str = 'cuda:0'
    seed: int = 42
    sampling_eps: float = 1e-3
    use_pretrained_conditioning: bool = False
    mol_emb_id: str = "ibm/MoLFormer-XL-both-10pct" # "seyonec/PubChem10M_SMILES_BPE_450k" # "ibm/MoLFormer-XL-both-10pct"
    prot_emb_id: str = "facebook/esm2_t30_150M_UR50D" # "Rostlab/prot_bert" # "facebook/esm2_t30_150M_UR50D"
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
        #self.embedding_mol_prot = get_embeddings.Embed_Mol_Prot(self.mol_emb_id, self.prot_emb_id)

        self.task = self.cfg.training.task # "ligand_given_protein" # "protein_given_ligand" # "joint" # "all"
        self.sample_dir = os.path.join(self.work_dir, "samples")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.checkpoint_meta_dir = os.path.join(self.work_dir, "checkpoints-meta", "checkpoint.pth")
        self.tokenizer_mol = Tok_Mol(self.mol_emb_id)
        self.tokenizer_prot = Tok_Prot(self.prot_emb_id)
        os.makedirs(self.sample_dir, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        os.makedirs(self.checkpoint_meta_dir, exist_ok = True)
        self.device = 'cuda'
        #self.device = torch.device(self.dev_id)
        #f"cuda:0" if torch.cuda.is_available() else "cpu")
        # -----------------------
        # Setup DDP
        # -----------------------
        #self.local_rank = int(os.environ['LOCAL_RANK'])
        #torch.cuda.set_device(self.local_rank)
        #dist.init_process_group(backend='nccl')
        #self.device = torch.device(f'cuda:{self.local_rank}')
        #if dist.get_rank() == 0:
        wandb.login()

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

    def freeze_model(self, model, freeze_what):
        for name, param in model.named_parameters():
            if freeze_what in name:
                print(f"Freeza {name}")
                param.requires_grad = False
        return model


    def load_model(
            self,
            ):
        # build token graph
        self.graph_prot = graph_lib.get_graph(self.cfg, self.device, tokens=36)
        self.graph_lig = graph_lib.get_graph(self.cfg, self.device, tokens=2364)
        # build score model
        self.score_model = ProteinLigandDiffusionModel(self.cfg).to(self.device) #SEDD(self.cfg).to(self.device)

        if self.model_weights!= None:
            self.score_model.load_state_dict(torch.load(self.model_weights))
        
            # Freeze ligand side for protein-only training
        if self.task in ['protein_given_ligand', 'protein_only']:
            self.score_model = self.freeze_model(self.score_model, 'lig')        
        elif self.task in ['ligand_given_protein', 'ligand_only']:
            self.score_model = self.freeze_model(self.score_model, 'prot')

        self.score_model.train()
        #summary(self.score_model, input_size = (4, 512), batch_size = -1)
        model_summ.count_parameters(self.score_model)
        #self.score_model = DDP(self.score_model, device_ids = [self.local_rank], output_device = self.local_rank)
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
                        
    def optimize_fn(self, optimizer, params):
        self.state['scaler'].unscale_(self.state['optimizer'])
        torch.nn.utils.clip_grad_norm_(params, self.cfg.optim.grad_clip)

        self.state['scaler'].step(self.state['optimizer'])
        self.state['scaler'].update()

    def calc_loss(self, batch, task, t = None, training=True):
        #if training:
        #    self.score_model.train()
        #else:
        #    self.score_model.eval()
        #batch = batch.to(self.device)
        if t is None:
            t = (1 - self.sampling_eps) * torch.rand(batch['prot_tokens'].shape[0], device=self.device) + self.sampling_eps
        
        sigma, dsigma = self.noise(t)
        sigma_reshaped = sigma.reshape(-1)
        if task == "protein_only":
            protein_indices = self.graph_prot.sample_transition(batch['prot_tokens'].to(self.device), sigma[:, None])
            log_score = self.score_model(protein_indices=protein_indices, timesteps = sigma_reshaped, mode=task)
        elif task == "ligand_only":
            ligand_indices = self.graph_lig.sample_transition(batch['lig_tokens'].to(self.device), sigma[:, None])
            log_score = self.score_model(ligand_indices=ligand_indices, timesteps = sigma_reshaped, mode=task)
        elif task == "joint":
            protein_indices = self.graph_prot.sample_transition(batch['prot_tokens'].to(self.device), sigma[:, None])
            ligand_indices = self.graph_lig.sample_transition(batch['lig_tokens'].to(self.device), sigma[:, None])
            #print(f"protein_indices_shape= {protein_indices.shape}, train={training}")
            #print(f"ligand indices= {ligand_indices.shape}, train={training}")
            log_score = self.score_model(protein_indices=protein_indices, ligand_indices=ligand_indices, timesteps = sigma_reshaped, mode=task)
            #print(f"log_score dim protein: {log_score[0].shape} \n logscore dim lig: {log_score[1].shape}")
        elif task == "ligand_given_protein":
            ligand_indices = self.graph.sample_transition(batch['lig_tokens'].to(self.device), sigma[:, None])
            log_score = self.score_model(ligand_indices=ligand_indices, protein_seq_str = batch['protein_seq'], timesteps = sigma_reshaped, mode=task, use_pretrained_conditioning=self.use_pretrained_conditioning)
        elif task == "protein_given_ligand":
            protein_indices = self.graph.sample_transition(batch['prot_tokens'].to(self.device), sigma[:, None])
            log_score = self.score_model(protein_indices=protein_indices, ligand_smiles_str=batch['ligand_smiles'], timesteps = sigma_reshaped, mode=task, use_pretrained_conditioning=self.use_pretrained_conditioning)

        else:
            raise NotImplementedError(f"Task {task} not implemented")

        if task in ['protein_only', 'protein_given_ligand']:
            loss = self.graph_prot.score_entropy(log_score, sigma[:, None], protein_indices, batch['prot_tokens'])
        elif task in ['ligand_only', 'ligand_given_protein']:
            loss = self.graph_lig.score_entropy(log_score, sigma[:, None], ligand_indices, batch['lig_tokens'])
        elif task == 'joint':
            loss_prot = self.graph_prot.score_entropy(log_score[0], sigma[:, None], protein_indices, batch['prot_tokens']).mean()
            loss_lig = self.graph_lig.score_entropy(log_score[1], sigma[:, None], ligand_indices, batch['lig_tokens']).mean()
            loss = (loss_prot + loss_lig)/2
            loss = (dsigma[:, None] * loss) 
            loss_prot = (dsigma[:, None] * loss_prot)
            loss_lig = (dsigma[:, None] * loss_lig)
            return loss, loss_prot, loss_lig
        else:
            raise NotImplementedError(f"Task {task} not implemented")

        loss = (dsigma[:, None] * loss)#.mean()
        return loss

    def train_step(self, batch, task):
        scaler = self.state['scaler']
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
        scaler.scale(loss).backward()
        if False:
            if task != "joint":
                scaler.scale(loss).backward()
            else:
                scaler.scale(loss_prot).backward(retain_graph=True)
                scaler.scale(loss_lig).backward()

        self.state['step'] += 1
        self.optimize_fn(self.state['optimizer'], self.score_model.parameters())
        self.state['ema'].update(self.score_model.parameters())
        self.state['optimizer'].step()
        self.state['optimizer'].zero_grad()
        #self.state['model'] = self.score_model
        self.scheduler.step()

        if task == "joint":
            return loss, loss_prot, loss_lig

        return loss 


    def eval_step(self, batch, task, cond=None):
        with torch.no_grad():
            ema = self.state['ema']
            ema.store(self.score_model.parameters())
            ema.copy_to(self.score_model.parameters())

            #print(batch)
            if task == "all":
                task = np.random.choice(["ligand_given_protein", "protein_given_ligand", "joint"])
                #print(f"Training task: {task}")

            if task == "joint":
                loss, loss_prot, loss_lig = self.calc_loss(batch, task, training=False)
                loss = loss.mean()
                loss_prot = loss_prot.mean()
                loss_lig = loss_lig.mean()
                ema.restore(self.score_model.parameters())
                return loss, loss_prot, loss_lig 

            loss = self.calc_loss(batch, task, training=False).mean()
            ema.restore(self.score_model.parameters())
        return loss

    def setupsampler(self, datatype, sampling_shape): #sampling_shape = (batch['prot_tokens'].shape[0], self.cfg.model.length)
        ema = self.state['ema']
        ema.store(self.score_model.parameters())
        ema.copy_to(self.score_model.parameters())

        ema.restore(self.score_model.parameters())
        with torch.no_grad():
            if datatype == 'protein':
                graph_sampler = self.graph_prot
            elif datatype == 'ligand':
                graph_sampler = self.graph_lig

            sampling_fn = sampling_dd.get_sampling_fn(
                               self.cfg,
                               graph_sampler,
                               self.noise,
                               sampling_shape,
                               self.sampling_eps,
                               self.device)

        return sampling_fn

    def generation(self, sampling_fn, model, tokenizer, task, prot_indices = None, lig_indices = None):
        samples = sampling_fn(model, task, prot_indices = prot_indices, lig_indices = lig_indices)
        #print(f"these are untok samples: {samples}")
        #print(task)
        if task in ['protein_given_ligand', 'protein_only']:
            real_samples = tokenizer.protein_tokenizer.batch_decode(samples)
        elif task in ['ligand_given_protein', 'ligand_only']:
            real_samples = tokenizer.mol_tokenizer.batch_decode(samples)
        return real_samples

    def train(self, wandbproj, wandbname): 
        if True:#dist.get_rank() == 0:
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
        """
        TO-DO: add the train and eval step into the train loop
        """
        self.cfg.optim.lr = float(self.cfg.optim.lr)
        self.setup_loaders()
        self.load_model()
        self.optim_state()
        self.scheduler = WarmupCosineLR(
                            self.state['optimizer'],
                            warmup_steps=self.cfg.optim.warmup,
                            max_steps=self.cfg.training.n_iters,
                            base_lr=self.cfg.optim.lr/10,
                            max_lr=self.cfg.optim.lr)

        self.sampler_prot = self.setupsampler('protein', (self.cfg.training.batch_size, self.cfg.data.max_protein_len))
        self.sampler_lig = self.setupsampler('ligand', (self.cfg.training.batch_size, self.cfg.data.max_ligand_len))

        #if self.cfg.training.snapshot_sampling:
        #    sampling_shape = (self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum), self.cfg.model.length)
        #    self.sampling_fn = sampling.get_sampling_fn(self.cfg, self.graph, self.noise, sampling_shape, self.sampling_eps, self.device)

        num_train_steps = self.cfg.training.n_iters
        print(f"Starting training loop at step {self.initial_step}.")
        print(f"Training for {self.cfg.training.epochs} epochs or {num_train_steps} steps.")

        step = self.state['step']
        eval_loss_list = []
        train_iter = iter(self.train_ds)  # Reset iterator at start of epoch
        eval_iter = iter(self.eval_ds)

        for ep in tqdm(range(self.cfg.training.epochs)):
            #self.train_sampler.set_epoch(ep)  # Ensures data is shuffled differently each epoch
            train_iter = iter(self.train_ds)  # Reset iterator at start of epoch            
            eval_iter = iter(self.eval_ds)

            """
            Training loop
            1. Get train batch
            2. Tokenize train batch
            3. Get embeddings for train batch
            4. Train step
            5. Log train loss
            """
            
            #while True:
            for _ in tqdm(range(len(self.train_ds))):
                if step >= num_train_steps:
                    print(f"Reached max training steps: {num_train_steps}. Ending training.")
                    return  # Exit training early

                try:
                    batch = next(train_iter)
                    if batch.get('ligand_smiles'):
                        batch['lig_tokens'] = self.tokenizer_mol.tokenize(batch['ligand_smiles'])['input_ids']
                    if batch.get('protein_seq'):
                        batch ['prot_tokens'] = self.tokenizer_prot.tokenize(batch['protein_seq'])['input_ids']
                    #print(f"Batch ligand smiles: {batch['ligand_smiles']}, protein seq: {batch['protein_seq']}")
                    #print(f"Batch ligand tokens: {batch['lig_tokens']}, protein tokens: {batch['prot_tokens']}")
                    #print(f"Batch ligand tokens shape: {batch['lig_tokens'].shape}, protein tokens shape: {batch['prot_tokens'].shape}")
                except StopIteration:
                    # End of dataset for this epoch
                    break
                
                if self.task == "joint":
                    loss, loss_prot, loss_lig = self.train_step(batch, self.task)
                else:
                    loss = self.train_step(batch, self.task)#, esm_cond=esm_cond, mol_cond=mol_cond)
                step = self.state['step']
                # Logging and checkpointing
                wandb.log({"training_loss": loss.item()})
                if self.task == "joint":
                    wandb.log({"training_loss_prot": loss_prot.item()})
                    wandb.log({"training_loss_lig": loss_lig.item()})
                if step % self.cfg.training.log_freq == 0:
                    print(f"epoch: {ep}, step: {step}, training_loss: {loss.item():.5e}")
                if step % self.cfg.training.eval_freq == 0 and step > 0:
                    try:
                        batch_eval = next(eval_iter)
                        if batch_eval.get('ligand_smiles'):
                            batch_eval['lig_tokens'] = self.tokenizer_mol.tokenize(batch_eval['ligand_smiles'])['input_ids']
                        if batch_eval.get('protein_seq'):
                            batch_eval['prot_tokens'] = self.tokenizer_prot.tokenize(batch_eval['protein_seq'])['input_ids']
                        ##### Turn back on before actual training!!!!!!!! #####
                        if True:
                            if self.task != 'joint':
                                eval_loss = self.eval_step(batch_eval, self.task)#, esm_cond=esm_cond, mol_cond=mol_cond)
                                
                                wandb.log({"eval_loss": eval_loss.item()})
                            else:
                                eval_loss, eval_loss_prot, eval_loss_lig = self.eval_step(batch_eval, self.task) 
                                wandb.log({"eval_loss_prot": eval_loss_prot.item()})
                                wandb.log({"eval_loss_lig": eval_loss_lig.item()})
                                wandb.log({"eval_loss": eval_loss.item()})
                            print(f"epoch: {ep}, step: {step}, evaluation_loss: {eval_loss.item():.5e}")
                            eval_loss_list.append(eval_loss.item())
                            if eval_loss.item() <= min(eval_loss_list):
                                utils.save_checkpoint(f'{self.checkpoint_meta_dir}/check_{wandbproj}_{wandbname}.pth', self.state)
                        
                        if self.task in ['protein_given_ligand', 'joint', 'protein_only']:
                            if self.task != 'protein_only':
                                protein_gen = self.generation(
                                                        self.sampler_prot,
                                                        self.score_model,
                                                        self.tokenizer_prot,
                                                        task='protein_given_ligand',
                                                        prot_indices = None,
                                                        lig_indices = batch_eval['lig_tokens'])
                        
                            else:
                                protein_gen = self.generation(
                                                        self.sampler_prot,
                                                        self.score_model,
                                                        self.tokenizer_prot,
                                                        task='protein_only',
                                                        prot_indices=None,
                                                        lig_indices=None)

                            print("Generated Proteins given ligands!!!!!")
                            for pgen in protein_gen:
                                print(pgen)

                        if self.task in ['ligand_given_protein', 'joint']:
                            ligand_gen_given_prot = self.generation(
                                                        self.sampler_lig,
                                                        self.score_model,
                                                        self.tokenizer_mol,
                                                        task='ligand_given_protein',
                                                        prot_indices = batch_eval['prot_tokens'],
                                                        lig_indices = None)

                            print("Generated Ligands given proteins!!!!!!")
                            for lgen in ligand_gen_given_prot:
                                print(lgen)

                    except StopIteration:
                        break
                    
                        # End of eval dataset for this epoch
                """
                Evaluation step
                1. Get eval batch
                2. Tokenize eval batch
                3. Get embeddings for eval batch
                4. Eval step
                5. Log eval loss
                6. Save checkpoint if needed
                """

            eval_iter = iter(self.eval_ds)

            for _ in range(len(self.eval_ds)):
                try:
                    batch_eval = next(eval_iter)
                    if batch_eval.get('ligand_smiles'):
                        batch_eval['lig_tokens'] = self.tokenizer_mol.tokenize(batch_eval['ligand_smiles'])['input_ids']
                    if batch_eval.get('protein_seq'):
                        batch_eval['prot_tokens'] = self.tokenizer_prot.tokenize(batch_eval['protein_seq'])['input_ids']
                    ##### Turn back on before actual training!!!!!!!! #####
                    if True:
                        if self.task != 'joint':
                            eval_loss = self.eval_step(batch_eval, self.task)#, esm_cond=esm_cond, mol_cond=mol_cond)
                            
                            wandb.log({"eval_loss": eval_loss.item()})
                        else:
                            eval_loss, eval_loss_prot, eval_loss_lig = self.eval_step(batch_eval, self.task) 
                            wandb.log({"eval_loss_prot": eval_loss_prot.item()})
                            wandb.log({"eval_loss_lig": eval_loss_lig.item()})
                            wandb.log({"eval_loss": eval_loss.item()})
                        print(f"epoch: {ep}, step: {step}, evaluation_loss: {eval_loss.item():.5e}")
                        eval_loss_list.append(eval_loss.item())
                        if eval_loss.item() <= min(eval_loss_list):
                            utils.save_checkpoint(f'{self.checkpoint_meta_dir}/check_{wandbproj}_{wandbname}.pth', self.state)
                    
                    if self.task in ['protein_given_ligand', 'joint', 'protein_only']:
                        if self.task != 'protein_only':
                            protein_gen = self.generation(
                                                    self.sampler_prot,
                                                    self.score_model,
                                                    self.tokenizer_prot,
                                                    task='protein_given_ligand',
                                                    prot_indices = None,
                                                    lig_indices = batch_eval['lig_tokens'])
                    
                        else:
                            protein_gen = self.generation(
                                                    self.sampler_prot,
                                                    self.score_model,
                                                    self.tokenizer_prot,
                                                    task='protein_only',
                                                    prot_indices=None,
                                                    lig_indices=None)
                                                                                                                               
                        print("Generated Proteins given ligands!!!!!")
                        for pgen in protein_gen:
                            print(pgen)
                                                                                                                               
                    if self.task in ['ligand_given_protein', 'joint']:
                        ligand_gen_given_prot = self.generation(
                                                    self.sampler_lig,
                                                    self.score_model,
                                                    self.tokenizer_mol,
                                                    task='ligand_given_protein',
                                                    prot_indices = batch_eval['prot_tokens'],
                                                    lig_indices = None)
                                                                                                                               
                        print("Generated Ligands given proteins!!!!!!")
                        for lgen in ligand_gen_given_prot:
                            print(lgen)
                                                                                                                               
                except StopIteration:
                    break

               #     try:
               #         batch_eval = next(eval_iter)
               #         batch_eval['lig_tokens'] = self.tokenizer_mol.tokenize(batch_eval['ligand_smiles'])['input_ids']
               #         batch_eval['prot_tokens'] = self.tokenizer_prot.tokenize(batch_eval['protein_seq'])['input_ids']
               #         eval_loss = self.eval_step(batch_eval, self.state['model'], self.task)#, esm_cond=esm_cond, mol_cond=mol_cond)
               #         print(f"epoch: {ep}, step: {step}, evaluation_loss: {eval_loss.item():.5e}")
               #     except StopIteration:
               #         # End of eval dataset for this epoch
               #         break
               # 
               # wandb.log({"eval_loss": eval_loss.item()})
               # eval_loss_list.append(eval_loss.item())
               # if eval_loss.item() <= min(eval_loss_list):
               #     utils.save_checkpoint(f'{self.checkpoint_meta_dir}/check.pth', self.state)


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




























if False:
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    import wandb
    
    # -----------------------
    # Setup DDP
    # -----------------------
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{local_rank}')
    
    # -----------------------
    # Initialize W&B (only on rank 0)
    # -----------------------
    if dist.get_rank() == 0:
        wandb.init(project="discrete-diffusion", name="cosine_scheduler_ddp")
    
    # -----------------------
    # Dataset and Dataloader
    # -----------------------
    train_dataset = MyDataset()
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    
    # -----------------------
    # Model and optimizer
    # -----------------------
    model = MyDiffusionModel().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # -----------------------
    # Training Loop
    # -----------------------
    num_epochs = 10
    num_timesteps = model.num_timesteps  # or however many timesteps your model has
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        # Accumulate per-timestep metrics
        epoch_loss_t = torch.zeros(num_timesteps, device=device)
        epoch_acc_t = torch.zeros(num_timesteps, device=device)
        count_t = torch.zeros(num_timesteps, device=device)
        
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            # Forward pass
            loss_t, pred_t, target_t = model(batch)  
            # loss_t: [num_timesteps]
            # pred_t: [batch_size, seq_len, num_classes] per timestep
            # target_t: [batch_size, seq_len] per timestep
    
            # Backward and step
            total_loss = loss_t.mean()
            total_loss.backward()
            optimizer.step()
            
            # Per-timestep accuracy
            pred_classes = pred_t.argmax(dim=-1)  # [batch, seq_len]
            acc_t = (pred_classes == target_t).float().mean(dim=1)  # mean over seq_len
            
            # Accumulate metrics
            epoch_loss_t += loss_t.detach()
            epoch_acc_t += acc_t.detach()
            count_t += 1
    
        # Average across batches
        epoch_loss_t /= count_t
        epoch_acc_t /= count_t
        
        # Reduce across GPUs
        dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_acc_t, op=dist.ReduceOp.SUM)
        epoch_loss_t /= dist.get_world_size()
        epoch_acc_t /= dist.get_world_size()
        
        # Log to W&B (only on rank 0)
        if dist.get_rank() == 0:
            timestep_table = wandb.Table(columns=["timestep", "loss", "accuracy"])
            for t in range(num_timesteps):
                timestep_table.add_data(int(t), float(epoch_loss_t[t]), float(epoch_acc_t[t]))
            wandb.log({
                "epoch": epoch,
                "timestep_metrics": timestep_table,
                "overall_loss": float(epoch_loss_t.mean()),
                "overall_accuracy": float(epoch_acc_t.mean())
            })
    
