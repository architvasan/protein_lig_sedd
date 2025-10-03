"""
Training script for SMILES data with proper tokenization.
"""

# Optional Intel extensions for Aurora XPU support
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
    INTEL_AVAILABLE = True
except ImportError:
    INTEL_AVAILABLE = False
    print("⚠️  Intel extensions not found, running on CPU/CUDA")

import datetime
import os
import sys
import gc
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

from itertools import chain
import wandb
import numpy as np
import torch
from tqdm import tqdm
import yaml
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import protlig_dd.processing.losses as losses
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
from protlig_dd.processing.noise_lib import sample_timesteps_curriculum
from protlig_dd.processing.noise_lib import get_curriculum_stats
import protlig_dd.utils.utils as utils
from protlig_dd.data.tokenize import Tok_Mol, Tok_SmilesPE
from protlig_dd.model.ema import ExponentialMovingAverage
from protlig_dd.utils.lr_scheduler import WarmupCosineLR
import protlig_dd.sampling.sampling as sampling
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import json


class SMILESDataset(torch.utils.data.Dataset):
    """Dataset class for processed SMILES data."""

    def __init__(self, data_file, tokenizer):
        self.data = torch.load(data_file, weights_only=False)
        self.tokenizer = tokenizer
        print(f"Loaded {len(self.data)} SMILES from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Tokenize SMILES on-the-fly
        smiles = sample['ligand_smiles'] if 'ligand_smiles' in sample else sample['smiles']
        tokens = self.tokenizer.tokenize([smiles])['input_ids'].squeeze(0)
        return tokens


def safe_getattr(obj, path, default=None):
    """Safely get nested attributes with default fallback."""
    try:
        parts = path.split('.')
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current
    except AttributeError:
        return default


@dataclass
class SMILESTrainer:
    """
    Trainer for SMILES data with proper tokenization.
    """
    work_dir: str
    config_file: str
    datafile: str = '/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/input_data/processed_pubchem.pt'
    dev_id: str = 'cuda:0'
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    force_fresh_start: bool = False
    
    def __post_init__(self):
        """Initialize trainer components."""
        print(f"SMILES Trainer initialized. Device: {self.dev_id}")

        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Setup device
        self.device = self.setup_device(self.dev_id)
        print(f"✅ Using device: {self.device}")

        # Load configuration
        with open(self.config_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        self.cfg = utils.Config(dictionary=cfg_dict)

        # Setup directories
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.sample_dir = os.path.join(self.work_dir, "samples")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = Tok_SmilesPE(vocab_file = "/lus/flare/projects/FoundEpidem/xlian/protein_lig_sedd/VocabFiles/vocab_spe.txt",
                                      spe_file = "/lus/flare/projects/FoundEpidem/xlian/protein_lig_sedd/VocabFiles/SPE_ChEMBL.txt") #Tok_Mol()

        # Device-specific setup
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            self.use_amp = True
            self.scaler = None
            print("✅ CUDA mixed precision will be enabled")
        elif device_type == 'xpu':
            self.use_amp = True
            self.scaler = None
            print("✅ XPU mixed precision will be enabled")
        else:
            self.use_amp = False
            self.scaler = None
            print(f"✅ {device_type.upper()} training without mixed precision")

        # Verify config sections
        required_sections = ['model', 'training', 'optim', 'data', 'noise']
        for section in required_sections:
            if not hasattr(self.cfg, section):
                print(f"⚠️  Warning: Missing config section '{section}' - using defaults")
                setattr(self.cfg, section, utils.Config(dictionary={}))

    def setup_device(self, dev_id):
        """Setup device with cross-platform compatibility."""
        if dev_id == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print("🚀 Auto-detected: CUDA GPU")
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device("xpu:0")
                print("🔥 Auto-detected: Intel XPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🍎 Auto-detected: Apple Silicon MPS")
            else:
                device = torch.device("cpu")
                print("💻 Auto-detected: CPU")
        elif dev_id.startswith("xpu"):
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device(dev_id)
                print(f"🔥 Using Intel XPU: {dev_id}")
            else:
                print("⚠️  XPU not available, falling back to CPU")
                device = torch.device("cpu")
        elif dev_id == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🍎 Using Apple Silicon MPS")
            else:
                print("⚠️  MPS not available, falling back to CPU")
                device = torch.device("cpu")
        elif dev_id.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device(dev_id)
                print(f"🚀 Using CUDA GPU: {dev_id}")
            else:
                print("⚠️  CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device(dev_id)
            print(f"💻 Using device: {dev_id}")

        return device
    
    def setup_data_loaders(self):
        """Setup data loaders for SMILES data."""

        if Path(self.datafile).exists():
            print(f"Loading SMILES dataset: {self.datafile}")
            self.setup_custom_data_loaders()
        else:
            raise FileNotFoundError(f"SMILES data file not found: {self.datafile}")

    def setup_custom_data_loaders(self):
        """Setup data loaders for SMILES data."""
        from torch.utils.data import DataLoader, random_split

        # Load dataset with tokenizer
        dataset = SMILESDataset(self.datafile, self.tokenizer)

        # Split into train/val
        train_ratio = safe_getattr(self.cfg, 'data.train_ratio', 0.9)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Create data loaders
        batch_size = safe_getattr(self.cfg, 'training.batch_size', 16)
        num_workers = safe_getattr(self.cfg, 'training.num_workers', 0)

        device_type = str(self.device).split(':')[0]
        pin_memory = device_type in ['cuda', 'xpu']

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"Batch size: {batch_size}")
    
    def setup_model(self):
        """Setup model for SMILES training."""
        print("Setting up model for SMILES...")
        
        # Build graph for absorbing diffusion
        self.graph = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.tokens)
        
        # Build noise schedule
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        
        # Build model - use SMILES-only model
        from protlig_dd.model.transformer_v100 import SEDD
        self.model = SEDD(self.cfg).to(self.device)
        print(f"✅ Using SEDD model for SMILES-only training")
        
        # Setup EMA
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), 
            decay=self.cfg.training.ema
        )
        
        print(f"Model ready. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    # ---------------------- SMILES generation helpers ----------------------
    def decode_smiles(self, token_tensor):
        """Decode token ids (1D tensor or list) to a SMILES string using the Mol tokenizer."""
        try:
            if torch.is_tensor(token_tensor):
                ids = token_tensor.cpu().tolist()
            else:
                ids = list(token_tensor)

            # Use the AutoTokenizer.decode if available
            tok = getattr(self.tokenizer, 'mol_tokenizer', None)
            if tok is not None and hasattr(tok, 'decode'):
                # remove padding tokens when decoding
                return tok.decode([int(x) for x in ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            else:
                # Fallback: join ids as string (rare)
                return ''.join([str(x) for x in ids])
        except Exception as e:
            print(f"⚠️  Error decoding SMILES tokens: {e}")
            return ''

    def setup_smiles_sampler(self, batch_size: int = 1, max_length: int = 128, eps: float = 1e-5):
        """Create a PC sampler for SMILES generation using the repo sampling framework."""
        sampling_shape = (batch_size, max_length)
        sampling_fn = sampling.get_sampling_fn(
            config=self.cfg,
            graph=self.graph,
            noise=self.noise,
            batch_dims=sampling_shape,
            eps=eps,
            device=self.device
        )
        return sampling_fn

    def generate_smiles_rigorous(self, num_samples: int = 10, max_length: int = 128):
        """Generate SMILES sequences using the CTMC rigorous sampler."""
        print(f"🧪 Generating {num_samples} SMILES (rigorous sampler) ...")
        self.model.eval()
        generated = []

        with torch.no_grad():
            # use EMA weights for generation
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

            sampler = self.setup_smiles_sampler(batch_size=num_samples, max_length=max_length)

            # Model wrapper matching sampling framework expectations
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model

                def __call__(self, x=None, sigma=None, **kwargs):
                    # Called by sampler using ligand_indices / timesteps
                    if 'ligand_indices' in kwargs and 'timesteps' in kwargs:
                        ligand_indices = kwargs['ligand_indices']
                        timesteps = kwargs['timesteps']
                        mode = kwargs.get('mode', 'ligand_only')

                        if mode in ('ligand_only', 'ligand_given_protein'):
                            return self.model(ligand_indices, timesteps)
                        else:
                            raise ValueError(f"SMILES model only supports ligand modes, got {mode}")

                    elif x is not None and sigma is not None:
                        # Legacy positional interface
                        return self.model(x, sigma)
                    else:
                        raise ValueError(f"ModelWrapper called with invalid args: x={x}, sigma={sigma}, kwargs={list(kwargs.keys())}")

                def eval(self):
                    self.model.eval(); return self
                def train(self, mode=True):
                    self.model.train(mode); return self
                def parameters(self):
                    return self.model.parameters()
                def state_dict(self):
                    return self.model.state_dict()
                def to(self, device):
                    self.model.to(device); return self
                @property
                def device(self):
                    return next(self.model.parameters()).device

            model_wrapper = ModelWrapper(self.model)
            samples = sampler(model_wrapper, task="ligand_only")

            for i in range(num_samples):
                sample_tokens = samples[i] if len(samples.shape) > 1 else samples
                seq = self.decode_smiles(sample_tokens)
                validity = Chem.MolFromSmiles(seq) is not None and len(seq) > 0
                generated.append({'sample_id': i, 
                                    'raw_tokens': sample_tokens[:max_length].cpu().tolist() if torch.is_tensor(sample_tokens) else sample_tokens, 
                                    'smiles': seq, 
                                    'validity': validity})
            self.ema.restore(self.model.parameters())

        self.model.train()
        return generated


    def generate_smiles(self, num_samples: int = 10, max_length: int = 128, sampling_method: str = 'rigorous'):
        if sampling_method == 'rigorous':
            return self.generate_smiles_rigorous(num_samples, max_length)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    def generate_and_save_smiles(self, step: int, num_samples: int = 100):
        """Generate SMILES samples using the saved checkpoint (EMA weights)."""
        print(f"🧪 Generating {num_samples} SMILES at step {step} using checkpoint weights...")

        try:
            # Get generation parameters from config
            sampling_method = safe_getattr(self.cfg, 'sampling.method', 'rigorous')
            max_len = safe_getattr(self.cfg, 'data.max_ligand_len', 128)

            # Switch to eval mode and use EMA weights for generation
            self.model.eval()
            with torch.no_grad():
                # Store current weights and load EMA weights
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model.parameters())

                # Generate SMILES using the EMA weights
                print("  Sampling in progress (this may show debug output)...")
                generated = self.generate_smiles(num_samples, max_length=max_len, sampling_method=sampling_method)
                print(f"  ✅ Generated {len(generated)} samples")

                # Restore training weights
                self.ema.restore(self.model.parameters())

            # Switch back to training mode
            self.model.train()

            # Save generated samples to sample_dir
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            jsonl_path = os.path.join(self.sample_dir, f'samples_step_{step}_{timestamp}.jsonl')
            txt_path = os.path.join(self.sample_dir, f'samples_step_{step}_{timestamp}.txt')

            with open(jsonl_path, 'w') as jf, open(txt_path, 'w') as tf:
                for rec in generated:
                    jf.write(json.dumps(rec) + '\n')
                    tf.write((rec.get('smiles', '') or '') + '\n')

            print(f"✅ Saved {len(generated)} generated SMILES to:")
            print(f"   JSONL: {jsonl_path}")
            print(f"   TXT: {txt_path}")

            # Log to wandb
            wandb.log({
                'samples/generated_count': len(generated),
                'samples/last_file': jsonl_path,
                'samples/method': sampling_method,
                'samples/step': step
            }, step=step)

            return generated

        except Exception as e:
            print(f"⚠️  Error during SMILES generation at step {step}: {e}")
            import traceback
            traceback.print_exc()
            return []


    def setup_optimizer(self):
        """Setup optimizer."""
        print("Setting up optimizer...")
        
        self.optimizer = losses.get_optimizer(
            self.cfg,
            chain(self.model.parameters(), self.noise.parameters())
        )
        
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            self.use_amp = True
            print("✅ Using CUDA mixed precision training")
        elif device_type == 'xpu':
            if INTEL_AVAILABLE:
                print("🔧 Applying IPEX optimization for Intel XPU...")
                try:
                    # Move model to device first
                    self.model.to(self.device)
                    # Apply IPEX optimization
                    self.model, self.optimizer = ipex.optimize(
                        self.model,
                        optimizer=self.optimizer
                    )
                    self.scaler = torch.xpu.amp.GradScaler()
                    self.use_amp = True
                    print("✅ IPEX optimization applied with XPU mixed precision")
                except Exception as e:
                    print(f"⚠️  IPEX optimization failed: {e}")
                    print("   Continuing without IPEX optimization...")
                    self.scaler = None
                    self.use_amp = False
            else:
                print("⚠️  Intel extensions not available - skipping IPEX optimization")
                self.scaler = None
                self.use_amp = False
        else:
            self.scaler = None
            self.use_amp = False
            print(f"✅ Using {device_type.upper()} without mixed precision")
        
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.cfg.optim.warmup,
            max_steps=self.cfg.training.n_iters,
            base_lr=self.cfg.optim.lr * 0.1,
            max_lr=self.cfg.optim.lr,
            min_lr=self.cfg.optim.lr * 0.01
        )
        
        self.state = {
            'optimizer': self.optimizer,
            'scaler': self.scaler,
            'scheduler': self.scheduler,
            'model': self.model.state_dict(),
            'noise': self.noise.state_dict(),
            'ema': self.ema.state_dict(),
            'step': 0,
            'epoch': 0
        }
        
        print("Optimizer ready.")

    def compute_loss(self, batch):

        # Sample timesteps
        if (hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled and
            hasattr(self.cfg.curriculum, 'probabilistic') and self.cfg.curriculum.probabilistic):
            # Use probabilistic curriculum: bias timestep sampling toward lower noise early in training

            t = sample_timesteps_curriculum(
                batch_size=batch.shape[0],
                device=self.device,
                training_step=self.state['step'],
                preschool_time=getattr(self.cfg.curriculum, 'preschool_time', 5000),
                curriculum_type=getattr(self.cfg.curriculum, 'difficulty_ramp', 'exponential'),
                bias_strength=getattr(self.cfg.curriculum, 'bias_strength', 2.0)
            )
            if self.state['step'] % 100 == 0:
                stats = get_curriculum_stats(t, self.state['step'],
                                           getattr(self.cfg.curriculum, 'preschool_time', 5000))
                print(f"📊 Probabilistic curriculum: step={self.state['step']}, "
                      f"progress={stats['curriculum_progress']:.2f}, "
                      f"avg_t={stats['mean_timestep']:.3f}, "
                      f"low_noise%={stats['low_noise_fraction']*100:.1f}, "
                      f"high_noise%={stats['high_noise_fraction']*100:.1f}")
        else:
            # Standard uniform timestep sampling
            t = torch.rand(batch.shape[0], device=self.device) * (1 - 1e-3) + 1e-3

        sigma, dsigma = self.noise(t)

        # Sample transition (add noise to the data)
        # Apply curriculum learning
        if hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled:
            perturbed_batch = self.graph.sample_transition_curriculum(
                batch,
                sigma,  # Fixed: removed [:, None] as graph functions handle broadcasting internally
                self.state['step'],
                preschool_time=self.cfg.curriculum.preschool_time,
                curriculum_type=getattr(self.cfg.curriculum, 'difficulty_ramp', 'exponential')
            )
        else:
            perturbed_batch = self.graph.sample_transition(batch, sigma) 

        #print('batch',batch.shape,  'perturbed_batch',perturbed_batch.shape)
        #print('batch',batch,  'perturbed_batch',perturbed_batch)
        #exit()

        # Validate perturbed_batch is now 2D: [batch_size, sequence_length]
        if perturbed_batch.dim() != 2:
            raise ValueError(f"perturbed_batch must be 2D [batch_size, seq_len], got {perturbed_batch.dim()}D with shape {perturbed_batch.shape}")

        # Forward pass with device-aware autocast and shape validation
        try:
            device_type = str(self.device).split(':')[0]
            if self.use_amp and device_type == 'xpu':
                with torch.xpu.amp.autocast():
                    log_score = self.model(perturbed_batch, sigma)
                    loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                    # Weight by dsigma for better training dynamics
                    #loss = (dsigma[:, None] * loss).mean()
            elif self.use_amp and device_type == 'cuda':
                with torch.amp.autocast('cuda'):
                    sigma_expanded = sigma[:, None].expand(-1, batch.shape[1])

                    log_score = self.model(perturbed_batch, sigma)
                    loss = self.graph.score_entropy(log_score, sigma_expanded, perturbed_batch, batch)

                    # Weight by dsigma for better training dynamics
                    loss = (dsigma[:, None] * loss).mean()
            else:
                # No autocast for CPU/MPS
                log_score = self.model(perturbed_batch, sigma)
                loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

                # Weight by dsigma for better training dynamics
                loss = (dsigma[:, None] * loss).mean()
        except Exception as e:
            if "Wrong shape" in str(e) or "einops" in str(e).lower():
                print(f"❌ Shape error in model forward pass:")
                print(f"   perturbed_batch shape: {perturbed_batch.shape}")
                print(f"   sigma shape: {sigma.shape}")
                print(f"   batch shape: {batch.shape}")
                print(f"   Error: {e}")

                # Try to fix the shape issue
                if perturbed_batch.dim() > 2:
                    print(f"🔧 Attempting to fix perturbed_batch shape...")
                    perturbed_batch = perturbed_batch.view(perturbed_batch.shape[0], -1)
                    print(f"   Fixed shape: {perturbed_batch.shape}")

                    # Retry the forward pass
                    device_type = str(self.device).split(':')[0]
                    if self.use_amp and device_type == 'xpu':
                        with torch.xpu.amp.autocast():
                            log_score = self.model(perturbed_batch, sigma)
                            loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                            loss = (dsigma[:, None] * loss).mean()
                    elif self.use_amp and device_type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            log_score = self.model(perturbed_batch, sigma)
                            loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                            loss = (dsigma[:, None] * loss).mean()
                    else:
                        log_score = self.model(perturbed_batch, sigma)
                        loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                        loss = (dsigma[:, None] * loss).mean()
                else:
                    raise e
            else:
                raise e
        
        return loss

    def train(self, project_name: str, run_name: str):
        """Main training loop for SMILES."""
        print("Starting SMILES training...")
        
        # Setup components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_optimizer()
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=self.cfg.__dict__
        )
        
        # Loss function will be computed inline using compute_loss method
        
        # Training loop
        step = 0
        max_steps = self.cfg.training.n_iters
        
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            # Track running loss for the epoch
            running_loss = 0.0
            running_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                if step >= max_steps:
                    print(f"Reached max steps: {max_steps}")
                    return
                
                batch = batch.to(self.device)
                #print(batch[0])

                # Ensure batch is 2D: [batch_size, sequence_length]
                if batch.dim() > 2:
                    #print(f"WARNING: Batch has {batch.dim()} dimensions, reshaping from {batch.shape}")
                    batch = batch.view(batch.shape[0], -1)
                    #print(f"Reshaped batch to: {batch.shape}")

                # Compute loss using the inline method
                loss = self.compute_loss(batch) / self.cfg.training.accum

                # Accumulate for epoch statistics (use un-reduced scalar)
                try:
                    running_loss += loss.item()
                except Exception:
                    # If loss is a tensor requiring grad, detach then convert
                    running_loss += loss.detach().cpu().item()
                running_batches += 1
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Update EMA and scheduler
                self.ema.update(self.model.parameters())
                self.scheduler.step()
                
                # Logging
                if step % 1000 == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': step,
                        'train/epoch': epoch
                    }, step=step)

                    print(f"Step {step}, Loss: {loss.item():.6f}")
                
                if step % 10000 == 0 and step > 0:
                    self.save_checkpoint(step)
                    self.generate_and_save_smiles(step)
                
                step += 1
                ######## for curriculum learning ########
                self.state['step'] = step
                ######## for curriculum learning ########
                
            # End of epoch: compute and log epoch-level training loss
            if running_batches > 0:
                epoch_train_loss = running_loss / running_batches
            else:
                epoch_train_loss = float('nan')

            # Run validation pass
            epoch_val_loss = None
            if hasattr(self, 'val_loader'):
                self.model.eval()
                val_running = 0.0
                val_batches = 0
                with torch.no_grad():
                    for vbatch in self.val_loader:
                        vbatch = vbatch.to(self.device)
                        if vbatch.dim() > 2:
                            vbatch = vbatch.view(vbatch.shape[0], -1)
                        vloss = self.compute_loss(vbatch)
                        try:
                            val_running += vloss.item()
                        except Exception:
                            val_running += vloss.detach().cpu().item()
                        val_batches += 1

                if val_batches > 0:
                    epoch_val_loss = val_running / val_batches
                else:
                    epoch_val_loss = float('nan')

                # Return to train mode for next epoch
                self.model.train()

            # Log epoch-level metrics to wandb and stdout
            wandb.log({
                'train/epoch_loss': epoch_train_loss,
                'train/epoch': epoch,
            })
            if epoch_val_loss is not None:
                wandb.log({'val/loss': epoch_val_loss, 'val/epoch': epoch})

            print(f"Epoch {epoch} summary: train_loss={epoch_train_loss:.6f}, val_loss={epoch_val_loss}")

            # ---------------- Generate SMILES at epoch end ----------------
            num_gen = 100
            sampling_method = safe_getattr(self.cfg, 'sampling.method', 'rigorous')
            max_len = safe_getattr(self.cfg, 'data.max_ligand_len', 128)

            generated = self.generate_smiles(num_gen, max_length=max_len, sampling_method=sampling_method)

            # Save generated samples to sample_dir
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            jsonl_path = os.path.join(self.sample_dir, f'samples_epoch_{epoch}_{timestamp}.jsonl')
            txt_path = os.path.join(self.sample_dir, f'samples_epoch_{epoch}_{timestamp}.txt')

            with open(jsonl_path, 'w') as jf, open(txt_path, 'w') as tf:
                for rec in generated:
                    jf.write(json.dumps(rec) + '\n')
                    tf.write((rec.get('smiles', '') or '') + '\n')

            print(f"Saved {len(generated)} generated SMILES to: {jsonl_path} and {txt_path}")

            # Log example and count to wandb
            wandb.log({'samples/generated_count': len(generated), 'samples/last_file': jsonl_path, 'samples/method': sampling_method}, step=step)

        wandb.finish()
        print("Training completed!")

    def save_checkpoint(self, step):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
        
        self.state.update({
            'step': step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        })
        
        torch.save(self.state, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SMILES Diffusion Model")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--datafile", type=str, 
                       default="/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/input_data/processed_pubchem.pt",
                       help="SMILES data file")
    parser.add_argument("--device", type=str, default="xpu:0", help="Device to use")
    parser.add_argument("--project", type=str, default="smiles-diffusion", help="Wandb project")
    parser.add_argument("--name", type=str, default="smiles-run", help="Wandb run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    trainer = SMILESTrainer(
        work_dir=args.work_dir,
        config_file=args.config,
        datafile=args.datafile,
        dev_id=args.device,
        seed=args.seed
    )
    
    trainer.train(args.project, args.name)


if __name__ == "__main__":
    main()
