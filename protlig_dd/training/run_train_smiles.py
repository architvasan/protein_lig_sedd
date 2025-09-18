"""
Training script for SMILES data with proper tokenization.
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import protlig_dd.processing.losses as losses
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
import protlig_dd.utils.utils as utils
from protlig_dd.data.data import get_dataloaders
from protlig_dd.data.tokenize import Tok_Mol
from protlig_dd.model.ema import ExponentialMovingAverage
from protlig_dd.utils.lr_scheduler import WarmupCosineLR
from protlig_dd.training.smiles_losses import get_smiles_loss_fn


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
        self.tokenizer = Tok_Mol()
        print("✅ SMILES tokenizer initialized")

        # Device-specific setup
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            self.use_amp = True
            self.scaler = None
            print("✅ CUDA mixed precision will be enabled")
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
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🍎 Auto-detected: Apple Silicon MPS")
            else:
                device = torch.device("cpu")
                print("💻 Auto-detected: CPU")
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
        print("Setting up SMILES data loaders...")

        if Path(self.datafile).exists():
            print(f"Loading SMILES dataset: {self.datafile}")
            self.setup_custom_data_loaders()
        else:
            raise FileNotFoundError(f"SMILES data file not found: {self.datafile}")

        print(f"SMILES data loaders ready.")

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
        pin_memory = device_type == 'cuda'

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
    
    def setup_optimizer(self):
        """Setup optimizer."""
        print("Setting up optimizer...")
        
        self.optimizer = losses.get_optimizer(
            self.cfg,
            chain(self.model.parameters(), self.noise.parameters())
        )
        
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
            print("✅ Using CUDA mixed precision training")
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
        
        # Build SMILES-specific loss function
        self.loss_fn = get_smiles_loss_fn(self.noise, self.graph, train=True)
        
        # Training loop
        step = 0
        max_steps = self.cfg.training.n_iters
        
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                if step >= max_steps:
                    print(f"Reached max steps: {max_steps}")
                    return
                
                batch = batch.to(self.device)
                
                # Forward pass with fixed autocast
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = self.loss_fn(self.model, batch)
                else:
                    loss = self.loss_fn(self.model, batch)
                
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
                self.ema.update()
                self.scheduler.step()
                
                # Logging
                if step % 100 == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': step,
                        'train/epoch': epoch
                    })
                
                # Save checkpoint
                if step % 5000 == 0:
                    self.save_checkpoint(step)
                
                step += 1
        
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
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
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
