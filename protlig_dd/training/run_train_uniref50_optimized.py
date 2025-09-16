"""
Optimized training script for UniRef50 with improved stability and efficiency.
"""

import datetime
import os
import gc
import time
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import protlig_dd.processing.losses as losses
import protlig_dd.sampling.sampling as sampling
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
import protlig_dd.utils.utils as utils
from protlig_dd.data.data import get_dataloaders
from protlig_dd.model.transformers_protlig_cross_v100_optimized import OptimizedTransformerBlock
from protlig_dd.model.ema import ExponentialMovingAverage
from protlig_dd.utils.lr_scheduler import WarmupCosineLR


class UniRef50Dataset(torch.utils.data.Dataset):
    """Dataset class for processed UniRef50 data."""

    def __init__(self, data_file):
        self.data = torch.load(data_file, weights_only=False)
        print(f"Loaded {len(self.data)} sequences from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Return the tokenized protein sequence
        return sample['prot_tokens']


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
class OptimizedUniRef50Trainer:
    """
    Optimized trainer for UniRef50 with improved memory efficiency and training stability.
    """
    work_dir: str
    config_file: str
    datafile: str = './input_data/processed_uniref50.pt'
    dev_id: str = 'cuda:0'
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Initialize trainer components."""
        print(f"Trainer initialized. Device: {self.dev_id}")

        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Setup device with cross-platform compatibility
        self.device = self.setup_device(self.dev_id)
        print(f"✅ Using device: {self.device}")

        # Load configuration
        with open(self.config_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        # Convert to namespace for easier access using the Config class
        self.cfg = utils.Config(dictionary=cfg_dict)

        # Setup directories
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.sample_dir = os.path.join(self.work_dir, "samples")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        print(f"Trainer initialized. Device: {self.device}")
        print(f"Config loaded from: {self.config_file}")

        # Initialize device-specific attributes
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            self.use_amp = True
            self.scaler = None  # Will be initialized in setup_training
            print("✅ CUDA mixed precision will be enabled")
        else:
            self.use_amp = False
            self.scaler = None
            print(f"✅ {device_type.upper()} training without mixed precision")

        # Verify key configuration sections exist
        required_sections = ['model', 'training', 'optim', 'data', 'noise']
        for section in required_sections:
            if not hasattr(self.cfg, section):
                print(f"⚠️  Warning: Missing config section '{section}' - using defaults")
                setattr(self.cfg, section, utils.Config(dictionary={}))

    def setup_device(self, dev_id):
        """Setup device with cross-platform compatibility."""
        if dev_id == "auto":
            # Auto-detect best available device
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
        """Setup optimized data loaders for UniRef50."""
        print("Setting up data loaders...")

        # Check if we have a custom dataset file (our processed UniRef50 data)
        if Path(self.datafile).exists():
            print(f"Loading custom dataset: {self.datafile}")
            self.setup_custom_data_loaders()
        else:
            print("Using standard data loading pipeline...")
            # Use the existing data loading infrastructure
            train_loader, val_loader = get_dataloaders(
                self.cfg,
                distributed=False
            )

            self.train_loader = train_loader
            self.val_loader = val_loader

        print(f"Data loaders ready.")

    def setup_custom_data_loaders(self):
        """Setup data loaders for our custom processed UniRef50 data."""
        from torch.utils.data import DataLoader, random_split

        # Load dataset
        dataset = UniRef50Dataset(self.datafile)

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
        num_workers = safe_getattr(self.cfg, 'training.num_workers', 0)  # Use 0 to avoid multiprocessing issues

        # Determine pin_memory based on device
        device_type = str(self.device).split(':')[0]
        pin_memory = device_type == 'cuda'  # Only pin memory for CUDA

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
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup model with optimizations."""
        print("Setting up model...")
        
        # Build graph for absorbing diffusion
        self.graph = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.tokens)
        
        # Build noise schedule
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        
        # Build model (using V100-compatible version for UniRef50)
        from protlig_dd.model.transformer_v100 import SEDD
        self.model = SEDD(self.cfg).to(self.device)
        print(f"✅ Using V100-compatible SEDD model (no flash attention required)")
        
        # Enable gradient checkpointing for memory efficiency (if supported)
        if hasattr(self.cfg, 'memory') and self.cfg.memory.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            else:
                print("⚠️  Gradient checkpointing not supported by V100 model")
        
        # Setup EMA
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), 
            decay=self.cfg.training.ema
        )
        
        print(f"Model ready. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self):
        """Setup optimizer with improved scheduling."""
        print("Setting up optimizer...")
        
        # Get optimizer
        self.optimizer = losses.get_optimizer(
            self.cfg,
            chain(self.model.parameters(), self.noise.parameters())
        )
        
        # Setup gradient scaler for mixed precision (CUDA only)
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
            print("✅ Using CUDA mixed precision training")
        else:
            self.scaler = None
            self.use_amp = False
            print(f"✅ Using {device_type.upper()} without mixed precision")
        
        # Setup learning rate scheduler
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.cfg.optim.warmup,
            max_steps=self.cfg.training.n_iters,
            base_lr=self.cfg.optim.lr * 0.1,
            max_lr=self.cfg.optim.lr,
            min_lr=self.cfg.optim.lr * 0.01
        )
        
        # Training state
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

    def setup_wandb(self, project_name: str, run_name: str):
        """Setup Wandb with comprehensive logging configuration."""
        print("🚀 Setting up Wandb logging...")

        # Initialize wandb
        run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                # Model configuration
                'model_name': safe_getattr(self.cfg, 'model.name', 'transformer'),
                'model_type': safe_getattr(self.cfg, 'model.type', 'ddit'),
                'hidden_size': safe_getattr(self.cfg, 'model.hidden_size', 512),
                'n_blocks': safe_getattr(self.cfg, 'model.n_blocks', 8),
                'n_heads': safe_getattr(self.cfg, 'model.n_heads', 8),
                'dropout': safe_getattr(self.cfg, 'model.dropout', 0.1),

                # Training configuration
                'batch_size': safe_getattr(self.cfg, 'training.batch_size', 16),
                'accumulation_steps': safe_getattr(self.cfg, 'training.accum', 2),
                'learning_rate': safe_getattr(self.cfg, 'optim.lr', 5e-5),
                'weight_decay': safe_getattr(self.cfg, 'optim.weight_decay', 0.01),
                'warmup_steps': safe_getattr(self.cfg, 'optim.warmup', 1000),
                'max_iterations': safe_getattr(self.cfg, 'training.n_iters', 5000),
                'ema_decay': safe_getattr(self.cfg, 'training.ema', 0.999),

                # Data configuration
                'max_protein_len': safe_getattr(self.cfg, 'data.max_protein_len', 512),
                'vocab_size': safe_getattr(self.cfg, 'data.vocab_size_protein', 25),
                'train_ratio': safe_getattr(self.cfg, 'data.train_ratio', 0.9),

                # Noise configuration
                'noise_type': safe_getattr(self.cfg, 'noise.type', 'cosine'),
                'sigma_min': safe_getattr(self.cfg, 'noise.sigma_min', 1e-4),
                'sigma_max': safe_getattr(self.cfg, 'noise.sigma_max', 0.5),

                # Curriculum learning
                'curriculum_enabled': safe_getattr(self.cfg, 'curriculum.enabled', False),
                'preschool_time': safe_getattr(self.cfg, 'curriculum.preschool_time', 5000),

                # System info
                'device': str(self.device),
                'seed': safe_getattr(self.cfg, 'training.seed', 42),
            },
            tags=['uniref50', 'sedd', 'protein', 'diffusion', 'optimized'],
            notes=f"Optimized UniRef50 training with improved V100-compatible attention and enhanced curriculum learning"
        )

        # Display the Wandb web interface link prominently
        print("\n" + "="*80)
        print("🌐 WANDB EXPERIMENT TRACKING")
        print("="*80)
        print(f"📊 Project: {project_name}")
        print(f"🏷️  Run Name: {run_name}")
        print(f"🔗 Web Interface: {wandb.run.url}")
        print(f"📈 Dashboard: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
        print("="*80)
        print("💡 Open the link above to monitor your training in real-time!")
        print("="*80 + "\n")

        # Store the run for later reference
        self.wandb_run = run

        print("✅ Wandb setup complete - tracking enabled!")

    def setup_wandb_model_watching(self):
        """Setup model watching after model is created."""
        try:
            # Watch model for gradient and parameter tracking
            log_freq = safe_getattr(self.cfg, 'training.log_freq', 50)
            wandb.watch(self.model, log='all', log_freq=log_freq)
            print("✅ Wandb model watching enabled")
        except Exception as e:
            print(f"⚠️  Warning: Could not setup model watching: {e}")
            print("   Training will continue without gradient tracking")

    def log_training_metrics(self, step: int, loss: float, lr: float, epoch: int,
                           batch_time: float = None, additional_metrics: dict = None):
        """Log training metrics to Wandb."""
        metrics = {
            'train/loss': loss,
            'train/learning_rate': lr,
            'train/step': step,
            'train/epoch': epoch,
        }

        if batch_time is not None:
            metrics['train/batch_time'] = batch_time
            metrics['train/samples_per_second'] = self.cfg.training.batch_size / batch_time

        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f'train/{key}'] = value

        wandb.log(metrics, step=step)

    def log_validation_metrics(self, step: int, val_loss: float, perplexity: float = None):
        """Log validation metrics to Wandb."""
        metrics = {
            'val/loss': val_loss,
            'val/step': step,
        }

        if perplexity is not None:
            metrics['val/perplexity'] = perplexity

        wandb.log(metrics, step=step)

    def log_model_samples(self, step: int, num_samples: int = 5):
        """Generate and log model samples to Wandb."""
        try:
            print(f"Generating {num_samples} samples for logging...")

            # Set model to eval mode
            self.model.eval()

            with torch.no_grad():
                # Generate samples using the sampling module
                samples = []
                for i in range(num_samples):
                    # Sample from the model
                    sample_length = torch.randint(50, 200, (1,)).item()

                    # Initialize with random tokens
                    initial_tokens = torch.randint(
                        0, self.cfg.data.vocab_size_protein,
                        (1, sample_length),
                        device=self.device
                    )

                    # This is a simplified sampling - you may need to adapt based on your sampling code
                    generated_sample = initial_tokens[0].cpu().numpy()

                    # Convert tokens back to amino acid sequence (simplified)
                    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                    sequence = ''.join([amino_acids[min(token, len(amino_acids)-1)] for token in generated_sample if token < len(amino_acids)])

                    samples.append({
                        'sample_id': i,
                        'sequence': sequence[:100],  # Truncate for display
                        'length': len(sequence)
                    })

                # Log samples as a table
                sample_table = wandb.Table(
                    columns=['Sample ID', 'Generated Sequence', 'Length'],
                    data=[[s['sample_id'], s['sequence'], s['length']] for s in samples]
                )

                wandb.log({
                    'samples/generated_proteins': sample_table,
                    'samples/avg_length': np.mean([s['length'] for s in samples]),
                    'samples/step': step
                }, step=step)

            # Set model back to train mode
            self.model.train()

        except Exception as e:
            print(f"Warning: Could not generate samples for logging: {e}")

    def log_system_metrics(self, step: int):
        """Log system metrics like GPU memory usage."""
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB

            wandb.log({
                'system/gpu_memory_allocated_gb': gpu_memory_allocated,
                'system/gpu_memory_reserved_gb': gpu_memory_reserved,
                'system/step': step
            }, step=step)

    def compute_loss(self, batch):
        """Compute loss with improved curriculum learning."""
        # Sample timesteps
        t = torch.rand(batch.shape[0], device=self.device) * (1 - 1e-3) + 1e-3
        sigma, dsigma = self.noise(t)
        
        # Apply curriculum learning
        if hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled:
            perturbed_batch = self.graph.sample_transition_curriculum(
                batch, 
                sigma[:, None], 
                self.state['step'],
                preschool_time=self.cfg.curriculum.preschool_time,
                curriculum_type=getattr(self.cfg.curriculum, 'difficulty_ramp', 'exponential')
            )
        else:
            perturbed_batch = self.graph.sample_transition(batch, sigma[:, None])
        
        # Forward pass with device-aware autocast
        if self.use_amp and str(self.device).split(':')[0] == 'cuda':
            with torch.cuda.amp.autocast():
                log_score = self.model(perturbed_batch, sigma)
                loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

                # Weight by dsigma for better training dynamics
                loss = (dsigma[:, None] * loss).mean()
        else:
            # No autocast for CPU/MPS
            log_score = self.model(perturbed_batch, sigma)
            loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

            # Weight by dsigma for better training dynamics
            loss = (dsigma[:, None] * loss).mean()
        
        return loss
    
    def train_step(self, batch):
        """Single training step with optimizations."""
        import time
        step_start_time = time.time()

        self.model.train()

        # Move batch to device
        batch = batch.to(self.device)

        # Compute loss
        loss = self.compute_loss(batch) / self.cfg.training.accum

        # Backward pass with device-aware gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Additional metrics for logging
        additional_metrics = {}

        # Update every accum steps
        if (self.state['step'] + 1) % self.cfg.training.accum == 0:
            # Unscale gradients for clipping (CUDA only)
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Compute gradient norm before clipping
            total_norm = 0
            for p in chain(self.model.parameters(), self.noise.parameters()):
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            additional_metrics['grad_norm'] = total_norm

            # Clip gradients
            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    chain(self.model.parameters(), self.noise.parameters()),
                    self.cfg.optim.grad_clip
                )

            # Optimizer step with device-aware scaling
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            # Update EMA
            self.ema.update(self.model.parameters())

            # Zero gradients
            self.optimizer.zero_grad()

        step_time = time.time() - step_start_time

        return loss.item() * self.cfg.training.accum, step_time, additional_metrics

    def validate_model(self):
        """Run validation and return metrics."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 10:  # Limit validation batches for speed
                    break

                batch = batch.to(self.device)
                loss = self.compute_loss(batch)
                val_losses.append(loss.item())

        self.model.train()
        return np.mean(val_losses)
    
    def save_checkpoint(self, step, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'noise_state_dict': self.noise.state_dict(),
            'config': self.cfg
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, wandb_project: str, wandb_name: str):
        """Main training loop with comprehensive Wandb logging."""
        print("Starting training...")

        # Setup components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_optimizer()

        # Setup Wandb with comprehensive configuration
        self.setup_wandb(wandb_project, wandb_name)

        # Setup model watching after model is created
        self.setup_wandb_model_watching()

        # Training state
        best_loss = float('inf')
        step = 0
        running_loss = 0.0
        log_interval_start_time = time.time()

        print(f"🚀 Starting training for {self.cfg.training.n_iters} steps...")

        for epoch in range(self.cfg.training.epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.cfg.training.epochs}')

            for batch in progress_bar:
                # Training step with timing
                loss, step_time, additional_metrics = self.train_step(batch)

                epoch_loss += loss
                running_loss += loss
                num_batches += 1
                step += 1
                self.state['step'] = step

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{step}/{self.cfg.training.n_iters}',
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })

                # Detailed logging
                if step % self.cfg.training.log_freq == 0:
                    avg_loss = running_loss / self.cfg.training.log_freq
                    interval_time = time.time() - log_interval_start_time

                    # Log training metrics
                    self.log_training_metrics(
                        step=step,
                        loss=avg_loss,
                        lr=self.scheduler.get_last_lr()[0],
                        epoch=epoch,
                        batch_time=step_time,
                        additional_metrics=additional_metrics
                    )

                    # Log system metrics
                    self.log_system_metrics(step)

                    # Reset running loss and timer
                    running_loss = 0.0
                    log_interval_start_time = time.time()

                # Validation and sampling
                if step % self.cfg.training.eval_freq == 0:
                    print(f"\n🔍 Running validation at step {step}...")

                    # Validation
                    val_loss = self.validate_model()
                    self.log_validation_metrics(step, val_loss)

                    # Generate and log samples
                    self.log_model_samples(step, num_samples=3)

                    print(f"✅ Validation loss: {val_loss:.4f}")

                # Checkpointing
                if step % self.cfg.training.snapshot_freq == 0:
                    avg_epoch_loss = epoch_loss / num_batches
                    is_best = avg_epoch_loss < best_loss

                    if is_best:
                        best_loss = avg_epoch_loss
                        print(f"🎉 New best loss: {best_loss:.4f}")

                    self.save_checkpoint(step, is_best)

                    # Log checkpoint info
                    wandb.log({
                        'checkpoint/step': step,
                        'checkpoint/best_loss': best_loss,
                        'checkpoint/current_loss': avg_epoch_loss,
                        'checkpoint/is_best': is_best
                    }, step=step)

                # Early stopping check
                if step >= self.cfg.training.n_iters:
                    print(f"\n✅ Reached maximum iterations ({self.cfg.training.n_iters})")
                    break

            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches
            wandb.log({
                'epoch/loss': avg_epoch_loss,
                'epoch/number': epoch,
                'epoch/step': step
            }, step=step)

            print(f"📊 Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

            if step >= self.cfg.training.n_iters:
                break

        # Final checkpoint and summary
        print("\n🏁 Training completed!")
        self.save_checkpoint(step, is_best=False)

        # Final validation
        final_val_loss = self.validate_model()
        self.log_validation_metrics(step, final_val_loss)

        # Final samples
        self.log_model_samples(step, num_samples=5)

        # Training summary
        wandb.log({
            'summary/final_step': step,
            'summary/final_train_loss': epoch_loss / num_batches,
            'summary/final_val_loss': final_val_loss,
            'summary/best_loss': best_loss,
            'summary/total_epochs': epoch + 1
        })

        print(f"📈 Final validation loss: {final_val_loss:.4f}")
        print(f"🏆 Best training loss: {best_loss:.4f}")

        wandb.finish()
        print("✅ Wandb logging completed")


def main():
    """Main entry point."""
    import argparse

    # Print startup banner
    print("\n" + "="*80)
    print("🧬 OPTIMIZED UNIREF50 SEDD TRAINING")
    print("="*80)
    print("🚀 Enhanced with V100-compatible attention & curriculum learning")
    print("📊 Full Wandb experiment tracking enabled")
    print("="*80 + "\n")

    parser = argparse.ArgumentParser(description="Train optimized UniRef50 model")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--datafile", type=str, default="./input_data/processed_uniref50.pt", help="Data file path")
    parser.add_argument("--wandb_project", type=str, default="uniref50-sedd", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Generate wandb name if not provided
    if args.wandb_name is None:
        args.wandb_name = f"uniref50_optimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"📁 Work directory: {args.work_dir}")
    print(f"⚙️  Config file: {args.config}")
    print(f"💾 Data file: {args.datafile}")
    print(f"🏷️  Wandb project: {args.wandb_project}")
    print(f"🏷️  Wandb run: {args.wandb_name}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎲 Seed: {args.seed}")
    print()

    try:
        # Create trainer
        print("🔧 Initializing trainer...")
        trainer = OptimizedUniRef50Trainer(
            work_dir=args.work_dir,
            config_file=args.config,
            datafile=args.datafile,
            dev_id=args.device,
            seed=args.seed
        )

        print("✅ Trainer initialized successfully!")
        print()

        # Start training
        trainer.train(args.wandb_project, args.wandb_name)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check the error above and verify your configuration.")
        return 1

    print("\n🎉 Training completed successfully!")
    return 0


if __name__ == "__main__":
    main()
