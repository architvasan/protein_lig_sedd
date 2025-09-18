#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) version of UniRef50 SEDD training.
Supports multi-GPU training with proper synchronization and scaling.
"""

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from typing import Optional
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from protlig_dd.training.run_train_uniref50_optimized import OptimizedUniRef50Trainer


@dataclass
class DDPUniRef50Trainer(OptimizedUniRef50Trainer):
    """
    Distributed Data Parallel version of UniRef50 trainer.
    Extends the optimized trainer with DDP capabilities.
    """
    local_rank: int = 0
    world_size: int = 1
    
    def __post_init__(self):
        """Initialize DDP-specific components after dataclass initialization."""
        super().__post_init__()
        self.is_main_process = self.local_rank == 0
        
    def setup_ddp(self, rank: int, world_size: int):
        """Initialize DDP process group."""
        self.local_rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Set device for this process
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        print(f"🚀 DDP initialized: rank {rank}/{world_size}, device {self.device}")
    
    def setup_data_loaders(self):
        """Setup data loaders with distributed sampling."""
        print(f"Setting up data loaders (rank {self.local_rank})...")
        
        # Load and split data
        data = torch.load(self.datafile, map_location='cpu')
        
        if isinstance(data, dict):
            sequences = data.get('sequences', data.get('data', []))
        elif isinstance(data, list):
            sequences = data
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
        
        print(f"Loaded {len(sequences)} sequences")
        
        # Create datasets
        from protlig_dd.data.uniref50_dataset import UniRef50Dataset
        
        # Split data
        train_size = int(self.cfg.data.train_ratio * len(sequences))
        val_size = len(sequences) - train_size
        
        train_sequences = sequences[:train_size]
        val_sequences = sequences[train_size:train_size + val_size]
        
        train_dataset = UniRef50Dataset(
            train_sequences,
            max_length=self.cfg.data.max_protein_len,
            vocab_size=self.cfg.data.vocab_size_protein
        )
        
        val_dataset = UniRef50Dataset(
            val_sequences,
            max_length=self.cfg.data.max_protein_len,
            vocab_size=self.cfg.data.vocab_size_protein
        )
        
        # Create distributed samplers
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True,
            drop_last=True
        )
        
        self.val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=False,
            drop_last=False
        )
        
        # Calculate effective batch size per GPU
        # Total effective batch size = batch_size * accum * world_size
        batch_size = self.cfg.training.batch_size
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=self.val_sampler,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        if self.is_main_process:
            print(f"Train dataset: {len(train_dataset)} samples")
            print(f"Val dataset: {len(val_dataset)} samples")
            print(f"Batch size per GPU: {batch_size}")
            print(f"Effective batch size: {batch_size * self.cfg.training.accum * self.world_size}")
            print(f"Train batches per GPU: {len(self.train_loader)}")
            print(f"Val batches per GPU: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup model with DDP wrapping."""
        if self.is_main_process:
            print("Setting up model...")
        
        # Build graph for absorbing diffusion
        from protlig_dd.processing import graph_lib
        self.graph = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.tokens)
        
        # Build noise schedule
        from protlig_dd.processing import noise_lib
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        
        # Build model
        from protlig_dd.model.transformer_v100 import SEDD
        self.model = SEDD(self.cfg).to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,  # Set to True if you have unused parameters
            broadcast_buffers=True
        )
        
        # Setup EMA on the underlying model
        from protlig_dd.training.ema import EMA
        self.ema = EMA(self.model.module.parameters(), decay=self.cfg.training.ema)
        
        if self.is_main_process:
            print(f"✅ Model wrapped with DDP on {self.world_size} GPUs")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"📊 Total parameters: {total_params:,}")
            print(f"📊 Trainable parameters: {trainable_params:,}")
    
    def setup_optimizer(self):
        """Setup optimizer with learning rate scaling for DDP."""
        if self.is_main_process:
            print("Setting up optimizer...")
        
        # Scale learning rate by world size (linear scaling rule)
        base_lr = self.cfg.optim.lr
        scaled_lr = base_lr * self.world_size
        
        # Create optimizer config with scaled learning rate
        from omegaconf import OmegaConf
        optim_cfg = OmegaConf.create(self.cfg.optim)
        optim_cfg.lr = scaled_lr
        
        # Get optimizer
        from protlig_dd.training import losses
        from itertools import chain
        self.optimizer = losses.get_optimizer(
            optim_cfg,
            chain(self.model.parameters(), self.noise.parameters())
        )
        
        # Setup gradient scaler for mixed precision
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # Scale warmup steps by world size to maintain same effective warmup
        warmup_steps = self.cfg.optim.warmup * self.world_size
        
        # Setup learning rate scheduler
        from protlig_dd.training.lr_scheduler import WarmupCosineLR
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=self.cfg.training.n_iters,
            base_lr=scaled_lr * 0.1,
            max_lr=scaled_lr,
            min_lr=scaled_lr * 0.01
        )
        
        if self.is_main_process:
            print(f"✅ Optimizer setup complete")
            print(f"   Base LR: {base_lr:.2e}")
            print(f"   Scaled LR: {scaled_lr:.2e} (x{self.world_size})")
            print(f"   Warmup steps: {warmup_steps}")
    
    def setup_wandb(self, project_name: str, run_name: str):
        """Setup Wandb logging (only on main process)."""
        if not self.is_main_process:
            return
        
        super().setup_wandb(project_name, run_name)
        
        # Add DDP-specific config
        wandb.config.update({
            'ddp/world_size': self.world_size,
            'ddp/effective_batch_size': self.cfg.training.batch_size * self.cfg.training.accum * self.world_size,
            'ddp/scaled_lr': self.cfg.optim.lr * self.world_size,
        })
    
    def train_step(self, batch):
        """DDP-aware training step."""
        step_start_time = time.time()

        self.model.train()

        # Move batch to device
        batch = batch.to(self.device, non_blocking=True)

        # Ensure batch is 2D
        if batch.dim() > 2:
            batch = batch.view(batch.shape[0], -1)

        # Compute loss
        loss = self.compute_loss(batch) / self.cfg.training.accum

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        step_time = time.time() - step_start_time
        return loss.item() * self.cfg.training.accum, step_time, {}

    def train(self, wandb_project: str, wandb_name: str):
        """Main DDP training loop."""
        if self.is_main_process:
            print("Starting DDP training...")

        # Setup components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_optimizer()

        # Setup Wandb (only on main process)
        self.setup_wandb(wandb_project, wandb_name)

        # Training state
        best_loss = float('inf')
        step = 0
        start_epoch = 0
        running_loss = 0.0

        if self.is_main_process:
            print(f"🚀 Starting DDP training for {self.cfg.training.n_iters} steps...")

        for epoch in range(start_epoch, self.cfg.training.epochs):
            # Set epoch for distributed sampler
            self.train_sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            # Only show progress bar on main process
            if self.is_main_process:
                progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.cfg.training.epochs}')
                data_iter = progress_bar
            else:
                data_iter = self.train_loader

            for batch in data_iter:
                # Training step
                loss, step_time, additional_metrics = self.train_step(batch)

                epoch_loss += loss
                running_loss += loss
                num_batches += 1
                step += 1

                # Update progress bar (main process only)
                if self.is_main_process:
                    progress_bar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                        'step': f'{step}/{self.cfg.training.n_iters}',
                        'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                    })

                # Gradient accumulation and optimizer step
                if step % self.cfg.training.accum == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.ema.update(self.model.module.parameters())  # Note: .module for DDP
                    self.optimizer.zero_grad()

                # Logging (main process only)
                if self.is_main_process and step % self.cfg.training.log_freq == 0:
                    avg_loss = running_loss / self.cfg.training.log_freq

                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': step,
                        'train/epoch': epoch + 1,
                        'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1024**3,
                    }, step=step)

                    running_loss = 0.0

                # Evaluation (main process only)
                if self.is_main_process and step % self.cfg.training.eval_freq == 0:
                    val_loss = self.validate_model()

                    wandb.log({
                        'val/loss': val_loss,
                        'val/step': step,
                    }, step=step)

                    # Save best model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self.save_checkpoint(step, epoch+1, best_loss, is_best=True)

                if step >= self.cfg.training.n_iters:
                    break

            if step >= self.cfg.training.n_iters:
                break

        if self.is_main_process:
            print("🎉 DDP training completed!")
            wandb.finish()
    
    def sync_and_reduce_loss(self, loss_tensor):
        """Synchronize and reduce loss across all processes."""
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor / self.world_size
    
    def cleanup_ddp(self):
        """Clean up DDP process group."""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_ddp_environment():
    """Setup environment variables for DDP."""
    # Set CUDA device ordering to be consistent
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Set NCCL environment variables for better performance
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
    
    # Disable tokenizers parallelism to avoid warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train_ddp_worker(rank: int, world_size: int, args):
    """Worker function for DDP training."""
    try:
        # Setup DDP environment
        setup_ddp_environment()
        
        # Create trainer
        trainer = DDPUniRef50Trainer(
            work_dir=args.work_dir,
            config_file=args.config,
            datafile=args.datafile,
            dev_id=f'cuda:{rank}',
            seed=args.seed + rank,  # Different seed per process
            force_fresh_start=args.fresh,
            sampling_method=args.sampling_method,
            local_rank=rank
        )
        
        # Setup DDP
        trainer.setup_ddp(rank, world_size)
        
        # Run training
        trainer.train(args.wandb_project, args.wandb_name)
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(trainer, 'cleanup_ddp'):
            trainer.cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train UniRef50 model with DDP")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--datafile", type=str, default="./input_data/processed_uniref50.pt", help="Data file path")
    parser.add_argument("--wandb_project", type=str, default="uniref50-sedd-ddp", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sampling_method", type=str, choices=["rigorous", "simple"], 
                       default="rigorous", help="Sampling method")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. DDP training requires CUDA.")
        return 1
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("⚠️  Only 1 GPU detected. Consider using single-GPU training instead.")
        print("   Use run_train_uniref50_optimized.py for single-GPU training.")
    
    print(f"🚀 Starting DDP training on {world_size} GPUs")
    print(f"📁 Work directory: {args.work_dir}")
    print(f"⚙️  Config file: {args.config}")
    print(f"💾 Data file: {args.datafile}")
    print(f"🧬 Sampling method: {args.sampling_method}")
    
    # Spawn processes for DDP training
    mp.spawn(
        train_ddp_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
    
    print("🎉 DDP training completed!")
    return 0


if __name__ == "__main__":
    exit(main())
