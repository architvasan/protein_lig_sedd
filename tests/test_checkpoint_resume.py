#!/usr/bin/env python3
"""
Unit tests for checkpoint loading and training resumption.
Tests the critical functionality of saving/loading model state and continuing training.
"""

from mpi4py import MPI
import pytest
import torch
import tempfile
import shutil
import os
import yaml
from pathlib import Path
import numpy as np

# Import your trainer class
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from protlig_dd.training.run_train_uniref_ddp_polaris import OptimizedUniRef50Trainer


class TestCheckpointResume:
    """Test checkpoint saving and loading functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_config(self, temp_dir):
        """Create minimal config for testing."""
        config = {
        # Curriculum learning settings
        'curriculum': {
            'enabled': True,
            'probabilistic': True,
            'preschool_time': 2500,
            # Alternative curriculum settings (commented in YAML):
            # 'difficulty_ramp': 'exponential',
        },
        # Data configuration
        'data': {
            'cache_dir': 'data',
            'max_ligand_len': 128,
            'max_protein_len': 32,
            'train': 'uniref50',
            'train_ratio': 0.95,
            'use_structure': False,
            'val_ratio': 0.05,
            'valid': 'uniref50',
            'vocab_size_ligand': 2364,
            'vocab_size_protein': 36,
        },

        # Evaluation settings
        'eval': {
            'batch_size': 16,
            'perplexity': True,
            'perplexity_batch_size': 8,
        },

        # Graph configuration
        'graph': {
            'file': 'data',
            'report_all': False,
            'type': 'absorb',
        },

        # Hydra configuration (for experiment tracking)
        'hydra': {
            'run': {
                'dir': 'exp_local/uniref50/${now:%Y.%m.%d}/${now:%H%M%S}',
            },
        },

        # Learning rate schedule
        'lr_schedule': {
            'max_steps': 5000000,
            'min_lr_ratio': 0.1,
            'type': 'cosine_with_warmup',
            'warmup_steps': 2000,
        },

        # Memory optimization settings
        'memory': {
            'gradient_checkpointing': True,
            'max_memory_per_gpu': 0.9,
            'mixed_precision': True,
        },

        # Model architecture
        'model': {
            'cond_dim': 128,
            'device': 'cuda:0',
            'dropout': 0.1,
            'esm_dim': 640,
            'hidden_size': 256,
            'length': 32,
            'molformer_dim': 768,
            'n_blocks_lig': 8,
            'n_blocks_prot': 20,
            'n_heads': 4,
            'name': 'medium',
            'scale_by_sigma': True,
            'type': 'ddit',
        },

        # Monitoring and logging
        'monitoring': {
            'log_gradients': True,
            'log_weights': False,
            'sample_frequency': 5000,
        },

        # GPU configuration
        'ngpus': 1,

        # Noise schedule
        'noise': {
            'eps': 0.02,
            'sigma_max': 0.95,
            'sigma_min': 0.01,
            'type': 'cosine',
        },

        # Optimizer settings
        'optim': {
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-08,
            'grad_clip': 1.0,
            'lr': 0.00002,
            'optimizer': 'AdamW',
            'warmup': 2500,
            'weight_decay': 0.01,
        },

        # Sampling configuration
        'sampling': {
            'noise_removal': True,
            'predictor': 'euler',
            'steps': 50,
        },

        # Token vocabulary size
        'tokens': 25,

        # Training configuration
        'training': {
            'accum': 1,
            'batch_size': 4,
            'ema': 0.999,
            'epochs': 5,
            'eval_freq': 2000,
            'force_reprocess': False,
            'log_freq': 20,
            'max_samples': 50000000,
            'n_iters': 25000,
            'num_workers': 4,
            'seed': 42,
            'snapshot_freq': 1000,
            'snapshot_freq_for_preemption': 1000,
            'snapshot_sampling': True,
            'task': 'protein_only',
            'weight': 'standard',
        },

        # Work directory
        'work_dir': '/Users/ramanathana/Work/Protein-Ligand-SEDD/protein_lig_sedd',

        # Defaults (Hydra-specific)
        'defaults': [
            '_self_',
            {'model': 'medium'},
        ],
    }
    #    config = {
    #        'tokens': 28,
    #        'model': {
    #            'd_model': 128,
    #            'n_layers': 2,
    #            'n_heads': 4,
    #            'vocab_size': 25,  # 20 amino acids + special tokens
    #        },
    #        'training': {
    #            'batch_size': 4,
    #            'n_iters': 20,
    #            'log_freq': 5,
    #            'eval_freq': 10,
    #            'save_freq': 10,
    #            'accum': 1,
    #            'ema': 0.999
    #        },
    #        'optim': {
    #            'lr': 1e-4,
    #            'grad_clip': 1.0,
    #            'weight_decay': 0.01,
    #        },
    #        'data': {
    #            'max_protein_len': 32,
    #        },
    #        'noise': {
    #            'sigma_min': 0.01,
    #            'sigma_max': 1.0,
    #        },
    #        'graph': {
    #            'type': 'absorb'
    #        },
    #        'noise': {
    #            'type': 'cosine',
    #            'sigma_min': 0.01,
    #            'sigma_max': 0.95
    #            }
    #    }
        
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def dummy_data(self, temp_dir):
        """Create dummy training data."""
        # Create synthetic protein sequences
        vocab_size = 25
        seq_len = 32
        num_samples = 100
        
        sample_list = []
        # Generate random sequences (representing tokenized proteins)
        sequences = torch.randint(0, vocab_size-1, (num_samples, seq_len))
        
        for seq in sequences:
            sample_list.append({'prot_tokens': seq})
        
        data_path = os.path.join(temp_dir, 'dummy_data.pt')
        torch.save(sample_list, data_path)
        print(sample_list)
        
        return data_path
    
    def test_checkpoint_creation(self, temp_dir, minimal_config, dummy_data):
        """Test that checkpoints are created during training."""
        trainer = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',  # Use CPU for testing
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True  # Skip evaluations for speed
        )
        
        # Run a few training steps
        trainer.train(wandb_project='test_chk', wandb_name='test_chk')
        
        # Check that checkpoint was created
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        assert os.path.exists(checkpoint_dir), "Checkpoint directory should be created"
        
        # Should have at least one checkpoint file
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0, "At least one checkpoint should be saved"
        
        # Check checkpoint contains required keys
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 
                        'ema_state_dict', 'step', 'epoch', 'loss']
        for key in required_keys:
            assert key in checkpoint, f"Checkpoint should contain {key}"
    
    def test_checkpoint_resume(self, temp_dir, minimal_config, dummy_data):
        """Test resuming training from checkpoint."""
        # First training run
        trainer1 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True
        )
        
        # Train for a few steps
        trainer1.train(wandb_project = 'test_checkpointing', wandb_name = 'test_chck')
        
        # Get the final state
        final_step_1 = trainer1.state['step']
        final_loss_1 = trainer1.state.get('loss', 0)
        
        # Get model parameters before
        model_params_1 = {name: param.clone() for name, param in trainer1.model.named_parameters()}
        
        # Second training run - resume from checkpoint
        trainer2 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,  # Same seed
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=False  # This should load checkpoint
        )
        
        # Check that state was restored
        assert trainer2.state['step'] == final_step_1, f"Step should be restored: {trainer2.state['step']} vs {final_step_1}"
        
        # Check that model parameters were restored
        model_params_2 = {name: param for name, param in trainer2.model.named_parameters()}
        
        for name in model_params_1:
            torch.testing.assert_close(
                model_params_1[name], 
                model_params_2[name],
                msg=f"Parameter {name} should be restored from checkpoint"
            )
    
    def test_fresh_start_ignores_checkpoint(self, temp_dir, minimal_config, dummy_data):
        """Test that force_fresh_start ignores existing checkpoints."""
        # First training run
        trainer1 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True
        )
        
        trainer1.train(wandb_project='test_chk', wandb_name='test_chk')
        final_step_1 = trainer1.state['step']
        
        # Second training run with fresh start
        trainer2 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=True  # This should ignore checkpoint
        )
        
        # Should start from step 0
        assert trainer2.state['step'] == 0, "Fresh start should begin from step 0"
        assert trainer2.state['step'] != final_step_1, "Fresh start should not resume from checkpoint"
    
    def test_checkpoint_with_different_config(self, temp_dir, minimal_config, dummy_data):
        """Test behavior when config changes between runs."""
        # First training run
        trainer1 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True
        )
        
        trainer1.train(wandb_project='test_chk', wandb_name='test_chk')
        
        # Modify config (change learning rate)
        with open(minimal_config, 'r') as f:
            config = yaml.safe_load(f)
        
        config['optim']['lr'] = 2e-4  # Different learning rate
        
        modified_config = os.path.join(temp_dir, 'modified_config.yaml')
        with open(modified_config, 'w') as f:
            yaml.dump(config, f)
        
        # Second training run with modified config
        trainer2 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=modified_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=False
        )
        
        # Should still load checkpoint but use new config
        # (This tests that model weights are restored but optimizer uses new LR)
        assert trainer2.scheduler.get_last_lr()[0] == 2e-4, "New learning rate should be used"
    
    def test_checkpoint_corruption_handling(self, temp_dir, minimal_config, dummy_data):
        """Test handling of corrupted checkpoint files."""
        # Create trainer and run training
        trainer1 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True
        )
        
        trainer1.train(wandb_project='test_chk', wandb_name='test_chk')
        
        # Corrupt the checkpoint file
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        if checkpoint_files:
            corrupt_file = os.path.join(checkpoint_dir, checkpoint_files[0])
            with open(corrupt_file, 'w') as f:
                f.write("corrupted data")
        
        # Try to resume - should handle gracefully
        trainer2 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=minimal_config,
            datafile=dummy_data,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=False
        )
        
        # Should start fresh when checkpoint is corrupted
        assert trainer2.state['step'] == 0, "Should start fresh when checkpoint is corrupted"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
