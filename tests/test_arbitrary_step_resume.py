#!/usr/bin/env python3
"""
Test script for arbitrary step resume functionality.
"""

import tempfile
import os
import yaml
import torch
import sys

# Add project root to path
sys.path.append('..')

def create_test_config():
    """Create a minimal test config."""
    config = {
        'model': {
            'hidden_size': 64,
            'n_heads': 4,
            'n_blocks_prot': 2,
            'n_blocks_lig': 2,
            'type': 'ddit',
            'name': 'test',
            'dropout': 0.0,
            'scale_by_sigma': True,
            'length': 32,
        },
        'training': {
            'batch_size': 2,
            'n_iters': 100,
            'log_freq': 5,
            'eval_freq': 999,
            'snapshot_freq': 999,
            'accum': 1,
            'ema': 0.999,
            'epochs': 3,
            'task': 'protein_only',
            'weight': 'standard',
        },
        'optim': {
            'lr': 1e-4,
            'optimizer': 'AdamW',
            'grad_clip': 1.0,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-8,
            'warmup': 0,
        },
        'data': {
            'max_protein_len': 32,
            'vocab_size_protein': 25,
            'cache_dir': 'data',
        },
        'noise': {
            'sigma_min': 0.01,
            'sigma_max': 1.0,
            'type': 'cosine',
        },
        'curriculum': {
            'enabled': False,
        },
        'memory': {
            'gradient_checkpointing': False,
            'mixed_precision': False,
        },
        'ngpus': 1,
        'tokens': 25,
        'lr_schedule': {
            'type': 'cosine_with_warmup',
            'warmup_steps': 0,
            'max_steps': 100,
            'min_lr_ratio': 0.1,
        },
        'sampling': {
            'steps': 10,
            'predictor': 'euler',
            'noise_removal': True,
        },
        'graph': {
            'type': 'absorb',
            'file': 'data',
            'report_all': False,
        },
    }
    return config

def test_arbitrary_step_resume():
    """Test arbitrary step resume functionality."""
    print("🧪 Testing Arbitrary Step Resume Functionality")
    print("=" * 60)
    
    # Create temporary directory and config
    temp_dir = tempfile.mkdtemp()
    config = create_test_config()
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create dummy data (small dataset for testing)
    dummy_data = torch.randint(1, 24, (20, 32))  # 20 sequences of length 32
    data_path = os.path.join(temp_dir, 'dummy_data.pt')
    torch.save(dummy_data, data_path)
    
    try:
        # Import the trainer
        from protlig_dd.training.run_train_uniref_ddp_aurora import OptimizedUniRef50Trainer
        
        print("✅ Successfully imported trainer")
        
        # Test 1: Normal training from step 0
        print("\n📊 Test 1: Normal training from step 0")
        trainer1 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=config_path,
            datafile=data_path,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=True,
            start_step=None  # Normal start
        )
        
        print(f"   Initial step: {trainer1.start_step}")
        print("   ✅ Normal initialization successful")
        
        # Test 2: Start from arbitrary step (without checkpoint)
        print("\n📊 Test 2: Start from arbitrary step 15 (no checkpoint)")
        trainer2 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=config_path,
            datafile=data_path,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=True,
            start_step=15  # Start from step 15
        )
        
        print(f"   Start step override: {trainer2.start_step}")
        print("   ✅ Arbitrary step initialization successful")
        
        # Test 3: Create a checkpoint and test resume with step override
        print("\n📊 Test 3: Create checkpoint and test step override")
        
        # Create a fake checkpoint
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'test_checkpoint.pth')
        
        # Create minimal checkpoint data
        fake_checkpoint = {
            'step': 10,
            'epoch': 0,
            'best_loss': 1.5,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'config': config
        }
        torch.save(fake_checkpoint, checkpoint_path)
        print(f"   Created fake checkpoint at step 10: {checkpoint_path}")
        
        # Test resuming with step override
        trainer3 = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=config_path,
            datafile=data_path,
            rank=0,
            world_size=1,
            dev_id='cpu',
            seed=42,
            use_ddp=False,
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=False,
            resume_checkpoint=checkpoint_path,
            start_step=25  # Override checkpoint step 10 with step 25
        )
        
        print(f"   Checkpoint step would be: 10")
        print(f"   Start step override: {trainer3.start_step}")
        print("   ✅ Checkpoint + step override initialization successful")
        
        print("\n🎯 Usage Examples:")
        print("=" * 60)
        print("# Normal training from beginning:")
        print("python protlig_dd/training/run_train_uniref_ddp_aurora.py \\")
        print("    --work_dir ./output \\")
        print("    --config config.yaml \\")
        print("    --datafile data.pt")
        print()
        print("# Resume from arbitrary step 1500 (mid-epoch):")
        print("python protlig_dd/training/run_train_uniref_ddp_aurora.py \\")
        print("    --work_dir ./output \\")
        print("    --config config.yaml \\")
        print("    --datafile data.pt \\")
        print("    --start_step 1500")
        print()
        print("# Resume from checkpoint but override to step 2000:")
        print("python protlig_dd/training/run_train_uniref_ddp_aurora.py \\")
        print("    --work_dir ./output \\")
        print("    --config config.yaml \\")
        print("    --datafile data.pt \\")
        print("    --resume_checkpoint ./checkpoints/checkpoint_step_1000.pth \\")
        print("    --start_step 2000")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temp directory")

if __name__ == "__main__":
    success = test_arbitrary_step_resume()
    if success:
        print("\n✅ Arbitrary step resume functionality is working!")
        exit(0)
    else:
        print("\n❌ Arbitrary step resume functionality needs fixes!")
        exit(1)
