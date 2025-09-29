#!/usr/bin/env python3
"""
DDP-compatible checkpoint resume test.
This tests checkpoint functionality in a distributed setting.
"""

import torch
import tempfile
import shutil
import os
import yaml
import sys
from pathlib import Path

# Add the project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def create_ddp_config(temp_dir, world_size=2, batch_size_per_gpu=2):
    """Create a DDP-compatible config."""
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
            'length': 16,
            'd_model': 64,
            'cond_dim': 64,
            'esm_dim': 64,
            'molformer_dim': 64,
        },
        'training': {
            'batch_size': batch_size_per_gpu,  # Per-GPU batch size
            'n_iters': 10,
            'log_freq': 3,
            'eval_freq': 999,
            'snapshot_freq': 5,
            'accum': 1,  # No gradient accumulation for DDP
            'ema': 0.999,
            'epochs': 1,
            'task': 'protein_only',
            'weight': 'standard',
            'num_workers': 0,  # Avoid multiprocessing issues
            'seed': 42,
        },
        'optim': {
            'lr': 1e-4 * world_size,  # Scale learning rate
            'optimizer': 'AdamW',
            'grad_clip': 1.0,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-8,
            'warmup': 0,
        },
        'data': {
            'max_protein_len': 16,
            'max_ligand_len': 16,
            'vocab_size_protein': 25,
            'vocab_size_ligand': 100,
            'cache_dir': 'data',
            'train_ratio': 0.8,
            'val_ratio': 0.2,
        },
        'noise': {
            'sigma_min': 0.01,
            'sigma_max': 1.0,
            'type': 'cosine',
            'eps': 0.02,
        },
        'curriculum': {
            'enabled': False,
        },
        'memory': {
            'gradient_checkpointing': False,
            'mixed_precision': False,
            'max_memory_per_gpu': 0.9,
        },
        'ngpus': world_size,
        'tokens': 25,
        'lr_schedule': {
            'type': 'cosine_with_warmup',
            'warmup_steps': 0,
            'max_steps': 20,
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
        'eval': {
            'batch_size': batch_size_per_gpu,
            'perplexity': False,  # Disable for speed
        },
        'monitoring': {
            'log_gradients': False,
            'log_weights': False,
            'sample_frequency': 999,  # Disable sampling
        },
    }
    
    config_path = os.path.join(temp_dir, 'ddp_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def create_dummy_data(temp_dir):
    """Create dummy data compatible with DDP."""
    vocab_size = 25
    seq_len = 16
    num_samples = 40  # More samples for DDP
    
    sequences = torch.randint(1, vocab_size-1, (num_samples, seq_len))
    
    data_path = os.path.join(temp_dir, 'ddp_dummy_data.pt')
    torch.save(sequences, data_path)
    
    return data_path

def test_ddp_checkpoint():
    """Test checkpoint functionality with DDP setup."""
    print("🧪 Testing DDP Checkpoint Functionality")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # Create test files
        config_path = create_ddp_config(temp_dir, world_size=2)
        data_path = create_dummy_data(temp_dir)
        
        print(f"📝 Created DDP config: {config_path}")
        print(f"📊 Created data: {data_path}")
        
        # Import trainer
        try:
            from protlig_dd.training.run_train_uniref_ddp_polaris import OptimizedUniRef50Trainer
        except ImportError as e:
            print(f"❌ Cannot import trainer: {e}")
            return False
        
        print("\n🚀 Testing first training run (rank 0)...")
        
        # First training run (simulate rank 0)
        try:
            trainer1 = OptimizedUniRef50Trainer(
                work_dir=temp_dir,
                config_file=config_path,
                datafile=data_path,
                rank=0,
                world_size=2,  # Simulate 2-GPU setup
                dev_id='cpu',  # Use CPU to avoid GPU issues
                seed=42,
                use_ddp=True,  # Enable DDP mode
                use_wandb=False,
                minimal_mode=True,
                force_fresh_start=True,
                disable_generation_tests=True,
            )
            
            print("✅ Trainer 1 initialized successfully")
            
        except Exception as e:
            print(f"❌ Trainer 1 initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Run training
        try:
            print("🏃 Starting training...")
            trainer1.train()
            
            final_step_1 = trainer1.state['step']
            print(f"✅ First run completed at step {final_step_1}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check checkpoint was created
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            print("❌ No checkpoint directory found!")
            return False
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if len(checkpoint_files) == 0:
            print("❌ No checkpoint files found!")
            return False
        
        print(f"💾 Found {len(checkpoint_files)} checkpoint files")
        
        # Test checkpoint loading
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            print("✅ Checkpoint loaded successfully")
            
            # Verify checkpoint structure
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'step']
            for key in required_keys:
                if key not in checkpoint:
                    print(f"❌ Missing checkpoint key: {key}")
                    return False
                print(f"✅ Found checkpoint key: {key}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False
        
        print("\n🔄 Testing checkpoint resume (rank 0)...")
        
        # Second training run - resume from checkpoint
        try:
            trainer2 = OptimizedUniRef50Trainer(
                work_dir=temp_dir,
                config_file=config_path,
                datafile=data_path,
                rank=0,
                world_size=2,
                dev_id='cpu',
                seed=42,
                use_ddp=True,
                use_wandb=False,
                minimal_mode=True,
                force_fresh_start=False,  # Should load checkpoint
                disable_generation_tests=True,
            )
            
            restored_step = trainer2.state['step']
            print(f"🔍 Restored step: {restored_step}")
            
            if restored_step == final_step_1:
                print("✅ Step correctly restored from checkpoint!")
            else:
                print(f"❌ Step mismatch! Expected {final_step_1}, got {restored_step}")
                return False
                
        except Exception as e:
            print(f"❌ Checkpoint resume failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🆕 Testing fresh start override...")
        
        # Third training run - fresh start
        try:
            trainer3 = OptimizedUniRef50Trainer(
                work_dir=temp_dir,
                config_file=config_path,
                datafile=data_path,
                rank=0,
                world_size=2,
                dev_id='cpu',
                seed=42,
                use_ddp=True,
                use_wandb=False,
                minimal_mode=True,
                force_fresh_start=True,  # Should ignore checkpoint
                disable_generation_tests=True,
            )
            
            fresh_step = trainer3.state['step']
            print(f"🔍 Fresh start step: {fresh_step}")
            
            if fresh_step == 0:
                print("✅ Fresh start correctly ignored checkpoint!")
            else:
                print(f"❌ Fresh start failed! Expected step 0, got {fresh_step}")
                return False
                
        except Exception as e:
            print(f"❌ Fresh start test failed: {e}")
            return False
        
        print("\n🎉 All DDP checkpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temp directory: {temp_dir}")

if __name__ == "__main__":
    success = test_ddp_checkpoint()
    if success:
        print("\n✅ DDP checkpoint test PASSED!")
        exit(0)
    else:
        print("\n❌ DDP checkpoint test FAILED!")
        exit(1)
