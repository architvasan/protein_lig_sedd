#!/usr/bin/env python3
"""
Test script to verify dtype consistency fixes for Aurora DDP training.
"""

import torch
import tempfile
import os
import yaml
import sys

# Add project root to path
sys.path.append('.')

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
            'n_iters': 10,
            'log_freq': 5,
            'eval_freq': 999,
            'snapshot_freq': 999,
            'accum': 1,
            'ema': 0.999,
            'epochs': 1,
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
    }
    return config

def test_dtype_consistency():
    """Test that dtype consistency fixes work."""
    print("🧪 Testing Dtype Consistency Fixes")
    print("=" * 50)
    
    # Create temporary directory and config
    temp_dir = tempfile.mkdtemp()
    config = create_test_config()
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create dummy data
    dummy_data = torch.randint(1, 24, (20, 32))  # 20 sequences of length 32
    data_path = os.path.join(temp_dir, 'dummy_data.pt')
    torch.save(dummy_data, data_path)
    
    try:
        # Import the trainer
        from protlig_dd.training.run_train_uniref_ddp_aurora import OptimizedUniRef50Trainer
        
        print("✅ Successfully imported trainer")
        
        # Create trainer instance (without DDP for testing)
        trainer = OptimizedUniRef50Trainer(
            work_dir=temp_dir,
            config_file=config_path,
            datafile=data_path,
            rank=0,
            world_size=1,
            dev_id='cpu',  # Use CPU to avoid GPU/XPU issues
            seed=42,
            use_ddp=False,  # Disable DDP for testing
            use_wandb=False,
            minimal_mode=True,
            force_fresh_start=True,
        )
        
        print("✅ Trainer initialized successfully")
        
        # Setup components
        trainer.setup_data_loaders()
        trainer.setup_model()
        trainer.setup_optimizer()
        
        print("✅ All components setup successfully")
        
        # Test dtype consistency
        model_dtype = next(trainer.model.parameters()).dtype
        print(f"📊 Model dtype: {model_dtype}")
        
        # Create test batch
        test_batch = torch.randint(1, 24, (2, 32), device=trainer.device)
        print(f"📊 Test batch dtype: {test_batch.dtype}")
        
        # Test compute_loss with dtype consistency
        try:
            trainer.model.train()
            loss = trainer.compute_loss(test_batch)
            print(f"✅ Compute loss successful: {loss.item():.6f}")
            print(f"📊 Loss dtype: {loss.dtype}")
        except Exception as e:
            print(f"❌ Compute loss failed: {e}")
            return False
        
        # Test simple generation
        try:
            print("\n🧬 Testing simple generation...")
            sequences = trainer.generate_protein_sequences_simple(
                num_samples=2, 
                max_length=16, 
                num_diffusion_steps=5
            )
            print(f"✅ Simple generation successful: {len(sequences)} sequences")
            for i, seq in enumerate(sequences):
                print(f"   Sequence {i}: {seq['sequence'][:20]}... (length: {seq['length']})")
        except Exception as e:
            print(f"❌ Simple generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 All dtype consistency tests passed!")
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
        print(f"🧹 Cleaned up temp directory")

if __name__ == "__main__":
    success = test_dtype_consistency()
    if success:
        print("\n✅ Dtype consistency fixes are working!")
        exit(0)
    else:
        print("\n❌ Dtype consistency fixes need more work!")
        exit(1)
