#!/usr/bin/env python3
"""
Simple checkpoint resume test that you can run manually.
This creates a minimal training setup, saves a checkpoint, and tests resuming.
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

def create_test_config(temp_dir):
    """Create a minimal config for testing."""
    config = {
        'model': {
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 4,
            'vocab_size': 25,
        },
        'training': {
            'batch_size': 2,
            'n_iters': 10,  # Very short training
            'log_freq': 5,
            'eval_freq': 999,  # No evaluation
            'save_freq': 5,    # Save checkpoint at step 5
            'accum': 1,
        },
        'optim': {
            'lr': 1e-4,
            'grad_clip': 1.0,
            'weight_decay': 0.01,
        },
        'data': {
            'max_protein_len': 16,  # Very short sequences
        },
        'noise': {
            'sigma_min': 0.01,
            'sigma_max': 1.0,
        }
    }
    
    config_path = os.path.join(temp_dir, 'test_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def create_dummy_data(temp_dir):
    """Create dummy protein sequence data."""
    # Create 50 random sequences of length 16
    vocab_size = 25
    seq_len = 16
    num_samples = 50
    
    sequences = torch.randint(1, vocab_size-1, (num_samples, seq_len))  # Avoid 0 (mask token)
    
    data_path = os.path.join(temp_dir, 'dummy_sequences.pt')
    torch.save(sequences, data_path)
    
    return data_path

def test_checkpoint_resume():
    """Main test function."""
    print("🧪 Testing Checkpoint Resume Functionality")
    print("=" * 50)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # Create test files
        config_path = create_test_config(temp_dir)
        data_path = create_dummy_data(temp_dir)
        
        print(f"📝 Created config: {config_path}")
        print(f"📊 Created data: {data_path}")
        
        # Import trainer
        from protlig_dd.training.run_train_uniref_ddp_polaris import OptimizedUniRef50Trainer
        
        print("\n🚀 Starting first training run...")
        
        # First training run
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
            minimal_mode=True
        )
        
        # Train for a few steps
        trainer1.train()
        
        # Get final state
        final_step_1 = trainer1.state['step']
        final_loss_1 = trainer1.state.get('loss', 0)
        
        print(f"✅ First run completed at step {final_step_1}")
        print(f"📉 Final loss: {final_loss_1:.4f}")
        
        # Save model parameters for comparison
        model_params_1 = {}
        for name, param in trainer1.model.named_parameters():
            model_params_1[name] = param.clone().detach()
        
        # Check checkpoint was created
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            print(f"💾 Found {len(checkpoint_files)} checkpoint files")
        else:
            print("❌ No checkpoint directory found!")
            return False
        
        print("\n🔄 Starting second training run (resume from checkpoint)...")
        
        # Second training run - should resume
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
            force_fresh_start=False  # Should load checkpoint
        )
        
        # Check if state was restored
        restored_step = trainer2.state['step']
        print(f"🔍 Restored step: {restored_step}")
        
        if restored_step == final_step_1:
            print("✅ Step correctly restored from checkpoint!")
        else:
            print(f"❌ Step mismatch! Expected {final_step_1}, got {restored_step}")
            return False
        
        # Check if model parameters were restored
        params_match = True
        for name, param in trainer2.model.named_parameters():
            if name in model_params_1:
                if not torch.allclose(param, model_params_1[name], atol=1e-6):
                    print(f"❌ Parameter {name} not restored correctly!")
                    params_match = False
                    break
        
        if params_match:
            print("✅ Model parameters correctly restored from checkpoint!")
        else:
            print("❌ Model parameters not restored correctly!")
            return False
        
        print("\n🆕 Testing fresh start (ignore checkpoint)...")
        
        # Third training run - fresh start
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
            force_fresh_start=True  # Should ignore checkpoint
        )
        
        fresh_step = trainer3.state['step']
        print(f"🔍 Fresh start step: {fresh_step}")
        
        if fresh_step == 0:
            print("✅ Fresh start correctly ignored checkpoint!")
        else:
            print(f"❌ Fresh start failed! Expected step 0, got {fresh_step}")
            return False
        
        print("\n🎉 All checkpoint tests passed!")
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
    success = test_checkpoint_resume()
    if success:
        print("\n✅ Checkpoint resume test PASSED!")
        exit(0)
    else:
        print("\n❌ Checkpoint resume test FAILED!")
        exit(1)
