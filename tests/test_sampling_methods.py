#!/usr/bin/env python3
"""
Test script to compare rigorous CTMC sampling vs simple heuristic sampling
in the UniRef50 optimized trainer.
"""

import sys
import os
from pathlib import Path
import torch
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock missing dependencies
class MockWandb:
    @staticmethod
    def init(*args, **kwargs):
        pass
    @staticmethod
    def log(*args, **kwargs):
        pass
    @staticmethod
    def finish():
        pass

class MockConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, MockConfig(value))
            else:
                setattr(self, key, value)

# Mock the missing modules
sys.modules['wandb'] = MockWandb()
sys.modules['omegaconf'] = type('MockOmegaConf', (), {'OmegaConf': type('OC', (), {'create': lambda x: MockConfig(x)})})()

from protlig_dd.training.run_train_uniref50_optimized import OptimizedUniRef50Trainer


def create_minimal_config():
    """Create a minimal configuration for testing."""
    config = {
        'model': {
            'dim': 256,
            'hidden_size': 256,
            'n_heads': 8,
            'n_blocks': 4,
            'n_blocks_prot': 4,
            'cond_dim': 128,
            'dropout': 0.1,
            'scale_by_sigma': True
        },
        'data': {
            'vocab_size_protein': 25,
            'max_protein_len': 256
        },
        'graph': {
            'type': 'absorb'
        },
        'noise': {
            'type': 'cosine',
            'sigma_min': 1e-4,
            'sigma_max': 0.5,
            'eps': 0.02
        },
        'sampling': {
            'predictor': 'euler',
            'steps': 50,
            'noise_removal': True
        },
        'training': {
            'eval_freq': 100,
            'ema': 0.9999
        },
        'optim': {
            'lr': 1e-4,
            'warmup': 1000
        },
        'tokens': 25
    }
    return config


def create_dummy_data(num_sequences=100, max_length=128):
    """Create dummy protein sequence data for testing."""
    # Amino acid vocabulary (20 standard + special tokens)
    vocab_size = 25
    
    data = []
    for i in range(num_sequences):
        # Random sequence length
        seq_len = torch.randint(30, max_length, (1,)).item()
        
        # Random protein tokens (avoid absorbing token which is vocab_size-1)
        tokens = torch.randint(0, vocab_size-1, (seq_len,))
        
        # Create dummy amino acid sequence
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequence = ''.join([amino_acids[t % 20] for t in tokens])
        
        data.append({
            'protein_seq': sequence,
            'prot_tokens': tokens,
            'length': seq_len
        })
    
    return data


def test_sampling_methods():
    """Test both sampling methods."""
    print("🧪 TESTING SAMPLING METHODS")
    print("=" * 80)
    
    # Setup
    work_dir = "./test_sampling"
    os.makedirs(work_dir, exist_ok=True)
    
    # Create config file
    config = create_minimal_config()
    config_file = os.path.join(work_dir, "test_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dummy data file
    dummy_data = create_dummy_data(50, 128)
    data_file = os.path.join(work_dir, "test_data.pt")
    torch.save(dummy_data, data_file)
    
    print(f"📁 Test directory: {work_dir}")
    print(f"⚙️  Config file: {config_file}")
    print(f"💾 Data file: {data_file}")
    
    try:
        # Test rigorous sampling
        print("\n🔬 Testing Rigorous CTMC Sampling")
        print("-" * 50)
        
        trainer_rigorous = OptimizedUniRef50Trainer(
            work_dir=work_dir,
            config_file=config_file,
            datafile=data_file,
            dev_id='cpu',  # Use CPU for testing
            seed=42,
            sampling_method="rigorous"
        )
        
        # Initialize trainer components
        trainer_rigorous.setup_data()
        trainer_rigorous.setup_model()
        trainer_rigorous.setup_optimizer()
        
        # Test rigorous sampling
        rigorous_sequences = trainer_rigorous.generate_protein_sequences(
            num_samples=3, max_length=50, sampling_method="rigorous"
        )
        
        print(f"✅ Generated {len(rigorous_sequences)} sequences with rigorous method")
        for i, seq in enumerate(rigorous_sequences[:2]):
            print(f"   Sample {i+1}: {seq['sequence'][:40]}... (len={seq['length']})")
        
        # Test simple sampling
        print("\n🎲 Testing Simple Heuristic Sampling")
        print("-" * 50)
        
        trainer_simple = OptimizedUniRef50Trainer(
            work_dir=work_dir,
            config_file=config_file,
            datafile=data_file,
            dev_id='cpu',  # Use CPU for testing
            seed=42,
            sampling_method="simple"
        )
        
        # Initialize trainer components
        trainer_simple.setup_data()
        trainer_simple.setup_model()
        trainer_simple.setup_optimizer()
        
        # Test simple sampling
        simple_sequences = trainer_simple.generate_protein_sequences(
            num_samples=3, max_length=50, sampling_method="simple",
            num_diffusion_steps=20, temperature=1.0
        )
        
        print(f"✅ Generated {len(simple_sequences)} sequences with simple method")
        for i, seq in enumerate(simple_sequences[:2]):
            print(f"   Sample {i+1}: {seq['sequence'][:40]}... (len={seq['length']})")
        
        # Compare methods
        print("\n📊 COMPARISON")
        print("-" * 50)
        
        rigorous_valid = len([s for s in rigorous_sequences if s['sequence']]) 
        simple_valid = len([s for s in simple_sequences if s['sequence']])
        
        print(f"Rigorous CTMC: {rigorous_valid}/{len(rigorous_sequences)} valid sequences")
        print(f"Simple Heuristic: {simple_valid}/{len(simple_sequences)} valid sequences")
        
        if rigorous_valid > 0:
            avg_len_rigorous = sum(s['length'] for s in rigorous_sequences if s['sequence']) / rigorous_valid
            print(f"Rigorous avg length: {avg_len_rigorous:.1f}")
        
        if simple_valid > 0:
            avg_len_simple = sum(s['length'] for s in simple_sequences if s['sequence']) / simple_valid
            print(f"Simple avg length: {avg_len_simple:.1f}")
        
        # Test the unified interface
        print("\n🔄 Testing Unified Interface")
        print("-" * 50)
        
        # Test method switching
        unified_rigorous = trainer_rigorous.generate_protein_sequences(
            num_samples=2, max_length=50, sampling_method="rigorous"
        )
        unified_simple = trainer_rigorous.generate_protein_sequences(
            num_samples=2, max_length=50, sampling_method="simple", 
            num_diffusion_steps=15, temperature=0.8
        )
        
        print(f"✅ Unified interface - Rigorous: {len(unified_rigorous)} sequences")
        print(f"✅ Unified interface - Simple: {len(unified_simple)} sequences")
        
        print("\n🎉 ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"🧹 Cleaned up test directory: {work_dir}")


def main():
    """Main test function."""
    print("🚀 SAMPLING METHODS TEST SUITE")
    print("=" * 80)
    print("Testing both rigorous CTMC and simple heuristic sampling methods")
    print("for protein sequence generation in UniRef50 trainer.")
    print()
    
    success = test_sampling_methods()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\n📖 Usage in training:")
        print("  # Use rigorous CTMC sampling (default)")
        print("  python protlig_dd/training/run_train_uniref50_optimized.py --sampling_method rigorous ...")
        print("  # Use simple heuristic sampling")
        print("  python protlig_dd/training/run_train_uniref50_optimized.py --sampling_method simple ...")
    else:
        print("❌ TESTS FAILED!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
