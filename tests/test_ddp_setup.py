#!/usr/bin/env python3
"""
Test script to validate DDP setup and hyperparameter scaling.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time


def test_ddp_worker(rank, world_size):
    """Test DDP functionality on each worker."""
    try:
        # Setup DDP
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        
        print(f"Rank {rank}: Initialized on device {device}")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        # Wrap with DDP
        if torch.cuda.is_available():
            ddp_model = DDP(model, device_ids=[rank])
        else:
            ddp_model = DDP(model)
        
        # Test forward pass
        x = torch.randn(8, 100).to(device)
        y = ddp_model(x)
        
        print(f"Rank {rank}: Forward pass successful, output shape: {y.shape}")
        
        # Test backward pass
        loss = y.sum()
        loss.backward()
        
        print(f"Rank {rank}: Backward pass successful")
        
        # Test synchronization
        tensor = torch.tensor([rank], dtype=torch.float32).to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(world_size))
        
        if tensor.item() == expected_sum:
            print(f"Rank {rank}: Synchronization test PASSED (sum={tensor.item()})")
        else:
            print(f"Rank {rank}: Synchronization test FAILED (expected {expected_sum}, got {tensor.item()})")
        
        # Cleanup
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"Rank {rank}: Error - {e}")
        raise


def test_hyperparameter_scaling():
    """Test hyperparameter scaling calculations."""
    print("\n🧮 Testing Hyperparameter Scaling")
    print("=" * 50)
    
    base_config = {
        'batch_size': 16,
        'accum': 4,
        'lr': 5e-5,
        'warmup': 5000,
    }
    
    for num_gpus in [1, 2, 4, 8]:
        effective_batch = base_config['batch_size'] * base_config['accum'] * num_gpus
        scaled_lr = base_config['lr'] * num_gpus
        scaled_warmup = base_config['warmup'] * num_gpus
        
        print(f"\n{num_gpus} GPUs:")
        print(f"  Effective batch size: {effective_batch}")
        print(f"  Scaled learning rate: {scaled_lr:.2e}")
        print(f"  Scaled warmup steps: {scaled_warmup}")
        
        # Memory estimation (rough)
        memory_per_gpu = 12  # GB baseline
        if effective_batch > 256:
            memory_per_gpu *= 1.2  # Slight increase for larger batches
        
        print(f"  Est. memory per GPU: {memory_per_gpu:.1f} GB")


def test_environment():
    """Test the environment setup."""
    print("🔍 Testing Environment")
    print("=" * 30)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Check distributed
    nccl_available = torch.distributed.is_nccl_available()
    print(f"NCCL available: {nccl_available}")
    
    gloo_available = torch.distributed.is_gloo_available()
    print(f"Gloo available: {gloo_available}")
    
    return cuda_available and num_gpus >= 2 and nccl_available


def main():
    print("🧪 DDP Setup Validation")
    print("=" * 40)
    
    # Test environment
    ddp_ready = test_environment()
    
    # Test hyperparameter scaling
    test_hyperparameter_scaling()
    
    if not ddp_ready:
        print("\n⚠️  DDP requirements not met. Skipping DDP test.")
        print("Requirements:")
        print("  - CUDA available")
        print("  - 2+ GPUs")
        print("  - NCCL backend available")
        return
    
    # Test DDP functionality
    print("\n🚀 Testing DDP Functionality")
    print("=" * 40)
    
    world_size = min(torch.cuda.device_count(), 4)  # Test with up to 4 GPUs
    print(f"Testing with {world_size} processes...")
    
    try:
        mp.spawn(
            test_ddp_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("\n✅ DDP test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ DDP test failed: {e}")
        return
    
    print("\n🎉 All tests passed! DDP setup is ready.")
    print("\nNext steps:")
    print("1. Run: ./run_train_uniref50_ddp.sh")
    print("2. Monitor training with: nvidia-smi -l 1")
    print("3. Check Wandb dashboard for metrics")


if __name__ == "__main__":
    main()
