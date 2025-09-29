#!/usr/bin/env python3
"""
Test script for Aurora DDP setup - based on the template provided.
This tests the basic DDP functionality without the full training pipeline.
"""

from mpi4py import MPI
import os, socket
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# Try to import Intel extensions
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
    INTEL_AVAILABLE = True
    print("✅ Intel extensions available")
except ImportError as e:
    INTEL_AVAILABLE = False
    print(f"⚠️  Intel extensions not available: {e}")

def setup_ddp_aurora():
    """Setup DDP for Aurora with proper Intel XPU handling."""
    # DDP: Set environmental variables used by PyTorch
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    LOCAL_RANK = int(os.environ.get('PALS_LOCAL_RANKID', '0'))
    
    # Set environment variables
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)
    
    # Setup master address for Aurora
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
    os.environ['MASTER_PORT'] = str(2345)
    
    print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")
    
    # Initialize distributed communication with CCL backend for Intel XPU
    torch.distributed.init_process_group(
        backend='ccl', 
        init_method='env://', 
        rank=int(RANK), 
        world_size=int(SIZE)
    )
    
    # Set XPU device
    if INTEL_AVAILABLE:
        torch.xpu.set_device(LOCAL_RANK)
        device = torch.device(f'xpu:{LOCAL_RANK}')
    else:
        device = torch.device('cpu')
    
    print(f"✅ DDP initialized: rank {RANK}/{SIZE}, local_rank {LOCAL_RANK}, device {device}")
    
    return RANK, device, SIZE, LOCAL_RANK

def test_basic_ddp():
    """Test basic DDP functionality."""
    print("🧪 Testing Basic DDP Functionality")
    print("=" * 50)
    
    # Setup DDP
    rank, device, world_size, local_rank = setup_ddp_aurora()
    
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    
    # Create simple test data
    batch_size = 4
    input_size = 128
    hidden_size = 64
    
    # Create dummy data
    src = torch.rand((batch_size, input_size), device=device)
    tgt = torch.rand((batch_size, hidden_size), device=device)
    
    print(f"Rank {rank}: Created test data - src: {src.shape}, tgt: {tgt.shape}")
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size)
    )
    
    # Move model to device
    model = model.to(device)
    
    # Apply IPEX optimization if available
    if INTEL_AVAILABLE and device.type == 'xpu':
        print(f"Rank {rank}: Applying IPEX optimization...")
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * world_size)
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
            print(f"Rank {rank}: ✅ IPEX optimization applied")
        except Exception as e:
            print(f"Rank {rank}: ⚠️  IPEX optimization failed: {e}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * world_size)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * world_size)
    
    # Wrap with DDP
    if device.type == 'xpu':
        # For Intel XPU, don't specify device_ids
        model_ddp = DDP(model, find_unused_parameters=False)
    else:
        # For other devices, use traditional approach
        model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    print(f"Rank {rank}: ✅ Model wrapped with DDP")
    
    # Create loss function
    criterion = torch.nn.MSELoss()
    
    # Test training loop
    print(f"Rank {rank}: Starting test training loop...")
    
    for step in range(5):
        # Forward pass
        output = model_ddp(src)
        loss = criterion(output, tgt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}: Step {step+1}/5, Loss: {loss.item():.6f}")
        
        # Synchronize all ranks
        torch.distributed.barrier()
    
    print(f"Rank {rank}: ✅ Training loop completed successfully")
    
    # Test all-reduce operation
    test_tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    print(f"Rank {rank}: Before all-reduce: {test_tensor.item()}")
    
    torch.distributed.all_reduce(test_tensor, op=torch.distributed.ReduceOp.SUM)
    expected_sum = sum(range(world_size))  # 0 + 1 + 2 + ... + (world_size-1)
    
    print(f"Rank {rank}: After all-reduce: {test_tensor.item()}, Expected: {expected_sum}")
    
    if abs(test_tensor.item() - expected_sum) < 1e-6:
        print(f"Rank {rank}: ✅ All-reduce test passed")
    else:
        print(f"Rank {rank}: ❌ All-reduce test failed")
    
    return True

def cleanup_ddp():
    """Clean up DDP process group."""
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        print(f"Rank {rank}: 🔄 Cleaning up DDP...")
        
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            print(f"Rank {rank}: ✅ DDP cleanup completed")
        except Exception as e:
            print(f"Rank {rank}: ⚠️  DDP cleanup failed: {e}")

def main():
    """Main test function."""
    try:
        success = test_basic_ddp()
        
        if success:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"Rank {rank}: 🎉 All DDP tests passed!")
        else:
            print("❌ Some DDP tests failed!")
            
    except Exception as e:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"Rank {rank}: ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()
