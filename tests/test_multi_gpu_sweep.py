#!/usr/bin/env python3
"""
Test script to validate multi-GPU hyperparameter sweep functionality.
"""

import os
import time
import subprocess
import torch
from pathlib import Path


def test_gpu_availability():
    """Test GPU availability and CUDA setup."""
    print("🔍 Testing GPU Availability")
    print("=" * 40)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available. Cannot test multi-GPU sweep.")
        return False
    
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("⚠️  Only 1 GPU available. Multi-GPU testing limited.")
    
    # List GPU details
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    return True


def test_cuda_visible_devices():
    """Test CUDA_VISIBLE_DEVICES functionality."""
    print("\n🎯 Testing CUDA_VISIBLE_DEVICES")
    print("=" * 40)
    
    # Test script that prints visible GPU
    test_script = '''
import torch
import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
if torch.cuda.is_available():
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
else:
    print("CUDA not available")
'''
    
    # Test different CUDA_VISIBLE_DEVICES settings
    for gpu_setting in ["0", "1", "0,1", "2,3"]:
        print(f"\nTesting CUDA_VISIBLE_DEVICES={gpu_setting}:")
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_setting
        
        try:
            result = subprocess.run(
                ['python', '-c', test_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✅ Success:")
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
            else:
                print("❌ Failed:")
                print(f"    {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")


def test_concurrent_processes():
    """Test running concurrent processes on different GPUs."""
    print("\n⚡ Testing Concurrent GPU Processes")
    print("=" * 40)
    
    # Simple GPU test script
    gpu_test_script = '''
import torch
import time
import os
import sys

gpu_id = int(sys.argv[1])
duration = int(sys.argv[2])

print(f"Process starting on GPU {gpu_id} for {duration}s")

if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    
    # Create some tensors to use GPU memory
    x = torch.randn(1000, 1000, device=device)
    
    for i in range(duration):
        # Do some computation
        y = torch.matmul(x, x.T)
        time.sleep(1)
        print(f"GPU {gpu_id}: Step {i+1}/{duration}")
    
    print(f"Process on GPU {gpu_id} completed")
else:
    print(f"CUDA not available for GPU {gpu_id}")
'''
    
    # Save test script
    test_script_path = Path("temp_gpu_test.py")
    with open(test_script_path, 'w') as f:
        f.write(gpu_test_script)
    
    try:
        # Start processes on different GPUs
        processes = []
        gpu_ids = [0, 1, 2, 3]
        available_gpus = min(torch.cuda.device_count(), len(gpu_ids))
        
        print(f"Starting {available_gpus} concurrent processes...")
        
        for i in range(available_gpus):
            gpu_id = gpu_ids[i]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            process = subprocess.Popen(
                ['python', str(test_script_path), str(gpu_id), '5'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((process, gpu_id))
            print(f"  Started process on GPU {gpu_id}")
        
        # Wait for all processes to complete
        print("\nWaiting for processes to complete...")
        for process, gpu_id in processes:
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"✅ GPU {gpu_id} process completed successfully")
                # Print last few lines of output
                lines = stdout.strip().split('\n')
                for line in lines[-2:]:
                    print(f"    {line}")
            else:
                print(f"❌ GPU {gpu_id} process failed:")
                print(f"    {stderr}")
    
    finally:
        # Clean up test script
        if test_script_path.exists():
            test_script_path.unlink()


def test_hyperparameter_sweep_dry_run():
    """Test hyperparameter sweep in dry run mode."""
    print("\n🧪 Testing Hyperparameter Sweep (Dry Run)")
    print("=" * 50)

    # Check if sweep script exists
    sweep_script = Path("hyperparameter_sweep.py")
    if not sweep_script.exists():
        print("❌ hyperparameter_sweep.py not found")
        return

    # Check if config exists
    config_files = [
        'configs/config_uniref50.yaml',
        'configs/config_uniref50_optimized.yaml',
        'configs/config_uniref50_stable.yaml'
    ]

    config_file = None
    for cf in config_files:
        if Path(cf).exists():
            config_file = cf
            break

    if not config_file:
        print("❌ No config file found. Tried:")
        for cf in config_files:
            print(f"    {cf}")
        return

    print(f"Using config: {config_file}")

    # Test with different GPU configurations
    gpu_configs = ["0", "0,1", "0,1,2,3"]

    for gpus in gpu_configs:
        print(f"\nTesting with GPUs: {gpus}")

        cmd = [
            'python', 'hyperparameter_sweep.py',
            '--base_config', config_file,
            '--work_dir', './test_sweep',
            '--datafile', './input_data/processed_uniref50.pt',
            '--sweep_type', 'predefined',
            '--gpus', gpus,
            '--dry_run'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("✅ Dry run successful")
                # Show GPU assignment info
                lines = result.stdout.split('\n')
                for line in lines:
                    if ('physical GPU' in line.lower() or
                        'cuda_visible_devices' in line.lower() or
                        'would execute' in line.lower()):
                        print(f"    {line}")
            else:
                print("❌ Dry run failed:")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("❌ Dry run timeout")
        except Exception as e:
            print(f"❌ Error: {e}")


def test_cuda_device_mapping():
    """Test CUDA device mapping with CUDA_VISIBLE_DEVICES."""
    print("\n🔍 Testing CUDA Device Mapping")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs: {total_gpus}")

    # Test the key insight: when CUDA_VISIBLE_DEVICES=N, cuda:0 maps to physical GPU N
    test_script = '''
import torch
import os

cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

if torch.cuda.is_available():
    print(f"Visible GPU count: {torch.cuda.device_count()}")

    # This should always be cuda:0 when CUDA_VISIBLE_DEVICES is set to a single GPU
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # Get device properties
    props = torch.cuda.get_device_properties(0)
    print(f"Device 0 name: {props.name}")
    print(f"Device 0 memory: {props.total_memory / 1024**3:.1f} GB")

    # Test tensor creation
    x = torch.randn(100, 100, device=device)
    print(f"Tensor created successfully on {x.device}")
else:
    print("CUDA not available")
'''

    # Test mapping for each GPU
    for gpu_id in range(min(4, total_gpus)):
        print(f"\n🧪 Testing physical GPU {gpu_id} → cuda:0 mapping:")

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            result = subprocess.run(
                ['python', '-c', test_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print("✅ Mapping successful:")
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
            else:
                print("❌ Mapping failed:")
                print(f"    {result.stderr}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run all tests."""
    print("🧪 Multi-GPU Hyperparameter Sweep Test Suite")
    print("=" * 60)
    
    # Test 1: GPU availability
    if not test_gpu_availability():
        return
    
    # Test 2: CUDA_VISIBLE_DEVICES
    test_cuda_visible_devices()
    
    # Test 3: Concurrent processes
    test_concurrent_processes()
    
    # Test 4: CUDA device mapping
    test_cuda_device_mapping()

    # Test 5: Hyperparameter sweep dry run
    test_hyperparameter_sweep_dry_run()

    print("\n🎉 Multi-GPU testing completed!")
    print("\nNext steps:")
    print("1. Run: python test_gpu_isolation.py  # More detailed GPU tests")
    print("2. Run: ./run_hyperparam_sweep.sh --dry-run")
    print("3. Run: ./run_hyperparam_sweep.sh --gpus 0,1")
    print("4. Monitor with: nvidia-smi -l 1")


if __name__ == "__main__":
    main()
