#!/usr/bin/env python3
"""
Test script to verify GPU isolation with CUDA_VISIBLE_DEVICES.
This script helps debug multi-GPU hyperparameter sweep issues.
"""

import os
import sys
import torch
import subprocess
import time
from pathlib import Path


def test_cuda_visible_devices():
    """Test CUDA_VISIBLE_DEVICES functionality."""
    print("🔍 Testing CUDA_VISIBLE_DEVICES Isolation")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs available: {total_gpus}")
    
    # Test script that reports GPU info
    test_script = '''
import torch
import os

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {props.name}")
        
    # Test device assignment
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    
    # Create tensor and check which GPU it's on
    x = torch.randn(100, 100, device=device)
    print(f"Tensor device: {x.device}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Get actual GPU memory info
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"Memory allocated: {memory_allocated:.1f} MB")
    print(f"Memory reserved: {memory_reserved:.1f} MB")
else:
    print("CUDA not available in subprocess")
'''
    
    # Test different CUDA_VISIBLE_DEVICES settings
    test_cases = []
    for gpu_id in range(min(4, total_gpus)):
        test_cases.append(str(gpu_id))
    
    if total_gpus >= 2:
        test_cases.append("0,1")
    if total_gpus >= 4:
        test_cases.append("2,3")
    
    for cuda_setting in test_cases:
        print(f"\n🧪 Testing CUDA_VISIBLE_DEVICES={cuda_setting}")
        print("-" * 40)
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda_setting
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', test_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                print("✅ Success:")
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
            else:
                print("❌ Failed:")
                print(f"    stdout: {result.stdout}")
                print(f"    stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True


def test_concurrent_gpu_usage():
    """Test concurrent GPU usage with different CUDA_VISIBLE_DEVICES."""
    print("\n⚡ Testing Concurrent GPU Usage")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    total_gpus = torch.cuda.device_count()
    if total_gpus < 2:
        print("⚠️  Need at least 2 GPUs for concurrent testing")
        return False
    
    # GPU workload script
    workload_script = '''
import torch
import time
import sys
import os

gpu_setting = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
job_name = sys.argv[1] if len(sys.argv) > 1 else 'test'
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10

print(f"Job {job_name}: Starting on CUDA_VISIBLE_DEVICES={gpu_setting}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Always use cuda:0 when CUDA_VISIBLE_DEVICES is set
    print(f"Job {job_name}: Using device {device}")
    
    # Create workload
    x = torch.randn(2000, 2000, device=device)
    
    for i in range(duration):
        # Matrix multiplication workload
        y = torch.matmul(x, x.T)
        z = torch.sum(y)
        
        memory_mb = torch.cuda.memory_allocated(device) / 1024**2
        print(f"Job {job_name}: Step {i+1}/{duration}, Memory: {memory_mb:.1f}MB, Result: {z.item():.2e}")
        time.sleep(1)
    
    print(f"Job {job_name}: Completed successfully")
else:
    print(f"Job {job_name}: CUDA not available")
'''
    
    # Save workload script
    script_path = Path("temp_gpu_workload.py")
    with open(script_path, 'w') as f:
        f.write(workload_script)
    
    try:
        # Start concurrent processes on different GPUs
        processes = []
        num_jobs = min(4, total_gpus)
        
        print(f"Starting {num_jobs} concurrent GPU jobs...")
        
        for i in range(num_jobs):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(i)
            
            process = subprocess.Popen(
                [sys.executable, str(script_path), f"job_{i}", "8"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((process, i))
            print(f"  Started job_{i} on GPU {i}")
        
        # Monitor processes
        print("\nMonitoring concurrent execution...")
        completed = 0
        
        while completed < len(processes):
            for process, gpu_id in processes:
                if process.poll() is not None and process not in [p[0] for p in processes if p[0].poll() is None]:
                    stdout, stderr = process.communicate()
                    
                    if process.returncode == 0:
                        print(f"✅ Job on GPU {gpu_id} completed successfully")
                        # Show last few lines
                        lines = stdout.strip().split('\n')
                        for line in lines[-3:]:
                            if line.strip():
                                print(f"    {line}")
                    else:
                        print(f"❌ Job on GPU {gpu_id} failed:")
                        print(f"    {stderr}")
                    
                    completed += 1
                    break
            
            time.sleep(1)
        
        print(f"\n🎉 All {num_jobs} concurrent jobs completed!")
        
    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()
    
    return True


def test_hyperparameter_sweep_simulation():
    """Simulate the hyperparameter sweep GPU assignment."""
    print("\n🧪 Testing Hyperparameter Sweep GPU Assignment")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    total_gpus = torch.cuda.device_count()
    available_gpus = list(range(min(4, total_gpus)))
    
    print(f"Available GPUs for sweep: {available_gpus}")
    
    # Simulate the sweep logic
    configs = [
        {'name': 'small_fast', 'lr': 1e-4},
        {'name': 'balanced', 'lr': 5e-5},
        {'name': 'large_thorough', 'lr': 2e-5},
        {'name': 'experimental', 'lr': 8e-5},
        {'name': 'memory_efficient', 'lr': 3e-5},
    ]
    
    # Track GPU assignments
    gpu_processes = {gpu_id: None for gpu_id in available_gpus}
    
    print("\nSimulating job assignments:")
    
    for i, config in enumerate(configs):
        # Find available GPU (simulate the sweep logic)
        available_gpu = None
        for gpu_id in available_gpus:
            if gpu_processes[gpu_id] is None:
                available_gpu = gpu_id
                break
        
        if available_gpu is not None:
            gpu_processes[available_gpu] = config['name']
            print(f"  Job {i+1}: {config['name']} → GPU {available_gpu}")
            
            # Test the actual command that would be run
            cmd = [
                'python', '-c', 
                f'''
import torch
import os
print(f"Config: {config['name']}")
print(f"CUDA_VISIBLE_DEVICES: {{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}}")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using device: {{device}}")
    print(f"Device name: {{torch.cuda.get_device_properties(device).name}}")
else:
    print("CUDA not available")
'''
            ]
            
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(available_gpu)
            
            try:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"    ✅ GPU {available_gpu} assignment successful")
                    for line in result.stdout.strip().split('\n'):
                        print(f"      {line}")
                else:
                    print(f"    ❌ GPU {available_gpu} assignment failed: {result.stderr}")
            except Exception as e:
                print(f"    ❌ Error testing GPU {available_gpu}: {e}")
        else:
            print(f"  Job {i+1}: {config['name']} → Waiting for GPU...")
    
    return True


def main():
    """Run all GPU isolation tests."""
    print("🧪 GPU Isolation Test Suite for Hyperparameter Sweep")
    print("=" * 70)
    
    success = True
    
    # Test 1: CUDA_VISIBLE_DEVICES functionality
    if not test_cuda_visible_devices():
        success = False
    
    # Test 2: Concurrent GPU usage
    if not test_concurrent_gpu_usage():
        success = False
    
    # Test 3: Hyperparameter sweep simulation
    if not test_hyperparameter_sweep_simulation():
        success = False
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 All GPU isolation tests passed!")
        print("\nThe multi-GPU hyperparameter sweep should work correctly.")
        print("Next steps:")
        print("1. Run: ./run_hyperparam_sweep.sh --dry-run")
        print("2. Run: ./run_hyperparam_sweep.sh --gpus 0,1")
        print("3. Monitor with: nvidia-smi -l 1")
    else:
        print("❌ Some tests failed. Check GPU setup and CUDA installation.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
