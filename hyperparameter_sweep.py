#!/usr/bin/env python3
"""
Hyperparameter optimization sweep for UniRef50 SEDD training.
Creates multiple config variations and runs training jobs.
"""

import os
import sys
import json
import yaml
import itertools
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class HyperparameterSweep:
    """Hyperparameter sweep manager for UniRef50 training."""
    
    def __init__(self, base_config_path: str, work_dir: str, datafile: str):
        self.base_config_path = base_config_path
        self.work_dir = work_dir
        self.datafile = datafile
        self.sweep_dir = os.path.join(work_dir, f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.sweep_dir, exist_ok=True)
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """Define the hyperparameter search space."""
        return {
            # Model architecture parameters
            'model.hidden_size': [512, 768, 1024],
            'model.n_heads': [8, 12, 16],
            'model.n_blocks_prot': [6, 8, 12],
            'model.dropout': [0.1, 0.15, 0.2],
            'model.cond_dim': [128, 256, 512],
            
            # Training parameters
            'training.batch_size': [16, 32, 64],
            'training.accum': [2, 4, 8],
            'training.ema': [0.999, 0.9995, 0.9999],
            
            # Optimizer parameters
            'optim.lr': [1e-5, 5e-5, 1e-4, 2e-4],
            'optim.weight_decay': [0.01, 0.05, 0.1],
            'optim.warmup': [1000, 5000, 10000],
            'optim.beta2': [0.95, 0.99, 0.999],
            'optim.grad_clip': [0.5, 1.0, 2.0],
            
            # Noise schedule parameters
            'noise.type': ['cosine', 'linear'],
            'noise.sigma_max': [0.5, 0.8, 1.0],
            'noise.eps': [0.02, 0.05, 0.1],
            
            # Sampling parameters
            'sampling.steps': [50, 100, 200],
            'sampling.predictor': ['euler'],  # Keep consistent for now
            
            # Data parameters
            'data.max_protein_len': [256, 512, 1024],
            'data.train_ratio': [0.9, 0.95],
            
            # Curriculum learning
            'curriculum.enabled': [True, False],
            'curriculum.preschool_time': [5000, 10000, 20000],
            
            # System parameters
            'sampling_method': ['rigorous', 'simple'],
        }
    
    def get_predefined_configs(self) -> List[Dict[str, Any]]:
        """Get a set of predefined promising configurations."""
        return [
            # Small, fast configuration for quick iteration
            {
                'name': 'small_fast',
                'model.hidden_size': 512,
                'model.n_heads': 8,
                'model.n_blocks_prot': 6,
                'model.dropout': 0.1,
                'model.cond_dim': 128,
                'training.batch_size': 32,
                'training.accum': 4,
                'training.n_iters': 50000,
                'optim.lr': 1e-4,
                'optim.warmup': 5000,
                'noise.sigma_max': 0.5,
                'sampling.steps': 50,
                'data.max_protein_len': 256,
                'sampling_method': 'simple',
            },
            
            # Medium configuration with rigorous sampling
            {
                'name': 'medium_rigorous',
                'model.hidden_size': 768,
                'model.n_heads': 12,
                'model.n_blocks_prot': 8,
                'model.dropout': 0.15,
                'model.cond_dim': 256,
                'training.batch_size': 32,
                'training.accum': 4,
                'training.n_iters': 100000,
                'optim.lr': 5e-5,
                'optim.warmup': 10000,
                'noise.sigma_max': 0.8,
                'sampling.steps': 100,
                'data.max_protein_len': 512,
                'sampling_method': 'rigorous',
            },
            
            # Large configuration for best quality
            {
                'name': 'large_quality',
                'model.hidden_size': 1024,
                'model.n_heads': 16,
                'model.n_blocks_prot': 12,
                'model.dropout': 0.1,
                'model.cond_dim': 512,
                'training.batch_size': 16,
                'training.accum': 8,
                'training.n_iters': 200000,
                'optim.lr': 2e-5,
                'optim.warmup': 15000,
                'noise.sigma_max': 0.8,
                'sampling.steps': 200,
                'data.max_protein_len': 512,
                'sampling_method': 'rigorous',
            },
            
            # High learning rate experiment
            {
                'name': 'high_lr_experiment',
                'model.hidden_size': 768,
                'model.n_heads': 12,
                'model.n_blocks_prot': 8,
                'model.dropout': 0.2,
                'model.cond_dim': 256,
                'training.batch_size': 64,
                'training.accum': 2,
                'training.n_iters': 75000,
                'optim.lr': 2e-4,
                'optim.warmup': 5000,
                'optim.weight_decay': 0.05,
                'noise.sigma_max': 0.5,
                'sampling.steps': 100,
                'data.max_protein_len': 512,
                'sampling_method': 'simple',
            },
            
            # Curriculum learning focus
            {
                'name': 'curriculum_focus',
                'model.hidden_size': 768,
                'model.n_heads': 12,
                'model.n_blocks_prot': 8,
                'model.dropout': 0.15,
                'model.cond_dim': 256,
                'training.batch_size': 32,
                'training.accum': 4,
                'training.n_iters': 150000,
                'optim.lr': 1e-4,
                'optim.warmup': 10000,
                'curriculum.enabled': True,
                'curriculum.preschool_time': 20000,
                'noise.sigma_max': 0.8,
                'sampling.steps': 100,
                'data.max_protein_len': 512,
                'sampling_method': 'rigorous',
            },
        ]
    
    def create_config_from_params(self, params: Dict[str, Any], config_name: str) -> str:
        """Create a config file from parameter dictionary."""
        config = self.base_config.copy()
        
        # Apply parameter overrides
        for param_path, value in params.items():
            if param_path == 'name':
                continue
            if param_path == 'sampling_method':
                continue  # This will be passed as command line argument
                
            # Navigate nested dictionary structure
            keys = param_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        # Save config file
        config_path = os.path.join(self.sweep_dir, f"config_{config_name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def run_training_job(self, config_path: str, job_name: str, sampling_method: str = 'rigorous', 
                        dry_run: bool = False) -> Optional[subprocess.Popen]:
        """Run a single training job."""
        wandb_name = f"sweep_{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = [
            'python', '-m', 'protlig_dd.training.run_train_uniref50_optimized',
            '--work_dir', self.work_dir,
            '--config', config_path,
            '--datafile', self.datafile,
            '--wandb_project', 'uniref50_hyperparam_sweep',
            '--wandb_name', wandb_name,
            '--sampling_method', sampling_method,
            '--device', 'cuda:0',
        ]
        
        if dry_run:
            print(f"🔍 DRY RUN - Would execute: {' '.join(cmd)}")
            return None
        
        print(f"🚀 Starting job: {job_name}")
        print(f"   Config: {config_path}")
        print(f"   Wandb: {wandb_name}")
        print(f"   Sampling: {sampling_method}")
        
        # Create log file
        log_file = os.path.join(self.sweep_dir, f"log_{job_name}.txt")
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        return process
    
    def run_predefined_sweep(self, dry_run: bool = False, max_concurrent: int = 2):
        """Run sweep with predefined configurations."""
        configs = self.get_predefined_configs()
        
        print(f"🎯 Running predefined hyperparameter sweep")
        print(f"📁 Sweep directory: {self.sweep_dir}")
        print(f"🔧 Number of configurations: {len(configs)}")
        print(f"⚡ Max concurrent jobs: {max_concurrent}")
        print()
        
        # Create summary file
        summary_file = os.path.join(self.sweep_dir, "sweep_summary.json")
        summary = {
            'sweep_type': 'predefined',
            'start_time': datetime.now().isoformat(),
            'configs': configs,
            'base_config': self.base_config_path,
            'work_dir': self.work_dir,
            'datafile': self.datafile,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Run jobs
        active_processes = []
        
        for i, config_params in enumerate(configs):
            config_name = config_params.get('name', f'config_{i:03d}')
            sampling_method = config_params.get('sampling_method', 'rigorous')
            
            # Create config file
            config_path = self.create_config_from_params(config_params, config_name)
            
            # Wait if we have too many concurrent jobs
            while len(active_processes) >= max_concurrent:
                # Check for completed processes
                active_processes = [p for p in active_processes if p.poll() is None]
                if len(active_processes) >= max_concurrent:
                    time.sleep(30)  # Wait 30 seconds before checking again
            
            # Start new job
            process = self.run_training_job(config_path, config_name, sampling_method, dry_run)
            if process:
                active_processes.append(process)
            
            print(f"✅ Job {i+1}/{len(configs)} started: {config_name}")
            time.sleep(5)  # Brief delay between job starts
        
        # Wait for all jobs to complete
        if not dry_run:
            print(f"\n⏳ Waiting for all jobs to complete...")
            for process in active_processes:
                process.wait()
        
        print(f"\n🎉 Sweep completed!")
        print(f"📊 Check results in Wandb project: uniref50_hyperparam_sweep")
        print(f"📁 Logs and configs saved in: {self.sweep_dir}")

    def run_random_sweep(self, num_configs: int = 10, dry_run: bool = False, max_concurrent: int = 2):
        """Run random hyperparameter sweep."""
        import random

        grid = self.get_hyperparameter_grid()

        print(f"🎲 Running random hyperparameter sweep")
        print(f"📁 Sweep directory: {self.sweep_dir}")
        print(f"🔧 Number of random configurations: {num_configs}")
        print(f"⚡ Max concurrent jobs: {max_concurrent}")
        print()

        # Generate random configurations
        configs = []
        for i in range(num_configs):
            config = {'name': f'random_{i:03d}'}
            for param, values in grid.items():
                config[param] = random.choice(values)
            configs.append(config)

        # Create summary file
        summary_file = os.path.join(self.sweep_dir, "random_sweep_summary.json")
        summary = {
            'sweep_type': 'random',
            'start_time': datetime.now().isoformat(),
            'num_configs': num_configs,
            'configs': configs,
            'base_config': self.base_config_path,
            'work_dir': self.work_dir,
            'datafile': self.datafile,
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Run jobs
        active_processes = []

        for i, config_params in enumerate(configs):
            config_name = config_params.get('name', f'random_{i:03d}')
            sampling_method = config_params.get('sampling_method', 'rigorous')

            # Create config file
            config_path = self.create_config_from_params(config_params, config_name)

            # Wait if we have too many concurrent jobs
            while len(active_processes) >= max_concurrent:
                active_processes = [p for p in active_processes if p.poll() is None]
                if len(active_processes) >= max_concurrent:
                    time.sleep(30)

            # Start new job
            process = self.run_training_job(config_path, config_name, sampling_method, dry_run)
            if process:
                active_processes.append(process)

            print(f"✅ Job {i+1}/{num_configs} started: {config_name}")
            time.sleep(5)

        # Wait for all jobs to complete
        if not dry_run:
            print(f"\n⏳ Waiting for all jobs to complete...")
            for process in active_processes:
                process.wait()

        print(f"\n🎉 Random sweep completed!")
        print(f"📊 Check results in Wandb project: uniref50_hyperparam_sweep")
        print(f"📁 Logs and configs saved in: {self.sweep_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for UniRef50 training")
    parser.add_argument("--base_config", type=str, required=True, 
                       help="Path to base configuration file")
    parser.add_argument("--work_dir", type=str, required=True,
                       help="Working directory for experiments")
    parser.add_argument("--datafile", type=str, required=True,
                       help="Path to processed UniRef50 data file")
    parser.add_argument("--max_concurrent", type=int, default=2,
                       help="Maximum number of concurrent training jobs")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing them")
    parser.add_argument("--sweep_type", type=str, choices=["predefined", "random"],
                       default="predefined", help="Type of sweep to run")
    parser.add_argument("--num_random", type=int, default=10,
                       help="Number of random configurations (for random sweep)")

    args = parser.parse_args()
    
    print("🧬 UNIREF50 HYPERPARAMETER SWEEP")
    print("=" * 60)
    print(f"📁 Base config: {args.base_config}")
    print(f"📁 Work directory: {args.work_dir}")
    print(f"💾 Data file: {args.datafile}")
    print(f"⚡ Max concurrent jobs: {args.max_concurrent}")
    print(f"🔍 Dry run: {args.dry_run}")
    print()
    
    # Validate inputs
    if not os.path.exists(args.base_config):
        print(f"❌ Base config file not found: {args.base_config}")
        return 1
    
    if not os.path.exists(args.datafile):
        print(f"❌ Data file not found: {args.datafile}")
        return 1
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Create and run sweep
    sweep = HyperparameterSweep(args.base_config, args.work_dir, args.datafile)

    if args.sweep_type == "predefined":
        sweep.run_predefined_sweep(dry_run=args.dry_run, max_concurrent=args.max_concurrent)
    elif args.sweep_type == "random":
        sweep.run_random_sweep(num_configs=args.num_random, dry_run=args.dry_run,
                              max_concurrent=args.max_concurrent)

    return 0


if __name__ == "__main__":
    exit(main())
