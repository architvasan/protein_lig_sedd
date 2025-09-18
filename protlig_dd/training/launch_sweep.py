#!/usr/bin/env python3
"""
Launch wandb hyperparameter sweep for UniRef50 training.
Super lazy script - just run it and it handles everything!
"""

import wandb
import yaml
import os
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Launch wandb sweep for UniRef50")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--config", type=str, required=True, help="Base config file path")
    parser.add_argument("--datafile", type=str, default="./input_data/processed_uniref50.pt", help="Data file path")
    parser.add_argument("--project", type=str, default="uniref50-sweep", help="Wandb project name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--count", type=int, default=20, help="Number of sweep runs")
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml", help="Sweep config file")
    
    args = parser.parse_args()
    
    print("🚀 LAZY WANDB SWEEP LAUNCHER")
    print("="*50)
    print(f"📁 Work dir: {args.work_dir}")
    print(f"⚙️  Config: {args.config}")
    print(f"💾 Data: {args.datafile}")
    print(f"🏷️  Project: {args.project}")
    print(f"🖥️  Device: {args.device}")
    print(f"🔢 Runs: {args.count}")
    print("="*50)
    
    # Load sweep configuration
    sweep_config_path = os.path.join(os.path.dirname(__file__), args.sweep_config)
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Update count if specified
    sweep_config['count'] = args.count
    
    # Create the sweep
    print("🎯 Creating wandb sweep...")
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"✅ Sweep created: {sweep_id}")
    
    # Define the training function for the sweep
    def train_with_sweep():
        # Initialize wandb run
        wandb.init()
        
        # Get hyperparameters from wandb
        config = wandb.config
        
        # Import here to avoid issues
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from protlig_dd.training.run_train_uniref50_optimized import OptimizedUniRef50Trainer
        
        # Create trainer with sweep parameters
        trainer = OptimizedUniRef50Trainer(
            work_dir=args.work_dir,
            config_file=args.config,
            datafile=args.datafile,
            dev_id=args.device,
            seed=42,
            force_fresh_start=True,  # Always fresh start for sweeps
            sampling_method="simple",  # Use simple sampling for speed
            epochs_override=config.epochs  # Use sweep epochs
        )
        
        # Generate unique run name
        run_name = f"sweep_{datetime.now().strftime('%m%d_%H%M%S')}_lr{config.learning_rate:.1e}_bs{config.batch_size}"
        
        # Start training
        trainer.train(args.project, run_name)
    
    print(f"🏃 Starting sweep agent...")
    print(f"💡 You can also run additional agents with:")
    print(f"   wandb agent {sweep_id}")
    print()
    
    # Run the sweep agent
    wandb.agent(sweep_id, train_with_sweep, count=args.count)
    
    print("🎉 Sweep completed!")

if __name__ == "__main__":
    main()
