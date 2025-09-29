#!/usr/bin/env python3
"""
Config dictionary format for UniRef50 training configuration.
This provides a Python dictionary equivalent of configs/config_uniref50.yaml
"""

def get_uniref_config():
    """
    Returns the complete configuration dictionary for UniRef50 training.
    
    Returns:
        dict: Complete configuration dictionary
    """
    config = {
        # Curriculum learning settings
        'curriculum': {
            'enabled': True,
            'probabilistic': True,
            'preschool_time': 2500,
            # Alternative curriculum settings (commented in YAML):
            # 'difficulty_ramp': 'exponential',
        },
        
        # Data configuration
        'data': {
            'cache_dir': 'data',
            'max_ligand_len': 128,
            'max_protein_len': 512,
            'train': 'uniref50',
            'train_ratio': 0.95,
            'use_structure': False,
            'val_ratio': 0.05,
            'valid': 'uniref50',
            'vocab_size_ligand': 2364,
            'vocab_size_protein': 36,
        },
        
        # Evaluation settings
        'eval': {
            'batch_size': 16,
            'perplexity': True,
            'perplexity_batch_size': 8,
        },
        
        # Graph configuration
        'graph': {
            'file': 'data',
            'report_all': False,
            'type': 'absorb',
        },
        
        # Hydra configuration (for experiment tracking)
        'hydra': {
            'run': {
                'dir': 'exp_local/uniref50/${now:%Y.%m.%d}/${now:%H%M%S}',
            },
        },
        
        # Learning rate schedule
        'lr_schedule': {
            'max_steps': 5000000,
            'min_lr_ratio': 0.1,
            'type': 'cosine_with_warmup',
            'warmup_steps': 2000,
        },
        
        # Memory optimization settings
        'memory': {
            'gradient_checkpointing': True,
            'max_memory_per_gpu': 0.9,
            'mixed_precision': True,
        },
        
        # Model architecture
        'model': {
            'cond_dim': 512,
            'device': 'cuda:0',
            'dropout': 0.1,
            'esm_dim': 640,
            'hidden_size': 1024,
            'length': 512,
            'molformer_dim': 768,
            'n_blocks_lig': 8,
            'n_blocks_prot': 20,
            'n_heads': 16,
            'name': 'medium',
            'scale_by_sigma': True,
            'type': 'ddit',
        },
        
        # Monitoring and logging
        'monitoring': {
            'log_gradients': True,
            'log_weights': False,
            'sample_frequency': 5000,
        },
        
        # GPU configuration
        'ngpus': 1,
        
        # Noise schedule
        'noise': {
            'eps': 0.02,
            'sigma_max': 0.95,
            'sigma_min': 0.01,
            'type': 'cosine',
        },
        
        # Optimizer settings
        'optim': {
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-08,
            'grad_clip': 1.0,
            'lr': 0.00002,
            'optimizer': 'AdamW',
            'warmup': 2500,
            'weight_decay': 0.01,
        },
        
        # Sampling configuration
        'sampling': {
            'noise_removal': True,
            'predictor': 'euler',
            'steps': 50,
        },
        
        # Token vocabulary size
        'tokens': 36,
        
        # Training configuration
        'training': {
            'accum': 4,
            'batch_size': 16,
            'ema': 0.999,
            'epochs': 5,
            'eval_freq': 2000,
            'force_reprocess': False,
            'log_freq': 20,
            'max_samples': 50000000,
            'n_iters': 25000,
            'num_workers': 4,
            'seed': 42,
            'snapshot_freq': 1000,
            'snapshot_freq_for_preemption': 1000,
            'snapshot_sampling': True,
            'task': 'protein_only',
            'weight': 'standard',
        },
        
        # Work directory
        'work_dir': '/Users/ramanathana/Work/Protein-Ligand-SEDD/protein_lig_sedd',
        
        # Defaults (Hydra-specific)
        'defaults': [
            '_self_',
            {'model': 'medium'},
        ],
    }
    
    return config


def get_minimal_test_config():
    """
    Returns a minimal configuration for testing purposes.
    
    Returns:
        dict: Minimal test configuration
    """
    base_config = get_uniref_config()
    
    # Override with minimal settings for testing
    test_overrides = {
        'training': {
            **base_config['training'],
            'batch_size': 4,
            'n_iters': 20,
            'log_freq': 5,
            'eval_freq': 10,
            'snapshot_freq': 10,
            'accum': 1,
        },
        'data': {
            **base_config['data'],
            'max_protein_len': 64,
            'max_ligand_len': 32,
        },
        'model': {
            **base_config['model'],
            'hidden_size': 128,
            'n_blocks_lig': 2,
            'n_blocks_prot': 4,
            'n_heads': 4,
        },
        'memory': {
            **base_config['memory'],
            'gradient_checkpointing': False,
            'mixed_precision': False,
        },
    }
    
    # Deep merge the overrides
    config = base_config.copy()
    for key, value in test_overrides.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    return config


def get_ddp_config(world_size=8, batch_size_per_gpu=4):
    """
    Returns configuration optimized for DDP training.
    
    Args:
        world_size (int): Number of GPUs/processes
        batch_size_per_gpu (int): Batch size per GPU
    
    Returns:
        dict: DDP-optimized configuration
    """
    base_config = get_uniref_config()
    
    # Calculate effective batch size and learning rate scaling
    effective_batch_size = world_size * batch_size_per_gpu
    lr_scale = world_size  # Linear scaling rule
    
    ddp_overrides = {
        'training': {
            **base_config['training'],
            'batch_size': batch_size_per_gpu,  # Per-GPU batch size
            'accum': 1,  # Disable gradient accumulation for DDP
        },
        'optim': {
            **base_config['optim'],
            'lr': base_config['optim']['lr'] * lr_scale,  # Scale learning rate
        },
        'ngpus': world_size,
        'memory': {
            **base_config['memory'],
            'gradient_checkpointing': True,  # Enable for memory efficiency
            'mixed_precision': True,
        },
    }
    
    # Deep merge the overrides
    config = base_config.copy()
    for key, value in ddp_overrides.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    return config


if __name__ == "__main__":
    # Example usage
    print("=== Full UniRef Config ===")
    full_config = get_uniref_config()
    print(f"Model hidden size: {full_config['model']['hidden_size']}")
    print(f"Training batch size: {full_config['training']['batch_size']}")
    print(f"Learning rate: {full_config['optim']['lr']}")
    
    print("\n=== Minimal Test Config ===")
    test_config = get_minimal_test_config()
    print(f"Model hidden size: {test_config['model']['hidden_size']}")
    print(f"Training iterations: {test_config['training']['n_iters']}")
    
    print("\n=== DDP Config (8 GPUs) ===")
    ddp_config = get_ddp_config(world_size=8, batch_size_per_gpu=4)
    print(f"Per-GPU batch size: {ddp_config['training']['batch_size']}")
    print(f"Scaled learning rate: {ddp_config['optim']['lr']}")
    print(f"Number of GPUs: {ddp_config['ngpus']}")
