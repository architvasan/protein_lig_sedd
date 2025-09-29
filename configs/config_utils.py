#!/usr/bin/env python3
"""
Utility functions for working with configuration files.
Provides conversion between YAML and dictionary formats.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Union
from config_uniref_dict import get_uniref_config, get_minimal_test_config, get_ddp_config


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_yaml_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


def dict_to_namespace(config_dict: Dict[str, Any]) -> object:
    """
    Convert configuration dictionary to namespace object for attribute access.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        object: Namespace object with attribute access
    """
    class ConfigNamespace:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(**value))
                else:
                    setattr(self, key, value)
        
        def __repr__(self):
            items = []
            for key, value in self.__dict__.items():
                if isinstance(value, ConfigNamespace):
                    items.append(f"{key}=ConfigNamespace(...)")
                else:
                    items.append(f"{key}={value}")
            return f"ConfigNamespace({', '.join(items)})"
    
    return ConfigNamespace(**config_dict)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        dict: Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def create_config_variants():
    """
    Create various configuration variants and save them as YAML files.
    """
    configs_dir = Path(__file__).parent
    
    # Full UniRef config
    full_config = get_uniref_config()
    save_yaml_config(full_config, configs_dir / "config_uniref_full.yaml")
    print("✅ Saved: config_uniref_full.yaml")
    
    # Minimal test config
    test_config = get_minimal_test_config()
    save_yaml_config(test_config, configs_dir / "config_uniref_test.yaml")
    print("✅ Saved: config_uniref_test.yaml")
    
    # DDP configs for different scales
    for world_size in [4, 8, 16, 32]:
        ddp_config = get_ddp_config(world_size=world_size, batch_size_per_gpu=4)
        save_yaml_config(ddp_config, configs_dir / f"config_uniref_ddp_{world_size}gpu.yaml")
        print(f"✅ Saved: config_uniref_ddp_{world_size}gpu.yaml")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary for required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_sections = [
        'model', 'training', 'optim', 'data', 'noise'
    ]
    
    required_fields = {
        'model': ['hidden_size', 'n_heads', 'type'],
        'training': ['batch_size', 'n_iters', 'lr'],
        'optim': ['lr', 'optimizer'],
        'data': ['max_protein_len', 'vocab_size_protein'],
        'noise': ['sigma_min', 'sigma_max', 'type'],
    }
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required section: {section}")
            return False
    
    # Check required fields within sections
    for section, fields in required_fields.items():
        if section in config:
            for field in fields:
                if field not in config[section]:
                    print(f"❌ Missing required field: {section}.{field}")
                    return False
    
    print("✅ Configuration validation passed")
    return True


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("📋 Configuration Summary:")
    print("=" * 50)
    
    # Model info
    if 'model' in config:
        model = config['model']
        print(f"🏗️  Model: {model.get('name', 'unknown')} ({model.get('type', 'unknown')})")
        print(f"   Hidden size: {model.get('hidden_size', 'N/A')}")
        print(f"   Heads: {model.get('n_heads', 'N/A')}")
        print(f"   Protein blocks: {model.get('n_blocks_prot', 'N/A')}")
        print(f"   Ligand blocks: {model.get('n_blocks_lig', 'N/A')}")
    
    # Training info
    if 'training' in config:
        training = config['training']
        print(f"🚀 Training:")
        print(f"   Batch size: {training.get('batch_size', 'N/A')}")
        print(f"   Iterations: {training.get('n_iters', 'N/A')}")
        print(f"   Accumulation: {training.get('accum', 'N/A')}")
        print(f"   Task: {training.get('task', 'N/A')}")
    
    # Optimizer info
    if 'optim' in config:
        optim = config['optim']
        print(f"⚡ Optimizer:")
        print(f"   Type: {optim.get('optimizer', 'N/A')}")
        print(f"   Learning rate: {optim.get('lr', 'N/A')}")
        print(f"   Weight decay: {optim.get('weight_decay', 'N/A')}")
    
    # Data info
    if 'data' in config:
        data = config['data']
        print(f"📊 Data:")
        print(f"   Max protein length: {data.get('max_protein_len', 'N/A')}")
        print(f"   Max ligand length: {data.get('max_ligand_len', 'N/A')}")
        print(f"   Protein vocab size: {data.get('vocab_size_protein', 'N/A')}")
        print(f"   Ligand vocab size: {data.get('vocab_size_ligand', 'N/A')}")
    
    # Hardware info
    ngpus = config.get('ngpus', 1)
    print(f"🖥️  Hardware: {ngpus} GPU(s)")
    
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    print("🔧 Config Utils Demo")
    print("=" * 50)
    
    # Load and validate the original YAML config
    try:
        yaml_config = load_yaml_config("config_uniref50.yaml")
        print("✅ Loaded YAML config successfully")
        validate_config(yaml_config)
        print_config_summary(yaml_config)
    except FileNotFoundError:
        print("⚠️  Original YAML config not found, using dictionary version")
        dict_config = get_uniref_config()
        validate_config(dict_config)
        print_config_summary(dict_config)
    
    # Create config variants
    print("\n🏭 Creating config variants...")
    create_config_variants()
    
    # Demo namespace conversion
    print("\n🔄 Testing namespace conversion...")
    config = get_minimal_test_config()
    ns = dict_to_namespace(config)
    print(f"Namespace access: ns.model.hidden_size = {ns.model.hidden_size}")
    print(f"Namespace access: ns.training.batch_size = {ns.training.batch_size}")
