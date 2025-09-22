#!/usr/bin/env python3
"""
Setup script for Wandb integration with SEDD training.
This script helps configure Wandb for experiment tracking.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_wandb_installation():
    """Check if wandb is installed."""
    try:
        import wandb
        print(f"✅ Wandb is installed (version: {wandb.__version__})")
        return True
    except ImportError:
        print("❌ Wandb is not installed")
        return False


def install_wandb():
    """Install wandb."""
    print("📦 Installing wandb...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ Wandb installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install wandb")
        return False


def setup_wandb_login():
    """Setup wandb login."""
    try:
        import wandb
        
        # Check if already logged in
        try:
            wandb.api.api_key
            print("✅ Wandb is already logged in")
            return True
        except:
            pass
        
        print("\n🔐 Setting up Wandb login...")
        print("You have several options:")
        print("1. Interactive login (recommended)")
        print("2. Set API key manually")
        print("3. Use offline mode")
        
        choice = input("\nChoose option (1/2/3): ").strip()
        
        if choice == "1":
            print("\n🌐 Opening browser for Wandb login...")
            print("If browser doesn't open, go to: https://wandb.ai/authorize")
            wandb.login()
            
        elif choice == "2":
            api_key = input("\n🔑 Enter your Wandb API key: ").strip()
            if api_key:
                wandb.login(key=api_key)
                print("✅ Wandb login successful")
            else:
                print("❌ No API key provided")
                return False
                
        elif choice == "3":
            print("📴 Setting up offline mode...")
            os.environ['WANDB_MODE'] = 'offline'
            print("✅ Wandb will run in offline mode")
            
        else:
            print("❌ Invalid choice")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error setting up wandb: {e}")
        return False


def create_wandb_config():
    """Create a default wandb configuration."""
    config_content = """# Wandb Configuration for SEDD Training

# Project settings
project: uniref50-sedd
entity: your-wandb-username  # Replace with your wandb username

# Default tags
tags:
  - uniref50
  - sedd
  - protein
  - diffusion

# Notes template
notes: |
  Optimized UniRef50 training with improved attention mechanism
  and curriculum learning strategies.

# Settings
save_code: true
"""
    
    config_file = Path("wandb_config.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created wandb config: {config_file}")
    print("📝 Please edit wandb_config.yaml to set your username")


def test_wandb_setup():
    """Test wandb setup."""
    try:
        import wandb
        
        print("\n🧪 Testing wandb setup...")
        
        # Initialize a test run
        run = wandb.init(
            project="sedd-test",
            name="setup-test",
            mode="offline"  # Use offline mode for testing
        )
        
        # Log some test data
        wandb.log({"test_metric": 1.0})
        
        # Finish the run
        wandb.finish()
        
        print("✅ Wandb test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Wandb test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Wandb for SEDD training")
    print("=" * 50)
    
    # Check installation
    if not check_wandb_installation():
        if not install_wandb():
            print("❌ Setup failed: Could not install wandb")
            return 1
    
    # Setup login
    if not setup_wandb_login():
        print("❌ Setup failed: Could not setup wandb login")
        return 1
    
    # Create config
    create_wandb_config()
    
    # Test setup
    if not test_wandb_setup():
        print("⚠️  Warning: Wandb test failed, but basic setup is complete")
    
    print("\n" + "=" * 50)
    print("🎉 Wandb setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit wandb_config.yaml with your settings")
    print("2. Run your training script")
    print("3. Monitor experiments at https://wandb.ai")
    print("\n🚀 Ready to track your SEDD experiments!")
    
    return 0


if __name__ == "__main__":
    exit(main())
