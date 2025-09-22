#!/usr/bin/env python3
"""
Create a 10k sample subset dataset specifically for hyperparameter sweeps.
This keeps the main training dataset unchanged while providing a smaller dataset for faster experimentation.
"""

import os
import sys
import torch
import random
import argparse
from pathlib import Path


def create_sweep_dataset(input_file: str, output_file: str, max_samples: int = 10000, seed: int = 42):
    """Create a subset dataset for hyperparameter sweeps."""
    
    print(f"🔄 Creating sweep dataset from {input_file}")
    print(f"📊 Target samples: {max_samples}")
    print(f"🎲 Random seed: {seed}")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return False
    
    # Load original dataset
    print("📥 Loading original dataset...")
    try:
        data = torch.load(input_file, weights_only=False)
        print(f"✅ Loaded {len(data)} sequences")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Check if we need to subsample
    if len(data) <= max_samples:
        print(f"ℹ️  Dataset already has {len(data)} samples (≤ {max_samples})")
        print("📋 Copying original dataset...")
        subset_data = data
    else:
        print(f"✂️  Subsampling from {len(data)} to {max_samples} samples...")
        
        # Set random seed for reproducible sampling
        random.seed(seed)
        
        # Create random subset
        subset_data = random.sample(data, max_samples)
        print(f"✅ Created subset with {len(subset_data)} samples")
    
    # Save subset dataset
    print(f"💾 Saving to {output_file}...")
    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the subset
        torch.save(subset_data, output_file)
        print(f"✅ Saved {len(subset_data)} samples to {output_file}")
        
        # Print statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Original size: {len(data) if 'data' in locals() else 'N/A'}")
        print(f"   Subset size: {len(subset_data)}")
        print(f"   Reduction: {(1 - len(subset_data)/len(data))*100:.1f}%" if len(data) > len(subset_data) else "   No reduction needed")
        
        # Analyze sequence lengths if possible
        try:
            if subset_data and isinstance(subset_data[0], dict) and 'prot_tokens' in subset_data[0]:
                lengths = [len(sample['prot_tokens']) for sample in subset_data[:100]]  # Sample first 100
                print(f"   Avg length (sample): {sum(lengths)/len(lengths):.1f}")
                print(f"   Min length (sample): {min(lengths)}")
                print(f"   Max length (sample): {max(lengths)}")
        except:
            pass  # Skip if we can't analyze
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create subset dataset for hyperparameter sweeps")
    parser.add_argument("--input", type=str, default="./input_data/processed_uniref50.pt",
                       help="Input dataset file")
    parser.add_argument("--output", type=str, default="./input_data/processed_uniref50_sweep.pt",
                       help="Output subset dataset file")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum number of samples in subset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling")
    
    args = parser.parse_args()
    
    print("🧬 UniRef50 Sweep Dataset Creator")
    print("=" * 50)
    
    # Create subset dataset
    success = create_sweep_dataset(
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    if success:
        print("\n🎉 SUCCESS!")
        print(f"📁 Sweep dataset created: {args.output}")
        print("\n🚀 Usage in hyperparameter sweep:")
        print(f"   Update datafile path to: {args.output}")
        print("   Or use --datafile parameter in sweep script")
        print("\n💡 Benefits:")
        print("   - Faster training iterations")
        print("   - Quicker hyperparameter exploration")
        print("   - More experiments per GPU-hour")
        print("=" * 50)
        return 0
    else:
        print("\n❌ FAILED!")
        print("Check the error messages above for details.")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    exit(main())
