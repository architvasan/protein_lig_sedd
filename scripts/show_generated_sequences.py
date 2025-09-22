#!/usr/bin/env python3
"""
Display examples of generated protein and protein-ligand sequences
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys

def load_vocabulary():
    """Load the vocabulary mapping for decoding sequences."""
    try:
        # Load vocabulary from the processed data
        vocab_file = "input_data/vocab.json"
        if Path(vocab_file).exists():
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            print(f"✅ Loaded vocabulary with {len(vocab)} tokens")
            return vocab
        else:
            print("⚠️  Vocabulary file not found, using default amino acid mapping")
            # Standard amino acid vocabulary (common in protein models)
            amino_acids = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'Z', 'B', 'ABSORB']
            vocab = {str(i): aa for i, aa in enumerate(amino_acids)}
            return vocab
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return None

def load_generated_samples():
    """Load the generated samples from the evaluation."""
    try:
        from protlig_dd.training.run_train_uniref50_optimized import OptimizedUniRef50Trainer
        from protlig_dd.utils import utils
        import yaml
        
        # Load configuration
        config_file = "configs/config_uniref50_stable.yaml"
        with open(config_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = utils.Config(dictionary=cfg_dict)
        
        # Create trainer
        trainer = OptimizedUniRef50Trainer(
            work_dir=".",
            config_file=config_file,
            datafile="./input_data/processed_uniref50.pt",
            dev_id="cpu",
            seed=42
        )
        
        # Setup components
        trainer.setup_custom_data_loaders()
        trainer.setup_model()
        
        # Load checkpoint
        checkpoint_path = "checkpoints/best_checkpoint.pth"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.model.eval()
        
        # Generate a few samples for display
        print("🎲 Generating sample sequences for display...")
        generated_samples = []
        
        with torch.no_grad():
            for i in range(10):  # Generate 10 samples for display
                # Initialize with absorbing states
                vocab_size = 37  # From config
                absorbing_token = vocab_size - 1
                max_length = 128  # Shorter for display
                
                sample = torch.full((1, max_length), absorbing_token, dtype=torch.long)
                
                # Simple diffusion sampling
                num_steps = 25
                for step in range(num_steps):
                    t = torch.tensor([1.0 - step / num_steps])
                    
                    with torch.autocast(device_type='cpu', enabled=False):
                        logits = trainer.model(sample, t)
                    
                    # Temperature sampling
                    temperature = 1.0
                    probs = torch.softmax(logits / temperature, dim=-1)
                    new_sample = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, max_length)
                    
                    # Update sample
                    mask = torch.rand(1, max_length) < (step + 1) / num_steps
                    sample = torch.where(mask, new_sample, sample)
                
                generated_samples.append(sample.squeeze(0))
        
        return generated_samples, trainer
        
    except Exception as e:
        print(f"❌ Error generating samples: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def decode_sequence(sequence, vocab, max_display_length=100):
    """Decode a sequence of token IDs to readable format."""
    try:
        # Convert tensor to list if needed
        if torch.is_tensor(sequence):
            sequence = sequence.tolist()
        
        # Decode tokens
        decoded = []
        for token_id in sequence:
            if str(token_id) in vocab:
                token = vocab[str(token_id)]
                if token not in ['PAD', 'ABSORB']:  # Skip padding and absorbing tokens
                    decoded.append(token)
            else:
                decoded.append(f'UNK_{token_id}')
        
        # Join and truncate for display
        decoded_str = ''.join(decoded)
        if len(decoded_str) > max_display_length:
            decoded_str = decoded_str[:max_display_length] + '...'
        
        return decoded_str, len([t for t in decoded if t not in ['PAD', 'ABSORB']])
        
    except Exception as e:
        print(f"❌ Error decoding sequence: {e}")
        return "ERROR", 0

def analyze_sequence_properties(sequence_str):
    """Analyze basic properties of a protein sequence."""
    if not sequence_str or sequence_str == "ERROR":
        return {}
    
    # Count amino acid types
    aa_counts = {}
    for aa in sequence_str:
        if aa.isalpha():
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    total_aa = sum(aa_counts.values())
    if total_aa == 0:
        return {}
    
    # Basic properties
    properties = {
        'length': total_aa,
        'unique_amino_acids': len(aa_counts),
        'most_common': max(aa_counts.items(), key=lambda x: x[1]) if aa_counts else ('N/A', 0),
        'composition': {aa: count/total_aa*100 for aa, count in aa_counts.items()}
    }
    
    return properties

def display_training_examples(trainer):
    """Show some examples from the training data for comparison."""
    print("\n📚 TRAINING DATA EXAMPLES")
    print("=" * 60)
    
    try:
        # Get a batch from training data
        for i, batch in enumerate(trainer.train_loader):
            if i >= 1:  # Just get first batch
                break
        
        vocab = load_vocabulary()
        if vocab is None:
            return
        
        # Show first 3 training examples
        for i in range(min(3, batch.shape[0])):
            sequence = batch[i]
            decoded_str, length = decode_sequence(sequence, vocab)
            properties = analyze_sequence_properties(decoded_str)
            
            print(f"\n🧬 Training Example {i+1}:")
            print(f"   Raw tokens: {sequence[:20].tolist()}... (showing first 20)")
            print(f"   Decoded: {decoded_str}")
            print(f"   Length: {length} amino acids")
            if properties:
                print(f"   Unique AAs: {properties['unique_amino_acids']}")
                print(f"   Most common: {properties['most_common'][0]} ({properties['most_common'][1]} times)")
    
    except Exception as e:
        print(f"❌ Error displaying training examples: {e}")

def main():
    """Main function to display generated sequences."""
    print("🧬 GENERATED PROTEIN SEQUENCE EXAMPLES")
    print("=" * 80)
    
    # Load vocabulary
    vocab = load_vocabulary()
    if vocab is None:
        print("❌ Cannot proceed without vocabulary")
        return
    
    # Load generated samples
    generated_samples, trainer = load_generated_samples()
    if not generated_samples:
        print("❌ No generated samples available")
        return
    
    print(f"✅ Generated {len(generated_samples)} sample sequences")
    
    # Display generated examples
    print(f"\n🎲 GENERATED SEQUENCE EXAMPLES")
    print("=" * 60)
    
    for i, sample in enumerate(generated_samples):
        decoded_str, length = decode_sequence(sample, vocab)
        properties = analyze_sequence_properties(decoded_str)
        
        print(f"\n🧬 Generated Example {i+1}:")
        print(f"   Raw tokens: {sample[:20].tolist()}... (showing first 20)")
        print(f"   Decoded: {decoded_str}")
        print(f"   Length: {length} amino acids")
        
        if properties:
            print(f"   Unique AAs: {properties['unique_amino_acids']}")
            if properties['most_common'][0] != 'N/A':
                print(f"   Most common: {properties['most_common'][0]} ({properties['most_common'][1]} times)")
            
            # Show top 5 amino acids
            top_aa = sorted(properties['composition'].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top amino acids: {', '.join([f'{aa}({pct:.1f}%)' for aa, pct in top_aa])}")
    
    # Show training examples for comparison
    if trainer:
        display_training_examples(trainer)
    
    # Summary analysis
    print(f"\n📊 SEQUENCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    all_lengths = []
    all_unique_counts = []
    all_compositions = {}
    
    for sample in generated_samples:
        decoded_str, length = decode_sequence(sample, vocab)
        properties = analyze_sequence_properties(decoded_str)
        
        if properties:
            all_lengths.append(length)
            all_unique_counts.append(properties['unique_amino_acids'])
            
            for aa, pct in properties['composition'].items():
                if aa not in all_compositions:
                    all_compositions[aa] = []
                all_compositions[aa].append(pct)
    
    if all_lengths:
        print(f"📏 Length Statistics:")
        print(f"   Mean: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
        print(f"   Range: {min(all_lengths)} - {max(all_lengths)}")
        
        print(f"\n🔤 Amino Acid Diversity:")
        print(f"   Mean unique AAs: {np.mean(all_unique_counts):.1f} ± {np.std(all_unique_counts):.1f}")
        print(f"   Range: {min(all_unique_counts)} - {max(all_unique_counts)}")
        
        print(f"\n🧪 Most Common Amino Acids Across All Samples:")
        avg_compositions = {aa: np.mean(pcts) for aa, pcts in all_compositions.items()}
        top_overall = sorted(avg_compositions.items(), key=lambda x: x[1], reverse=True)[:10]
        for aa, avg_pct in top_overall:
            print(f"   {aa}: {avg_pct:.1f}% (±{np.std(all_compositions[aa]):.1f}%)")
    
    print(f"\n✅ Sequence analysis complete!")
    print(f"📁 Generated sequences show the current model's capabilities")
    print(f"🎯 Compare with training examples to assess quality")

if __name__ == "__main__":
    main()
