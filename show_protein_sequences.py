#!/usr/bin/env python3
"""
Display properly decoded protein sequences from the generated samples
"""

import torch
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_vocabulary():
    """Load the actual vocabulary mapping."""
    vocab_file = "input_data/vocab.json"
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    # Create reverse mapping (token_id -> amino_acid)
    id_to_token = {v: k for k, v in vocab.items()}
    print(f"✅ Loaded vocabulary: {vocab}")
    return id_to_token

def load_and_decode_samples():
    """Load generated samples and decode them properly."""
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
        
        trainer.setup_custom_data_loaders()
        trainer.setup_model()
        
        # Load checkpoint
        checkpoint_path = "checkpoints/best_checkpoint.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        # Load vocabulary
        vocab = load_vocabulary()
        
        print("🎲 Generating protein sequences...")
        generated_samples = []
        
        with torch.no_grad():
            for i in range(15):  # Generate 15 samples
                # Initialize sequence
                vocab_size = len(vocab)  # Should be 25
                max_length = 100  # Reasonable length for display
                
                # Start with mask tokens (token 4)
                sample = torch.full((1, max_length), 4, dtype=torch.long)  # <mask> token
                
                # Simple diffusion sampling
                num_steps = 20
                for step in range(num_steps):
                    t = torch.tensor([1.0 - step / num_steps])
                    
                    with torch.autocast(device_type='cpu', enabled=False):
                        logits = trainer.model(sample, t)
                    
                    # Temperature sampling
                    temperature = 0.8  # Lower temperature for more focused sampling
                    probs = torch.softmax(logits / temperature, dim=-1)
                    new_sample = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, max_length)
                    
                    # Gradually replace mask tokens
                    replace_prob = (step + 1) / num_steps
                    mask = torch.rand(1, max_length) < replace_prob
                    sample = torch.where(mask, new_sample, sample)
                
                generated_samples.append(sample.squeeze(0))
        
        return generated_samples, trainer, vocab
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None

def decode_sequence(sequence, vocab):
    """Decode a sequence properly."""
    if torch.is_tensor(sequence):
        sequence = sequence.tolist()
    
    decoded = []
    for token_id in sequence:
        if token_id in vocab:
            token = vocab[token_id]
            # Only include actual amino acids (skip special tokens)
            if token in 'ACDEFGHIKLMNPQRSTVWY':
                decoded.append(token)
        
    return ''.join(decoded)

def analyze_protein_sequence(sequence_str):
    """Analyze properties of a protein sequence."""
    if not sequence_str:
        return {}
    
    # Amino acid properties
    hydrophobic = set('AILMFPWV')
    polar = set('NQST')
    charged_positive = set('RHK')
    charged_negative = set('DE')
    aromatic = set('FWY')
    
    total = len(sequence_str)
    if total == 0:
        return {}
    
    properties = {
        'length': total,
        'hydrophobic_pct': sum(1 for aa in sequence_str if aa in hydrophobic) / total * 100,
        'polar_pct': sum(1 for aa in sequence_str if aa in polar) / total * 100,
        'positive_pct': sum(1 for aa in sequence_str if aa in charged_positive) / total * 100,
        'negative_pct': sum(1 for aa in sequence_str if aa in charged_negative) / total * 100,
        'aromatic_pct': sum(1 for aa in sequence_str if aa in aromatic) / total * 100,
        'composition': {aa: sequence_str.count(aa) / total * 100 for aa in set(sequence_str)}
    }
    
    return properties

def display_training_examples(trainer, vocab):
    """Show training examples for comparison."""
    print("\n📚 TRAINING DATA EXAMPLES")
    print("=" * 80)
    
    # Get training batch
    for i, batch in enumerate(trainer.train_loader):
        if i >= 1:
            break
    
    for i in range(min(5, batch.shape[0])):
        sequence = batch[i]
        decoded = decode_sequence(sequence, vocab)
        properties = analyze_protein_sequence(decoded)
        
        print(f"\n🧬 Training Example {i+1}:")
        print(f"   Sequence: {decoded[:80]}{'...' if len(decoded) > 80 else ''}")
        print(f"   Length: {len(decoded)} amino acids")
        if properties:
            print(f"   Hydrophobic: {properties['hydrophobic_pct']:.1f}%")
            print(f"   Polar: {properties['polar_pct']:.1f}%")
            print(f"   Charged+: {properties['positive_pct']:.1f}%")
            print(f"   Charged-: {properties['negative_pct']:.1f}%")

def main():
    """Main function."""
    print("🧬 PROTEIN SEQUENCE ANALYSIS")
    print("=" * 80)
    
    # Generate and decode samples
    generated_samples, trainer, vocab = load_and_decode_samples()
    if not generated_samples:
        return
    
    print(f"\n🎲 GENERATED PROTEIN SEQUENCES")
    print("=" * 80)
    
    all_properties = []
    
    for i, sample in enumerate(generated_samples):
        decoded = decode_sequence(sample, vocab)
        properties = analyze_protein_sequence(decoded)
        
        if len(decoded) > 0:  # Only show non-empty sequences
            print(f"\n🧬 Generated Protein {i+1}:")
            print(f"   Sequence: {decoded}")
            print(f"   Length: {len(decoded)} amino acids")
            
            if properties:
                all_properties.append(properties)
                print(f"   Composition: {len(set(decoded))} unique amino acids")
                print(f"   Hydrophobic: {properties['hydrophobic_pct']:.1f}%")
                print(f"   Polar: {properties['polar_pct']:.1f}%")
                print(f"   Charged+: {properties['positive_pct']:.1f}%")
                print(f"   Charged-: {properties['negative_pct']:.1f}%")
                print(f"   Aromatic: {properties['aromatic_pct']:.1f}%")
                
                # Show most common amino acids
                top_aa = sorted(properties['composition'].items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   Top AAs: {', '.join([f'{aa}({pct:.1f}%)' for aa, pct in top_aa])}")
    
    # Show training examples
    if trainer:
        display_training_examples(trainer, vocab)
    
    # Summary statistics
    if all_properties:
        print(f"\n📊 GENERATED SEQUENCES SUMMARY")
        print("=" * 80)
        
        lengths = [p['length'] for p in all_properties]
        hydrophobic = [p['hydrophobic_pct'] for p in all_properties]
        polar = [p['polar_pct'] for p in all_properties]
        
        print(f"📏 Length Statistics:")
        print(f"   Mean: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"   Range: {min(lengths)} - {max(lengths)}")
        
        print(f"\n🧪 Chemical Properties (Average):")
        print(f"   Hydrophobic: {np.mean(hydrophobic):.1f}% ± {np.std(hydrophobic):.1f}%")
        print(f"   Polar: {np.mean(polar):.1f}% ± {np.std(polar):.1f}%")
        print(f"   Charged+: {np.mean([p['positive_pct'] for p in all_properties]):.1f}%")
        print(f"   Charged-: {np.mean([p['negative_pct'] for p in all_properties]):.1f}%")
        print(f"   Aromatic: {np.mean([p['aromatic_pct'] for p in all_properties]):.1f}%")
        
        # Overall amino acid frequency
        all_compositions = {}
        for props in all_properties:
            for aa, pct in props['composition'].items():
                if aa not in all_compositions:
                    all_compositions[aa] = []
                all_compositions[aa].append(pct)
        
        print(f"\n🔤 Most Frequent Amino Acids (Across All Generated Sequences):")
        avg_compositions = {aa: np.mean(pcts) for aa, pcts in all_compositions.items()}
        top_overall = sorted(avg_compositions.items(), key=lambda x: x[1], reverse=True)
        for aa, avg_pct in top_overall:
            print(f"   {aa}: {avg_pct:.1f}% ± {np.std(all_compositions[aa]):.1f}%")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
