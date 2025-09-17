"""
Create a test protein dataset for SEDD training.
This script creates a manageable dataset from various sources for testing.
"""

import os
import torch
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm
import json
from collections import OrderedDict
from transformers import GPT2TokenizerFast
import random


class TestProteinDatasetCreator:
    """Create a test protein dataset from multiple sources."""

    def __init__(self, output_dir: str = "./input_data", num_sequences: int = 5000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.num_sequences = num_sequences

        # Amino acid vocabulary
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

        print(f"Creating test protein dataset")
        print(f"Output directory: {self.output_dir}")
        print(f"Target sequences: {self.num_sequences}")

    def create_protein_tokenizer(self):
        """Create a protein-specific tokenizer."""
        print("Creating protein tokenizer...")

        # Create vocabulary
        all_tokens = self.special_tokens + self.amino_acids
        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))

        # Save vocabulary file
        vocab_file = self.output_dir / 'vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f)

        # Create empty merges file
        merges_file = self.output_dir / 'merges.txt'
        with open(merges_file, 'w') as f:
            f.write('#version: 0.2\n')

        # Initialize tokenizer
        tokenizer = GPT2TokenizerFast(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file),
            bos_token='<s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )

        print(f"Tokenizer created with vocabulary size: {len(vocab)}")
        return tokenizer

    def get_sample_sequences(self) -> List[str]:
        """Get sample protein sequences from various sources."""
        print("Generating sample protein sequences...")

        # Base sequences from known proteins
        base_sequences = [
            # Insulin (human)
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",

            # Lysozyme (human)
            "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",

            # Hemoglobin alpha chain
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",

            # Cytochrome c
            "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE",

            # Ubiquitin
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",

            # Green fluorescent protein (partial)
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",

            # Myoglobin (partial)
            "MGLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGDFGADAQGAMTKALELFRNDIAAKYKELGFQG",

            # Albumin (partial)
            "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESHAGCEKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
        ]

        sequences = []

        # Add base sequences
        for seq in base_sequences:
            if self.is_valid_protein_sequence(seq):
                sequences.append(seq)

        print(f"Added {len(sequences)} base sequences")

        # Generate variations of base sequences
        target_per_base = max(1, self.num_sequences // len(base_sequences))

        for base_seq in base_sequences:
            for _ in range(target_per_base):
                if len(sequences) >= self.num_sequences:
                    break

                # Create variations
                variant = self.create_sequence_variant(base_seq)
                if variant and self.is_valid_protein_sequence(variant):
                    sequences.append(variant)

        # Generate completely random sequences if needed
        while len(sequences) < self.num_sequences:
            random_seq = self.generate_random_sequence()
            if self.is_valid_protein_sequence(random_seq):
                sequences.append(random_seq)

        # Shuffle and trim to exact number
        random.shuffle(sequences)
        sequences = sequences[:self.num_sequences]

        print(f"Generated {len(sequences)} total sequences")
        return sequences

    def create_sequence_variant(self, base_seq: str, mutation_rate: float = 0.1) -> str:
        """Create a variant of a base sequence with random mutations."""
        seq_list = list(base_seq)
        seq_length = len(seq_list)

        # Number of mutations
        num_mutations = max(1, int(seq_length * mutation_rate))

        # Random positions to mutate
        positions = random.sample(range(seq_length), min(num_mutations, seq_length))

        for pos in positions:
            # Replace with random amino acid
            seq_list[pos] = random.choice(self.amino_acids)

        return ''.join(seq_list)

    def generate_random_sequence(self, min_length: int = 50, max_length: int = 300) -> str:
        """Generate a completely random protein sequence."""
        length = random.randint(min_length, max_length)
        return ''.join(random.choices(self.amino_acids, k=length))

    def is_valid_protein_sequence(self, seq: str) -> bool:
        """Check if a sequence is valid."""
        if not seq or len(seq) < 20 or len(seq) > 512:
            return False

        # Check if sequence contains only amino acids
        valid_chars = set(self.amino_acids)
        seq_chars = set(seq.upper())

        return seq_chars.issubset(valid_chars)

    def tokenize_sequences(self, sequences: List[str], tokenizer) -> List[Dict]:
        """Tokenize protein sequences."""
        print("Tokenizing sequences...")

        tokenized_data = []
        max_length = 512

        for seq in tqdm(sequences, desc="Tokenizing"):
            # Truncate if too long
            if len(seq) > max_length - 2:  # Account for special tokens
                seq = seq[:max_length - 2]

            # Tokenize
            tokens = tokenizer.encode(seq, add_special_tokens=True)

            # Pad to max_length
            if len(tokens) < max_length:
                tokens.extend([tokenizer.pad_token_id] * (max_length - len(tokens)))

            tokenized_data.append({
                'protein_seq': seq,
                'prot_tokens': torch.tensor(tokens, dtype=torch.long),
                'length': len(seq)
            })

        return tokenized_data

    def save_processed_data(self, tokenized_data: List[Dict], filename: str = "uniref50_subset.pt"):
        """Save processed data."""
        output_file = self.output_dir / filename

        print(f"Saving to {output_file}")
        torch.save(tokenized_data, output_file)

        # Save metadata
        metadata = {
            'num_sequences': len(tokenized_data),
            'max_length': max(item['length'] for item in tokenized_data),
            'min_length': min(item['length'] for item in tokenized_data),
            'avg_length': np.mean([item['length'] for item in tokenized_data]),
            'vocab_size': len(self.amino_acids) + len(self.special_tokens),
        }

        metadata_file = self.output_dir / f"{filename.replace('.pt', '_metadata.json')}"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(tokenized_data)} sequences")
        print(f"Metadata: {metadata}")

        return str(output_file)

    def create_dataset(self) -> str:
        """Main dataset creation pipeline."""
        print("Creating test protein dataset...")

        # Create tokenizer
        tokenizer = self.create_protein_tokenizer()

        # Get sequences
        sequences = self.get_sample_sequences()

        # Tokenize
        tokenized_data = self.tokenize_sequences(sequences, tokenizer)

        # Save
        output_file = self.save_processed_data(tokenized_data)

        print(f"✅ Dataset created successfully!")
        print(f"📁 File: {output_file}")
        print(f"📊 Sequences: {len(tokenized_data)}")

        return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create test protein dataset")
    parser.add_argument("--output_dir", type=str, default="./input_data",
                       help="Output directory")
    parser.add_argument("--num_sequences", type=int, default=5000,
                       help="Number of sequences to generate")
    parser.add_argument("--filename", type=str, default="uniref50_subset.pt",
                       help="Output filename")

    args = parser.parse_args()

    # Create dataset
    creator = TestProteinDatasetCreator(
        output_dir=args.output_dir,
        num_sequences=args.num_sequences
    )

    try:
        output_file = creator.create_dataset()

        print("\n" + "="*50)
        print("🎉 SUCCESS!")
        print(f"📁 Dataset saved to: {output_file}")
        print("🚀 Ready for training!")
        print("="*50)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())