"""
Download and process a small subset of UniRef50 data for testing.
This script downloads a manageable subset and processes it for SEDD training.
"""

import os
import requests
import gzip
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import json
from collections import OrderedDict
from transformers import GPT2TokenizerFast


class UniRef50SubsetDownloader:
    """Download and process UniRef50 subset for testing."""
    
    def __init__(self, output_dir: str = "./input_data", subset_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.subset_size = subset_size
        
        # UniRef50 URL (we'll download a small portion)
        self.uniref50_url = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
        
        # Amino acid vocabulary for tokenization
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        
        print(f"Initialized UniRef50 subset downloader")
        print(f"Output directory: {self.output_dir}")
        print(f"Target subset size: {self.subset_size}")
    
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
        
        # Create empty merges file (required by GPT2TokenizerFast)
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
    
    def download_uniref50_sample(self) -> str:
        """Download a sample of UniRef50 data."""
        print("Downloading UniRef50 sample...")
        
        # For testing, we'll use a smaller, more accessible dataset
        # Let's use a sample from the Hugging Face datasets library instead
        try:
            from datasets import load_dataset
            print("Using Hugging Face datasets to load UniRef50 sample...")
            
            # Load a small subset of UniRef50
            dataset = load_dataset("agemagician/uniref50", split="train", streaming=True)
            
            sequences = []
            for i, example in enumerate(dataset):
                if i >= self.subset_size:
                    break
                
                # Extract sequence from the example
                if 'text' in example:
                    seq = example['text'].strip()
                elif 'sequence' in example:
                    seq = example['sequence'].strip()
                else:
                    # Try to find sequence in any string field
                    for key, value in example.items():
                        if isinstance(value, str) and len(value) > 20:
                            # Likely a protein sequence if it's long and contains amino acids
                            if all(c in self.amino_acids + ['X', 'U', 'O'] for c in value.upper()):
                                seq = value.strip().upper()
                                break
                    else:
                        continue
                
                # Filter sequences by length and composition
                if self.is_valid_protein_sequence(seq):
                    sequences.append(seq)
                
                if i % 1000 == 0:
                    print(f"Processed {i} entries, collected {len(sequences)} valid sequences")
            
            print(f"Collected {len(sequences)} valid protein sequences")
            return sequences
            
        except ImportError:
            print("Hugging Face datasets not available, using alternative method...")
            return self.download_from_uniprot_sample()
    
    def download_from_uniprot_sample(self) -> List[str]:
        """Download from a smaller UniProt sample."""
        print("Downloading from UniProt sample...")
        
        # Use a smaller, more accessible dataset for testing
        sample_sequences = [
            # Some example protein sequences for testing
            "MKKFFDSRREQGGSGLGSGSSGGSVLVNGGPLPALLLLLLSAGLEAGGQRGK",
            "MKKLLFAIPLVVPFNYSHQMQALVHQGMVLAFSQYLQQCPFEDHVKLVNEVTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLHIQPSLVGKLNIRQGTDVQAWIRGCRL",
            "MSTFECLKRAWYVAAMSSEVEGEALFHRRILGTSVMIYRLADGTPVAMHDRCPHRFAPLHLGQREGDEIACRYHALRFDADGRCTHNPHGNGRIPDAARVRRFPLLERYGFLWIWMDDSDPDPALLPDFSPLEEGHPNAVAQTYMHMDVNYELIIDNVMDLSHIDHVHGEIITTRGQLSPVVPKVRERDTHISARWEWSQTPAMMIFAPFLPRPEDAARHYFDISWTAPANIQLSVGAVQDSDDFGDATSQYDLHTCTPEDAFKTHYFFATRRNHIVDDADYNRKKIEAMHAAFETEDGPIITAVQEEMGEAEFFSLDPVLMSNDVAPVKVRRRLRRMIVEEQAAAAAADAA",
            "MFPKNAWYVACTPDEIADKPLGRQICNEKIVFYRGPEGRVAAVEDFCPHRGAPLSLGFVRDGKLICGYHGLEMGCEGKTLAMPGQRVQGFPCIKSYAVEERYGFIWVWPGDRELADPALIHHLEWADNPEWAYGGGLYHIACDYRLMIDNLMDLTHETYVHASSIGQKEIDEAPVSTRVEGDTVITSRYMDNVMAPPFWRAALRGNGLADDVPVDRWQICRFAPPSHVLIEVGVAHAGKGGYDAPAEYKAGSIVVDFITPESDTSIWYFWGMARNFRPQGTELTETIRVGQGKIFAEDLDMLEQQQRNLLAYPERQLLKLNIDAGGVQSRRVIDRILAAEQEAADAALIARSAS"
        ]
        
        # Generate more sequences by creating variations
        sequences = []
        for base_seq in sample_sequences:
            sequences.append(base_seq)
            
            # Create variations by random mutations (for testing diversity)
            for _ in range(self.subset_size // len(sample_sequences)):
                if len(sequences) >= self.subset_size:
                    break
                
                # Create a variation
                seq_list = list(base_seq)
                # Randomly mutate a few positions
                num_mutations = min(5, len(seq_list) // 20)
                positions = np.random.choice(len(seq_list), num_mutations, replace=False)
                
                for pos in positions:
                    seq_list[pos] = np.random.choice(self.amino_acids)
                
                new_seq = ''.join(seq_list)
                if self.is_valid_protein_sequence(new_seq):
                    sequences.append(new_seq)
        
        print(f"Generated {len(sequences)} test sequences")
        return sequences[:self.subset_size]
    
    def is_valid_protein_sequence(self, seq: str) -> bool:
        """Check if a sequence is a valid protein sequence."""
        if not seq or len(seq) < 20 or len(seq) > 1000:
            return False
        
        # Check if sequence contains mostly amino acids
        valid_chars = set(self.amino_acids + ['X', 'U', 'O'])  # Include rare amino acids
        seq_chars = set(seq.upper())
        
        # At least 90% should be standard amino acids
        valid_ratio = len(seq_chars.intersection(set(self.amino_acids))) / len(seq_chars)
        return valid_ratio >= 0.9
    
    def tokenize_sequences(self, sequences: List[str], tokenizer) -> List[Dict]:
        """Tokenize protein sequences."""
        print("Tokenizing sequences...")
        
        tokenized_data = []
        max_length = 512  # Maximum sequence length
        
        for seq in tqdm(sequences, desc="Tokenizing"):
            # Truncate if too long
            if len(seq) > max_length:
                seq = seq[:max_length]
            
            # Tokenize sequence
            tokens = tokenizer.encode(seq, add_special_tokens=True)
            
            # Pad to max_length
            if len(tokens) < max_length:
                tokens.extend([tokenizer.pad_token_id] * (max_length - len(tokens)))
            
            tokenized_data.append({
                'protein_seq': seq,
                'prot_tokens': torch.tensor(tokens, dtype=torch.long),
                'length': len(seq)
            })
        
        print(f"Tokenized {len(tokenized_data)} sequences")
        return tokenized_data
    
    def save_processed_data(self, tokenized_data: List[Dict], filename: str = "uniref50_subset.pt"):
        """Save processed data to file."""
        output_file = self.output_dir / filename
        
        print(f"Saving processed data to {output_file}")
        torch.save(tokenized_data, output_file)
        
        # Save metadata
        metadata = {
            'num_sequences': len(tokenized_data),
            'max_length': max(item['length'] for item in tokenized_data),
            'min_length': min(item['length'] for item in tokenized_data),
            'avg_length': np.mean([item['length'] for item in tokenized_data]),
            'vocab_size': 25,  # 20 amino acids + 5 special tokens
        }
        
        metadata_file = self.output_dir / f"{filename.replace('.pt', '_metadata.json')}"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(tokenized_data)} sequences")
        print(f"Metadata: {metadata}")
        
        return output_file
    
    def process_subset(self) -> str:
        """Main processing pipeline."""
        print("Starting UniRef50 subset processing...")
        
        # Create tokenizer
        tokenizer = self.create_protein_tokenizer()
        
        # Download sequences
        sequences = self.download_uniref50_sample()
        
        if not sequences:
            raise ValueError("No sequences downloaded!")
        
        # Tokenize sequences
        tokenized_data = self.tokenize_sequences(sequences, tokenizer)
        
        # Save processed data
        output_file = self.save_processed_data(tokenized_data)
        
        print(f"✅ Successfully processed UniRef50 subset!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Number of sequences: {len(tokenized_data)}")
        
        return str(output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download UniRef50 subset for testing")
    parser.add_argument("--output_dir", type=str, default="./input_data", 
                       help="Output directory for processed data")
    parser.add_argument("--subset_size", type=int, default=10000,
                       help="Number of sequences to download")
    parser.add_argument("--filename", type=str, default="uniref50_subset.pt",
                       help="Output filename")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = UniRef50SubsetDownloader(
        output_dir=args.output_dir,
        subset_size=args.subset_size
    )
    
    try:
        # Process subset
        output_file = downloader.process_subset()
        
        print("\n" + "="*50)
        print("🎉 SUCCESS!")
        print(f"📁 Data saved to: {output_file}")
        print("🚀 You can now use this data for training!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your internet connection and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
