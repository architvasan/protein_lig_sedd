import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
import selfies as sf
from typing import List, Dict, Optional
try:
    from SmilesPE.tokenizer import *
    from protlig_dd.data.smiles_pair_encoders_functions import * #SMILES_SPE_Tokenizer
except ImportError as e:
    print(f"SmilesPE not installed: {e}")

@dataclass
class Tok_Mol:
    mol_model_id: str = "ibm/MoLFormer-XL-both-10pct"

    def __post_init__(self):
        # Load MolFormer models
        self.mol_tokenizer = AutoTokenizer.from_pretrained(self.mol_model_id, trust_remote_code=True, padding='max_length', max_length=128)#202)
    def tokenize(self, smiles_list):
        #Tokenize ligands & get embeddings
        mol_inputs = self.mol_tokenizer(smiles_list, padding='max_length', max_length=128, truncation=True, return_tensors="pt")#.to('cuda')
        return mol_inputs

@dataclass
class Tok_Prot :
    prot_model_id: str = "facebook/esm2_t30_150M_UR50D"

    def __post_init__(self):
        # Load ESM2 150M models
        self.protein_tokenizer = AutoTokenizer.from_pretrained(self.prot_model_id, padding='max_length', max_length=512)#1022)#,  # Pad to 512 tokens)AutoTokenizer.from_pretrained(self.mol_model_id, trust_remote_code=True, padding='max_length', max_length=202)
    def tokenize(self, protein_seq_list):
        protein_inputs = self.protein_tokenizer(protein_seq_list, padding='max_length', max_length=512, truncation=True, return_tensors="pt").to('cuda')
        return protein_inputs
    
@dataclass
class Tok_SmilesPE:
    mol_model_id: str = "ibm/MoLFormer-XL-both-10pct"
    maxlength: int = 128
    vocab_file: str = '../../VocabFiles/vocab_spe.txt'
    spe_file: str = '../../VocabFiles/SPE_ChEMBL.txt'

    def __post_init__(self):
        #vocab_file = '/lus/eagle/projects/FoundEpidem/xlian/smallmolec_campaign/VocabFiles/vocab_spe.txt'
        #spe_file = '/lus/eagle/projects/FoundEpidem/xlian/smallmolec_campaign/VocabFiles/SPE_ChEMBL.txt'
        self.mol_tokenizer = SMILES_SPE_Tokenizer(vocab_file=self.vocab_file, spe_file=self.spe_file)

    def tokenize(self, smiles_list):
        """Tokenize ligands & get embeddings"""
        mol_inputs = self.mol_tokenizer.encode_plus(
            smiles_list,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.maxlength,
            truncation=True,
            return_tensors="pt"
        )
        return mol_inputs

@dataclass
class Tok_Mol_Selfie:
    """
    SELFIES tokenizer for molecular strings.

    SELFIES (Self-Referencing Embedded Strings) is a 100% robust molecular
    string representation that guarantees valid molecules.

    Args:
        max_length: Maximum sequence length for padding/truncation
        padding: Padding strategy ('max_length', 'longest', 'do_not_pad')
        truncation: Whether to truncate sequences longer than max_length
        add_special_tokens: Whether to add [CLS] and [SEP] tokens
        semantic_constraints: SELFIES semantic constraints ('default', 'hypervalent', 'minimal')
    """
    max_length: int = 128
    padding: str = 'max_length'
    truncation: bool = True
    add_special_tokens: bool = True
    semantic_constraints: str = 'default'

    def __post_init__(self):
        """Initialize the SELFIES tokenizer and build vocabulary."""
        # Set semantic constraints for SELFIES
        if self.semantic_constraints != 'default':
            sf.set_semantic_constraints(self.semantic_constraints)

        # Build vocabulary from SELFIES alphabet
        # Get the semantic robust alphabet (all valid SELFIES symbols)
        self.alphabet = list(sorted(sf.get_semantic_robust_alphabet()))

        # Add special tokens
        self.special_tokens = {
            'pad_token': '[nop]',  # SELFIES no-operation token for padding
            'cls_token': '[CLS]',  # Start of sequence
            'sep_token': '[SEP]',  # End of sequence
            'unk_token': '[UNK]',  # Unknown token
            'mask_token': '[MASK]'  # Mask token for MLM tasks
        }

        # Add special tokens to alphabet if not present
        for token in self.special_tokens.values():
            if token not in self.alphabet:
                self.alphabet.append(token)

        # Create vocabulary mappings
        self.vocab_stoi = {token: idx for idx, token in enumerate(self.alphabet)}
        self.vocab_itos = {idx: token for token, idx in self.vocab_stoi.items()}
        self.vocab_size = len(self.alphabet)

        # Store special token IDs
        self.pad_token_id = self.vocab_stoi[self.special_tokens['pad_token']]
        self.cls_token_id = self.vocab_stoi[self.special_tokens['cls_token']]
        self.sep_token_id = self.vocab_stoi[self.special_tokens['sep_token']]
        self.unk_token_id = self.vocab_stoi[self.special_tokens['unk_token']]
        self.mask_token_id = self.vocab_stoi[self.special_tokens['mask_token']]

        print(f"✅ SELFIES tokenizer initialized")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Max length: {self.max_length}")
        print(f"   Semantic constraints: {self.semantic_constraints}")

    def smiles_to_selfies(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES string to SELFIES string.

        Args:
            smiles: SMILES string

        Returns:
            SELFIES string or None if encoding fails
        """
        try:
            return sf.encoder(smiles)
        except sf.EncoderError as e:
            print(f"⚠️  SELFIES encoding error for '{smiles}': {e}")
            return None

    def selfies_to_smiles(self, selfies: str) -> Optional[str]:
        """
        Convert SELFIES string to SMILES string.

        Args:
            selfies: SELFIES string

        Returns:
            SMILES string or None if decoding fails
        """
        try:
            return sf.decoder(selfies)
        except sf.DecoderError as e:
            print(f"⚠️  SELFIES decoding error for '{selfies}': {e}")
            return None

    def tokenize(self, smiles_list: List[str], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of SMILES strings into SELFIES tokens.

        Args:
            smiles_list: List of SMILES strings
            return_tensors: Format of returned tensors ('pt' for PyTorch, 'np' for NumPy)

        Returns:
            Dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask (1 for real tokens, 0 for padding)
                - selfies_strings: List of SELFIES strings (for reference)
        """
        batch_input_ids = []
        batch_attention_mask = []
        selfies_strings = []

        for smiles in smiles_list:
            # Convert SMILES to SELFIES
            selfies = self.smiles_to_selfies(smiles)
            if selfies is None:
                # If encoding fails, use empty sequence
                selfies = ""

            selfies_strings.append(selfies)

            # Split SELFIES into tokens
            tokens = list(sf.split_selfies(selfies)) if selfies else []

            # Add special tokens if requested
            if self.add_special_tokens:
                tokens = [self.special_tokens['cls_token']] + tokens + [self.special_tokens['sep_token']]

            # Convert tokens to IDs
            token_ids = [
                self.vocab_stoi.get(token, self.unk_token_id)
                for token in tokens
            ]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(token_ids)

            # Apply padding or truncation
            if self.padding == 'max_length':
                if len(token_ids) < self.max_length:
                    # Pad
                    padding_length = self.max_length - len(token_ids)
                    token_ids.extend([self.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)
                elif len(token_ids) > self.max_length and self.truncation:
                    # Truncate
                    token_ids = token_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]

            batch_input_ids.append(token_ids)
            batch_attention_mask.append(attention_mask)

        # Convert to tensors
        if return_tensors == "pt":
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        elif return_tensors == "np":
            input_ids = np.array(batch_input_ids, dtype=np.int64)
            attention_mask = np.array(batch_attention_mask, dtype=np.int64)
        else:
            input_ids = batch_input_ids
            attention_mask = batch_attention_mask

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'selfies_strings': selfies_strings
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to SMILES string.

        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens in decoding

        Returns:
            SMILES string
        """
        # Convert tensor to list if needed
        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().tolist()

        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            token = self.vocab_itos.get(token_id, self.special_tokens['unk_token'])

            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens.values():
                continue

            tokens.append(token)

        # Join tokens to form SELFIES string
        selfies = ''.join(tokens)

        # Convert SELFIES to SMILES
        smiles = self.selfies_to_smiles(selfies)
        return smiles if smiles else ""

    def batch_decode(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token IDs back to SMILES strings.

        Args:
            batch_token_ids: List of token ID lists or tensor
            skip_special_tokens: Whether to skip special tokens in decoding

        Returns:
            List of SMILES strings
        """
        if torch.is_tensor(batch_token_ids):
            batch_token_ids = batch_token_ids.cpu().tolist()

        return [self.decode(token_ids, skip_special_tokens) for token_ids in batch_token_ids]

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.vocab_stoi.copy()

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
