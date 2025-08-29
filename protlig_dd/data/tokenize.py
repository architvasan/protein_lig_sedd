import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass

class Tok_Mol:
    mol_model_id: str = "ibm/MoLFormer-XL-both-10pct"

    def __post_init__(self):
        # Load MolFormer models
        self.mol_tokenizer = AutoTokenizer.from_pretrained(self.mol_model_id, trust_remote_code=True, padding='max_length', max_length=202)
    def tokenize(self, smiles_list):
        #Tokenize ligands & get embeddings
        mol_inputs = self.mol_tokenizer(smiles_list, padding='max_length', max_length=202, truncation=True, return_tensors="pt").to('cuda')
        return mol_inputs

class Tok_Prot :
    prot_model_id: str = "facebook/esm2_t30_150M_UR50D"

    def __post_init__(self):
        # Load ESM2 150M models
        self.protein_tokenizer = AutoTokenizer.from_pretrained(self.prot_model_id, padding='max_length', max_length=1022)#,  # Pad to 512 tokens)AutoTokenizer.from_pretrained(self.mol_model_id, trust_remote_code=True, padding='max_length', max_length=202)
    def tokenize(self, smiles_list):
        protein_inputs = self.protein_tokenizer(protein_seq_list, padding='max_length', max_length=1022, truncation=True, return_tensors="pt").to('cuda')
        return protein_inputs