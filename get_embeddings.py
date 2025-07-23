import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

plinder_path = '/homes/nharwell/GenMolDPLM/src/data/processed_plinder_data/processed_plinder_data.pt'
data = torch.load(plinder_path, weights_only = False)

# Load MolFormer & ESM2 150M models
mol_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
protein_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

#Example: process the first N samples
N = 8
smiles_list = [sample['ligand_smiles'] for sample in data[:N]]
protein_seq_list = [sample['protein_seq'] for sample in data[:N]]

#Tokenize ligands & get embeddings
mol_inputs = mol_tokenizer(smiles_list, padding=True, return_tensors="pt")
with torch.no_grad():
    mol_outputs = mol_model(**mol_inputs)
mol_embeddings = mol_outputs.last_hidden_state #(batch, seq_len, hidden)

#Tokenize proteins & get embeddings
protein_inputs = protein_tokenizer(protein_seq_list, padding=True, return_tensors="pt")
with torch.no_grad():
    protein_outputs = protein_model(**protein_inputs)
protein_embeddings = protein_outputs.last_hidden_state # (batch, seq_len, hidden)

print("Mol embeddings shape:", mol_embeddings.shape)
print("Protein embeddings shape:", protein_embeddings.shape)
print(mol_embeddings)
print(protein_embeddings)


''' 
Molformer 50M Embedding Shape:
- 768-dimensional embedding
- molformer.embeddings.word_embeddings.weight [2362, 768]
- so vocab size is 2362

ESM2 150M Embedding Shape:
- 640-dimensional embedding
- esm.embeddings.position_embeddings.weight	[1028, 640]
- esm.embeddings.word_embeddings.weight	[33, 640]
- esm.embeddings.position_ids	[1, 1026]
- So sequence length up to 1026 & vocab size is 33 tokens
'''

# MolFormer Model: https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct 
# ESM2 Model: https://huggingface.co/facebook/esm2_t30_150M_UR50D 
# ESM2 Embedding Extraction example: https://www.kaggle.com/code/viktorfairuschin/extracting-esm-2-embeddings-from-fasta-files/notebook


