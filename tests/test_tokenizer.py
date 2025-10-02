import numpy 
import torch
import protlig_dd.data.tokenize as tokenize

example_smiles = [
  {"lig_smiles": "CCO"},
  {"lig_smiles": "CC(=O)O"},
  {"lig_smiles": "C1=CC=CC=C1"},
  {"lig_smiles": "CCN(CC)CC"},
  {"lig_smiles": "CC(C)O"},
  {"lig_smiles": "C1CCCCC1"},
  {"lig_smiles": "CCOC(=O)C"},
  {"lig_smiles": "C1=CC=C(C=C1)O"},
  {"lig_smiles": "CC(C)C(=O)O"},
  {"lig_smiles": "CC(C)CC(=O)O"},
  {"lig_smiles": "C1=CC=C2C(=C1)C=CC=C2"},
  {"lig_smiles": "CC(C)N"},
  {"lig_smiles": "CC(C)O"},
  {"lig_smiles": "C1=CC(=CC=C1Cl)Cl"},
  {"lig_smiles": "CCOC(=O)C1=CC=CC=C1"},
  {"lig_smiles": "C1CCC(CC1)O"},
  {"lig_smiles": "CCN(CC)C(=O)C"},
  {"lig_smiles": "C1=CC=C(C=C1)N"},
  {"lig_smiles": "CC(=O)NC1=CC=CC=C1"},
  {"lig_smiles": "CC(C)C1=CC=CC=C1"},
  {"lig_smiles": "C1=CC=C(C=C1)C(=O)O"},
  {"lig_smiles": "CCOC(=O)N"},
  {"lig_smiles": "C1=CC=C(C=C1)F"},
  {"lig_smiles": "CC(C)C(=O)N"},
  {"lig_smiles": "C1=CC(=CC=C1O)O"},
  {"lig_smiles": "CC(C)CC"},
  {"lig_smiles": "C1=CC=CN=C1"},
  {"lig_smiles": "C1=CC=C(C=C1)Br"},
  {"lig_smiles": "CCN(CC)CCO"},
  {"lig_smiles": "C1=CC=C(C=C1)C#N"},
  {"lig_smiles": "CC(C)C(O)C(=O)O"},
  {"lig_smiles": "C1=CC(=O)C=CC1=O"}
]

tokenize_smiles = tokenize.Tok_SmilesPE(vocab_file = "/lus/flare/projects/FoundEpidem/xlian/protein_lig_sedd/VocabFiles/vocab_spe.txt",
                                        spe_file = "/lus/flare/projects/FoundEpidem/xlian/protein_lig_sedd/VocabFiles/SPE_ChEMBL.txt")


for row in example_smiles:
    tokenized_row = tokenize_smiles.tokenize(row['lig_smiles'])
    print(tokenized_row)
