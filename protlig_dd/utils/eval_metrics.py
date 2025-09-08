from dataclasses import dataclass
import yaml
import inspect
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Contrib.SA_Score import sascorer
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from dataclasses import dataclass
# Evaluation imports for protein designability and ligand drug-likeness

@dataclass
class ProteinEvaluator:
    """Evaluate protein designability using various metrics"""
    prot_model_id: str = "facebook/esm2_t30_150M_UR50D"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    protein_sequences: list = None
    def __post_init__(self):
            self.protein_model = AutoModel.from_pretrained(self.prot_model_id).to('cuda')
            self.protein_tokenizer = AutoTokenizer.from_pretrained(self.prot_model_id, padding='max_length', max_length=1022)#,  # Pad to 512 tokens)
    
    def calculate_designability_score(self):
        """Calculate protein designability using structural and sequence-based metrics"""
        scores = {}
        
        # Basic sequence metrics
        scores['length'] = [len(seq) for seq in self.protein_sequences]
        scores['hydrophobicity'] = [self._calculate_hydrophobicity(seq) for seq in self.protein_sequences]
        scores['charge'] = [self._calculate_charge(seq) for seq in self.protein_sequences]
        scores['instability_index'] = [self._calculate_instability(seq) for seq in self.protein_sequences]
        
        # Secondary structure prediction confidence (if TAPE available)
        scores['esm_confidence'] = self._calculate_bert_confidence(self.protein_sequences)
        
        return scores
    def calculate_similarity(self, reference_sequences):
        """Calculate sequence similarity to reference set using simple identity metric"""
        from Bio import pairwise2
        from Bio.Seq import Seq
        from Bio.Align import substitution_matrices
        
        matrix = substitution_matrices.load("BLOSUM62")
        scores = []
        
        for seq in self.protein_sequences:
            max_score = 0
            for ref_seq in reference_sequences:
                alignments = pairwise2.align.globaldx(seq, ref_seq, matrix)
                if alignments:
                    score = alignments[0][2]  # Get the score of the best alignment
                    if score > max_score:
                        max_score = score
            scores.append(max_score / max(len(seq), len(ref_seq)))  # Normalize by length
        
        return scores
    def _calculate_hydrophobicity(self, sequence):
        """Calculate Kyte-Doolittle hydrophobicity scale"""
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        return sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / len(sequence)
    
    def _calculate_charge(self, sequence):
        """Calculate net charge at pH 7"""
        charge_scale = {
            'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1
        }
        return sum(charge_scale.get(aa, 0) for aa in sequence)
    
    def _calculate_instability(self, sequence):
        """Calculate instability index"""
        # Simplified implementation - normally would use full DIWV table
        unstable_pairs = {'DD', 'EE', 'KK', 'RR', 'HH', 'FF', 'WW', 'YY'}
        instability = sum(1 for i in range(len(sequence)-1) 
                         if sequence[i:i+2] in unstable_pairs)
        return instability / (len(sequence) - 1) if len(sequence) > 1 else 0
    
    def _calculate_bert_confidence(self, sequences):
        """Use TAPE BERT model to assess sequence naturalness"""
        confidences = []
        with torch.no_grad():
            for seq in sequences:
                if len(seq) > 1024:  # Skip very long sequences
                    confidences.append(0.0)
                    continue
                    
                tokens = self.protein_tokenizer(sequences, padding='max_length', max_length=1022, truncation=True, return_tensors="pt").to('cuda')
                
                outputs = self.protein_model(**tokens)
                # Use negative loss as confidence metric
                logits = outputs[0]
                probs = F.softmax(logits, dim=-1)
                confidence = torch.mean(torch.max(probs, dim=-1)[0]).item()
                confidences.append(confidence)
        
        return confidences

@dataclass
class LigandEvaluator:
    """Evaluate ligand drug-likeness and synthetic accessibility"""
    
    def __init__(self):
        pass
    
    def calculate_drug_likeness(self, smiles_list):
        """Calculate various drug-likeness metrics"""
        scores = {}
        valid_mols = []
        
        # Parse SMILES
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            valid_mols.append(mol)
        
        # Basic molecular properties
        scores['validity'] = [mol is not None for mol in valid_mols]
        scores['mw'] = [Descriptors.MolWt(mol) if mol else 0 for mol in valid_mols]
        scores['logp'] = [Crippen.MolLogP(mol) if mol else 0 for mol in valid_mols]
        scores['hbd'] = [Lipinski.NumHDonors(mol) if mol else 0 for mol in valid_mols]
        scores['hba'] = [Lipinski.NumHAcceptors(mol) if mol else 0 for mol in valid_mols]
        scores['tpsa'] = [Descriptors.TPSA(mol) if mol else 0 for mol in valid_mols]
        scores['rotatable_bonds'] = [Descriptors.NumRotatableBonds(mol) if mol else 0 for mol in valid_mols]
        
        # Lipinski's Rule of Five
        scores['lipinski_violations'] = []
        for mol in valid_mols:
            if mol is None:
                scores['lipinski_violations'].append(4)
                continue
            violations = 0
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            scores['lipinski_violations'].append(violations)
        
        # Synthetic Accessibility Score
        scores['sa_score'] = []
        for mol in valid_mols:
            if mol is not None:
                try:
                    sa_score = sascorer.calculateScore(mol)
                    scores['sa_score'].append(sa_score)
                except:
                    scores['sa_score'].append(10.0)  # Worst possible score
            else:
                scores['sa_score'].append(10.0)
        
        return scores
