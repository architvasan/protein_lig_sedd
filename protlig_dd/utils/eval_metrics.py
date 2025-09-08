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
    def calculate_similarity(self, sequence, reference_sequences):
        """Calculate sequence similarity to reference set using simple identity metric"""
        from Bio import pairwise2
        from Bio.Seq import Seq
        from Bio.Align import substitution_matrices
        
        matrix = substitution_matrices.load("BLOSUM62")
        scores = []
        
        for seq in sequence:
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
    
    def calculate_emb_cos_distance(self, sequences, reference_sequences):
        from sklearn.metrics.pairwise import cosine_distances
        from transformers import EsmForSequenceClassification

        mlm_model = EsmForSequenceClassification.from_pretrained(self.prot_model_id).to(self.device)
        mlm_model.eval()
        
        def get_embeddings(sequences):
            embeddings = []
            with torch.no_grad():
                for seq in sequences:
                    if len(seq) > 1024:
                        embeddings.append(torch.zeros(self.protein_model.config.hidden_size).cpu().numpy())
                        continue

                    tokens = self.protein_tokenizer(seq, return_tensors="pt", padding='max_length', max_length=1022, truncation=True).to('cuda')
                    outputs = mlm_model.base_model(**tokens, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1][0]
                    mask = tokens['attention_mask'][0].bool()
                    mean_embedding = last_hidden[mask].mean(dim=0)

                    embeddings.append(mean_embedding.cpu().numpy())
                    
            return embeddings
        
        prot_embs = get_embeddings(sequences)
        ref_embs = get_embeddings(reference_sequences)
        
        distances = []
        for prot_emb, ref_emb in zip(prot_embs, ref_embs):
            dist = cosine_distances([prot_emb], [ref_emb])[0][0]
            distances.append(dist)

        return distances
    
        
    def calculate_mlm_ppl(self, sequences, batch_size=8, mask_fraction=0.15, seed = 42):
        """
        Calculates the MLM score (pseudo-perplexity) for a list of protein sequences.

        This method leverages the ESM model's native Masked Language Model task. It
        randomly masks a fraction of tokens in each sequence and calculates the
        model's ability to predict the original tokens at those masked positions.
        A lower score indicates the model finds the sequence more probable or "natural".

        Args:
            sequences (list): A list of protein sequence strings to evaluate.
            batch_size (int): The number of sequences to process in each batch.
                              Adjust based on GPU memory. Defaults to 8.
            mask_fraction (float): The fraction of amino acid tokens to mask in each
                                   sequence. Defaults to 0.15.

        Returns:
            list: A list of float values, where each value is the pseudo-perplexity
                  score for the corresponding input sequence.
        """
        from transformers import EsmForMaskedLM

        if seed is not None:
            print(f"Using fixed random seed for masking: {seed}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        mlm_model = EsmForMaskedLM.from_pretrained(self.prot_model_id).to(self.device)
        mlm_model.eval()

        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Calculating ESM PPL"):
                batch_seqs = sequences[i:i + batch_size]

                # 1. Tokenize to get ground truth labels
                inputs = self.protein_tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1022  # Consistent with __post_init__
                )
                labels = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # 2. Create the masked input
                masked_input_ids = labels.clone()
                
                # Determine which tokens are actual amino acids (and can be masked)
                is_amino_acid = (labels != self.protein_tokenizer.cls_token_id) & \
                                (labels != self.protein_tokenizer.sep_token_id) & \
                                (labels != self.protein_tokenizer.pad_token_id)
                
                # For each sequence, randomly select positions to mask
                probability_matrix = torch.full(labels.shape, mask_fraction, device=self.device)
                masked_indices = torch.bernoulli(probability_matrix).bool() & is_amino_acid
                
                masked_input_ids[masked_indices] = self.protein_tokenizer.mask_token_id

                # 3. Get model's predictions (logits) for the masked input
                outputs = mlm_model(masked_input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # 4. Calculate the CrossEntropyLoss only at the masked positions
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, mlm_model.config.vocab_size), labels.view(-1))
                loss = loss.view(labels.size())

                # Sum the loss for the masked tokens and count them
                sum_masked_loss = torch.sum(loss * masked_indices, dim=1)
                num_masked = torch.sum(masked_indices, dim=1)
                
                # Avoid division by zero if a sequence has no masked tokens
                num_masked = torch.max(num_masked, torch.tensor(1.0, device=self.device))
                
                # Calculate the average negative log-likelihood over masked positions
                mean_nll_masked = sum_masked_loss / num_masked
                
                # Pseudo-perplexity is the exponential of the average loss
                pseudo_ppl = torch.exp(mean_nll_masked)
                
                scores.extend(pseudo_ppl.cpu().numpy().tolist())

            # Flatten the scores if batch_size > 1
            scores = [item for sublist in scores for item in (sublist if isinstance(sublist, list) else [sublist])]
                
        return scores

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
