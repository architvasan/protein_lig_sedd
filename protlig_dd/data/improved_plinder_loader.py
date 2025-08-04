#!/usr/bin/env python3
"""
Improved PLINDER Data Loader

This module provides a corrected data loader that properly interfaces with 
the PLINDER dataset using the official PLINDER API.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# PLINDER imports
from plinder.core import PlinderSystem, get_plindex, get_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ImprovedPLINDERProcessor:
    """
    Improved processor for the PLINDER dataset using the official API.
    
    This processor correctly interfaces with PLINDER to download and process
    protein-ligand interaction data for training the GenMolDPLM model.
    
    Supports both automatic PLINDER data loading and pointing to external
    PLINDER dataset directories for flexibility.
    """
    
    # Amino acid mapping for tokenization
    aa_mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20, '<pad>': 21
    }
    
    def __init__(self, 
                 output_dir: Union[str, Path], 
                 cache_dir: Optional[Union[str, Path]] = None,
                 plinder_data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the improved PLINDER processor.
        
        Args:
            output_dir: Directory to save processed data
            cache_dir: Directory to cache PLINDER data (optional)
            plinder_data_dir: Path to external PLINDER dataset directory (optional)
                            If provided, will use this as the data source instead of 
                            downloading/using default PLINDER cache
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.cache_dir = self.output_dir / "cache"
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Store external data directory if provided
        self.plinder_data_dir = Path(plinder_data_dir) if plinder_data_dir else None
        if self.plinder_data_dir:
            if not self.plinder_data_dir.exists():
                raise FileNotFoundError(f"PLINDER data directory not found: {self.plinder_data_dir}")
            logger.info(f"Using external PLINDER data directory: {self.plinder_data_dir}")
        
        logger.info(f"Initialized PLINDER processor. Output: {self.output_dir}, Cache: {self.cache_dir}")
        
        # Load PLINDER index
        logger.info("Loading PLINDER index...")
        if self.plinder_data_dir:
            # Try to load from external directory first
            self.plindex = self._load_plindex_from_external_dir()
        else:
            # Use default PLINDER API
            self.plindex = get_plindex()
        logger.info(f"Loaded PLINDER index with {len(self.plindex)} entries")
        
        # Load train split
        logger.info("Loading PLINDER train split...")
        if self.plinder_data_dir:
            self.split_data = self._load_split_from_external_dir()
        else:
            self.split_data = get_split()
        logger.info(f"Loaded split data with {len(self.split_data)} entries")
    
    def _load_plindex_from_external_dir(self) -> pd.DataFrame:
        """
        Load PLINDER index from external data directory.
        
        Returns:
            DataFrame containing PLINDER index
        """
        # Look for common PLINDER index file names
        possible_index_files = [
            'index.parquet',
            'plindex.parquet', 
            'dataset_index.parquet',
            'plinder_index.parquet',
            'index.csv',
            'plindex.csv'
        ]
        
        index_file = None
        for filename in possible_index_files:
            candidate = self.plinder_data_dir / filename
            if candidate.exists():
                index_file = candidate
                break
        
        if index_file is None:
            logger.warning(f"No PLINDER index file found in {self.plinder_data_dir}. Falling back to default API.")
            return get_plindex()
        
        logger.info(f"Loading PLINDER index from: {index_file}")
        
        try:
            if index_file.suffix == '.parquet':
                return pd.read_parquet(index_file)
            elif index_file.suffix == '.csv':
                return pd.read_csv(index_file)
            else:
                logger.warning(f"Unsupported index file format: {index_file.suffix}. Falling back to default API.")
                return get_plindex()
        except Exception as e:
            logger.warning(f"Failed to load index from {index_file}: {e}. Falling back to default API.")
            return get_plindex()
    
    def _load_split_from_external_dir(self) -> pd.DataFrame:
        """
        Load PLINDER split data from external data directory.
        
        Returns:
            DataFrame containing split information
        """
        # Look for common split file names
        possible_split_files = [
            'split.parquet',
            'splits.parquet',
            'train_val_test_split.parquet',
            'dataset_split.parquet',
            'split.csv',
            'splits.csv'
        ]
        
        split_file = None
        for filename in possible_split_files:
            candidate = self.plinder_data_dir / filename
            if candidate.exists():
                split_file = candidate
                break
        
        if split_file is None:
            logger.warning(f"No split file found in {self.plinder_data_dir}. Falling back to default API.")
            return get_split()
        
        logger.info(f"Loading split data from: {split_file}")
        
        try:
            if split_file.suffix == '.parquet':
                return pd.read_parquet(split_file)
            elif split_file.suffix == '.csv':
                return pd.read_csv(split_file)
            else:
                logger.warning(f"Unsupported split file format: {split_file.suffix}. Falling back to default API.")
                return get_split()
        except Exception as e:
            logger.warning(f"Failed to load split from {split_file}: {e}. Falling back to default API.")
            return get_split()
    
    def get_train_systems(self, max_samples: Optional[int] = None) -> List[str]:
        """
        Get list of system IDs for training.
        
        Args:
            max_samples: Maximum number of systems to return
            
        Returns:
            List of system IDs for training
        """
        # Get training split system IDs
        train_split = self.split_data[self.split_data['split'] == 'train']
        system_ids = train_split['system_id'].tolist()
        
        if max_samples:
            system_ids = system_ids[:max_samples]
        
        logger.info(f"Found {len(system_ids)} training systems")
        return system_ids
    
    def process_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a single PLINDER system.
        
        Args:
            system_id: PLINDER system ID
            
        Returns:
            Dictionary with processed features or None if processing failed
        """
        try:
            # Create PlinderSystem object
            system = PlinderSystem(system_id=system_id)
            
            # Get system metadata
            system_entry = self.plindex[self.plindex['system_id'] == system_id].iloc[0]
            
            # Get ligand SMILES from system object (more accurate)
            try:
                system_smiles = system.smiles
                if not system_smiles:
                    # Fall back to metadata table
                    ligand_smiles = system_entry.get('ligand_smiles', '')
                else:
                    # Use the first ligand SMILES from system
                    ligand_smiles = list(system_smiles.values())[0]
                    
                if not ligand_smiles:
                    logger.warning(f"No ligand SMILES found for system {system_id}")
                    return None
                    
            except Exception as e:
                logger.debug(f"Could not extract ligand SMILES for {system_id}: {e}")
                return None
            
            # Extract protein sequence from PlinderSystem
            try:
                sequences = system.sequences
                if not sequences:
                    logger.debug(f"No protein sequences found for system {system_id}")
                    return None
                
                # Get the first protein sequence (in case there are multiple chains)
                protein_seq = list(sequences.values())[0]
                if not protein_seq:
                    logger.debug(f"Empty protein sequence for system {system_id}")
                    return None
                    
            except Exception as e:
                logger.debug(f"Could not extract protein sequence for {system_id}: {e}")
                return None
            
            # Validate ligand SMILES
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES for system {system_id}: {ligand_smiles}")
                return None
            
            # Tokenize protein sequence
            protein_tokens = self.tokenize_protein_sequence(protein_seq)
            
            # Tokenize ligand SMILES
            ligand_tokens = self.tokenize_smiles(ligand_smiles)
            
            # Get binding affinity if available (correct column name)
            binding_affinity = system_entry.get('ligand_binding_affinity', None)
            if binding_affinity is not None and pd.notna(binding_affinity):
                try:
                    binding_affinity = float(binding_affinity)
                except (ValueError, TypeError):
                    binding_affinity = None
            else:
                binding_affinity = None
            
            # For now, skip structure processing since we need to implement proper structure extraction
            # In a full implementation, you would:
            # 1. Download the structure files using PlinderSystem
            # 2. Parse the structure to extract protein coordinates
            # 3. Identify binding pocket residues
            protein_coords = None
            interaction_mask = None
            
            # Generate 3D ligand coordinates from SMILES
            ligand_coords = self._generate_ligand_coords_from_smiles(ligand_smiles)
            
            # Create sample
            sample = {
                'system_id': system_id,
                'protein_seq': protein_seq,
                'protein_tokens': protein_tokens,
                'protein_coords': protein_coords,
                'ligand_smiles': ligand_smiles,
                'ligand_tokens': ligand_tokens,
                'ligand_coords': ligand_coords,
                'binding_value': binding_affinity,
                'interaction_mask': interaction_mask
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error processing system {system_id}: {e}")
            return None
    
    def _extract_protein_coords_from_structure(self, structure_data: Any, protein_seq: str) -> Optional[np.ndarray]:
        """
        Extract protein backbone coordinates from structure data.
        
        Args:
            structure_data: Structure data from PLINDER
            protein_seq: Protein sequence
            
        Returns:
            Array of backbone coordinates [seq_len, 3, 3] or None
        """
        try:
            # This is a simplified implementation
            # In practice, you'd need to properly parse the structure data
            # and extract N, CA, C coordinates for each residue
            
            # For now, return None to indicate no structure available
            # You can implement proper structure parsing here
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting protein coordinates: {e}")
            return None
    
    def _generate_ligand_coords_from_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """
        Generate 3D coordinates for ligand from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Array of 3D coordinates [num_atoms, 3] or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Extract coordinates
            conf = mol.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            
            return coords.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Error generating ligand coordinates for {smiles}: {e}")
            return None
    
    def tokenize_protein_sequence(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """
        Tokenize protein sequence to integers.
        
        Args:
            sequence: Amino acid sequence
            max_length: Maximum sequence length
            
        Returns:
            np.ndarray: Tokenized sequence
        """
        # Truncate if too long
        sequence = sequence[:max_length]
        
        # Convert to tokens
        tokens = np.array([self.aa_mapping.get(aa, self.aa_mapping['X']) for aa in sequence], dtype=np.int64)
        
        # Pad if needed
        if len(tokens) < max_length:
            padding = np.full(max_length - len(tokens), self.aa_mapping['<pad>'], dtype=np.int64)
            tokens = np.concatenate([tokens, padding])
        
        return tokens
    
    def tokenize_smiles(self, smiles: str, max_length: int = 128) -> np.ndarray:
        """
        Character-level tokenization of SMILES strings.
        
        Args:
            smiles: SMILES string
            max_length: Maximum sequence length
            
        Returns:
            np.ndarray: Tokenized SMILES
        """
        # Define a basic vocabulary for SMILES characters
        # In practice, you might want to use a more sophisticated tokenizer
        vocab = list("CNOSPFBrClI()[]+=\\#-@:123456789%/c.nsop")
        char_to_idx = {c: i+1 for i, c in enumerate(vocab)}  # Reserve 0 for padding
        
        # Tokenize
        tokens = np.zeros(max_length, dtype=np.int64)
        for i, char in enumerate(smiles[:max_length]):
            tokens[i] = char_to_idx.get(char, 0)  # Unknown chars map to 0
            
        return tokens
    
    def process_dataset(self, max_samples: Optional[int] = None, num_workers: int = 1) -> str:
        """
        Process PLINDER dataset and save processed samples.
        
        Args:
            max_samples: Maximum number of samples to process
            num_workers: Number of parallel workers (set to 1 for debugging)
            
        Returns:
            Path to saved processed data file
        """
        # Get training system IDs
        system_ids = self.get_train_systems(max_samples)
        
        # Process systems
        processed_samples = []
        logger.info(f"Processing {len(system_ids)} systems with {num_workers} workers")
        
        if num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.process_system, system_id) for system_id in system_ids]
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        sample = future.result()
                        if sample is not None:
                            processed_samples.append(sample)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i+1}/{len(system_ids)} systems")
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
        else:
            # Sequential processing (better for debugging)
            for i, system_id in enumerate(system_ids):
                try:
                    sample = self.process_system(system_id)
                    if sample is not None:
                        processed_samples.append(sample)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i+1}/{len(system_ids)} systems, {len(processed_samples)} successful")
                except Exception as e:
                    logger.error(f"Error processing system {system_id}: {e}")
        
        # Save processed data
        output_file = self.output_dir / "processed_plinder_data.pt"
        torch.save(processed_samples, output_file)
        logger.info(f"Saved {len(processed_samples)} processed samples to {output_file}")
        
        # Save metadata
        metadata = {
            'num_samples': len(processed_samples),
            'max_protein_len': max(len(sample['protein_seq']) for sample in processed_samples) if processed_samples else 0,
            'max_ligand_len': max(len(sample['ligand_smiles']) for sample in processed_samples) if processed_samples else 0,
            'has_structure': any(sample.get('protein_coords') is not None for sample in processed_samples),
            'has_binding': any(sample.get('binding_value') is not None for sample in processed_samples),
            'has_interactions': any(sample.get('interaction_mask') is not None for sample in processed_samples)
        }
        
        metadata_file = self.output_dir / "metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset metadata: {metadata}")
        
        return str(output_file)


def test_plinder_loader():
    """Test the improved PLINDER loader with a small subset."""
    
    # Create processor
    processor = ImprovedPLINDERProcessor(
        output_dir="./test_plinder_output",
        cache_dir="./plinder_cache"
    )
    
    # Process a small subset
    data_file = processor.process_dataset(max_samples=5, num_workers=1)
    
    # Test loading the processed data
    data = torch.load(data_file)
    logger.info(f"Successfully processed and loaded {len(data)} samples")
    
    # Print sample information
    if data:
        sample = data[0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        logger.info(f"Protein sequence length: {len(sample['protein_seq'])}")
        logger.info(f"Ligand SMILES: {sample['ligand_smiles']}")
        logger.info(f"Has binding value: {sample.get('binding_value') is not None}")
        logger.info(f"Has interaction mask: {sample.get('interaction_mask') is not None}")
    
    return data_file


if __name__ == "__main__":
    # Test the improved loader
    test_plinder_loader()
