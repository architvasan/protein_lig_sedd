#!/usr/bin/env python3
"""
Updated Data Pipeline for GenMolDPLM

This module integrates the improved PLINDER loader with the original data processing
workflow to create a complete data pipeline for training the unified model.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# Import the improved PLINDER processor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from improved_plinder_loader import ImprovedPLINDERProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class ImprovedProteinLigandDataset(Dataset):
    """
    Improved PyTorch dataset for protein-ligand interaction data.
    
    This dataset loads processed protein-ligand pairs from the improved
    PLINDER processor and provides them in a format suitable for training
    the unified model.
    """
    
    def __init__(self, data_file: Union[str, Path], max_protein_len: int = 1024, max_ligand_len: int = 128):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to processed data file (.pt)
            max_protein_len: Maximum protein sequence length
            max_ligand_len: Maximum ligand sequence length
        """
        self.data_file = Path(data_file)
        self.max_protein_len = max_protein_len
        self.max_ligand_len = max_ligand_len
        
        # Load data
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.data = torch.load(self.data_file, weights_only=False)
        logger.info(f"Loaded {len(self.data)} samples from {self.data_file}")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with sample data formatted for the unified model
        """
        sample = self.data[idx]
        
        # Create attention masks
        protein_seq_len = len(sample['protein_seq'])
        ligand_seq_len = len(sample['ligand_smiles'])
        
        protein_mask = torch.zeros(self.max_protein_len, dtype=torch.float)
        protein_mask[:min(protein_seq_len, self.max_protein_len)] = 1.0
        
        ligand_mask = torch.zeros(self.max_ligand_len, dtype=torch.float)
        ligand_mask[:min(ligand_seq_len, self.max_ligand_len)] = 1.0
        
        # Create processed sample for the unified model
        processed_sample = {
            'protein_mask': protein_mask,
            'ligand_mask': ligand_mask,
            'protein_seq': sample['protein_seq'],
            'ligand_smiles': sample['ligand_smiles'],
        }
        
        # Add protein coordinates if available
        if sample.get('protein_coords') is not None:
            processed_sample['protein_coords'] = torch.tensor(sample['protein_coords'], dtype=torch.float)
        
        # Add ligand coordinates if available
        if sample.get('ligand_coords') is not None:
            processed_sample['ligand_coords'] = torch.tensor(sample['ligand_coords'], dtype=torch.float)
        
        # Add binding value if available
        if sample.get('binding_value') is not None:
            processed_sample['binding_value'] = torch.tensor(sample['binding_value'], dtype=torch.float)
        
        # Add interaction mask if available
        if sample.get('interaction_mask') is not None:
            processed_sample['interaction_mask'] = torch.tensor(sample['interaction_mask'], dtype=torch.float)
        
        return processed_sample


def improved_protein_ligand_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Improved collate function for batching protein-ligand samples.
    
    This function handles variable-length sequences by creating
    appropriate padding and attention masks.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched samples
    """
    # Get batch size
    batch_size = len(batch)
    
    # Find max lengths in the batch
    max_protein_len = max(len(sample['protein_mask']) for sample in batch)
    max_ligand_len = max(len(sample['ligand_mask']) for sample in batch)
    
    # Initialize tensors with padding
    protein_tokens = torch.zeros((batch_size, max_protein_len), dtype=torch.long)
    protein_mask = torch.zeros((batch_size, max_protein_len), dtype=torch.float)
    ligand_tokens = torch.zeros((batch_size, max_ligand_len), dtype=torch.long)
    ligand_mask = torch.zeros((batch_size, max_ligand_len), dtype=torch.float)
    
    # Check what optional data is available
    has_protein_coords = all('protein_coords' in sample for sample in batch)
    has_ligand_coords = all('ligand_coords' in sample for sample in batch)
    has_binding = all('binding_value' in sample for sample in batch)
    has_interaction = all('interaction_mask' in sample for sample in batch)
    
    # Initialize optional tensors
    protein_coords = None
    ligand_coords = None
    binding_values = None
    interaction_masks = None
    
    if has_protein_coords:
        # For protein coords, we need to determine the structure shape
        sample_coords = batch[0]['protein_coords']
        if len(sample_coords.shape) == 3:  # [seq_len, 3, 3] format
            protein_coords = torch.zeros((batch_size, max_protein_len, 3, 3), dtype=torch.float)
        else:  # [seq_len, 3] format
            protein_coords = torch.zeros((batch_size, max_protein_len, 3), dtype=torch.float)
    
    if has_ligand_coords:
        # Ligand coords might have variable number of atoms
        max_atoms = max(sample['ligand_coords'].shape[0] for sample in batch)
        ligand_coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float)
    
    if has_binding:
        binding_values = torch.zeros(batch_size, dtype=torch.float)
    
    if has_interaction:
        interaction_masks = torch.zeros((batch_size, max_protein_len), dtype=torch.float)
    
    # Fill tensors
    protein_seqs = []
    ligand_smiles = []
    system_ids = []
    
    for i, sample in enumerate(batch):
        # Basic protein and ligand data
        protein_len = len(sample['protein_mask'])
        ligand_len = len(sample['ligand_mask'])
        
        protein_mask[i, :protein_len] = sample['protein_mask'][:protein_len]
        
        ligand_mask[i, :ligand_len] = sample['ligand_mask'][:ligand_len]
        
        # Optional data
        if has_protein_coords:
            coords = sample['protein_coords']
            protein_coords[i, :coords.shape[0]] = coords
        
        if has_ligand_coords:
            coords = sample['ligand_coords']
            ligand_coords[i, :coords.shape[0]] = coords
        
        if has_binding:
            binding_values[i] = sample['binding_value']
        
        if has_interaction:
            mask = sample['interaction_mask']
            interaction_masks[i, :len(mask)] = mask
        
        # Store metadata
        protein_seqs.append(sample['protein_seq'])
        ligand_smiles.append(sample['ligand_smiles'])
    
    # Create batched sample
    batched_sample = {
        'protein_mask': protein_mask,
        'ligand_mask': ligand_mask,
        'protein_seq': protein_seqs,
        'ligand_smiles': ligand_smiles,
    }
    
    # Add optional tensors if available
    if has_protein_coords:
        batched_sample['protein_coords'] = protein_coords
    
    if has_ligand_coords:
        batched_sample['ligand_coords'] = ligand_coords
    
    if has_binding:
        batched_sample['binding_value'] = binding_values
    
    if has_interaction:
        batched_sample['interaction_mask'] = interaction_masks
    
    return batched_sample








def create_improved_data_loaders(
    data_file: str,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_protein_len: int = 1024,
    max_ligand_len: int = 128,
    use_structure: bool = False,
    seed: int = 42,
    force_reprocess: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create improved DataLoaders using the updated PLINDER processor.
    
    Args:
        plinder_output_dir: Directory to save/load processed PLINDER data
        plinder_data_dir: Path to external PLINDER dataset directory (optional)
                         If provided, will use this as the data source instead of 
                         downloading/using default PLINDER cache
        max_samples: Maximum number of samples to process
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation
        max_protein_len: Maximum protein sequence length
        max_ligand_len: Maximum ligand sequence length
        use_structure: Whether to use structure information (placeholder)
        seed: Random seed for reproducibility
        force_reprocess: Whether to force reprocessing of data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(seed)
    
    # Create dataset
    dataset = ImprovedProteinLigandDataset(
        data_file=data_file,
        max_protein_len=max_protein_len,
        max_ligand_len=max_ligand_len
    )
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Split dataset into train, validation, and test sets
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    logger.info(f"Split dataset into {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=improved_protein_ligand_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=improved_protein_ligand_collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=improved_protein_ligand_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader



def ddp_data_loaders(
    data_file: str,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_protein_len: int = 1024,
    max_ligand_len: int = 128,
    use_structure: bool = False,
    seed: int = 42,
    force_reprocess: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create improved DataLoaders using the updated PLINDER processor.
    
    Args:
        plinder_output_dir: Directory to save/load processed PLINDER data
        plinder_data_dir: Path to external PLINDER dataset directory (optional)
                         If provided, will use this as the data source instead of 
                         downloading/using default PLINDER cache
        max_samples: Maximum number of samples to process
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation
        max_protein_len: Maximum protein sequence length
        max_ligand_len: Maximum ligand sequence length
        use_structure: Whether to use structure information (placeholder)
        seed: Random seed for reproducibility
        force_reprocess: Whether to force reprocessing of data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, DistributedSampler

    torch.manual_seed(seed)
    
    # Create dataset
    dataset = ImprovedProteinLigandDataset(
        data_file=data_file,
        max_protein_len=max_protein_len,
        max_ligand_len=max_ligand_len
    )
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Split dataset into train, validation, and test sets
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)
    logger.info(f"Split dataset into {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        #shuffle=True,
        num_workers=num_workers,
        collate_fn=improved_protein_ligand_collate_fn,
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        #shuffle=False,
        num_workers=num_workers,
        collate_fn=improved_protein_ligand_collate_fn,
        pin_memory=True,
        sampler=val_sampler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        #shuffle=False,
        num_workers=num_workers,
        collate_fn=improved_protein_ligand_collate_fn,
        pin_memory=True,
        sampler=test_sampler
    )
    
    return train_loader, val_loader, test_loader


def test_improved_data_pipeline():
    """Test the improved data pipeline."""
    
    logger.info("Testing improved data pipeline...")
    
    # Create data loaders with a small sample
    train_loader, val_loader, test_loader = create_improved_data_loaders(
        data_file="./test_improved_pipeline",
        max_samples=10,  # Small sample for testing
        batch_size=2,
        num_workers=0,  # Use 0 for debugging
        force_reprocess=True
    )
    
    # Test training loader
    logger.info("Testing training loader...")
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Batch {batch_idx + 1}:")
        logger.info(f"  Protein mask shape: {batch['protein_mask'].shape}")
        logger.info(f"  Ligand mask shape: {batch['ligand_mask'].shape}")
        
        if 'binding_value' in batch:
            logger.info(f"  Binding values: {batch['binding_value']}")
        
        if 'protein_coords' in batch:
            logger.info(f"  Protein coords shape: {batch['protein_coords'].shape}")
        
        if 'ligand_coords' in batch:
            logger.info(f"  Ligand coords shape: {batch['ligand_coords'].shape}")
        
        logger.info(f"  System IDs: {batch['system_ids']}")
        
        # Only check first batch
        break
    
    logger.info("Improved data pipeline test completed successfully!")
    
    return train_loader, val_loader, test_loader









def test_improved_data_pipeline():
    """Test the improved data pipeline."""
    
    logger.info("Testing improved data pipeline...")
    
    # Create data loaders with a small sample
    train_loader, val_loader, test_loader = create_improved_data_loaders(
        data_file="./test_improved_pipeline",
        max_samples=10,  # Small sample for testing
        batch_size=2,
        num_workers=0,  # Use 0 for debugging
        force_reprocess=True
    )
    
    # Test training loader
    logger.info("Testing training loader...")
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Batch {batch_idx + 1}:")
        logger.info(f"  Protein mask shape: {batch['protein_mask'].shape}")
        logger.info(f"  Ligand mask shape: {batch['ligand_mask'].shape}")
        
        if 'binding_value' in batch:
            logger.info(f"  Binding values: {batch['binding_value']}")
        
        if 'protein_coords' in batch:
            logger.info(f"  Protein coords shape: {batch['protein_coords'].shape}")
        
        if 'ligand_coords' in batch:
            logger.info(f"  Ligand coords shape: {batch['ligand_coords'].shape}")
        
        logger.info(f"  System IDs: {batch['system_ids']}")
        
        # Only check first batch
        break
    
    logger.info("Improved data pipeline test completed successfully!")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the improved pipeline
    test_improved_data_pipeline()
