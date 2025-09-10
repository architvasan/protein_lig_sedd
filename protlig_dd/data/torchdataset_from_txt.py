import os
import torch
from Bio import SeqIO
from rdkit import Chem
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
@dataclass
class Reader_Converter:
    filetype: str # 'fasta', 'tsv', 'csv'
    inp_file: str
    modalities: str # 'protein_seq', 'ligand_smiles', 'protein_ligand'
    out_file: str

    smiles_column: str | None = None
    protein_column: str | None = None
    save_freq: int = 1000
    def __post_init__(self):
        if self.filetype == 'csv':
            self.data_inp = pd.read_csv(inp_file)
        elif self.filetype == 'tsv':
            self.data_inp = pd.read_csv(inp_file, sep='\t', index=False)
        elif self.filetype == 'fasta':
            # Read from file
            with open(self.inp_file, 'r') as f:
                self.data_inp = []
                for line in f:
                    line = line.strip()
                    if not line.startswith('>') and line:  # Skip header lines and empty lines
                        self.data_inp.append(line)

    def build_dataset_pd(self):
        self.dict_list = []
        for it, (_, row) in enumerate(self.data_inp.iterrows()):
            dict_it = {}
            if self.modalities in ['protein_seq', 'protein_ligand']:
                dict_it['protein_seq'] = row[protein_column]
            if self.modalities in ['ligand_smiles', 'protein_ligand']:
                dict_it['ligand_smiles'] = row[smiles_column]
            self.dict_list.append(dict_it)
            if it % self.save_freq == 0 and it!= 0:
                self.save_pt()

    def build_dataset_fasta(self):
        self.dict_list = [{'protein_seq': dat} for dat in self.data_inp]
        
    def save_pt(self):
        torch.save(self.dict_list, self.out_file)

    def run(self):
        if self.filetype in ['csv', 'tsv']:
            self.build_dataset_pd()
        elif self.filetype == 'fasta':
            self.build_dataset_fasta()
            self.save_pt()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert biological data files to PyTorch format')
    
    # Required arguments
    parser.add_argument('--filetype', '-f', type=str, required=True,
                        choices=['fasta', 'tsv', 'csv'],
                        help='Input file type (fasta, tsv, or csv)')
    
    parser.add_argument('--inp_file', '-i', type=str, required=True,
                        help='Path to input file')
    
    parser.add_argument('--modalities', '-m', type=str, required=True,
                        choices=['protein_seq', 'ligand_smiles', 'protein_ligand'],
                        help='Data modalities to extract')
    
    parser.add_argument('--out_file', '-o', type=str, required=True,
                        help='Path to output PyTorch file')
    
    # Optional arguments
    parser.add_argument('--smiles_column', '-s', type=str, default=None,
                        help='Column name containing SMILES data (required for ligand_smiles or protein_ligand modalities)')
    
    parser.add_argument('--protein_column', '-p', type=str, default=None,
                        help='Column name containing protein sequences (required for protein_seq or protein_ligand modalities)')
    
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='Frequency of intermediate saves (default: 1000)')
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.inp_file):
        raise FileNotFoundError(f"Input file not found: {args.inp_file}")
    
    if args.modalities in ['protein_seq', 'protein_ligand'] and args.filetype != 'fasta' and not args.protein_column:
        raise ValueError("protein_column must be specified for protein_seq or protein_ligand modalities with csv/tsv files")
    
    if args.modalities in ['ligand_smiles', 'protein_ligand'] and not args.smiles_column:
        raise ValueError("smiles_column must be specified for ligand_smiles or protein_ligand modalities")
    
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Initialize and run converter
    converter = Reader_Converter(
        filetype=args.filetype,
        inp_file=args.inp_file,
        modalities=args.modalities,
        out_file=args.out_file,
        smiles_column=args.smiles_column,
        protein_column=args.protein_column,
        save_freq=args.save_freq
    )
    
    print(f"Converting {args.inp_file} to {args.out_file}")
    print(f"Filetype: {args.filetype}, Modalities: {args.modalities}")
    
    converter.run()
    
    print(f"Conversion complete! Output saved to {args.out_file}")


if __name__ == "__main__":
    main()    
