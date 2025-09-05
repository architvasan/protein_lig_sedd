import os
import torch
from Bio import SeqIO
from rdkit import Chem
from tqdm import tqdm

def pdb_to_seq( pdb_fn ) -> str:
    '''
    Given a pdb file, return the sequence of the protein as a string.
    '''

    to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

    seq = []
    seqstr = ''
    #print(pdb_fn)
    chain_id = 'A'
    with open(pdb_fn) as fp:
        for line in fp:
            if line!='':
                #for it_l, l in enumerate(line):
                #    print(it_l, l)
                #[print(line[i]) for i in range(len(line))]
                #print(line[1])
                if line.startswith("END" or "TER"):
                    seq.append(seqstr)
                    seqstr = ''

                if not line.startswith("ATOM"):
                    continue

                #print(line[21])
                if line[21]!=chain_id:
                    seq.append(seqstr)
                    seqstr = ''
                    chain_id = line[21]

                if line[12:16].strip() != "CA":
                    continue
                resName = line[17:20]
                if resName not in to1letter.keys():
                    seqstr += 'X'
                else:
                    seqstr += to1letter[resName]

    
    return seq



def _pdb_to_seq(pdb_path):
    """Extract protein sequence from a PDB file."""
    seqs = []
    with open(pdb_path, "r") as f:
        for record in SeqIO.parse(f, "pdb-seqres"):
            seqs.append(str(record.seq))
    return "".join(seqs)

def sdf_to_smiles(sdf_path):
    """Convert SDF file to canonical SMILES."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    smiles_list = []
    for mol in suppl:
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol, canonical=True))
    return smiles_list

def build_dataset(root_dir, output_file):
    ligand_smiles = []
    protein_seq = []

    for it_sys, folder in tqdm(enumerate(os.listdir(root_dir))):
        sys_path = os.path.join(root_dir, folder)
        if not os.path.isdir(sys_path):
            continue

        pdb_path = os.path.join(sys_path, "receptor.pdb")
        lig_dir = os.path.join(sys_path, "ligand_files")

        if not (os.path.exists(pdb_path) and os.path.isdir(lig_dir)):
            continue

        # Extract protein sequence
        seq = pdb_to_seq(pdb_path)

        # Extract ligand smiles
        for sdf_file in os.listdir(lig_dir):
            if sdf_file.endswith(".sdf"):
                sdf_path = os.path.join(lig_dir, sdf_file)
                smiles_list = sdf_to_smiles(sdf_path)
                for smi in smiles_list:
                    ligand_smiles.append(smi)
                    protein_seq.append(seq)  # repeat same protein seq for each ligand

        if it_sys % 1000 == 0 and it_sys !=0:
            data_dict = {
                "ligand_smiles": ligand_smiles,
                "protein_seq": protein_seq
            }
            save_it = int(it_sys/1000)
            torch.save(data_dict, f'{output_file}.{save_it}.pt')
            print(f"Saved dataset with {len(ligand_smiles)} entries to {output_file}")
            del(data_dict) 
            del(ligand_smiles)
            del(protein_seq)
            ligand_smiles = []
            protein_seq = []

    data_dict = {
        "ligand_smiles": ligand_smiles,
        "protein_seq": protein_seq
    }

    torch.save(data_dict, f'{output_file}.final.pt')
    print(f"Saved dataset with {len(ligand_smiles)} entries to {output_file}")

# Example usage:
root_directory = "/nfs/ml_lab/projects/ml_lab/avasan/plinder/dataset/plinder/2024-06/v2/systems"  # where 5y3l__2__... folders are
output_path = "output_torchdata/ligand_protein_dataset"
build_dataset(root_directory, output_path)

