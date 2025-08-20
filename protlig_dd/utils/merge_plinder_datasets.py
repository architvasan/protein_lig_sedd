import torch
from pathlib import Path
from tqdm import tqdm
def merge_pt_files(pt_files, output_file):
    merged_list = []
    for f in tqdm(pt_files):
        data = torch.load(f)  # each file is a dict: {'prot_seq': [...], 'lig_seq': [...]}
        prot_seqs = data['protein_seq']
        lig_seqs = data['ligand_smiles']
        
        if len(prot_seqs) != len(lig_seqs):
            raise ValueError(f"Length mismatch in {f}: {len(prot_seqs)} vs {len(lig_seqs)}")
        
        # Convert to list of dicts
        merged_list.extend([{'protein_seq': p[0], 'ligand_smiles': l} for p, l in zip(prot_seqs, lig_seqs) if len(p)==1])
    
    print(f"Total entries: {len(merged_list)}")
    # Now merged_list is in the desired format

    # Save merged list to a .pt file
    torch.save(merged_list, output_file)
    
    print(f"Merged {len(pt_files)} files into {len(merged_list)} entries")
    print(f"Saved to {output_file}")
 
if __name__ == "__main__":
    # Folder containing all .pt files
    pt_folder_1 = Path("input_data/output_torchdata_round1")
    pt_folder_2 = Path("input_data/output_torchdata_round2")
    pt_folder_3 = Path("input_data/output_torchdata_round3")

    pt_files_1 = sorted(list(pt_folder_1.glob("*.pt")))
    pt_files_2 = sorted(list(pt_folder_2.glob("*.pt")))
    pt_files_3 = sorted(list(pt_folder_3.glob("*.pt")))
    pt_files_1.extend(pt_files_2)
    del(pt_files_2)
    pt_files_1.extend(pt_files_3)
    del(pt_files_3)

    merge_pt_files(pt_files_1, 'input_data/merged_plinder.pt')
