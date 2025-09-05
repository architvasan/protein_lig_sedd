import argparse
import torch
import os
from transformers import AutoTokenizer

os.chdir("/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/protlig_dd")
from protlig_dd.utils.load_model import load_model
from protlig_dd.data.get_embeddings import Embed_Mol_Prot
import protlig_dd.sampling.sampling as sampling
from protlig_dd.model import utils as mutils

def generate(args):
    """
    Main function to load the model and generate ligands.
    """
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load the pre-trained SEDD model, graph, and noise schedule
    print(f"Loading model from: {args.model_path}...")
    score_model, graph, noise = load_model(args.model_path, device)
    score_model.eval()

    # 2. Prepare the conditional embeddings
    print("Generating conditional protein embedding...")
    try:
        embedder = Embed_Mol_Prot(
            mol_model_id=args.mol_model_id,
            prot_model_id=args.prot_model_id
        )
    except Exception as e:
        print(f"Failed to load embedding models from Hugging Face: {e}")
        print("Please check your internet connection and model IDs.")
        return
        
    # The embedder needs a list of proteins and a list of SMILES.
    # For unconditional ligand generation (given only a protein), we can pass a placeholder SMILES.
    protein_seq_list = [args.protein_sequence] * args.num_samples
    placeholder_smiles_list = ["C"] * args.num_samples # A single carbon atom is a simple placeholder

    # Generate embeddings. We only need the protein embedding for conditioning.
    _, _, mol_cond, esm_cond = embedder.process_embeddings(
        protein_seq_list=protein_seq_list,
        smiles_list=placeholder_smiles_list
    )
    esm_cond = esm_cond.to(device)
    mol_cond = mol_cond.to(device) # We pass the placeholder embedding as well
    print("Protein embedding generated.")

    # 3. Create the sampler with a conditioned score function
    # The sampling function from the library isn't directly designed to pass conditions.
    # The key trick here is to wrap the score_model's forward pass in a new function
    # that already includes our conditional embeddings.

    def conditioned_model(x, sigma):
        # This wrapper fixes the esm_cond and mol_cond arguments for the model call
        return score_model(x, sigma, esm_cond=esm_cond, mol_cond=mol_cond)

    # Define the shape of the tensors to be generated
    # The total length is defined in the model config (e.g., 1224 = 202 for ligand + 1022 for protein)
    sampling_shape = (args.num_samples, score_model.config.model.length)

    print(f"Setting up sampler with {args.steps} steps...")
    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        sampling_shape,
        'analytic', # Using the 'analytic' predictor as it's often effective
        args.steps,
        device=device
    )

    # 4. Run the generation process
    print("Starting ligand generation...")
    # The pc_sampler expects a model, so we pass our wrapper
    samples = sampling_fn(conditioned_model)
    print("Generation complete.")

    # 5. Decode the output tokens into SMILES strings
    print("Decoding generated tokens...")
    # The combined sequence contains ligand tokens followed by protein tokens.
    # We need to extract just the ligand part. Based on get_embeddings.py,
    # the MolFormer tokenizer uses a max_length of 202.
    ligand_token_len = 202 
    ligand_tokens = samples[:, :ligand_token_len]

    # Load the same tokenizer used for embedding to decode the results
    mol_tokenizer = AutoTokenizer.from_pretrained(args.mol_model_id, trust_remote_code=True)
    
    generated_smiles = mol_tokenizer.batch_decode(ligand_tokens, skip_special_tokens=True)

    print("\n--- Generated Ligands ---")
    for i, smiles in enumerate(generated_smiles):
        # The decoding might include padding or other artifacts, so we clean it up
        cleaned_smiles = smiles.replace(" ", "").strip()
        print(f"Sample {i+1}: {cleaned_smiles}")
    print("-------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a ligand conditioned on a protein sequence using a pre-trained SEDD model.")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the directory containing the pre-trained model checkpoint (e.g., 'checkpoints/checkpoint_10.pth')."
    )
    parser.add_argument(
        "--protein_sequence", 
        type=str, 
        required=True, 
        help="Amino acid sequence of the target protein."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1, 
        help="Number of ligands to generate."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=512, 
        help="Number of steps for the diffusion sampling process. More steps can lead to better quality."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run the generation on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--mol_model_id", 
        type=str, 
        default="ibm/MoLFormer-XL-both-10pct", 
        help="Hugging Face model ID for the molecule embedder (MoLFormer)."
    )
    parser.add_argument(
        "--prot_model_id", 
        type=str, 
        default="facebook/esm2_t30_150M_UR50D", 
        help="Hugging Face model ID for the protein embedder (ESM2)."
    )

    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()