import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO

# --- Configuration: PLEASE EDIT THESE VALUES ---

# 1. Path to your model checkpoint (use the pre-trained one for this test!)
CHECKPOINT_PATH = "/homes/xlian/esm/facebook_esm2_t33_650M_UR50D" # Example: using the official pre-trained model from Hugging Face
# Your local path:
CHECKPOINT_PATH = "/homes/xlian/esm/riseke_finetune/esm2-650m-noeval-2gpu-4/checkpoint-16000/"

# 2. List of FASTA files you want to analyze
FASTA_FILES = [
    #"../chimeric_sequences.fasta",
    "../../cdhit_filtered/enzyme_a_clustered.fasta", 
    # Add more file paths here
]

# 5. Output image file name
OUTPUT_IMAGE = "../mlm_distribution.png"

# 3. BATCH_SIZE for processing. Adjust based on your GPU memory.
BATCH_SIZE = 8

# 4. Fraction of tokens to mask for the MLM task
MASK_FRACTION = 0.15

# --- End of Configuration ---


def parse_fasta(file_path):
    """Parses a FASTA file using Biopython, yields (header, sequence) tuples."""
    if not Path(file_path).is_file():
        print(f"Warning: FASTA file not found at {file_path}. Skipping.")
        return
    for record in SeqIO.parse(file_path, "fasta"):
        yield record.id, str(record.seq)

def calculate_mlm_pseudo_perplexity(sequences, model, tokenizer, device, batch_size=8, mask_fraction=0.15):
    """
    Calculates a score based on the model's native MLM task.
    This is often called a "pseudo-perplexity" or "MLM score".
    A lower score indicates the model is more confident in the sequence's structure.
    """
    scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Calculating MLM Score"):
            batch = sequences[i:i+batch_size]
            
            # 1. Tokenize the original sequences to get the ground truth labels
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            labels = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # 2. Create the masked input
            masked_input_ids = labels.clone()
            
            # Find positions that are actual amino acids (not special tokens)
            is_token = (labels != tokenizer.cls_token_id) & \
                       (labels != tokenizer.sep_token_id) & \
                       (labels != tokenizer.pad_token_id)
            
            # For each sequence, randomly select positions to mask
            # `torch.rand` creates a tensor of random numbers between 0 and 1
            # We mask where the random number is less than our fraction
            probability_matrix = torch.full(labels.shape, mask_fraction, device=device)
            masked_indices = torch.bernoulli(probability_matrix).bool() & is_token
            
            # Replace selected positions with the mask token
            masked_input_ids[masked_indices] = tokenizer.mask_token_id

            # 3. Get model's predictions (logits) for the masked input
            outputs = model(masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 4. Calculate the loss ONLY at the masked positions
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
            loss = loss.view(labels.size())

            # Sum the loss only for the tokens that were masked
            sum_masked_loss = torch.sum(loss * masked_indices, dim=1)
            
            # Count how many tokens were actually masked in each sequence
            num_masked = torch.sum(masked_indices, dim=1)
            
            # Avoid division by zero for sequences where no token was masked
            num_masked = torch.max(num_masked, torch.tensor(1.0).to(device))
            
            # Calculate the average loss over the masked positions
            mean_nll_masked = sum_masked_loss / num_masked
            
            # The "pseudo-perplexity" is the exponential of this loss
            pseudo_ppl = torch.exp(mean_nll_masked)
            
            scores.extend(pseudo_ppl.cpu().numpy())
            
    return scores

# Process each FASTA file
def compute_list_fasta_files(fasta_files, model, tokenizer, device, batch_size, mask_fraction):
    results = {}
    for fasta_file in fasta_files:
        print(f"\n--- Processing file: {fasta_file} ---")
        sequences = [seq for header, seq in parse_fasta(fasta_file)]
        
        if not sequences:
            print(f"No sequences found in {fasta_file}. Skipping.")
            continue
        
        print(f"Found {len(sequences)} sequences.")
        
        # Calculate scores using the new MLM method
        mlm_scores = calculate_mlm_pseudo_perplexity(
            sequences, model, tokenizer, device, 
            batch_size=batch_size, mask_fraction=mask_fraction
        )
        results[fasta_file] = mlm_scores
    return results

def compute_list_models(fasta_file, model_list, tokenizer, device, batch_size, mask_fraction):
    results = {}
    for model_name, model in model_list.items():
        print(f"\n--- Processing with model: {model_name} ---")
        sequences = [seq for header, seq in parse_fasta(fasta_file)]
        
        if not sequences:
            print(f"No sequences found in {fasta_file}. Skipping.")
            continue
        
        print(f"Found {len(sequences)} sequences.")
        
        # Calculate scores using the new MLM method
        mlm_scores = calculate_mlm_pseudo_perplexity(
            sequences, model, tokenizer, device, 
            batch_size=batch_size, mask_fraction=mask_fraction
        )
        results[model_name] = mlm_scores
    return results


def plot_distributions(results_dict, output_path):
    """
    Plots score distributions from multiple files or models on one histogram.
    If keys look like file paths, labels by file; if keys look like model names, labels by model.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Heuristics: if all keys are paths, treat as FASTA files; else, treat as model names
    keys = list(results_dict.keys())
    if all("/" in k or "\\" in k for k in keys):
        label_type = "FASTA File"
        label_func = lambda k: Path(k).name
    else:
        label_type = "Model"
        label_func = lambda k: k

    max_val = 0
    for scores in results_dict.values():
        if scores:
            max_val = max(max_val, np.percentile(scores, 99.5)) 

    bins = np.linspace(0, max_val, 60) if max_val > 0 else 60

    for key, scores in results_dict.items():
        if not scores:
            continue
        
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        
        label = f"{label_func(key)}\n(N={len(scores)}, Mean={mean_score:.2f}, Median={median_score:.2f})"
        ax.hist(scores, bins=bins, alpha=0.6, label=label, density=True)

    ax.set_title("Distribution of MLM Pseudo-Perplexity Scores", fontsize=16, fontweight='bold')
    ax.set_xlabel("MLM PPL Score (Lower is more 'natural')", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title=label_type, fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to '{output_path}'")
    plt.show()


if __name__ == "__main__":
    # Set up device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ESM model and tokenizer
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = EsmForMaskedLM.from_pretrained(CHECKPOINT_PATH).to(device)
    model.eval() # Set model to evaluation mode

    all_results = {}

    '''
    all_results = compute_list_fasta_files(
        FASTA_FILES, model, tokenizer, device, 
        batch_size=BATCH_SIZE, mask_fraction=MASK_FRACTION
    )
    '''

    all_results = compute_list_models(
        FASTA_FILES[0],
        {
            "/homes/xlian/esm/facebook_esm2_t33_650M_UR50D": EsmForMaskedLM.from_pretrained("/homes/xlian/esm/facebook_esm2_t33_650M_UR50D").to(device),
            "/homes/xlian/esm/riseke_finetune/esm2-650m-noeval-2gpu-4/checkpoint-16000/": EsmForMaskedLM.from_pretrained("/homes/xlian/esm/riseke_finetune/esm2-650m-noeval-2gpu-4/checkpoint-16000/").to(device)
        },
        tokenizer,
        device,
        batch_size=BATCH_SIZE,
        mask_fraction=MASK_FRACTION
    )

    # Plot the results
    if all_results:
        plot_distributions(all_results, OUTPUT_IMAGE)

    else:
        print("\nNo data was processed. No plot will be generated.")