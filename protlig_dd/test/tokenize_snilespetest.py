import torch
from protlig_dd.data.tokenize import Tok_SmilesPE

def tokenize_smiles_list(smiles_list, max_length=128, pad_token_id=0):
    """
    Tokenize a list of SMILES strings individually using SMILESpe tokenizer.
    
    Args:
        smiles_list: List of SMILES strings
        max_length: Maximum sequence length for padding/truncation
        pad_token_id: Token ID to use for padding
    
    Returns:
        torch.Tensor: Tensor of shape [len(smiles_list), max_length]
    """
    # Initialize tokenizer
    tokenizer = Tok_SmilesPE()
    
    tokenized_sequences = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            # Tokenize individual SMILES string
            # Pass as single-item list to maintain expected input format
            tokens = tokenizer.tokenize([smiles])['input_ids']
            
            # Extract the sequence (remove batch dimension if present)
            if tokens.dim() > 1:
                sequence = tokens.squeeze(0)
            else:
                sequence = tokens
            
            # Convert to list for easier manipulation
            sequence = sequence.tolist() if torch.is_tensor(sequence) else sequence
            
            # Truncate if too long
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
                print(f"Warning: SMILES {i} truncated from {len(sequence)} to {max_length} tokens")
            
            # Pad if too short
            while len(sequence) < max_length:
                sequence.append(pad_token_id)
            
            tokenized_sequences.append(sequence)
            
        except Exception as e:
            print(f"Error tokenizing SMILES {i} ('{smiles}'): {e}")
            # Create a sequence of padding tokens for failed tokenization
            sequence = [pad_token_id] * max_length
            tokenized_sequences.append(sequence)
    
    # Convert to tensor
    tensor = torch.tensor(tokenized_sequences, dtype=torch.long)
    
    print(f"Tokenized {len(smiles_list)} SMILES into tensor of shape {tensor.shape}")
    return tensor

# Example usage:
if __name__ == "__main__":
    # Example SMILES list
    smiles_list = [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CCN(CC)CC",  # triethylamine
    ]
    
    # Tokenize the SMILES
    tokenized_tensor = tokenize_smiles_list(smiles_list, max_length=64)
    
    print("Tokenized tensor shape:", tokenized_tensor.shape)
    print("First few tokens of each SMILES:")
    for i, tokens in enumerate(tokenized_tensor):
        non_zero_tokens = tokens[tokens != 0][:10]  # Show first 10 non-padding tokens
        print(f"SMILES {i} ({smiles_list[i]}): {non_zero_tokens.tolist()}")