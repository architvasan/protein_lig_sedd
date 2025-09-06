import torch
from tqdm import tqdm

# --- 核心导入 ---
from protlig_dd.model.transformers_prot_lig import ProteinLigandSharedDiffusion
from protlig_dd.utils.utils import Config
from protlig_dd.data.tokenize import Tok_Mol
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
from protlig_dd.model.ema import ExponentialMovingAverage
import protlig_dd.sampling.sampling as sampling # <-- 关键：直接使用项目的采样模块
from protlig_dd.model import utils as mutils # <-- 关键：复用 model utils

@torch.no_grad()
def conditional_sampler_fn(
    model: ProteinLigandSharedDiffusion, 
    sampler, # The sampler function created by sampling.py
    protein_sequence: str,
    task: str = "ligand_given_protein"
    ):
    """
    A wrapper that adapts the project's sampler to perform conditional sampling.

    Args:
        model: The ProteinLigandSharedDiffusion model.
        sampler: The sampling function created by `sampling.get_pc_sampler`.
        protein_sequence (str): The protein sequence to condition on.
        task (str): The generation task.

    Returns:
        torch.Tensor: A tensor of the generated token IDs.
    """
    
    model.eval()

    # --- This is the key adaptation ---
    # The original sampling_score_fn in the sampler doesn't accept extra arguments.
    # We need to create a NEW score function that does, and "inject" it into the sampler.

    # 1. Create a conditional score function
    def conditional_score_fn(x, sigma):
        # This function will be called by the sampler's internal loop.
        # We pass the fixed conditional arguments here.
        # Note: The model's forward pass expects a list/batch of sequences.
        return model(
            noisy_tokens=x,
            timesteps=sigma,
            prot_seq=[protein_sequence] * x.shape[0], # Pass the protein sequence
            lig_seq=None,
            task=task
        )
    
    # 2. The sampler from sampling.py is a closure. We can't easily modify its
    #    internal score_fn. Instead, we can re-implement the sampling loop
    #    from get_pc_sampler, but using our new conditional_score_fn.
    #    This is the most robust way.

    # Extract components from the sampler's closure (if possible) or re-create them.
    # For simplicity and robustness, we re-create the components here.
    graph = sampler.__closure__[0].cell_contents
    noise = sampler.__closure__[1].cell_contents
    predictor = sampler.__closure__[3].cell_contents
    denoiser = sampler.__closure__[4].cell_contents
    steps = sampler.__closure__[5].cell_contents
    eps = sampler.__closure__[6].cell_contents
    device = sampler.__closure__[7].cell_contents
    
    # Get the target shape from the sampler's configuration
    batch_dims = sampler.__closure__[2].cell_contents
    
    # --- Re-implementing the pc_sampler loop from sampling.py ---
    x = graph.sample_limit(*batch_dims).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    for i in tqdm(range(steps), desc=f"Conditional Denoising for ligand"):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        
        # We need to adapt the score function for the predictor
        # The predictor expects a function of (x, t)
        # We wrap our conditional function
        def wrapped_score_fn_for_predictor(x, t_sigma):
             # The base score_fn from sampling.py expects the raw model output (logits)
             # and then it applies .exp(). Our model already outputs logits.
             logits = conditional_score_fn(x, t_sigma)
             return logits.exp() # Return the true score as expected by the predictor
        
        # The predictor's update_fn expects a score_fn, x, t, and step_size
        x = predictor.update_fn(wrapped_score_fn_for_predictor, x, t, dt)
        
    # Final denoising step
    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    # The denoiser also needs a compatible score function
    def wrapped_score_fn_for_denoiser(x, t_sigma):
        return conditional_score_fn(x, t_sigma).exp()
        
    x = denoiser.update_fn(wrapped_score_fn_for_denoiser, x, t)
    
    return x

# ==============================================================================
#                                  Main Program
# ==============================================================================
if __name__ == '__main__':
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config(yamlfile = '../../configs/config_ddp.yaml')
    config = config
    model = ProteinLigandSharedDiffusion(config).to('cuda')

    ligand_vocab_size = 2364 # As per ProteinLigandSharedDiffusion init
    graph = graph_lib.get_graph(config, DEVICE, tokens=ligand_vocab_size)
    noise = noise_lib.get_noise(config).to(DEVICE)
    
    # We define the shape of the output we want
    batch_size = 1
    target_length = config.data.max_ligand_len
    sampling_shape = (batch_size, target_length)
    
    # Use the official factory function to get the sampler
    sampler = sampling.get_sampling_fn(
        config=config,
        graph=graph,
        noise=noise,
        batch_dims=sampling_shape,
        eps=1e-5, # A small value for the final timestep
        device=DEVICE
    )
    print("-> Sampler built successfully.")

    # --- 3. Run the Conditional Sampling Task ---
    TARGET_PROTEIN = "MKKFFDSRREQGGSGLGSGSSGGSVLVNGGPLPALLLLLLSAGLEAGGQRGK"
    
    # The model might be untrained, but the full pipeline will run.
    # We need to use the ema handler for the correct API, even if it's not trained.
    ema_handler = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
    ema_handler.copy_to(model.parameters())

    generated_ligand_tokens = conditional_sampler_fn(
        model=model,
        sampler=sampler,
        protein_sequence=TARGET_PROTEIN,
        task="ligand_given_protein"
    )
    
    ema_handler.restore(model.parameters())

    # --- 4. Decode the Output ---
    print("\n--- Decoding the generated ligand ---")
    mol_tokenizer = Tok_Mol(model_id=config.model.mol_emb_id).tokenizer
    
    smiles_output = mol_tokenizer.decode(generated_ligand_tokens[0], skip_special_tokens=True)
    smiles_cleaned = smiles_output.replace(" ", "")

    print(f"Conditioning Protein (start): {TARGET_PROTEIN[:50]}...")
    print(f"Generated Ligand (SMILES): {smiles_cleaned}")