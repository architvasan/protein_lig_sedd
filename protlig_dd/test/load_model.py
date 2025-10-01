"""
Unit test for loading SMILES diffusion model.
Based on protlig_dd/training/run_train_smiles.py
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import required modules (following run_train_smiles.py)
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
import protlig_dd.utils.utils as utils
from protlig_dd.data.tokenize import Tok_Mol, Tok_SmilesPE
from protlig_dd.model.ema import ExponentialMovingAverage


class SMILESModelLoader:
    """Load SMILES diffusion model for testing, based on SMILESTrainer setup."""

    def __init__(self, config_path: str, device: torch.device = None):
        self.config_path = config_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self.cfg = self._load_config()

        # Initialize components
        self.model = None
        self.graph = None
        self.noise = None
        self.ema = None
        self.tokenizer = None

    def _load_config(self):
        """Load configuration from YAML file."""
        print(f"Loading config from: {self.config_path}")

        # Use the existing Config class from utils
        cfg = utils.Config(yamlfile=self.config_path)

        print(f"✅ Config loaded successfully")
        print(f"   Model type: {cfg.model.type}")
        print(f"   Tokens: {cfg.tokens}")
        print(f"   Device: {self.device}")

        return cfg

    def setup_tokenizer(self, tokenizer_type: str = "molformer"):
        """Setup tokenizer (MoLFormer or SmilesPE)."""
        print(f"Setting up {tokenizer_type} tokenizer...")

        if tokenizer_type.lower() == "molformer":
            self.tokenizer = Tok_Mol()
            print("✅ MoLFormer tokenizer initialized")
        elif tokenizer_type.lower() == "smilespe":
            self.tokenizer = Tok_SmilesPE()
            print("✅ SmilesPE tokenizer initialized")
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    def setup_model(self):
        """Setup model components (graph, noise, model, EMA)."""
        print("Setting up model for SMILES...")

        # Build graph for absorbing diffusion
        self.graph = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.tokens)
        print("✅ Graph initialized")

        # Build noise schedule
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        print("✅ Noise schedule initialized")

        # Build model - use SMILES-only model (following run_train_smiles.py)
        from protlig_dd.model.transformer_v100 import SEDD
        self.model = SEDD(self.cfg).to(self.device)
        print("✅ SEDD model initialized")

        # Setup EMA
        self.ema = ExponentialMovingAverage(
            self.model.parameters(),
            decay=self.cfg.training.ema
        )
        print("✅ EMA initialized")

        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model ready:")
        print(f"   Total parameters: {num_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {trainable_params / 1_000_000:.2f}M parameters")

        return self.model

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model state loaded")

        # Load EMA state if available
        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            print("✅ EMA state loaded")

        # Print checkpoint info
        if 'step' in checkpoint:
            print(f"   Checkpoint step: {checkpoint['step']}")
        if 'epoch' in checkpoint:
            print(f"   Checkpoint epoch: {checkpoint['epoch']}")

        return checkpoint

    def compute_loss(self, batch):
        """
        Compute diffusion loss following the pattern from run_train_smiles.py.

        Args:
            batch: Input batch of token IDs [batch_size, sequence_length]

        Returns:
            loss: Computed diffusion loss
        """
        if self.model is None or self.graph is None or self.noise is None:
            raise RuntimeError("Model components not initialized. Call setup_model() first.")

        # Ensure batch is 2D: [batch_size, sequence_length]
        if batch.dim() != 2:
            if batch.dim() > 2:
                batch = batch.view(batch.shape[0], -1)
            else:
                raise ValueError(f"Batch must be at least 2D, got {batch.dim()}D with shape {batch.shape}")

        # Sample timesteps
        t = torch.rand(batch.shape[0], device=self.device) * (1 - 1e-3) + 1e-3
        sigma, dsigma = self.noise(t)

        # Sample transition (add noise to the data)
        perturbed_batch = self.graph.sample_transition(batch, sigma[:, None])

        # Fix perturbed_batch shape if needed - reshape to 2D before validation
        if perturbed_batch.dim() > 2:
            perturbed_batch = perturbed_batch.view(perturbed_batch.shape[0], -1)

            # Also reshape batch to match perturbed_batch dimensions
            if batch.shape[1] != perturbed_batch.shape[1]:
                batch = batch.view(batch.shape[0], -1)
                # If still doesn't match, pad or truncate
                if batch.shape[1] != perturbed_batch.shape[1]:
                    target_len = perturbed_batch.shape[1]
                    if batch.shape[1] < target_len:
                        # Pad batch to match perturbed_batch
                        padding = target_len - batch.shape[1]
                        batch = torch.cat([batch, torch.zeros(batch.shape[0], padding, device=batch.device, dtype=batch.dtype)], dim=1)
                    else:
                        # Truncate batch to match perturbed_batch
                        batch = batch[:, :target_len]

        # Validate perturbed_batch is now 2D: [batch_size, sequence_length]
        if perturbed_batch.dim() != 2:
            raise ValueError(f"perturbed_batch must be 2D [batch_size, seq_len], got {perturbed_batch.dim()}D with shape {perturbed_batch.shape}")

        # Forward pass with device-aware autocast
        device_type = str(self.device).split(':')[0]
        if device_type == 'cuda':
            with torch.amp.autocast('cuda'):
                sigma_expanded = sigma[:, None].expand(-1, batch.shape[1])
                log_score = self.model(perturbed_batch, sigma)
                loss = self.graph.score_entropy(log_score, sigma_expanded, perturbed_batch, batch)
                # Weight by dsigma for better training dynamics
                loss = (dsigma[:, None] * loss).mean()
        else:
            # No autocast for CPU/MPS
            log_score = self.model(perturbed_batch, sigma)
            loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            # Weight by dsigma for better training dynamics
            loss = (dsigma[:, None] * loss).mean()

        return loss

    def test_loss_computation(self, batch_size: int = 2, seq_length: int = 128):
        """Test loss computation with dummy data."""
        print(f"Testing loss computation (batch_size={batch_size}, seq_length={seq_length})...")

        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")

        # Create dummy input (random token IDs)
        dummy_batch = torch.randint(0, self.cfg.tokens, (batch_size, seq_length), device=self.device)

        # Compute loss
        self.model.train()  # Set to training mode for loss computation
        try:
            loss = self.compute_loss(dummy_batch)
            print(f"✅ Loss computation successful")
            print(f"   Input shape: {dummy_batch.shape}")
            print(f"   Loss value: {loss.item():.6f}")
            print(f"   Loss dtype: {loss.dtype}")
            return loss
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            raise

    def test_forward_pass(self, batch_size: int = 2, seq_length: int = 128):
        """Test forward pass with dummy data."""
        print(f"Testing forward pass (batch_size={batch_size}, seq_length={seq_length})...")

        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")

        # Create dummy input (random token IDs)
        dummy_input = torch.randint(0, self.cfg.tokens, (batch_size, seq_length), device=self.device)
        dummy_timesteps = torch.rand(batch_size, device=self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(dummy_input, dummy_timesteps)
                print(f"✅ Forward pass successful")
                print(f"   Input shape: {dummy_input.shape}")
                print(f"   Output shape: {output.shape}")
                print(f"   Output dtype: {output.dtype}")
                return output
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                raise

    def get_model_info(self):
        """Get detailed model information."""
        if self.model is None:
            return "Model not initialized"

        info = {
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'config': {
                'hidden_size': self.cfg.model.hidden_size,
                'n_blocks_lig': self.cfg.model.n_blocks_lig,
                'n_heads': self.cfg.model.n_heads,
                'vocab_size': self.cfg.tokens,
                'max_length': self.cfg.model.length,
            }
        }
        return info


def load_fresh_model(config_path: str, device: torch.device = None, tokenizer_type: str = "molformer"):
    """
    Load a fresh (untrained) SMILES model.

    Args:
        config_path: Path to config YAML file
        device: Device to load model on
        tokenizer_type: "molformer" or "smilespe"

    Returns:
        tuple: (model_loader, model, config)
    """
    loader = SMILESModelLoader(config_path, device)
    loader.setup_tokenizer(tokenizer_type)
    model = loader.setup_model()

    return loader, model, loader.cfg


def load_model_from_checkpoint(config_path: str, checkpoint_path: str, device: torch.device = None):
    """
    Load model from checkpoint.

    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        tuple: (model_loader, model, config, checkpoint)
    """
    loader = SMILESModelLoader(config_path, device)
    loader.setup_tokenizer()
    loader.setup_model()
    checkpoint = loader.load_checkpoint(checkpoint_path)

    return loader, loader.model, loader.cfg, checkpoint


if __name__ == '__main__':
    print("=" * 80)
    print("SMILES Model Loading Unit Test")
    print("=" * 80)

    # Configuration
    CONFIG_FILE_PATH = '/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/configs/config_pubchem_smiles.yaml'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Test 1: Load fresh model
        print("\n🧪 Test 1: Loading fresh model...")
        loader, model, config = load_fresh_model(
            config_path=CONFIG_FILE_PATH,
            device=DEVICE,
            tokenizer_type="molformer"
        )

        # Test 2: Model info
        print("\n🧪 Test 2: Model information...")
        info = loader.get_model_info()
        for key, value in info.items():
            if key != 'config':
                print(f"   {key}: {value}")
        print("   Config:")
        for k, v in info['config'].items():
            print(f"     {k}: {v}")

        # Test 3: Forward pass
        print("\n🧪 Test 3: Forward pass test...")
        output = loader.test_forward_pass(batch_size=2, seq_length=64)

        # Test 4: Loss computation
        print("\n🧪 Test 4: Loss computation test...")
        loss = loader.test_loss_computation(batch_size=2, seq_length=64)

        print("\n✅ All tests passed!")
        print(f"Model loaded successfully with {info['trainable_params']:,} parameters")
        print(f"Forward pass output shape: {output.shape}")
        print(f"Loss computation successful: {loss.item():.6f}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()