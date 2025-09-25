"""
Optimized training script for UniRef50 with improved stability and efficiency.
"""
from mpi4py import MPI
import os, socket
import torch
### Import intel_extension_for_pytorch for running on aurora
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as torch_ccl
except:
    print("intel_extension_for_pytorch not found, running on CPU/cuda")
import datetime
import os
import sys
import gc
import time
# Add the project root to Python path to ensure protlig_dd module can be found
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")
from itertools import chain
import wandb
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from collections import OrderedDict
from transformers import GPT2TokenizerFast
import yaml
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import protlig_dd.processing.losses as losses
import protlig_dd.sampling.sampling as sampling
import protlig_dd.processing.graph_lib as graph_lib
import protlig_dd.processing.noise_lib as noise_lib
import protlig_dd.utils.utils as utils
from protlig_dd.data.data import get_dataloaders
from protlig_dd.model.transformers_protlig_cross_v100_optimized import OptimizedTransformerBlock
from protlig_dd.model.ema import ExponentialMovingAverage
from protlig_dd.utils.lr_scheduler import WarmupCosineLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
### Import torch.distributed for running on aurora
import torch.distributed as dist
import argparse
import datetime

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use ccl backend for Intel XPU instead of nccl
    dist.init_process_group("ccl", rank=rank, world_size=world_size)
    torch.xpu.set_device(rank)

def setup_ddp_polaris(rank, world_size):
    # DDP: Set environmental variables used by PyTorch
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    LOCAL_RANK = os.environ.get('PMI_LOCAL_RANK')
    local_rank = LOCAL_RANK
    print(f"{RANK=}")
    print(f"{LOCAL_RANK=}")
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR 
    os.environ['MASTER_PORT'] = str(2345)
    print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

    # DDP: initialize distributed communication with nccl backend
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))

    print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}, Local rank: {local_rank}")
    print(dist)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    print("DDPPP")
    return dist.get_rank(), device_id, dist.get_world_size()

class ProteinTokenizer:
    """Protein tokenizer based on the download script."""

    def __init__(self, vocab_file=None, merges_file=None):
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

        if vocab_file and merges_file:
            # Load existing tokenizer
            self.tokenizer = GPT2TokenizerFast(
                vocab_file=vocab_file,
                merges_file=merges_file,
                bos_token='<s>',
                eos_token='</s>',
                unk_token='<unk>',
                pad_token='<pad>',
                mask_token='<mask>'
            )
        else:
            # Create new tokenizer
            self.tokenizer = self._create_tokenizer()

    def _create_tokenizer(self):
        """Create protein tokenizer."""
        all_tokens = self.special_tokens + self.amino_acids
        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))

        # Create temporary files
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vocab, f)
            vocab_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('#version: 0.2\n')
            merges_file = f.name

        tokenizer = GPT2TokenizerFast(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token='<s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )

        # Clean up temp files
        os.unlink(vocab_file)
        os.unlink(merges_file)

        return tokenizer

    def tokenize_sequence(self, sequence, max_length=512):
        """Tokenize a single protein sequence."""
        # Truncate if needed
        if len(sequence) > max_length - 2:
            sequence = sequence[:max_length - 2]

        # Tokenize
        tokens = self.tokenizer.encode(sequence, add_special_tokens=True)

        # Pad
        if len(tokens) < max_length:
            tokens.extend([self.tokenizer.pad_token_id] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)


class UniRef50Dataset(torch.utils.data.Dataset):
    """Dataset class for processed UniRef50 data - supports both tokenized and untokenized data."""

    def __init__(self, data_file, tokenize_on_fly=False, max_length=512, use_streaming=False):
        self.data_file = data_file
        self.tokenize_on_fly = tokenize_on_fly
        self.max_length = max_length
        self.use_streaming = use_streaming

        if use_streaming:
            print(f"🌊 Using streaming mode for large file: {data_file}")
            # For streaming, we need to get the length without loading all data
            self._get_streaming_info()
        else:
            # Try memory mapping first for large files
            file_size_gb = os.path.getsize(data_file) / (1024**3)
            if file_size_gb > 50:  # If file is larger than 50GB
                print(f"📁 Large file detected ({file_size_gb:.1f}GB). Using memory mapping...")
                try:
                    self.data = torch.load(data_file, weights_only=False, map_location='cpu', mmap=True)
                except Exception as e:
                    print(f"⚠️  Memory mapping failed: {e}")
                    print("🌊 Falling back to streaming mode...")
                    self.use_streaming = True
                    self._get_streaming_info()
                    return
            else:
                print(f"📁 Loading file ({file_size_gb:.1f}GB) into memory...")
                self.data = torch.load(data_file, weights_only=False)

        if not self.use_streaming:
            self._check_tokenization_format()

    def _get_streaming_info(self):
        """Get dataset info for streaming mode without loading all data."""
        print("🔍 Analyzing file structure for streaming...")

        # Load just a small sample to understand the format
        try:
            # Try to load just the first few items
            sample_data = []
            with open(self.data_file, 'rb') as f:
                import pickle
                try:
                    # Try to load incrementally
                    for i in range(min(10, 1000)):  # Load first 10 items for analysis
                        try:
                            item = pickle.load(f)
                            sample_data.append(item)
                        except EOFError:
                            break
                except:
                    # If incremental loading fails, try loading the whole thing and take a sample
                    f.seek(0)
                    full_data = pickle.load(f)
                    if isinstance(full_data, list) and len(full_data) > 0:
                        sample_data = full_data[:10]
                        self.total_length = len(full_data)
                        print(f"📊 Detected {self.total_length} total sequences")
                    else:
                        raise ValueError("Unknown data format for streaming")

            if sample_data:
                self.sample_data = sample_data
                self._check_tokenization_format_streaming(sample_data)
                if not hasattr(self, 'total_length'):
                    # If we couldn't get total length, we'll need to count
                    print("⚠️  Could not determine total length. Will count during training.")
                    self.total_length = None
            else:
                raise ValueError("Could not load any sample data")

        except Exception as e:
            print(f"❌ Error setting up streaming: {e}")
            raise ValueError(f"Cannot setup streaming for file {self.data_file}: {e}")

    def _check_tokenization_format_streaming(self, sample_data):
        """Check tokenization format for streaming mode."""
        if len(sample_data) > 0:
            sample = sample_data[0]
            if isinstance(sample, dict) and 'prot_tokens' in sample:
                self.is_tokenized = True
                print(f"✅ Detected pre-tokenized data (streaming mode)")
            elif isinstance(sample, dict) and ('protein_seq' in sample or 'sequence' in sample):
                self.is_tokenized = False
                print(f"✅ Detected untokenized data (streaming mode)")
                if self.tokenize_on_fly:
                    self.tokenizer = ProteinTokenizer()
                    print("✅ Tokenizer initialized for streaming")
            elif isinstance(sample, str):
                self.is_tokenized = False
                print(f"✅ Detected raw sequence strings (streaming mode)")
                if self.tokenize_on_fly:
                    self.tokenizer = ProteinTokenizer()
                    print("✅ Tokenizer initialized for streaming")
            else:
                raise ValueError(f"Unknown data format in streaming mode: {type(sample)}")

    def _check_tokenization_format(self):
        """Check tokenization format for regular mode."""
        # Check if data is already tokenized
        if len(self.data) > 0:
            sample = self.data[0]
            if isinstance(sample, dict) and 'prot_tokens' in sample:
                self.is_tokenized = True
                print(f"Loaded {len(self.data)} pre-tokenized sequences from {self.data_file}")
            elif isinstance(sample, dict) and ('protein_seq' in sample or 'sequence' in sample):
                self.is_tokenized = False
                print(f"Loaded {len(self.data)} untokenized sequences from {self.data_file}")
                if self.tokenize_on_fly:
                    self.tokenizer = ProteinTokenizer()
                    print("✅ Tokenizer initialized for on-the-fly tokenization")
            elif isinstance(sample, str):
                self.is_tokenized = False
                print(f"Loaded {len(self.data)} raw sequence strings from {self.data_file}")
                if self.tokenize_on_fly:
                    self.tokenizer = ProteinTokenizer()
                    print("✅ Tokenizer initialized for on-the-fly tokenization")
            else:
                raise ValueError(f"Unknown data format in {self.data_file}. Expected dict with 'prot_tokens' or 'protein_seq'/'sequence', or raw strings.")
        else:
            raise ValueError(f"Empty dataset loaded from {self.data_file}")

    def __len__(self):
        if self.use_streaming:
            if self.total_length is not None:
                return self.total_length
            else:
                # If we don't know the length, we need to count (expensive)
                print("⚠️  Counting sequences in large file (this may take a while)...")
                count = 0
                try:
                    with open(self.data_file, 'rb') as f:
                        import pickle
                        data = pickle.load(f)
                        if isinstance(data, list):
                            count = len(data)
                        else:
                            count = 1
                    self.total_length = count
                    return count
                except Exception as e:
                    print(f"❌ Error counting sequences: {e}")
                    return 1000000  # Fallback estimate
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.use_streaming:
            return self._get_streaming_item(idx)
        else:
            sample = self.data[idx]

            if self.is_tokenized:
                # Return pre-tokenized data
                return sample['prot_tokens']
            else:
                # Handle untokenized data
                if not self.tokenize_on_fly:
                    raise ValueError("Data is not tokenized but tokenize_on_fly=False. Set tokenize_on_fly=True or use pre-tokenized data.")

                # Extract sequence
                if isinstance(sample, dict):
                    if 'protein_seq' in sample:
                        sequence = sample['protein_seq']
                    elif 'sequence' in sample:
                        sequence = sample['sequence']
                    else:
                        raise ValueError(f"Dict sample missing 'protein_seq' or 'sequence' key: {list(sample.keys())}")
                elif isinstance(sample, str):
                    sequence = sample
                else:
                    raise ValueError(f"Unknown sample type: {type(sample)}")

                # Tokenize on the fly
                return self.tokenizer.tokenize_sequence(sequence, self.max_length)

    def _get_streaming_item(self, idx):
        """Get item in streaming mode - loads data on demand."""
        try:
            # For streaming, we need to load the specific item
            # This is a simplified implementation - for very large files,
            # you might want to implement more sophisticated caching

            if not hasattr(self, '_cached_data') or self._cached_data is None:
                # Load the full data (this is still the bottleneck for very large files)
                print(f"🔄 Loading data for streaming access...")
                with open(self.data_file, 'rb') as f:
                    import pickle
                    self._cached_data = pickle.load(f)
                print(f"✅ Data loaded for streaming")

            sample = self._cached_data[idx]

            if self.is_tokenized:
                return sample['prot_tokens']
            else:
                if not self.tokenize_on_fly:
                    raise ValueError("Data is not tokenized but tokenize_on_fly=False.")

                # Extract sequence
                if isinstance(sample, dict):
                    if 'protein_seq' in sample:
                        sequence = sample['protein_seq']
                    elif 'sequence' in sample:
                        sequence = sample['sequence']
                    else:
                        raise ValueError(f"Dict sample missing sequence key: {list(sample.keys())}")
                elif isinstance(sample, str):
                    sequence = sample
                else:
                    raise ValueError(f"Unknown sample type: {type(sample)}")

                return self.tokenizer.tokenize_sequence(sequence, self.max_length)

        except Exception as e:
            print(f"❌ Error loading streaming item {idx}: {e}")
            raise

def safe_getattr(obj, path, default=None):
    """Safely get nested attributes with default fallback."""
    try:
        parts = path.split('.')
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current
    except AttributeError:
        return default

@dataclass
class OptimizedUniRef50Trainer:
    """
    Optimized trainer for UniRef50 with improved memory efficiency and training stability.
    """
    work_dir: str
    config_file: str
    datafile: str = './input_data/processed_uniref50.pt'
    rank: int| None = 0
    world_size: int| None = 1
    dev_id: str = 'xpu:0'
    seed: int = 42
    resume_checkpoint: Optional[str] = None
    force_fresh_start: bool = False
    sampling_method: str = "rigorous"  # "rigorous" or "simple"
    epochs_override: Optional[int] = None  # Override epochs for hyperparameter sweeps
    use_ddp: bool = True
    use_wandb: bool = True  # Toggle wandb logging on/off
    minimal_mode: bool = False  # Minimal mode for large-scale DDP debugging

    def __post_init__(self):
        """Initialize trainer components."""
        print(f"Trainer initialized. Device: {self.dev_id}")

        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.use_ddp:
            self.rank, self.device, self.world_size = setup_ddp_polaris(self.rank, self.world_size)
            #self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device(self.dev_id)
        print(f"{self.rank=}")
        print(f"{self.device=}")
        # Setup device with cross-platform compatibility
        #self.device = self.setup_device(self.rank)
        #print(f"✅ Using device: {self.device}")

        # Load configuration
        with open(self.config_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        # Convert to namespace for easier access using the Config class
        self.cfg = utils.Config(dictionary=cfg_dict)

        # Setup directories
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.sample_dir = os.path.join(self.work_dir, "samples")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        print(f"Config loaded from: {self.config_file}")

        # Setup logging
        self.setup_logging()

        # Initialize device-specific attributes
        device_type = str(self.device).split(':')[0]
        if device_type == 'xpu':
            self.use_amp = True
            self.scaler = None  # Will be initialized in setup_training
            print("✅ XPU mixed precision will be enabled")
        elif device_type == 'cuda':
            self.use_amp = True
            self.scaler = None  # Will be initialized in setup_training
            print("✅ CUDA mixed precision will be enabled")
        else:
            self.use_amp = False
            self.scaler = None
            print(f"✅ {device_type.upper()} training without mixed precision")

        # Verify key configuration sections exist
        required_sections = ['model', 'training', 'optim', 'data', 'noise', 'sampling']
        for section in required_sections:
            if not hasattr(self.cfg, section):
                print(f"⚠️  Warning: Missing config section '{section}' - using defaults")
                setattr(self.cfg, section, utils.Config(dictionary={}))

        # Log sampling method configuration
        print(f"🧬 Sampling method configured: {self.sampling_method}")
        if self.sampling_method == "rigorous":
            print(
                f"📊 Using CTMC sampling with {getattr(self.cfg.sampling, 'steps', 100)} steps\n"+\
                "🔧 Predictor: {getattr(self.cfg.sampling, 'predictor', 'euler')} \n"+\
                "🎯 Noise removal: {getattr(self.cfg.sampling, 'noise_removal', True)}"
                )
        else:
            print(f"   🎲 Using simple heuristic sampling")

        # Configure quick generation test frequency
        quick_gen_freq = getattr(self.cfg.training, 'quick_gen_freq', self.cfg.training.log_freq * 2)
        print(f"   🧪 Quick generation tests every {quick_gen_freq} steps")

    def setup_device(self, dev_id):
        """Setup device with cross-platform compatibility."""
        if dev_id == "auto":
            # Auto-detect best available device
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device("xpu:0")
                print("⚡ Auto-detected: Intel XPU")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
                print("🚀 Auto-detected: CUDA GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🍎 Auto-detected: Apple Silicon MPS")
            else:
                device = torch.device("cpu")
                print("💻 Auto-detected: CPU")
        elif dev_id == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🍎 Using Apple Silicon MPS")
            else:
                print("⚠️  MPS not available, falling back to CPU")
                device = torch.device("cpu")
        elif dev_id.startswith("xpu"):
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device(dev_id)
                print(f"⚡ Using Intel XPU: {dev_id}")
            else:
                print("⚠️  XPU not available, falling back to CPU")
                device = torch.device("cpu")
        elif dev_id.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device(dev_id)
                print(f"🚀 Using CUDA GPU: {dev_id}")
            else:
                print("⚠️  CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device(dev_id)
            print(f"💻 Using device: {dev_id}")

        return device
    
    def setup_data_loaders(self):
        """Setup optimized data loaders for UniRef50."""
        print("Setting up data loaders...")

        # Check if we have a custom dataset file (our processed UniRef50 data)
        if Path(self.datafile).exists():
            if self.use_ddp:
                print(f"Loading custom dataset {self.datafile} with ddp")
                self.setup_custom_data_loaders_ddp()
            else:
                print(f"Loading custom dataset: {self.datafile} sans ddp")
                self.setup_custom_data_loaders()
        else:
            print("Using standard data loading pipeline...")
            # Use the existing data loading infrastructure
            train_loader, val_loader = get_dataloaders(
                self.cfg,
                distributed=False
            )

            self.train_loader = train_loader
            self.val_loader = val_loader

        print(f"Data loaders ready.")

    def setup_custom_data_loaders(self):
        """Setup data loaders for our custom processed UniRef50 data."""
        from torch.utils.data import DataLoader, random_split

        # Load dataset with tokenization options
        max_length = safe_getattr(self.cfg, 'data.max_protein_len', 512)
        dataset = UniRef50Dataset(
            self.datafile,
            tokenize_on_fly=getattr(self, 'tokenize_on_fly', False),
            max_length=max_length,
            use_streaming=getattr(self, 'use_streaming', False)
        )

        # Split into train/val
        train_ratio = safe_getattr(self.cfg, 'data.train_ratio', 0.9)

        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Create data loaders
        batch_size = safe_getattr(self.cfg, 'training.batch_size', 16)
        num_workers = safe_getattr(self.cfg, 'training.num_workers', 0)  # Use 0 to avoid multiprocessing issues

        # Determine pin_memory based on device
        device_type = str(self.device).split(':')[0]
        pin_memory = device_type in ['cuda', 'xpu']  # Pin memory for CUDA and XPU

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    

    def setup_custom_data_loaders_ddp(self):
        """Setup data loaders for our custom processed UniRef50 data."""
        from torch.utils.data import DataLoader, random_split

        # Load dataset with tokenization options
        max_length = safe_getattr(self.cfg, 'data.max_protein_len', 512)
        dataset = UniRef50Dataset(
            self.datafile,
            tokenize_on_fly=getattr(self, 'tokenize_on_fly', False),
            max_length=max_length,
            use_streaming=getattr(self, 'use_streaming', False)
        )

        # Split into train/val
        train_ratio = safe_getattr(self.cfg, 'data.train_ratio', 0.9)

        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Create data loaders
        batch_size = safe_getattr(self.cfg, 'training.batch_size', 16)
        num_workers = 0 #safe_getattr(self.cfg, 'training.num_workers', 0)  # Use 0 to avoid multiprocessing issues

        # Determine pin_memory based on device
        device_type = str(self.device).split(':')[0]
        pin_memory = device_type in ['cuda', 'xpu']  # Pin memory for CUDA and XPU

        self.train_sampler = DistributedSampler(
                            train_dataset,
                            num_replicas=dist.get_world_size(),
                            rank=self.rank,
                            shuffle=True,
                            drop_last=False)

        print(f"{self.train_sampler=}")
        self.train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                                drop_last=False,
                                shuffle=False,  # Don't shuffle when using sampler
                                sampler=self.train_sampler)

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def setup_model(self):
        """Setup model with optimizations."""
        print("Setting up model...")
        
        # Build graph for absorbing diffusion
        self.graph = graph_lib.get_graph(self.cfg, self.device, tokens=self.cfg.tokens)
        
        # Build noise schedule
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        
        # Build model (using V100-compatible version for UniRef50)
        from protlig_dd.model.transformer_v100 import SEDD
        self.model = SEDD(self.cfg).to(self.device)
        print(f"✅ Using V100-compatible SEDD model (no flash attention required)")
        
        # Enable gradient checkpointing for memory efficiency (if supported)
        if hasattr(self.cfg, 'memory') and self.cfg.memory.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            else:
                print("⚠️  Gradient checkpointing not supported by V100 model")
        
        # Setup EMA
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), 
            decay=self.cfg.training.ema
        )
        
        print(f"Model ready. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    

    def setup_optimizer(self):
        """Setup optimizer with DDP-aware learning rate scaling."""
        print("Setting up optimizer...")

        # Calculate DDP-scaled learning rate (most stable approach)
        base_lr = self.cfg.optim.lr
        if self.use_ddp and self.world_size > 1:
            # Linear scaling with world size (most stable for DDP)
            scaled_lr = base_lr * self.world_size
            print(f"🔄 DDP Learning Rate Scaling:")
            print(f"   Base LR: {base_lr:.2e}")
            print(f"   World Size: {self.world_size}")
            print(f"   Scaled LR: {scaled_lr:.2e} (linear scaling)")
        else:
            scaled_lr = base_lr
            print(f"📊 Single GPU Learning Rate: {scaled_lr:.2e}")

        # Temporarily modify config for optimizer creation
        original_lr = self.cfg.optim.lr
        self.cfg.optim.lr = scaled_lr

        # Get optimizer with scaled learning rate
        self.optimizer = losses.get_optimizer(
            self.cfg,
            chain(self.model.parameters(), self.noise.parameters())
        )

        # Restore original config
        self.cfg.optim.lr = original_lr

        # Apply IPEX optimization for Intel XPU
        device_type = str(self.device).split(':')[0]
        if device_type == 'xpu':
            print("🔧 Applying IPEX optimization for Intel XPU...")
            self.model.to(self.device)
            self.model, self.optimizer = ipex.optimize(self.model, optimizer=self.optimizer)
            self.scaler = torch.xpu.amp.GradScaler()
            self.use_amp = True
            print("✅ Using XPU mixed precision training with IPEX optimization")
        elif device_type == 'cuda':
            self.model.to(self.device)
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
            print("✅ Using CUDA mixed precision training")
        else:
            self.scaler = None
            self.use_amp = False
            print(f"✅ Using {device_type.upper()} without mixed precision")
        
        # Setup learning rate scheduler with scaled learning rate
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=self.cfg.optim.warmup,
            max_steps=self.cfg.training.n_iters,
            base_lr=scaled_lr * 0.1,
            max_lr=scaled_lr,
            min_lr=scaled_lr * 0.01
        )

        print(f"✅ Scheduler configured with scaled LR: max={scaled_lr:.2e}, warmup_steps={self.cfg.optim.warmup}")
        
        # Training state
        self.state = {
            'optimizer': self.optimizer,
            'scaler': self.scaler,
            'scheduler': self.scheduler,
            'model': self.model.state_dict(),
            'noise': self.noise.state_dict(),
            'ema': self.ema.state_dict(),
            'step': 0,
            'epoch': 0
        }
        
        print("Optimizer ready.")

    def setup_logging(self):
        """Setup logging system - either wandb or file logging."""
        if self.use_wandb:
            print("📊 Wandb logging enabled")
        else:
            # Setup file logging
            log_dir = os.path.join(self.work_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)

            # Create log file with timestamp and rank
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if self.use_ddp:
                log_filename = f"training_rank{self.rank}_{timestamp}.log"
            else:
                log_filename = f"training_{timestamp}.log"

            self.log_file = os.path.join(log_dir, log_filename)

            # Write header
            with open(self.log_file, 'w') as f:
                f.write(f"# Training Log - Started {timestamp}\n")
                f.write(f"# Rank: {self.rank if self.use_ddp else 0}\n")
                f.write(f"# Device: {self.device}\n")
                f.write(f"# World Size: {self.world_size if self.use_ddp else 1}\n")
                f.write("# Format: step,epoch,loss,lr,time\n")

            print(f"📝 File logging enabled: {self.log_file}")

    def log_metrics(self, metrics, step=None):
        """Log metrics to wandb or file - ALL RANKS log everything."""
        # Skip logging in minimal mode to prevent sync issues
        if self.minimal_mode:
            return

        # ALL ranks log everything to prevent sync issues
        if self.use_wandb:
            wandb.log(metrics, step=step)
        else:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_file, 'a') as f:
                # Write metrics in a structured format
                metric_str = f"{timestamp}"
                if step is not None:
                    metric_str += f",step={step}"
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_str += f",{key}={value:.6f}"
                    else:
                        metric_str += f",{key}={value}"
                f.write(metric_str + "\n")

    def wrap_model_ddp(self):
        if self.use_ddp:
            # Get the local device ID (not the global rank)
            local_device_id = self.device #int(str(self.device).split(':')[1])
            self.model_ddp = DDP(
                                self.model,
                                device_ids=[local_device_id],
                                output_device=local_device_id,
                                find_unused_parameters=False)
            print(f"✅ Model wrapped with DDP on device {local_device_id}")
        else:
            self.model_ddp = self.model
            print("✅ Using model without DDP wrapper")

    def setup_wandb(self, project_name: str, run_name: str):
        """Setup Wandb with comprehensive logging configuration."""
        if not self.use_wandb:
            print("📝 Wandb disabled - using file logging instead")
            return

        print("🚀 Setting up Wandb logging...")

        # Initialize wandb only on rank 0
        if self.rank == 0:
            run = wandb.init(
                project=project_name,
                name=run_name,
                config={
                    # Model configuration
                    'model_name': safe_getattr(self.cfg, 'model.name', 'transformer'),
                    'model_type': safe_getattr(self.cfg, 'model.type', 'ddit'),
                    'hidden_size': safe_getattr(self.cfg, 'model.hidden_size', 512),
                    'n_blocks': safe_getattr(self.cfg, 'model.n_blocks', 8),
                    'n_heads': safe_getattr(self.cfg, 'model.n_heads', 8),
                    'dropout': safe_getattr(self.cfg, 'model.dropout', 0.1),

                    # Training configuration
                    'batch_size': safe_getattr(self.cfg, 'training.batch_size', 16),
                    'accumulation_steps': safe_getattr(self.cfg, 'training.accum', 2),
                    'base_learning_rate': safe_getattr(self.cfg, 'optim.lr', 5e-5),
                    'effective_learning_rate': safe_getattr(self.cfg, 'optim.lr', 5e-5) * (self.world_size if self.use_ddp else 1),
                    'world_size': self.world_size if self.use_ddp else 1,
                    'weight_decay': safe_getattr(self.cfg, 'optim.weight_decay', 0.01),
                    'warmup_steps': safe_getattr(self.cfg, 'optim.warmup', 1000),
                    'max_iterations': safe_getattr(self.cfg, 'training.n_iters', 5000),
                    'ema_decay': safe_getattr(self.cfg, 'training.ema', 0.999),

                    # Data configuration
                    'max_protein_len': safe_getattr(self.cfg, 'data.max_protein_len', 512),
                    'vocab_size': safe_getattr(self.cfg, 'data.vocab_size_protein', 25),
                    'train_ratio': safe_getattr(self.cfg, 'data.train_ratio', 0.9),

                    # Noise configuration
                    'noise_type': safe_getattr(self.cfg, 'noise.type', 'cosine'),
                    'sigma_min': safe_getattr(self.cfg, 'noise.sigma_min', 1e-4),
                    'sigma_max': safe_getattr(self.cfg, 'noise.sigma_max', 0.5),

                    # Curriculum learning
                    'curriculum_enabled': safe_getattr(self.cfg, 'curriculum.enabled', False),
                    'preschool_time': safe_getattr(self.cfg, 'curriculum.preschool_time', 5000),

                    # System info
                    'device': str(self.device),
                    'seed': safe_getattr(self.cfg, 'training.seed', 42),
                    'epochs': self.epochs_override if self.epochs_override is not None else safe_getattr(self.cfg, 'training.epochs', 50),
            },
            tags=['uniref50', 'sedd', 'protein', 'diffusion', 'optimized'],
            notes=f"Optimized UniRef50 training with improved V100-compatible attention and enhanced curriculum learning"
        )

            # Display the Wandb web interface link prominently
            print("\n" + "="*80)
            print("🌐 WANDB EXPERIMENT TRACKING")
            print("="*80)
            print(f"📊 Project: {project_name}")
            print(f"🏷️  Run Name: {run_name}")
            print(f"🔗 Web Interface: {wandb.run.url}")
            print(f"📈 Dashboard: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
            print("="*80)
            print("💡 Open the link above to monitor your training in real-time!")
            print("="*80 + "\n")

            # Store the run for later reference
            self.wandb_run = run

            print("✅ Wandb setup complete - tracking enabled!")

    def setup_wandb_model_watching(self):
        """Setup model watching after model is created."""
        if not self.use_wandb:
            return

        try:
            if self.rank == 0:  # Keep this rank check for wandb.watch
                # Watch model for gradient and parameter tracking
                log_freq = safe_getattr(self.cfg, 'training.log_freq', 50)
                wandb.watch(self.model, log='all', log_freq=log_freq)
                print("✅ Wandb model watching enabled")
        except Exception as e:
            print(f"⚠️  Warning: Could not setup model watching: {e}")
            print("   Training will continue without gradient tracking")

    def log_training_metrics(self, step: int, loss: float, lr: float, epoch: int,
                           batch_time: float = None, additional_metrics: dict = None):
        """Log training metrics - ALL RANKS participate."""
        # All ranks participate in logging computation

        metrics = {
            'train/loss': loss,
            'train/learning_rate': lr,
            'train/step': step,
            'train/epoch': epoch,
        }

        if batch_time is not None:
            metrics['train/batch_time'] = batch_time
            metrics['train/samples_per_second'] = self.cfg.training.batch_size / batch_time

        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f'train/{key}'] = value

        self.log_metrics(metrics, step=step)

    def log_to_wandb(self, metrics: dict, step: int = None):
        """Helper method to log to wandb or file - ALL RANKS log."""
        # All ranks log to prevent sync issues
        self.log_metrics(metrics, step=step)

    def log_validation_metrics(self, step: int, val_loss: float, perplexity: float = None, recon_loss: float = None):
        """Log validation metrics - ALL RANKS log."""
        # All ranks log to prevent sync issues

        metrics = {
            'val/loss': val_loss,
            'val/step': step,
        }

        if perplexity is not None:
            metrics['val/perplexity'] = perplexity

        if recon_loss is not None:
            metrics['val/reconstruction_loss'] = recon_loss

        wandb.log(metrics, step=step)

    def decode_sequence(self, sequence_tensor, vocab_mapping=None):
        """Decode a sequence tensor to amino acid string."""
        if vocab_mapping is None:
            # Load vocabulary mapping
            try:
                import json
                with open("input_data/vocab.json", 'r') as f:
                    vocab = json.load(f)
                id_to_token = {v: k for k, v in vocab.items()}
            except:
                # Fallback mapping
                id_to_token = {i: aa for i, aa in enumerate(['<s>', '<pad>', '</s>', '<unk>', '<mask>'] + list('ACDEFGHIKLMNPQRSTVWY'))}
        else:
            id_to_token = vocab_mapping

        if torch.is_tensor(sequence_tensor):
            sequence = sequence_tensor.tolist()
        else:
            sequence = sequence_tensor

        # Decode only amino acids (skip special tokens)
        amino_acids = []
        for token_id in sequence:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token in 'ACDEFGHIKLMNPQRSTVWY':
                    amino_acids.append(token)

        return ''.join(amino_acids)

    def setup_protein_sampler(self, batch_size: int = 1, max_length: int = 256):
        """Setup the rigorous CTMC sampler for protein generation."""
        sampling_shape = (batch_size, max_length)

        # Use the formal sampling framework
        sampling_fn = sampling.get_sampling_fn(
            config=self.cfg,
            graph=self.graph,
            noise=self.noise,
            batch_dims=sampling_shape,
            eps=1e-5,  # Small epsilon for sampling
            device=self.device
        )
        return sampling_fn

    def generate_protein_sequences_rigorous(self, num_samples: int = 10, max_length: int = 256):
        """Generate protein sequences using rigorous CTMC sampling."""
        print(f"🧬 Generating {num_samples} protein sequences using rigorous CTMC sampling...")

        self.model.eval()
        generated_sequences = []

        with torch.no_grad():
            # Apply EMA weights for generation
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

            try:
                # Setup sampler for single batch
                sampler = self.setup_protein_sampler(batch_size=num_samples, max_length=max_length)

                # Create a model wrapper that matches the expected sampling interface
                class ModelWrapper:
                    def __init__(self, model):
                        self.model = model

                    def __call__(self, x=None, sigma=None, **kwargs):
                        # Handle different calling conventions from the sampling framework
                        if 'protein_indices' in kwargs and 'timesteps' in kwargs:
                            # Called from sampling framework with keyword arguments
                            # Extract the relevant arguments for UniRef50 model
                            protein_indices = kwargs['protein_indices']
                            timesteps = kwargs['timesteps']
                            mode = kwargs.get('mode', 'protein_only')

                            # UniRef50 model expects (indices, sigma) format
                            # Use protein_indices as the main input for protein_only mode
                            # Use DDP model if available
                            model_to_use = self.model_ddp if hasattr(self, 'model_ddp') and self.use_ddp else self.model
                            if mode == 'protein_only':
                                return model_to_use(protein_indices, timesteps)
                            else:
                                raise ValueError(f"UniRef50 model only supports protein_only mode, got {mode}")

                        elif x is not None and sigma is not None:
                            # Called with positional arguments (legacy interface)
                            # Convert sigma to timesteps if needed
                            if hasattr(sigma, 'shape') and len(sigma.shape) > 0:
                                timesteps = sigma
                            else:
                                timesteps = sigma * torch.ones(x.shape[0], device=x.device)
                            # Call the model with proper interface (use DDP model if available)
                            model_to_use = self.model_ddp if hasattr(self, 'model_ddp') and self.use_ddp else self.model
                            return model_to_use(x, timesteps)
                        else:
                            raise ValueError(f"ModelWrapper called with invalid arguments. Got x={x}, sigma={sigma}, kwargs={list(kwargs.keys())}")

                    def eval(self):
                        """Set model to evaluation mode."""
                        self.model.eval()
                        return self

                    def train(self, mode=True):
                        """Set model to training mode."""
                        self.model.train(mode)
                        return self

                    def parameters(self):
                        """Return model parameters."""
                        return self.model.parameters()

                    def state_dict(self):
                        """Return model state dict."""
                        return self.model.state_dict()

                    def to(self, device):
                        """Move model to device."""
                        self.model.to(device)
                        return self

                    @property
                    def device(self):
                        """Get model device."""
                        return next(self.model.parameters()).device

                model_wrapper = ModelWrapper(self.model)

                # Generate samples using the formal framework
                # Note: The sampling framework expects a specific interface
                samples = sampler(model_wrapper, task="protein_only")

                # Process each generated sample
                for i in range(num_samples):
                    try:
                        if len(samples.shape) > 1:
                            sample_tokens = samples[i]
                        else:
                            sample_tokens = samples

                        decoded_sequence = self.decode_sequence(sample_tokens)

                        generated_sequences.append({
                            'sample_id': i,
                            'raw_tokens': sample_tokens[:50].cpu().tolist(),
                            'sequence': decoded_sequence,
                            'length': len(decoded_sequence),
                            'unique_amino_acids': len(set(decoded_sequence)) if decoded_sequence else 0
                        })
                    except Exception as e:
                        print(f"⚠️  Error processing sample {i}: {e}")
                        generated_sequences.append({
                            'sample_id': i,
                            'raw_tokens': [],
                            'sequence': '',
                            'length': 0,
                            'unique_amino_acids': 0
                        })

            except Exception as e:
                print(f"⚠️  Error in rigorous sampling: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple method
                print("🔄 Falling back to simple generation method...")
                generated_sequences = self.generate_protein_sequences_simple(num_samples, max_length)

            finally:
                # Restore original weights
                self.ema.restore(self.model.parameters())

        self.model.train()
        return generated_sequences

    def generate_protein_sequences_simple(self, num_samples: int = 10, max_length: int = 256, num_diffusion_steps: int = 50, temperature: float = 1.0):
        """Generate protein sequences using simple heuristic diffusion sampling."""
        print(f"🧬 Generating {num_samples} protein sequences using simple heuristic sampling...")

        self.model.eval()
        generated_sequences = []

        with torch.no_grad():
            for i in range(num_samples):
                try:
                    # Initialize with absorbing states (assuming vocab_size - 1 is absorbing)
                    vocab_size = self.cfg.data.vocab_size_protein + (1 if hasattr(self.cfg.graph, 'type') and self.cfg.graph.type == "absorb" else 0)
                    absorbing_token = vocab_size - 1

                    # Start with all absorbing tokens
                    sample = torch.full((1, max_length), absorbing_token, dtype=torch.long, device=self.device)

                    # Diffusion denoising process
                    for step in range(num_diffusion_steps):
                        # Compute timestep (from 1.0 to 0.0)
                        t = torch.tensor([1.0 - step / num_diffusion_steps], device=self.device)

                        # Get model predictions (use DDP model)
                        model_to_use = self.model_ddp if self.use_ddp else self.model
                        device_type = str(self.device).split(':')[0]
                        if device_type == 'xpu':
                            with torch.xpu.amp.autocast(enabled=False):
                                logits = model_to_use(sample, t)
                        elif device_type == 'cuda':
                            with torch.cuda.amp.autocast(enabled=False):
                                logits = model_to_use(sample, t)
                        else:
                            logits = model_to_use(sample, t)

                        # Temperature sampling
                        probs = torch.softmax(logits / temperature, dim=-1)

                        # Sample new tokens for each position
                        batch_size, seq_len, vocab_size_actual = probs.shape
                        probs_flat = probs.view(-1, vocab_size_actual)
                        new_tokens = torch.multinomial(probs_flat, 1).view(batch_size, seq_len)

                        # Gradually replace absorbing tokens with generated tokens
                        replace_prob = (step + 1) / num_diffusion_steps
                        mask = torch.rand(batch_size, seq_len, device=self.device) < replace_prob
                        sample = torch.where(mask, new_tokens, sample)

                    # Decode the generated sequence
                    decoded_sequence = self.decode_sequence(sample[0])

                    generated_sequences.append({
                        'sample_id': i,
                        'raw_tokens': sample[0][:50].cpu().tolist(),  # First 50 tokens for debugging
                        'sequence': decoded_sequence,
                        'length': len(decoded_sequence),
                        'unique_amino_acids': len(set(decoded_sequence)) if decoded_sequence else 0
                    })

                except Exception as e:
                    print(f"⚠️  Error generating sample {i}: {e}")
                    generated_sequences.append({
                        'sample_id': i,
                        'raw_tokens': [],
                        'sequence': '',
                        'length': 0,
                        'unique_amino_acids': 0
                    })

        self.model.train()
        return generated_sequences

    def generate_protein_sequences(self, num_samples: int = 10, max_length: int = 256,
                                 sampling_method: str = "rigorous", **kwargs):
        """
        Generate protein sequences with choice of sampling method.

        Args:
            num_samples: Number of sequences to generate
            max_length: Maximum sequence length
            sampling_method: "rigorous" (CTMC) or "simple" (heuristic)
            **kwargs: Additional arguments for simple method (num_diffusion_steps, temperature)
        """
        if sampling_method == "rigorous":
            return self.generate_protein_sequences_rigorous(num_samples, max_length)
        elif sampling_method == "simple":
            return self.generate_protein_sequences_simple(num_samples, max_length, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. Use 'rigorous' or 'simple'.")

    def quick_generation_test(self, step: int, epoch: int, num_samples: int = 3, max_length: int = 128):
        """Quick generation test during training to monitor generation quality - ALL RANKS participate."""
        # All ranks participate to prevent sync issues

        # Use same max length as training data if not specified
        if max_length is None:
            max_length = safe_getattr(self.cfg, 'data.max_protein_len', 512)

        print(f"\n🧬 Quick generation test - Step {step} (max_length: {max_length})")

        # Create samples directory
        samples_dir = os.path.join(self.work_dir, "generated_samples", "quick_generation")
        os.makedirs(samples_dir, exist_ok=True)

        try:
            # Use the configured sampling method for quick test
            start_time = time.time()

            if self.sampling_method == "rigorous":
                sequences = self.generate_protein_sequences_rigorous(num_samples, max_length)
            else:
                sequences = self.generate_protein_sequences_simple(
                    num_samples, max_length, num_diffusion_steps=15, temperature=1.0
                )

            generation_time = time.time() - start_time

            # Analyze generated sequences
            valid_sequences = [s for s in sequences if s['sequence']]
            valid_count = len(valid_sequences)

            if valid_count > 0:
                avg_length = np.mean([s['length'] for s in valid_sequences])
                avg_unique_aa = np.mean([s['unique_amino_acids'] for s in valid_sequences])

                # Show first sequence as example
                example_seq = valid_sequences[0]['sequence'][:40]
                print(f"   ✅ Generated {valid_count}/{num_samples} valid sequences")
                print(f"   📊 Avg length: {avg_length:.1f}, Avg unique AAs: {avg_unique_aa:.1f}")
                print(f"   🧬 Example: {example_seq}...")

                # Calculate ESM perplexity for quick generation samples (first 3 for speed)
                print("   🧬 Calculating ESM perplexity for quick generation samples...")
                quick_esm_perplexities = self.calculate_esm_perplexity(valid_sequences[:3])

                # Create sample table with full sequences and ESM perplexity
                sample_data = []
                for i, seq_info in enumerate(valid_sequences[:3]):  # Show first 3
                    esm_ppl = quick_esm_perplexities[i] if quick_esm_perplexities and i < len(quick_esm_perplexities) else None
                    sample_data.append([
                        i + 1,
                        seq_info['sequence'],  # Full sequence, no truncation
                        seq_info['length'],
                        seq_info['unique_amino_acids'],
                        f"{esm_ppl:.2f}" if esm_ppl is not None else "N/A"
                    ])

                quick_gen_table = wandb.Table(
                    columns=['Sample', 'Sequence', 'Length', 'Unique AAs', 'ESM Perplexity'],
                    data=sample_data
                )

                # Calculate quick ESM perplexity stats
                quick_esm_stats = {}
                if quick_esm_perplexities:
                    quick_esm_stats = {
                        'quick_gen/esm_perplexity_mean': np.mean(quick_esm_perplexities),
                        'quick_gen/esm_perplexity_std': np.std(quick_esm_perplexities),
                        'quick_gen/esm_perplexity_min': np.min(quick_esm_perplexities),
                        'quick_gen/esm_perplexity_max': np.max(quick_esm_perplexities),
                    }

                # Save samples to file system
                self.save_generation_samples(
                    sequences=sequences,
                    step=step,
                    epoch=epoch,
                    sample_type="quick_generation",
                    esm_perplexities=quick_esm_perplexities,
                    generation_time=generation_time,
                    sampling_method=self.sampling_method
                )

                # Log to wandb (rank 0 only)
                self.log_to_wandb({
                    'quick_gen/samples': quick_gen_table,
                    'quick_gen/valid_sequences': valid_count,
                    'quick_gen/total_sequences': num_samples,
                    'quick_gen/success_rate': valid_count / num_samples,
                    'quick_gen/avg_length': avg_length,
                    'quick_gen/avg_unique_aa': avg_unique_aa,
                    'quick_gen/generation_time': generation_time,
                    'quick_gen/sampling_method': self.sampling_method,
                    **quick_esm_stats  # Add ESM perplexity stats
                }, step=step)

                return True
            else:
                print(f"   ⚠️  No valid sequences generated ({num_samples} attempted)")
                self.log_to_wandb({
                    'quick_gen/valid_sequences': 0,
                    'quick_gen/total_sequences': num_samples,
                    'quick_gen/success_rate': 0.0,
                    'quick_gen/generation_time': generation_time,
                    'quick_gen/sampling_method': self.sampling_method
                }, step=step)
                return False

        except Exception as e:
            print(f"   ❌ Quick generation test failed: {e}")
            self.log_to_wandb({
                'quick_gen/error': str(e),
                'quick_gen/sampling_method': self.sampling_method
            }, step=step)
            return False

    def test_sampling_methods(self, num_samples: int = 5, max_length: int = 100):
        """Test and compare both sampling methods."""
        print("\n🧪 TESTING SAMPLING METHODS")
        print("=" * 60)

        try:
            # Test rigorous sampling
            print("🔬 Testing rigorous CTMC sampling...")
            rigorous_sequences = self.generate_protein_sequences_rigorous(num_samples, max_length)
            rigorous_success = len([s for s in rigorous_sequences if s['sequence']]) > 0

            print(f"   ✅ Rigorous sampling: {len(rigorous_sequences)} sequences generated")
            if rigorous_success:
                avg_length_rigorous = np.mean([s['length'] for s in rigorous_sequences if s['sequence']])
                print(f"   📊 Average length: {avg_length_rigorous:.1f}")
                print(f"   🧬 Sample: {rigorous_sequences[0]['sequence'][:50]}...")

            # Test simple sampling
            print("\n🎲 Testing simple heuristic sampling...")
            simple_sequences = self.generate_protein_sequences_simple(num_samples, max_length,
                                                                    num_diffusion_steps=20, temperature=1.0)
            simple_success = len([s for s in simple_sequences if s['sequence']]) > 0

            print(f"   ✅ Simple sampling: {len(simple_sequences)} sequences generated")
            if simple_success:
                avg_length_simple = np.mean([s['length'] for s in simple_sequences if s['sequence']])
                print(f"   📊 Average length: {avg_length_simple:.1f}")
                print(f"   🧬 Sample: {simple_sequences[0]['sequence'][:50]}...")

            # Summary
            print(f"\n📋 SAMPLING TEST SUMMARY:")
            print(f"   Rigorous CTMC: {'✅ PASS' if rigorous_success else '❌ FAIL'}")
            print(f"   Simple Heuristic: {'✅ PASS' if simple_success else '❌ FAIL'}")

            return rigorous_success and simple_success

        except Exception as e:
            print(f"❌ Sampling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_esm_perplexity(self, generated_sequences):
        """Calculate ESM perplexity for generated protein sequences."""
        try:
            from protlig_dd.utils.eval_metrics import ProteinEvaluator

            # Extract valid sequences
            valid_sequences = [seq_info['sequence'] for seq_info in generated_sequences
                             if seq_info['sequence'] and len(seq_info['sequence']) > 0]

            if not valid_sequences:
                print("⚠️  No valid sequences found for ESM perplexity calculation")
                return []

            print(f"📊 Calculating ESM perplexity for {len(valid_sequences)} sequences...")

            # Initialize ProteinEvaluator
            evaluator = ProteinEvaluator()

            # Calculate MLM perplexity using ESM model
            perplexities = evaluator.calculate_mlm_ppl(
                sequences=valid_sequences,
                batch_size=4,  # Small batch size to avoid memory issues
                mask_fraction=0.15,
                seed=42
            )

            print(f"✅ ESM perplexity calculation completed. Mean: {sum(perplexities)/len(perplexities):.2f}")
            return perplexities

        except Exception as e:
            print(f"⚠️  Error calculating ESM perplexity: {e}")
            import traceback
            traceback.print_exc()
            return []

    def analyze_sequence_properties(self, sequences):
        """Analyze biochemical properties of generated sequences."""
        if not sequences:
            return {}

        # Filter out empty sequences
        valid_sequences = [seq['sequence'] for seq in sequences if seq['sequence']]
        if not valid_sequences:
            return {'error': 'No valid sequences generated'}

        # Amino acid property groups (biochemical classification)
        hydrophobic = set('AILMFPWV')        # Hydrophobic/nonpolar amino acids
        polar = set('NQSTC')                 # Polar uncharged amino acids
        charged_positive = set('RHK')        # Positively charged (basic): Arg, His, Lys
        charged_negative = set('DE')         # Negatively charged (acidic): Asp, Glu
        aromatic = set('FWY')                # Aromatic amino acids
        small = set('AGST')                  # Small amino acids

        # Analyze each sequence
        properties = {
            'num_sequences': len(valid_sequences),
            'lengths': [len(seq) for seq in valid_sequences],
            'unique_aa_counts': [len(set(seq)) for seq in valid_sequences],
            'hydrophobic_pcts': [],
            'polar_pcts': [],
            'positive_pcts': [],
            'negative_pcts': [],
            'aromatic_pcts': [],
            'amino_acid_frequencies': {}
        }

        # Initialize amino acid frequency counter
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            properties['amino_acid_frequencies'][aa] = 0

        total_amino_acids = 0

        for seq in valid_sequences:
            if len(seq) == 0:
                continue

            # Count amino acid types
            hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic)
            polar_count = sum(1 for aa in seq if aa in polar)
            positive_count = sum(1 for aa in seq if aa in charged_positive)
            negative_count = sum(1 for aa in seq if aa in charged_negative)
            aromatic_count = sum(1 for aa in seq if aa in aromatic)

            # Calculate percentages
            seq_len = len(seq)
            properties['hydrophobic_pcts'].append(hydrophobic_count / seq_len * 100)
            properties['polar_pcts'].append(polar_count / seq_len * 100)
            properties['positive_pcts'].append(positive_count / seq_len * 100)
            properties['negative_pcts'].append(negative_count / seq_len * 100)
            properties['aromatic_pcts'].append(aromatic_count / seq_len * 100)

            # Count individual amino acids
            for aa in seq:
                if aa in properties['amino_acid_frequencies']:
                    properties['amino_acid_frequencies'][aa] += 1
                    total_amino_acids += 1

        # Convert counts to percentages
        if total_amino_acids > 0:
            for aa in properties['amino_acid_frequencies']:
                properties['amino_acid_frequencies'][aa] = (
                    properties['amino_acid_frequencies'][aa] / total_amino_acids * 100
                )

        # Calculate summary statistics
        properties['summary'] = {
            'avg_length': np.mean(properties['lengths']) if properties['lengths'] else 0,
            'std_length': np.std(properties['lengths']) if properties['lengths'] else 0,
            'min_length': min(properties['lengths']) if properties['lengths'] else 0,
            'max_length': max(properties['lengths']) if properties['lengths'] else 0,
            'avg_unique_aa': np.mean(properties['unique_aa_counts']) if properties['unique_aa_counts'] else 0,
            'avg_hydrophobic': np.mean(properties['hydrophobic_pcts']) if properties['hydrophobic_pcts'] else 0,
            'avg_polar': np.mean(properties['polar_pcts']) if properties['polar_pcts'] else 0,
            'avg_positive': np.mean(properties['positive_pcts']) if properties['positive_pcts'] else 0,
            'avg_negative': np.mean(properties['negative_pcts']) if properties['negative_pcts'] else 0,
            'avg_aromatic': np.mean(properties['aromatic_pcts']) if properties['aromatic_pcts'] else 0,
        }

        return properties

    def comprehensive_evaluation(self, step: int, epoch: int, num_samples: int = 15, sampling_method: str = "rigorous"):
        """Comprehensive evaluation including generation quality assessment - ALL RANKS participate."""
        # All ranks participate to prevent sync issues

        print(f"\n🔬 COMPREHENSIVE EVALUATION - Step {step}, Epoch {epoch}")
        print("=" * 80)

        try:
            # 1. Validation loss
            print("📊 Computing validation loss...")
            val_loss, recon_loss, fixed_noise_metrics = self.validate_model()
            print(f"   Validation Loss: {val_loss:.4f}"
                  f"   Reconstruction Loss: {recon_loss:.4f}" if recon_loss is not None else ""
                  f"   Fixed Noise Metrics: {fixed_noise_metrics}" if fixed_noise_metrics else "" 
                )

            # 2. Generate sequences using specified method
            print(f"🧬 Generating protein sequences using {sampling_method} method...")
            # Use same max length as training data for consistency
            eval_max_length = safe_getattr(self.cfg, 'data.max_protein_len', 512)

            if sampling_method == "rigorous":
                generated_sequences = self.generate_protein_sequences(
                    num_samples=num_samples,
                    max_length=eval_max_length,  # Match training data length
                    sampling_method="rigorous"
                )
            else:
                generated_sequences = self.generate_protein_sequences(
                    num_samples=num_samples,
                    max_length=eval_max_length,  # Match training data length
                    sampling_method="simple",
                    num_diffusion_steps=30,  # More steps for better quality
                    temperature=0.9  # Slightly focused sampling
                )

            # Log sampling method used
            self.log_to_wandb({
                'eval/sampling_method': sampling_method,
                'eval/num_generated_samples': len(generated_sequences)
            }, step=step)

            # 3. Analyze sequence properties
            print("🔍 Analyzing sequence properties...")
            properties = self.analyze_sequence_properties(generated_sequences)

            # 3.5. Calculate ESM perplexity for generated sequences
            print("🧬 Calculating ESM perplexity for generated sequences...")
            esm_perplexities = self.calculate_esm_perplexity(generated_sequences)

            # 4. Compare with training data (sample a few training sequences)
            print("📚 Sampling training data for comparison...")
            training_comparison = self.get_training_data_stats()

            # 5. Log comprehensive metrics to Wandb
            print("📈 Logging metrics to Wandb...")

            # Basic metrics
            wandb_metrics = {
                'eval/validation_loss': val_loss,
                'eval/step': step,
                'eval/epoch': epoch,
            }

            # Add reconstruction loss if available
            if recon_loss is not None:
                wandb_metrics['eval/reconstruction_loss'] = recon_loss

            # Add fixed noise level metrics
            for noise_key, noise_value in fixed_noise_metrics.items():
                if noise_value is not None:
                    wandb_metrics[f'eval/{noise_key}'] = noise_value

            # Generation quality metrics
            if 'error' not in properties:
                summary = properties['summary']
                wandb_metrics.update({
                    'generation/num_valid_sequences': properties['num_sequences'],
                    'generation/avg_length': summary['avg_length'],
                    'generation/std_length': summary['std_length'],
                    'generation/min_length': summary['min_length'],
                    'generation/max_length': summary['max_length'],
                    'generation/avg_unique_amino_acids': summary['avg_unique_aa'],
                    'generation/avg_hydrophobic_pct': summary['avg_hydrophobic'],
                    'generation/avg_polar_pct': summary['avg_polar'],
                    'generation/avg_positive_pct': summary['avg_positive'],
                    'generation/avg_negative_pct': summary['avg_negative'],
                    'generation/avg_aromatic_pct': summary['avg_aromatic'],
                })

                # Amino acid frequency distribution
                for aa, freq in properties['amino_acid_frequencies'].items():
                    wandb_metrics[f'generation/aa_freq_{aa}'] = freq

            # ESM perplexity metrics
            if esm_perplexities and len(esm_perplexities) > 0:
                wandb_metrics.update({
                    'generation/esm_perplexity_mean': np.mean(esm_perplexities),
                    'generation/esm_perplexity_std': np.std(esm_perplexities),
                    'generation/esm_perplexity_min': np.min(esm_perplexities),
                    'generation/esm_perplexity_max': np.max(esm_perplexities),
                    'generation/esm_perplexity_median': np.median(esm_perplexities),
                })

            # Training data comparison
            if training_comparison:
                for key, value in training_comparison.items():
                    wandb_metrics[f'training_data/{key}'] = value

            # Log to Wandb
            self.log_to_wandb(wandb_metrics, step=step)

            # 6. Create and log sample table with full sequences and ESM perplexity
            if generated_sequences:
                sample_data = []
                for i, seq_info in enumerate(generated_sequences[:10]):  # Show top 10
                    # Get corresponding ESM perplexity if available
                    esm_ppl = esm_perplexities[i] if esm_perplexities and i < len(esm_perplexities) else None

                    sample_data.append([
                        seq_info['sample_id'],
                        seq_info['sequence'],  # Full sequence, no truncation
                        seq_info['length'],
                        seq_info['unique_amino_acids'],
                        f"{esm_ppl:.2f}" if esm_ppl is not None else "N/A"
                    ])

                sample_table = wandb.Table(
                    columns=['Sample ID', 'Generated Sequence', 'Length', 'Unique AAs', 'ESM Perplexity'],
                    data=sample_data
                )

                self.log_to_wandb({
                    'samples/generated_proteins': sample_table,
                    'samples/step': step
                }, step=step)

            # 7. Print summary
            print(f"\n📋 EVALUATION SUMMARY:")
            print(f"   Validation Loss: {val_loss:.4f}")
            if recon_loss is not None:
                print(f"   Reconstruction Loss: {recon_loss:.4f}")

            # Print fixed noise level results
            if fixed_noise_metrics:
                print(f"   Fixed Noise Levels:")
                for noise_key, noise_value in sorted(fixed_noise_metrics.items()):
                    if noise_value is not None:
                        noise_level = noise_key.replace('fixed_noise_', '')
                        print(f"     σ={noise_level}: {noise_value:.4f}")

            if 'error' not in properties:
                print(f"   Generated Sequences: {properties['num_sequences']}")
                print(f"   Avg Length: {properties['summary']['avg_length']:.1f} ± {properties['summary']['std_length']:.1f}")
                print(f"   Avg Unique AAs: {properties['summary']['avg_unique_aa']:.1f}")
                print(f"   Composition: H={properties['summary']['avg_hydrophobic']:.1f}% P={properties['summary']['avg_polar']:.1f}% +={properties['summary']['avg_positive']:.1f}% -={properties['summary']['avg_negative']:.1f}%")

                # Show a few example sequences
                print(f"\n🧬 Example Generated Sequences:")
                for i, seq_info in enumerate(generated_sequences[:3]):
                    if seq_info['sequence']:
                        print(f"   {i+1}: {seq_info['sequence'][:60]}{'...' if len(seq_info['sequence']) > 60 else ''}")
            else:
                print(f"   ⚠️  Generation failed: {properties['error']}")

            # Save samples to file system (if generation was successful)
            if 'error' not in properties and 'generated_sequences' in locals():
                self.save_generation_samples(
                    sequences=generated_sequences,
                    step=step,
                    epoch=epoch,
                    sample_type="comprehensive_evaluation",
                    esm_perplexities=esm_perplexities,
                    generation_time=None,  # Not tracked in comprehensive eval
                    sampling_method=sampling_method,
                    validation_loss=val_loss,
                    properties=properties
                )

            return val_loss, properties

        except Exception as e:
            print(f"❌ Error during comprehensive evaluation: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), {'error': str(e)}

    def save_generation_samples(self, sequences, step, epoch, sample_type,
                              esm_perplexities=None, generation_time=None,
                              sampling_method=None, validation_loss=None, properties=None):
        """Save generated samples to file system for offline analysis."""
        import json
        from datetime import datetime

        try:
            # Create samples directory
            samples_dir = os.path.join(self.work_dir, "generated_samples", sample_type)
            os.makedirs(samples_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"step_{step:06d}_epoch_{epoch:03d}_{timestamp}.json"
            filepath = os.path.join(samples_dir, filename)

            # Prepare data for saving
            save_data = {
                'metadata': {
                    'step': step,
                    'epoch': epoch,
                    'sample_type': sample_type,
                    'timestamp': timestamp,
                    'sampling_method': sampling_method,
                    'generation_time': generation_time,
                    'validation_loss': validation_loss,
                    'num_sequences': len(sequences),
                    'config': {
                        'max_protein_len': safe_getattr(self.cfg, 'data.max_protein_len', 512),
                        'vocab_size': safe_getattr(self.cfg, 'data.vocab_size_protein', 25),
                        'model_hidden_size': safe_getattr(self.cfg, 'model.hidden_size', 768),
                        'model_n_heads': safe_getattr(self.cfg, 'model.n_heads', 12),
                    }
                },
                'sequences': [],
                'statistics': properties if properties else {}
            }

            # Add sequence data
            for i, seq_info in enumerate(sequences):
                seq_data = {
                    'id': seq_info.get('sample_id', i),
                    'sequence': seq_info.get('sequence', ''),
                    'length': seq_info.get('length', 0),
                    'unique_amino_acids': seq_info.get('unique_amino_acids', 0),
                    'raw_tokens': seq_info.get('raw_tokens', [])
                }

                # Add ESM perplexity if available
                if esm_perplexities and i < len(esm_perplexities):
                    seq_data['esm_perplexity'] = float(esm_perplexities[i])

                save_data['sequences'].append(seq_data)

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)

            print(f"💾 Saved {len(sequences)} samples to: {filepath}")

            # Also save a simple FASTA file for easy sequence analysis
            fasta_filename = f"step_{step:06d}_epoch_{epoch:03d}_{timestamp}.fasta"
            fasta_filepath = os.path.join(samples_dir, fasta_filename)

            with open(fasta_filepath, 'w') as f:
                for i, seq_info in enumerate(sequences):
                    if seq_info.get('sequence'):
                        seq_id = seq_info.get('sample_id', i)
                        esm_ppl = f"_ESM_PPL_{esm_perplexities[i]:.2f}" if esm_perplexities and i < len(esm_perplexities) else ""
                        f.write(f">sample_{seq_id}_step_{step}_epoch_{epoch}{esm_ppl}\n")
                        f.write(f"{seq_info['sequence']}\n")

            print(f"🧬 Saved FASTA sequences to: {fasta_filepath}")

        except Exception as e:
            print(f"⚠️  Error saving samples: {e}")
            import traceback
            traceback.print_exc()

    def get_training_data_stats(self, num_samples: int = 50):
        """Get statistics from training data for comparison."""
        try:
            # Sample some training sequences
            training_sequences = []
            sample_count = 0

            for batch in self.train_loader:
                if sample_count >= num_samples:
                    break

                # Decode sequences from this batch
                for i in range(min(batch.shape[0], num_samples - sample_count)):
                    sequence = self.decode_sequence(batch[i])
                    if sequence:  # Only include non-empty sequences
                        training_sequences.append(sequence)
                        sample_count += 1

                    if sample_count >= num_samples:
                        break

            if not training_sequences:
                return {}

            # Analyze training sequences using the same method
            training_seq_info = [{'sequence': seq} for seq in training_sequences]
            properties = self.analyze_sequence_properties(training_seq_info)

            if 'error' in properties:
                return {}

            # Return summary stats with 'training_' prefix
            return {
                'avg_length': properties['summary']['avg_length'],
                'std_length': properties['summary']['std_length'],
                'avg_unique_aa': properties['summary']['avg_unique_aa'],
                'avg_hydrophobic': properties['summary']['avg_hydrophobic'],
                'avg_polar': properties['summary']['avg_polar'],
                'avg_positive': properties['summary']['avg_positive'],
                'avg_negative': properties['summary']['avg_negative'],
                'avg_aromatic': properties['summary']['avg_aromatic'],
            }

        except Exception as e:
            print(f"⚠️  Could not analyze training data: {e}")
            return {}

    def log_system_metrics(self, step: int):
        """Log system metrics like GPU/XPU memory usage."""
        device_type = str(self.device).split(':')[0]

        if device_type == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
            xpu_memory_allocated = torch.xpu.memory_allocated() / 1024**3  # GB
            xpu_memory_reserved = torch.xpu.memory_reserved() / 1024**3   # GB

            self.log_to_wandb({
                'system/xpu_memory_allocated_gb': xpu_memory_allocated,
                'system/xpu_memory_reserved_gb': xpu_memory_reserved,
                'system/step': step
            }, step=step)
        elif device_type == 'cuda' and torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB

            self.log_to_wandb({
                'system/gpu_memory_allocated_gb': gpu_memory_allocated,
                'system/gpu_memory_reserved_gb': gpu_memory_reserved,
                'system/step': step
            }, step=step)

    def _get_device_memory_str(self):
        """Get device memory usage as a formatted string."""
        device_type = str(self.device).split(':')[0]

        if device_type == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
            return f'{torch.xpu.memory_allocated()/1024**3:.1f}GB'
        elif device_type == 'cuda' and torch.cuda.is_available():
            return f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        else:
            return 'N/A'

    def compute_loss(self, batch):
        """Compute loss with improved curriculum learning."""
        # Ensure batch is 2D: [batch_size, sequence_length]
        if batch.dim() != 2:
            print(f"WARNING: compute_loss received {batch.dim()}D batch with shape {batch.shape}")
            if batch.dim() > 2:
                batch = batch.view(batch.shape[0], -1)
                print(f"Reshaped batch to: {batch.shape}")
            else:
                raise ValueError(f"Batch must be at least 2D, got {batch.dim()}D with shape {batch.shape}")

        # Sample timesteps with optional probabilistic curriculum
        if (hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled and
            hasattr(self.cfg.curriculum, 'probabilistic') and self.cfg.curriculum.probabilistic):
            # Use probabilistic curriculum: bias timestep sampling toward lower noise early in training
            from protlig_dd.processing.noise_lib import sample_timesteps_curriculum
            t = sample_timesteps_curriculum(
                batch_size=batch.shape[0],
                device=self.device,
                training_step=self.state['step'],
                preschool_time=getattr(self.cfg.curriculum, 'preschool_time', 5000),
                curriculum_type=getattr(self.cfg.curriculum, 'difficulty_ramp', 'exponential'),
                bias_strength=getattr(self.cfg.curriculum, 'bias_strength', 2.0)
            )
            if self.state['step'] % 100 == 0:
                from protlig_dd.processing.noise_lib import get_curriculum_stats
                stats = get_curriculum_stats(t, self.state['step'],
                                           getattr(self.cfg.curriculum, 'preschool_time', 5000))
                print(f"📊 Probabilistic curriculum: step={self.state['step']}, "
                      f"progress={stats['curriculum_progress']:.2f}, "
                      f"avg_t={stats['mean_timestep']:.3f}, "
                      f"low_noise%={stats['low_noise_fraction']*100:.1f}, "
                      f"high_noise%={stats['high_noise_fraction']*100:.1f}")
        else:
            # Standard uniform timestep sampling
            t = torch.rand(batch.shape[0], device=self.device) * (1 - 1e-3) + 1e-3

        sigma, dsigma = self.noise(t)
        
        # Apply curriculum learning (sigma scaling approach)
        if (hasattr(self.cfg, 'curriculum') and self.cfg.curriculum.enabled and
            not getattr(self.cfg.curriculum, 'probabilistic', False)):
            # Use sigma scaling curriculum (original approach)
            perturbed_batch = self.graph.sample_transition_curriculum(
                batch,
                sigma,  # Fixed: removed [:, None] as graph functions handle broadcasting internally
                self.state['step'],
                preschool_time=self.cfg.curriculum.preschool_time,
                curriculum_type=getattr(self.cfg.curriculum, 'difficulty_ramp', 'exponential')
            )
        else:
            # Use standard transition (either no curriculum or probabilistic curriculum already applied)
            perturbed_batch = self.graph.sample_transition(batch, sigma)  # Fixed: removed [:, None]

        # Validate perturbed_batch is 2D: [batch_size, sequence_length]
        if perturbed_batch.dim() != 2:
            raise ValueError(f"perturbed_batch must be 2D [batch_size, seq_len], got {perturbed_batch.dim()}D with shape {perturbed_batch.shape}. This indicates an issue in the graph operations.")

        # Forward pass with device-aware autocast and shape validation
        try:
            device_type = str(self.device).split(':')[0]
            # Use DDP model for training
            model_to_use = self.model_ddp if self.use_ddp else self.model

            if self.use_amp and device_type == 'xpu':
                with torch.xpu.amp.autocast():
                    log_score = model_to_use(perturbed_batch, sigma)
                    loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            elif self.use_amp and device_type == 'cuda':
                with torch.cuda.amp.autocast():
                    log_score = model_to_use(perturbed_batch, sigma)
                    loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

                    # Weight by dsigma for better training dynamics
                    loss = (dsigma[:, None] * loss).mean()
            else:
                # No autocast for CPU/MPS
                log_score = model_to_use(perturbed_batch, sigma)
                loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

                # Weight by dsigma for better training dynamics
                loss = (dsigma[:, None] * loss).mean()
        except Exception as e:
            if "Wrong shape" in str(e) or "einops" in str(e).lower():
                print(f"❌ Shape error in model forward pass:")
                print(f"   perturbed_batch shape: {perturbed_batch.shape}")
                print(f"   sigma shape: {sigma.shape}")
                print(f"   Error: {e}")

                # Try to fix the shape issue
                if perturbed_batch.dim() > 2:
                    print(f"🔧 Attempting to fix perturbed_batch shape...")
                    perturbed_batch = perturbed_batch.view(perturbed_batch.shape[0], -1)
                    print(f"   Fixed shape: {perturbed_batch.shape}")

                    # Retry the forward pass with DDP model
                    model_to_use = self.model_ddp if self.use_ddp else self.model
                    device_type = str(self.device).split(':')[0]
                    if self.use_amp and device_type == 'xpu':
                        with torch.xpu.amp.autocast():
                            log_score = model_to_use(perturbed_batch, sigma)
                            loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                            loss = (dsigma[:, None] * loss).mean()
                    elif self.use_amp and device_type == 'cuda':
                        with torch.cuda.amp.autocast():
                            log_score = model_to_use(perturbed_batch, sigma)
                            loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                            loss = (dsigma[:, None] * loss).mean()
                    else:
                        log_score = model_to_use(perturbed_batch, sigma)
                        loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
                        loss = (dsigma[:, None] * loss).mean()
                else:
                    raise e
            else:
                raise e
        
        return loss
    
    def train_step(self, batch):
        """Single training step with optimizations."""
        import time
        step_start_time = time.time()

        self.model.train()

        # Move batch to device and ensure correct shape
        batch = batch.to(self.device)

        # Ensure batch is 2D: [batch_size, sequence_length]
        if batch.dim() > 2:
            print(f"WARNING: Batch has {batch.dim()} dimensions, reshaping from {batch.shape}")
            batch = batch.view(batch.shape[0], -1)
            print(f"Reshaped batch to: {batch.shape}")

        # Compute loss with DDP-aware accumulation
        if self.use_ddp:
            # For DDP, disable gradient accumulation to avoid sync issues
            loss = self.compute_loss(batch)
            effective_accum = 1
        else:
            loss = self.compute_loss(batch) / self.cfg.training.accum
            effective_accum = self.cfg.training.accum

        # Backward pass with device-aware gradient scaling
        # Note: For DDP, we disabled gradient accumulation to avoid sync issues
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Additional metrics for logging
        additional_metrics = {}

        # Update every accum steps (simplified for DDP)
        if self.use_ddp:
            # For DDP: update every step to avoid sync issues
            should_update = True
        else:
            should_update = (self.state['step'] + 1) % self.cfg.training.accum == 0

        if should_update:
            # Unscale gradients for clipping (CUDA only)
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Compute gradient norm before clipping
            total_norm = 0
            for p in chain(self.model.parameters(), self.noise.parameters()):
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            additional_metrics['grad_norm'] = total_norm

            # Clip gradients
            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    chain(self.model.parameters(), self.noise.parameters()),
                    self.cfg.optim.grad_clip
                )

            # Optimizer step with device-aware scaling
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            # Update EMA
            self.ema.update(self.model.parameters())

            # Zero gradients
            self.optimizer.zero_grad()

        step_time = time.time() - step_start_time

        return loss.item() * effective_accum, step_time, additional_metrics

    def eval_reconstruction_loss(self, batch):
        """Evaluate model at very low timestep - approximates reconstruction task."""
        try:
            print("calculating recon loss")
            # Use small timestep that gives meaningful dsigma (above noise minimum)
            # This approximates the reconstruction task while keeping score_entropy meaningful
            t = torch.full((batch.shape[0],), 0.01, device=self.device)

            # Get sigma and dsigma from noise scheduler (proper way)
            sigma, dsigma = self.noise(t)

            # Debug print to see what we're getting
            print(f"Reconstruction: t={t[0]:.6f}, sigma={sigma.mean():.6f}, dsigma={dsigma.mean():.6f}")

            # Forward pass with minimal noise - get score (use DDP model for evaluation too)
            model_to_use = self.model_ddp if self.use_ddp else self.model
            score = model_to_use(batch, sigma)

            # Use the graph's score_entropy method for proper loss computation
            # Expand sigma to match sequence dimension if needed
            if len(sigma.shape) == 1 and len(batch.shape) == 2:
                sigma_expanded = sigma[:, None].expand(-1, batch.shape[1])
            else:
                sigma_expanded = sigma

            entropy = self.graph.score_entropy(score, sigma_expanded, batch, batch)

            # Weight by dsigma similar to training loss for consistency
            weighted_entropy = (dsigma[:, None] * entropy).mean()

            # Debug print to see final values
            print(f"Reconstruction: entropy_mean={entropy.mean():.6f}, weighted_entropy={weighted_entropy:.6f}")

            return weighted_entropy.item()

        except Exception as e:
            print(f"Warning: Could not compute reconstruction loss: {e}")
            return None

    def eval_fixed_noise_levels(self, batch):
        """Evaluate at specific noise levels regardless of curriculum using proper score entropy."""
        try:
            results = {}
            # Fixed timesteps to always test (independent of curriculum)
            # These correspond to different points in the diffusion process
            fixed_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]

            for timestep in fixed_timesteps:
                # Use noise scheduler to get sigma and dsigma for this timestep
                t = torch.full((batch.shape[0],), timestep, device=self.device)
                sigma, dsigma = self.noise(t)

                # Sample noisy version at this noise level using sample_transition
                x_noisy = self.graph.sample_transition(batch, sigma)

                # Forward pass at this specific noise level - get score (use DDP model)
                model_to_use = self.model_ddp if self.use_ddp else self.model
                score = model_to_use(x_noisy, sigma)

                # Use proper score entropy for loss computation
                # Expand sigma to match sequence dimension if needed
                if len(sigma.shape) == 1 and len(batch.shape) == 2:
                    sigma_expanded = sigma[:, None].expand(-1, batch.shape[1])
                else:
                    sigma_expanded = sigma

                entropy = self.graph.score_entropy(score, sigma_expanded, x_noisy, batch)

                # Weight by dsigma like training loss for proper comparison
                weighted_loss = (dsigma[:, None] * entropy).mean()

                # Debug print
                print(f"Timestep {timestep}: sigma = {sigma.mean():.3f}, dsigma = {dsigma.mean():.3f}, entropy = {entropy.mean():.4f}, weighted_loss = {weighted_loss:.4f}")

                results[f'fixed_timestep_{timestep}'] = weighted_loss.item()

            return results

        except Exception as e:
            print(f"Warning: Could not compute fixed noise level evaluation: {e}")
            return {}

    def validate_model(self):
        """Run validation and return metrics including reconstruction and fixed noise losses."""
        self.model.eval()
        val_losses = []
        recon_losses = []
        fixed_noise_metrics = {}

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 10:  # Limit validation batches for speed
                    break

                batch = batch.to(self.device)

                # 1. Standard validation loss computation
                try:
                    loss = self.compute_loss(batch)
                    val_losses.append(loss.item())
                except AttributeError:
                    # Fallback: compute loss directly without state
                    t = torch.rand(batch.shape[0], device=self.device)
                    sigma = self.noise(t)[0]

                    # Forward pass (use DDP model)
                    model_to_use = self.model_ddp if self.use_ddp else self.model
                    logits = model_to_use(batch, sigma)

                    # Simple cross-entropy loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch.view(-1),
                        ignore_index=1  # Ignore padding
                    )
                    val_losses.append(loss.item())

                # 2. Reconstruction loss (noise-agnostic metric)
                recon_loss = self.eval_reconstruction_loss(batch)
                if recon_loss is not None:
                    recon_losses.append(recon_loss)

                # 3. Fixed noise level evaluation (curriculum-agnostic)
                if i == 0:  # Only compute on first batch for speed
                    batch_fixed_noise = self.eval_fixed_noise_levels(batch)
                    for key, value in batch_fixed_noise.items():
                        if key not in fixed_noise_metrics:
                            fixed_noise_metrics[key] = []
                        fixed_noise_metrics[key].append(value)

        self.model.train()

        print(recon_losses)
        # Average all metrics
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_recon_loss = np.mean(recon_losses) if recon_losses else None

        # Average fixed noise metrics
        avg_fixed_noise = {}
        for key, values in fixed_noise_metrics.items():
            avg_fixed_noise[key] = np.mean(values) if values else None

        return avg_val_loss, avg_recon_loss, avg_fixed_noise
    
    def save_checkpoint(self, step, epoch=0, best_loss=float('inf'), is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'best_loss': best_loss,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'noise_state_dict': self.noise.state_dict(),
            'config': self.cfg
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")

    def cleanup_ddp(self):
        """Clean up DDP process group safely."""
        if self.use_ddp and torch.distributed.is_initialized():
            print("🔄 Cleaning up DDP process group...")
            try:
                # Try barrier with timeout, but don't fail if it times out
                torch.distributed.barrier(timeout=30)  # 30 second timeout
                print("✅ Final barrier completed")
            except Exception as e:
                print(f"⚠️  Final barrier failed (continuing cleanup): {e}")

            try:
                torch.distributed.destroy_process_group()
                print("✅ DDP process group destroyed")
            except Exception as e:
                print(f"⚠️  DDP cleanup failed: {e}")

    def train(self, wandb_project: str, wandb_name: str):
        """Main training loop with comprehensive Wandb logging."""
        print("Starting training...")

        # Setup components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_optimizer()
        self.wrap_model_ddp()
        # Setup Wandb with comprehensive configuration
        self.setup_wandb(wandb_project, wandb_name)

        # Setup model watching after model is created
        self.setup_wandb_model_watching()

        # Training state
        best_loss = float('inf')
        step = 0
        start_epoch = 0
        running_loss = 0.0
        log_interval_start_time = time.time()

        # Check for existing checkpoint to resume from (unless forcing fresh start)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
        if self.resume_checkpoint:
            checkpoint_path = self.resume_checkpoint

        if not self.force_fresh_start and os.path.exists(checkpoint_path):
            print(f"📂 Found existing checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Load optimizer state
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load scheduler state
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                # Load training state
                step = checkpoint.get('step', 0)
                start_epoch = checkpoint.get('epoch', 0)
                best_loss = checkpoint.get('best_loss', float('inf'))

                # Initialize state if not exists
                if not hasattr(self, 'state'):
                    self.state = {'step': step}
                else:
                    self.state['step'] = step

                print(f"✅ Resumed from checkpoint:")
                print(f"   Step: {step}")
                print(f"   Epoch: {start_epoch}")
                print(f"   Best loss: {best_loss:.4f}")

            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                print("Starting training from scratch...")
                step = 0
                start_epoch = 0
                best_loss = float('inf')
        else:
            if self.force_fresh_start:
                print("🆕 Force fresh start enabled. Starting training from scratch...")
            else:
                print("🆕 No existing checkpoint found. Starting training from scratch...")

        # Initialize state if not exists
        if not hasattr(self, 'state'):
            self.state = {'step': step}

        print(f"🚀 Starting training for {self.cfg.training.n_iters} steps (from step {step})...")

        # Initial generation test to verify setup
        if step == 0:
            print("\n🧪 Running initial generation test to verify setup...")
            # Use shorter length for initial test to be faster, but still reasonable
            initial_max_length = min(100, safe_getattr(self.cfg, 'data.max_protein_len', 512))
            initial_test_success = self.quick_generation_test(step, 0, num_samples=2, max_length=initial_max_length)
            if initial_test_success:
                print("✅ Initial generation test passed - training can proceed")
            else:
                print("⚠️  Initial generation test had issues - but continuing training")

        # Use epochs override priority: command line > wandb config > config file
        if self.epochs_override is not None:
            total_epochs = self.epochs_override
        elif wandb.run and 'epochs' in wandb.config:
            total_epochs = wandb.config.epochs
        else:
            total_epochs = self.cfg.training.epochs

        print(f"🔄 Training for {total_epochs} epochs (override: {self.epochs_override}, config: {self.cfg.training.epochs})")

        for epoch in range(start_epoch, total_epochs):
            # Set epoch for distributed sampler to ensure proper shuffling
            if self.use_ddp and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
                print(f"🔄 Set sampler epoch to {epoch} for proper data shuffling")

            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')

            for batch in progress_bar:
                # Training step with timing
                loss, step_time, additional_metrics = self.train_step(batch)

                epoch_loss += loss
                running_loss += loss
                num_batches += 1
                step += 1
                self.state['step'] = step

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': f'{step}/{self.cfg.training.n_iters}',
                    'device_mem': self._get_device_memory_str()
                })

                # Detailed logging
                if step % self.cfg.training.log_freq == 0:
                    avg_loss = running_loss / self.cfg.training.log_freq
                    interval_time = time.time() - log_interval_start_time

                    # Log training metrics
                    self.log_training_metrics(
                        step=step,
                        loss=avg_loss,
                        lr=self.scheduler.get_last_lr()[0],
                        epoch=epoch,
                        batch_time=step_time,
                        additional_metrics=additional_metrics
                    )

                    # Log system metrics
                    self.log_system_metrics(step)

                    # Reset running loss and timer
                    running_loss = 0.0
                    log_interval_start_time = time.time()

                # Log reconstruction and fixed noise losses every 100 steps (RANK 0 ONLY)
                if step % self.cfg.training.log_freq * 5 == 0 and step > 0 and self.rank == 0:
                    print(f"🔍 Computing reconstruction and fixed noise losses at step {step}...")
                    try:
                        # Get a single batch for loss computation
                        sample_batch = next(iter(self.val_loader))
                        sample_batch = sample_batch.to(self.device)

                        # Use smaller batch size to avoid OOM - take first 4 samples
                        small_batch_size = min(4, sample_batch.shape[0])
                        small_batch = sample_batch[:small_batch_size]

                        # Compute reconstruction loss (t=0, no noise)
                        recon_loss = self.eval_reconstruction_loss(small_batch)

                        # Compute fixed noise level losses
                        fixed_noise_metrics = self.eval_fixed_noise_levels(small_batch)

                        # Log to wandb
                        metrics_to_log = {
                            'train/step': step,
                            'train/epoch': epoch,
                        }

                        if recon_loss is not None:
                            metrics_to_log['train/reconstruction_loss'] = recon_loss
                            print(f"   📊 Reconstruction Loss: {recon_loss:.4f}")
                        else:
                            print("   ⚠️  Reconstruction loss is None")

                        for noise_key, noise_value in fixed_noise_metrics.items():
                            if noise_value is not None:
                                metrics_to_log[f'train/{noise_key}'] = noise_value
                                print(f"   📊 {noise_key}: {noise_value:.4f}")
                            else:
                                print(f"   ⚠️  {noise_key}: None (skipped)")
                            wandb.log(metrics_to_log, step=step)

                    except Exception as e:
                        print(f"⚠️  Warning: Could not compute reconstruction/fixed noise losses: {e}")
                        # Continue training even if logging fails

                # Note: No barrier needed here - only rank 0 does evaluation, others continue normally

                # Quick generation test (more frequent than comprehensive evaluation)
                quick_gen_freq = getattr(self.cfg.training, 'quick_gen_freq', self.cfg.training.log_freq * 10)
                if step % quick_gen_freq == 0 and step > 0:
                    self.quick_generation_test(step, epoch)

                # Comprehensive evaluation (skip in minimal mode)
                if not self.minimal_mode and step % self.cfg.training.eval_freq == 0:
                    val_loss, generation_properties = self.comprehensive_evaluation(
                        step, epoch, num_samples=10, sampling_method=self.sampling_method
                    )

                    # Update best loss tracking
                    if val_loss < best_loss:
                        best_loss = val_loss
                        print(f"🎉 New best validation loss: {val_loss:.4f}")

                    print(f"✅ Evaluation completed. Val loss: {val_loss:.4f}")

                # Checkpointing
                if step % self.cfg.training.snapshot_freq == 0:
                    avg_epoch_loss = epoch_loss / num_batches
                    is_best = avg_epoch_loss < best_loss

                    if is_best:
                        best_loss = avg_epoch_loss
                        print(f"🎉 New best loss: {best_loss:.4f}")

                    self.save_checkpoint(step, epoch, best_loss, is_best)

                    # Log checkpoint info
                    self.log_to_wandb({
                        'checkpoint/step': step,
                        'checkpoint/best_loss': best_loss,
                        'checkpoint/current_loss': avg_epoch_loss,
                        'checkpoint/is_best': is_best
                    }, step=step)

                # Early stopping check
                if step >= self.cfg.training.n_iters:
                    print(f"\n✅ Reached maximum iterations ({self.cfg.training.n_iters})")
                    break

            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches
            self.log_to_wandb({
                'epoch/loss': avg_epoch_loss,
                'epoch/number': epoch,
                'epoch/step': step
            }, step=step)

            print(f"📊 Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

            # End-of-epoch comprehensive evaluation
            print(f"\n🎯 END-OF-EPOCH EVALUATION - Epoch {epoch+1}")
            val_loss, generation_properties = self.comprehensive_evaluation(
                step, epoch+1, num_samples=20  # More samples at end of epoch
            )

            # Update best loss and save best model
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"🏆 New best model! Validation loss: {val_loss:.4f}")
                self.save_checkpoint(step, epoch+1, best_loss, is_best=True)

            # Log epoch summary
            self.log_to_wandb({
                'epoch/number': epoch + 1,
                'epoch/avg_train_loss': avg_epoch_loss,
                'epoch/val_loss': val_loss,
                'epoch/best_loss': best_loss,
                'epoch/step': step
            }, step=step)

            if step >= self.cfg.training.n_iters:
                break

        # Final checkpoint and comprehensive evaluation
        print("\n🏁 Training completed!")
        self.save_checkpoint(step, epoch+1, best_loss, is_best=False)

        # Final comprehensive evaluation
        print("\n🎊 FINAL COMPREHENSIVE EVALUATION")
        final_val_loss, final_generation_properties = self.comprehensive_evaluation(
            step, epoch+1, num_samples=30  # More samples for final evaluation
        )

        # Training summary
        self.log_to_wandb({
            'summary/final_step': step,
            'summary/final_train_loss': epoch_loss / num_batches,
            'summary/final_val_loss': final_val_loss,
            'summary/best_loss': best_loss,
            'summary/total_epochs': epoch + 1,
            'summary/training_completed': True
        })

        print(f"\n🎉 TRAINING SUMMARY:")
        print(f"   Total Steps: {step}")
        print(f"   Total Epochs: {epoch + 1}")
        print(f"   Final Validation Loss: {final_val_loss:.4f}")
        print(f"   Best Validation Loss: {best_loss:.4f}")
        if 'error' not in final_generation_properties:
            print(f"   Final Generation Quality:")
            summary = final_generation_properties['summary']
            print(f"     - Avg Sequence Length: {summary['avg_length']:.1f}")
            print(f"     - Avg Unique AAs: {summary['avg_unique_aa']:.1f}")
            print(f"     - Composition Balance: H={summary['avg_hydrophobic']:.1f}% P={summary['avg_polar']:.1f}%")

        print(f"📈 Final validation loss: {final_val_loss:.4f}")
        print(f"🏆 Best training loss: {best_loss:.4f}")

        # Clean up DDP process group
        self.cleanup_ddp()

        # Cleanup logging
        if self.use_wandb:
            wandb.finish()
            print("✅ Wandb logging completed")
        else:
            print("✅ File logging completed")


def main():
    
    print("\n" + "="*80)
    print("🧬 OPTIMIZED UNIREF50 SEDD TRAINING")
    print("="*80)
    print("🚀 Enhanced with V100-compatible attention & curriculum learning")
    print("📊 Full Wandb experiment tracking enabled")
    print("="*80 + "\n")

    parser = argparse.ArgumentParser(description="Train optimized UniRef50 model")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--datafile", type=str, default="./input_data/processed_uniref50.pt", help="Data file path")
    parser.add_argument("--wandb_project", type=str, default="uniref50-sedd", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start (ignore existing checkpoints)")
    parser.add_argument("--sampling_method", type=str, default="rigorous",
                       choices=["rigorous", "simple"],
                       help="Sampling method: 'rigorous' (CTMC) or 'simple' (heuristic)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs (useful for hyperparameter sweeps)")
    parser.add_argument("--tokenize_on_fly", action="store_true",
                       help="Tokenize untokenized data on the fly (use with raw sequence data)")
    parser.add_argument("--use_streaming", action="store_true",
                       help="Use streaming mode for very large files (>50GB)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging and use file logging instead")
    parser.add_argument("--minimal_mode", action="store_true",
                       help="Minimal mode: disable evaluations and complex logging for large-scale DDP debugging")

    args = parser.parse_args()

    # Generate wandb name if not provided
    if args.wandb_name is None:
        args.wandb_name = f"uniref50_optimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"📁 Work directory: {args.work_dir}")
    print(f"⚙️  Config file: {args.config}")
    print(f"💾 Data file: {args.datafile}")
    print(f"🏷️  Wandb project: {args.wandb_project}")
    print(f"🏷️  Wandb run: {args.wandb_name}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🧬 Sampling method: {args.sampling_method}")
    print(f"🔤 Tokenize on fly: {args.tokenize_on_fly}")
    print(f"🌊 Use streaming: {args.use_streaming}")
    print(f"📊 Wandb logging: {'disabled' if args.no_wandb else 'enabled'}")
    print()

    try:
        # Create trainer
        print("🔧 Initializing trainer...")
        trainer = OptimizedUniRef50Trainer(
            work_dir=args.work_dir,
            config_file=args.config,
            datafile=args.datafile,
            rank=0,
            world_size=0,
            dev_id=args.device,
            seed=args.seed,
            force_fresh_start=args.fresh,
            sampling_method=args.sampling_method,
            epochs_override=args.epochs,
            use_wandb=not args.no_wandb,
            minimal_mode=args.minimal_mode
        )

        # Set tokenization and streaming options
        trainer.tokenize_on_fly = args.tokenize_on_fly
        trainer.use_streaming = args.use_streaming

        print("✅ Trainer initialized successfully!")
        print()

        # Start training
        trainer.train(args.wandb_project, args.wandb_name)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check the error above and verify your configuration.")

        # Cleanup DDP even if training failed
        try:
            if 'trainer' in locals() and hasattr(trainer, 'cleanup_ddp'):
                trainer.cleanup_ddp()
        except Exception as cleanup_error:
            print(f"⚠️  DDP cleanup failed: {cleanup_error}")

        return 1

    print("\n🎉 Training completed successfully!")
    return 0


if __name__ == "__main__":
    main()
