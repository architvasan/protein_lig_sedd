# pip installs (if needed):
# pip install torch torchvision torchaudio
# pip install fair-esm transformers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)
from . import rotary
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim+1, dim)
        
    def forward(self, x):
        return self.embedding(x)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for attending between tracks
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context, x_mask=None, context_mask=None):
        """
        x: [B, L_x, D] - query sequence
        context: [B, L_c, D] - key/value sequence from other track
        """
        x_res = x
        x_norm = self.norm(x)
        
        # Create attention mask if needed
        attn_mask = None
        if context_mask is not None:
            # Convert boolean mask to float mask for attention
            # True = keep, False = mask out
            # MHA expects True = mask out, so invert
            key_padding_mask = ~context_mask
        else:
            key_padding_mask = None
            
        x_ca, _ = self.cross_attn(
            x_norm, context, context,
            key_padding_mask=key_padding_mask
        )
        x = x_res + self.dropout(x_ca)
        return x

class ProteinEncoderESM(nn.Module):
    """
    Wraps a pretrained ESM-2 and exposes:
      - encode(seq_strs): returns per-residue embeddings [B, Lp, Dp]
                          and a pooled [B, Dp] (CLS) embedding
    """
    def __init__(self, esm_variant: str = "esm2_t33_650M_UR50D", device="cuda"):
        super().__init__()
        import esm  # fair-esm
        load_fn = getattr(esm.pretrained, esm_variant)
        self.esm_model, self.alphabet = load_fn()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()
        self.device = device
        self.esm_model.to(device)

    @torch.no_grad()
    def encode(self, seq_strs):
        """
        seq_strs: List[str] of protein sequences (standard letters)
        Returns:
            per_res_emb: [B, Lp, Dp]
            pooled_emb:  [B, Dp]  (mean-pooled over residues excluding padding)
            attn_mask:   [B, Lp]  (True = keep)
        """
        # ESM wants list of (name, sequence)
        batch = [("seq", s) for s in seq_strs]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        out = self.esm_model(tokens, repr_layers=[33], return_contacts=False)
        # Layer 33 is the final for esm2_t33_650M; change if using other variants
        token_reps = out["representations"][33]  # [B, Lp+special, Dp]

        # ESM adds BOS (start) token; remove BOS for per-residue; keep CLS via BOS position if desired.
        # Convention: take token_reps[:, 1:-1] for residues (drop BOS/EOS), but some variants only add BOS.
        # We'll be robust: trim BOS; keep rest as residues; ignore any final EOS if present.
        # Alphabet padding idx:
        pad_idx = self.alphabet.padding_idx
        padding_mask = tokens != pad_idx  # [B, Lp+special]
        # Try removing first token (BOS)
        per_res = token_reps[:, 1:, :]
        per_res_mask = padding_mask[:, 1:]

        # If model includes EOS, you can drop it as well by checking sequence lengths; keeping it is usually harmless.
        # Simple pooled embedding: mean over non-pad residues
        lengths = per_res_mask.sum(dim=1).clamp(min=1)
        pooled = (per_res * per_res_mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)

        return per_res, pooled, per_res_mask


# --------- Ligand encoder (Molformer via HF) ----------
class LigandEncoderMolformer(nn.Module):
    """
    Wraps a pretrained Molformer (Hugging Face).
    You may need to pick a specific checkpoint, e.g.:
      - "DeepChem/molformer-embeddings" (example name) or
      - a Huawei Noah "Molformer" variant on HF.
    Adjust model_name to a checkpoint you have locally or can download.
    """
    def __init__(self, model_name: str = "ibm-research/MoLFormer-XL-both-10pct", device="cuda"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def encode(self, smiles_list):
        """
        smiles_list: List[str] of SMILES
        Returns:
            per_tok_emb: [B, Ll, Dl]
            pooled_emb:  [B, Dl] (CLS or mean)
            attn_mask:   [B, Ll] (True = keep)
        """
        toks = self.tokenizer(
            smiles_list,
            padding='max_length',
            truncation=True,
            max_length=202,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(**toks)
        hidden = out.last_hidden_state  # [B, Ll, Dl]
        mask = toks["attention_mask"].bool()  # [B, Ll]
        # pooled = CLS (token 0) if exists, else mean
        pooled = hidden[:, 0, :] if (hidden.size(1) > 0) else hidden.mean(dim=1)
        return hidden, pooled, mask



class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout, cond_dim, mlp_ratio=4, enable_cross_attention=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio
        self.enable_cross_attention = enable_cross_attention
        
        self.norm1 = LayerNorm(self.dim)

        # Self-attention components
        self.attn_qkv = nn.Linear(self.dim, 3 * self.dim)
        self.attn = nn.MultiheadAttention(self.dim, self.n_heads, dropout=self.dropout, batch_first=True)
        self.attn_out = nn.Linear(self.dim, self.dim)
        
        # Optional cross-attention to other track
        if self.enable_cross_attention:
            self.cross_attn = CrossAttentionLayer(self.dim, self.n_heads, dropout)
        
        self.dropout1 = nn.Dropout(self.dropout)
        
        self.norm2 = LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * self.mlp_ratio, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.dim * self.mlp_ratio, self.dim, bias=True),
        )
        self.dropout2 = nn.Dropout(self.dropout)
        
        self.adaLN_modulation = nn.Linear(self.cond_dim, 6 * self.dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, other_track=None, other_mask=None, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        
        # Self-attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            #print(cos)
            #print(sin)
            #print(qkv)
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        
        q, k, v = qkv.unbind(dim=2)
        q = rearrange(q, "b s h d -> s b (h d)")
        k = rearrange(k, "b s h d -> s b (h d)")
        v = rearrange(v, "b s h d -> s b (h d)")
        
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.transpose(0, 1)
        
        x = bias_dropout_scale_fn(self.attn_out(attn_out), None, gate_msa, x_skip, self.dropout)
        
        # Cross-attention to other track (if enabled and other track provided)
        if self.enable_cross_attention and other_track is not None:
            x = self.cross_attn(x, other_track, x_mask=None, context_mask=other_mask)
        
        # MLP
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), 
            None, gate_mlp, x, self.dropout
        )
        
        return x


class DualTrackDiffusionTransformer(nn.Module):
    """
    A dual-track diffusion transformer that can handle both protein and ligand tracks
    with optional cross-attention between them.
    """
    def __init__(self, config, vocab_size_protein, vocab_size_ligand):
        super().__init__()
        self.config = config
        self.enable_cross_attention = getattr(config.model, 'enable_cross_attention', True)
        
        # Separate embeddings for each track
        self.protein_embed = EmbeddingLayer(config.model.hidden_size, vocab_size_protein)
        self.ligand_embed = EmbeddingLayer(config.model.hidden_size, vocab_size_ligand)
        
        # Shared timestep embedder
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        
        # Separate rotary embeddings for each track (could be shared)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
        
        # Transformer blocks
        self.n_blocks = config.model.n_blocks
        self.blocks_prot = nn.ModuleList([
            DDiTBlock(
                config.model.hidden_size, 
                config.model.n_heads, 
                config.model.dropout, 
                config.model.cond_dim,
                enable_cross_attention=self.enable_cross_attention
            ) for _ in range(self.n_blocks)
        ])

        self.blocks_lig = nn.ModuleList([
            DDiTBlock(
                config.model.hidden_size, 
                config.model.n_heads, 
                config.model.dropout, 
                config.model.cond_dim,
                enable_cross_attention=self.enable_cross_attention
            ) for _ in range(self.n_blocks)
        ])
        
        # Separate output layers for each vocabulary
        self.protein_output = DDitFinalLayer(config.model.hidden_size, vocab_size_protein, config.model.cond_dim)
        self.ligand_output = DDitFinalLayer(config.model.hidden_size, vocab_size_ligand, config.model.cond_dim)
        
        self.scale_by_sigma = config.model.scale_by_sigma
        self.absorb = config.graph.type == "absorb"

    def forward_single_track(
        self, 
        indices, 
        sigma, 
        track_type="protein",
        other_track_hidden=None,
        other_track_mask=None
    ):
        """
        Forward pass for a single track with optional cross-attention to another track
        """
        # Embed tokens based on track type
        if track_type == "protein":
            x = self.protein_embed(indices)
            output_layer = self.protein_output
        elif track_type == 'ligand':  # ligand
            x = self.ligand_embed(indices)
            output_layer = self.ligand_output
        else:
            raise ValueError('track type not supported')

        # Timestep conditioning
        c = F.silu(self.sigma_map(sigma))
        
        # Rotary embeddings
        rotary_cos_sin = self.rotary_emb(x)
        
        # Pass through transformer blocks
        if track_type == 'protein':
            with torch.cuda.amp.autocast(dtype=torch.float32):
                for i in range(self.n_blocks):
                    x = self.blocks_prot[i](
                        x, 
                        rotary_cos_sin, 
                        c, 
                        other_track=other_track_hidden,
                        other_mask=other_track_mask,
                        seqlens=None
                    )

        elif track_type == 'ligand':
            with torch.cuda.amp.autocast(dtype=torch.float32):
                for i in range(self.n_blocks):
                    x = self.blocks_lig[i](
                        x, 
                        rotary_cos_sin, 
                        c, 
                        other_track=other_track_hidden,
                        other_mask=other_track_mask,
                        seqlens=None
                    )

        else:
            raise ValueError("track type not supported")
        
        # Final output projection
        x = output_layer(x, c)
        
        # Scale by sigma if needed
        if self.scale_by_sigma and self.absorb:
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)
        # How does this scaling work?
        # This scaling adjusts the logits based on the noise level (sigma).
        # It helps the model to account for the varying uncertainty in the input data
        # at different timesteps of the diffusion process.
        # The term esigm1_log is log(exp(sigma) - 1), which relates to the noise schedule.
        # The subtraction of np.log(x.shape[-1] - 1) normalizes the logits
        # by the vocabulary size minus one (excluding padding).
        # This ensures that the model's predictions are calibrated correctly
        # across different noise levels.
     

        # Zero out the logits at the input token positions
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
       # How does torch.scatter work here?
       # It replaces the logits at the positions specified by 'indices' with zeros.
       # This prevents the model from predicting the original token at those positions.
       # For example, if indices = [2, 0, 1] for a sequence of length 3,
       # it will set x[:, 0, 2], x[:, 1, 0], x[:, 2, 1] to zero. 
        #print(f"Single track shape {x.size()}")
        return x

    def forward_joint(
        self,
        protein_indices,
        ligand_indices,
        sigma,
        protein_mask=None,
        ligand_mask=None
    ):
        """
        Joint forward pass with cross-attention between tracks
        """
        # Embed both tracks
        protein_hidden = self.protein_embed(protein_indices)
        ligand_hidden = self.ligand_embed(ligand_indices)
        
        # Timestep conditioning
        c = F.silu(self.sigma_map(sigma))
        
        # Rotary embeddings for both tracks
        protein_rotary = self.rotary_emb(protein_hidden)
        ligand_rotary = self.rotary_emb(ligand_hidden)
        
        # Pass through transformer blocks with cross-attention
        with torch.cuda.amp.autocast(dtype=torch.float32):
            for i in range(self.n_blocks):
                # Update protein track (attending to ligand)
                new_protein = self.blocks_prot[i](
                    protein_hidden,
                    protein_rotary,
                    c,
                    other_track=ligand_hidden if self.enable_cross_attention else None,
                    other_mask=ligand_mask if self.enable_cross_attention else None,
                    seqlens=None
                )
                
                # Update ligand track (attending to protein)
                new_ligand = self.blocks_lig[i](
                    ligand_hidden,
                    ligand_rotary,
                    c,
                    other_track=protein_hidden if self.enable_cross_attention else None,
                    other_mask=protein_mask if self.enable_cross_attention else None,
                    seqlens=None
                )
                
                protein_hidden = new_protein
                ligand_hidden = new_ligand
            
            # Final output projections
            protein_logits = self.protein_output(protein_hidden, c)
            ligand_logits = self.ligand_output(ligand_hidden, c)
        
        # Scale by sigma if needed
        if self.scale_by_sigma and self.absorb:
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(protein_logits.dtype)[:, None, None]
            protein_logits = protein_logits - esigm1_log - np.log(protein_logits.shape[-1] - 1)
            ligand_logits = ligand_logits - esigm1_log - np.log(ligand_logits.shape[-1] - 1)
        
        # Zero out the logits at input positions
        protein_logits = torch.scatter(protein_logits, -1, protein_indices[..., None], torch.zeros_like(protein_logits[..., :1]))
        ligand_logits = torch.scatter(ligand_logits, -1, ligand_indices[..., None], torch.zeros_like(ligand_logits[..., :1]))
        
        return protein_logits, ligand_logits


class ProteinLigandDiffusionModel(nn.Module):
    """
    Main model class that handles different training modes:
    - protein_only: Train protein diffusion without ligand
    - ligand_only: Train ligand diffusion without protein
    - ligand_given_protein: Train ligand diffusion conditioned on protein
    - protein_given_ligand: Train protein diffusion conditioned on ligand
    - joint: Train both with cross-attention
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get vocab sizes
        vocab_size_protein = 36 + (1 if config.graph.type == "absorb" else 0)
        vocab_size_ligand = 2364 + (1 if config.graph.type == "absorb" else 0)
        
        # Initialize the dual-track transformer
        self.transformer = DualTrackDiffusionTransformer(
            config, 
            vocab_size_protein, 
            vocab_size_ligand
        )
        
        # Optional: pretrained encoders for conditioning (can be removed if not needed)
        self.prot_enc = ProteinEncoderESM(device=config.model.device)
        self.lig_enc = LigandEncoderMolformer(device=config.model.device)
        # Projectors for pretrained embeddings
        self.prot_proj = nn.Linear(1280, config.model.hidden_size)  # ESM dim to hidden
        self.lig_proj = nn.Linear(768, config.model.hidden_size)   # Molformer dim to hidden


    def forward(
        self,
        protein_indices=None,
        ligand_indices=None,
        timesteps=None,
        protein_mask=None,
        ligand_mask=None,
        mode="joint",
        protein_seq_str=None,  # For pretrained conditioning
        ligand_smiles_str=None,  # For pretrained conditioning
        use_pretrained_conditioning=False,  # Override config
    ):
        """
        Forward pass with different modes
        
        Args:
            protein_indices: [B, L_p] protein token indices
            ligand_indices: [B, L_l] ligand token indices
            timesteps: [B] diffusion timesteps
            protein_mask: [B, L_p] attention mask for protein
            ligand_mask: [B, L_l] attention mask for ligand
            mode: Training mode - "protein_only", "ligand_only", "ligand_given_protein", 
                  "protein_given_ligand", "joint"
            protein_seq_str: List of protein sequences (for pretrained conditioning)
            ligand_smiles_str: List of SMILES strings (for pretrained conditioning)
        """
        
        if mode == "protein_only":
            # Train only protein diffusion without any ligand information
            return self.transformer.forward_single_track(
                protein_indices, 
                timesteps, 
                track_type="protein",
                other_track_hidden=None,
                other_track_mask=None
            )
        
        elif mode == "ligand_only":
            # Train only ligand diffusion without any protein information
            return self.transformer.forward_single_track(
                ligand_indices,
                timesteps,
                track_type="ligand", 
                other_track_hidden=None,
                other_track_mask=None
            )
        
        elif mode == "ligand_given_protein":
            # Train ligand diffusion conditioned on protein
            if use_pretrained_conditioning and protein_seq_str is not None:
                # Use pretrained protein encoder
                with torch.no_grad():
                    p_per, _, p_mask = self.prot_enc.encode(protein_seq_str)
                protein_hidden = self.prot_proj(p_per)
                protein_mask = p_mask
            else:
                # Use protein diffusion track embeddings
                protein_hidden = self.transformer.protein_embed(protein_indices)
                # Get intermediate representations by passing through some blocks
                c = F.silu(self.transformer.sigma_map(timesteps))
                rotary_cos_sin = self.transformer.rotary_emb(protein_hidden)
                for i in range(self.transformer.n_blocks // 2):  # Use first half of blocks
                    protein_hidden = self.transformer.blocks_prot[i](
                        protein_hidden, rotary_cos_sin, c, None, None, None
                    )
            
            return self.transformer.forward_single_track(
                ligand_indices,
                timesteps,
                track_type="ligand",
                other_track_hidden=protein_hidden,
                other_track_mask=protein_mask
            )
        
        elif mode == "protein_given_ligand":
            # Train protein diffusion conditioned on ligand
            if use_pretrained_conditioning and ligand_smiles_str is not None:
                # Use pretrained ligand encoder
                with torch.no_grad():
                    l_per, _, l_mask = self.lig_enc.encode(ligand_smiles_str)
                ligand_hidden = self.lig_proj(l_per)
                ligand_mask = l_mask
            else:
                # Use ligand diffusion track embeddings
                ligand_hidden = self.transformer.ligand_embed(ligand_indices)
                c = F.silu(self.transformer.sigma_map(timesteps))
                rotary_cos_sin = self.transformer.rotary_emb(ligand_hidden)
                for i in range(self.transformer.n_blocks // 2):
                    ligand_hidden = self.transformer.blocks_lig[i](
                        ligand_hidden, rotary_cos_sin, c, None, None, None
                    )
            
            return self.transformer.forward_single_track(
                protein_indices,
                timesteps,
                track_type="protein",
                other_track_hidden=ligand_hidden,
                other_track_mask=ligand_mask
            )
        
        elif mode == "joint":
            # Train both tracks jointly with cross-attention
            return self.transformer.forward_joint(
                protein_indices,
                ligand_indices,
                timesteps,
                protein_mask,
                ligand_mask
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def disable_cross_attention(self):
        """Utility method to disable cross-attention in all blocks"""
        for block in self.transformer.blocks:
            block.enable_cross_attention = False
    
    def enable_cross_attention(self):
        """Utility method to enable cross-attention in all blocks"""
        for block in self.transformer.blocks:
            block.enable_cross_attention = True
