# pip installs (if needed):
# pip install torch torchvision torchaudio
# pip install fair-esm transformers

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
from typing import Optional
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
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings




class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_dim, dim)#vocab_dim + 1, dim)
        # Optionally initialize manually
        #torch.nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        #self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))#(vocab_dim+1, dim)))
        #torch.nn.init.xavier_uniform_(self.embedding)
        #torch.nn.init.uniform_(self.embedding)
        #torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        #print(self.embedding)
        #print(x.size)
        #print("x.shape:", x.shape)
        #print("x.min():", x.min().item())
        #print("x.max():", x.max().item())
        #print("embedding.num_embeddings:", self.embedding.num_embeddings)
        return self.embedding(x)
        #return self.embedding[x]

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

# --------- Protein encoder (ESM-2) ----------

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


# --------- Shared latent projections ----------
class SharedLatentProjector(nn.Module):
    """
    Projects protein and ligand embeddings to a shared latent dim.
    """
    def __init__(self, d_prot: int, d_lig: int, d_latent: int):
        super().__init__()
        self.prot_proj = nn.Sequential(
            nn.Linear(d_prot, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.lig_proj = nn.Sequential(
            nn.Linear(d_lig, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward_protein(self, x):  # [B, Lp, Dp] or [B, Dp]
        return self.prot_proj(x)

    def forward_ligand(self, x):   # [B, Ll, Dl] or [B, Dl]
        return self.lig_proj(x)


# --------- A simple Transformer block with optional cross-attention ----------
class CrossAttentionBlock(nn.Module):
    """
    A minimal Transformer encoder block that supports optional cross-attention
    to a conditioning sequence.
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_mask=None, cond=None, cond_mask=None):
        # Self-attention
        #x_res = x
        #x = self.norm1(x)
        #x_sa, _ = self.self_attn(x, x, x, key_padding_mask=(~x_mask) if x_mask is not None else None)
        #x = x_res + x_sa

        # Cross-attention (if cond provided)
        if cond is not None:
            x_res = x
            x_norm = self.norm1(x)
            cond_kv = cond
            x_ca, _ = self.cross_attn(
                x_norm, cond_kv, cond_kv,
                key_padding_mask=(~cond_mask) if cond_mask is not None else None
            )
            x = x_res + x_ca

        # FFN
        x_res = x
        x = self.norm3(x)
        x = x_res + self.ff(x)
        return x

class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, dropout, cond_dim, mlp_ratio=4):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = LayerNorm(self.dim)

        # qkv projection ( needed for rotary)
        self.attn_qkv = nn.Linear(self.dim, 3 * self.dim)
        self.attn = nn.MultiheadAttention(self.dim, self.n_heads, dropout=self.dropout, batch_first=True)
        self.attn_out = nn.Linear(self.dim, self.dim)

        self.cross_attn = CrossAttentionBlock(self.dim, self.n_heads, self.dim)
        self.dropout1 = nn.Dropout(self.dropout)

        self.norm2 = LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * self.mlp_ratio, bias = True),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.dim * self.mlp_ratio, self.dim, bias = True),
        )
        self.dropout2 = nn.Dropout(self.dropout)

        self.adaLN_modulation = nn.Linear(self.cond_dim, 6 * self.dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        #if self.cond_dim is not None:
        #    self.cond_proj = nn.Linear(self.cond_dim, self.dim * 2)

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None, cond=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype
        #print(x.size())
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        
        # split q, k, v and merge heads
        q, k, v = qkv.unbind(dim=2)  # each [b, s, h, d]
        q = rearrange(q, "b s h d -> s b (h d)")
        k = rearrange(k, "b s h d -> s b (h d)")
        v = rearrange(v, "b s h d -> s b (h d)")

        # run MultiheadAttention
        attn_out, _ = self.attn(q, k, v)  # [s, b, d_model]
        attn_out = attn_out.transpose(0, 1)  # [b, s, d_model]

        ## back to [b, s, d]
        #attn_out = attn_out.transpose(0, 1)  # [b, s, d_model]

        #qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        #if seqlens is None:
        #    cu_seqlens = torch.arange(
        #        0, (batch_size + 1) * seq_len, step=seq_len,
        #        dtype=torch.int32, device=qkv.device
        #    )
        #else:
        #    cu_seqlens = seqlens.cumsum(-1)
        #x = flash_attn_varlen_qkvpacked_func(
        #    qkv, cu_seqlens, seq_len, 0., causal=False)

        #x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(attn_out), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)

        x = self.cross_attn(x, x_mask=None, cond=cond, cond_mask=None)
        return x

# --------- Shared diffusion transformer (token-level) ----------
class SharedDiffusionTransformer(nn.Module):
    """
    A lightweight diffusion transformer that denoises discrete tokens.
    It supports conditioning via cross-attn on a shared latent sequence.

    Inputs:
      noisy_tokens:       [B, L, vocab] as logits or integer tokens (you decide)
      token_emb:          embedding layer to map integers -> vectors [B, L, D]
      cond_seq:           [B, Lc, D] conditioning sequence in shared latent space
      t_emb:              [B, D] diffusion timestep embedding (optional)
    """
    def __init__(self, config, tokens):#hidden_size, vocab_size, cond_dim, n_heads, dropout, scale_by_sigma, graph_type = 'absorb'):#self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048):
        super().__init__()
        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = tokens + (1 if self.absorb else 0) #config.
        #print(f"{vocab_size=}")
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        #self.esm_proj = nn.Linear(config.model.esm_dim, config.model.cond_dim)# if config.model.use_esm else None
        #self.mol_proj = nn.Linear(config.model.molformer_dim, config.model.cond_dim)# if config.model.use_molformer else None
        #self.esm_norm = nn.BatchNorm1d(self.config.model.cond_dim)
        #self.mol_norm = nn.BatchNorm1d(self.config.model.cond_dim)
        print(f"{config.model.dropout=}")
    
        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.dropout, config.model.cond_dim) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma
    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )
    def forward(self, indices, sigma, cond_seq=None, src_mask=None, cond_mask=None):

        #print(indices.size)
        #print("ESM cond")
        #print(esm_cond.size())
        x = self.vocab_embed(indices)
        #print("vocab embedded")
        print(f"{x.size()=}")
        c = F.silu(self.sigma_map(sigma))

        #print(c.size())
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None, cond=cond_seq, cond_mask=cond_mask)

            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0

        #print("x.shape:", x.shape)
        #print("indices.shape:", indices.shape)
        #print("indices.max():", indices.max().item())
        #print("x.shape[-1]:", x.shape[-1])
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        #x = x.index_fill(-1, indices, 0)  # only works if indices are flat and dim matches

        return x

# --------- End-to-end wrapper ----------
class ProteinLigandSharedDiffusion(nn.Module):
    """
    End-to-end:
      - encodes protein/ligand sequences
      - projects to shared latent space
      - builds a conditioning sequence (co-attend or concat)
      - denoises (discrete diffusion) via shared transformer

    Supports three modes:
      1) ligand_given_protein
      2) protein_given_ligand
      3) joint (both condition each other or condition on concat)
    """
    def __init__(
        self,
        config):
        #d_latent=512,
        #vocab_prot=30,    # set your protein vocab size (discrete diffusion tokens)
        #vocab_lig=128,    # set your ligand vocab size
        #n_layers=6, n_heads=8, d_ff=2048, device="cpu"
        #):
        super().__init__()
        esm_variant = "esm2_t33_650M_UR50D"
        mol_model_id = "ibm/MoLFormer-XL-both-10pct"

        self.device = config.model.device

        # Encoders
        
        self.prot_enc = ProteinEncoderESM(esm_variant="esm2_t33_650M_UR50D", device=self.device) 
        self.lig_enc  = LigandEncoderMolformer()# device=device) #molformer_ckpt,

        # Infer dims by doing tiny dummy forwards (lazy approach) – or set manually if you prefer.
        # We'll do a tiny dry-run on init with simple inputs to grab dims.
        with torch.no_grad():
            p_per, p_pool, _ = self.prot_enc.encode(["A"])   # single AA; ESM still returns Dp
            l_per, l_pool, _ = self.lig_enc.encode(["C"])    # single char SMILES; tokenizer handles

        d_prot = p_per.size(-1)
        d_lig  = l_per.size(-1)
        print(f"Protein encoder output dim: {d_prot}")
        print(f"Ligand encoder output dim: {d_lig}")
        # Projectors to shared latent
        self.projector = SharedLatentProjector(d_prot, d_lig, config.model.hidden_size)

        # Two diffusion heads (one for each vocabulary) sharing the same backbone weights?
        # Option A: single backbone, separate output heads
        self.shared_backbone = nn.ModuleDict({
            "ligand": SharedDiffusionTransformer(config, tokens = 2363),#vocab_size=vocab_lig, d_model=d_latent,
                                                 #n_layers=n_layers, n_heads=n_heads, d_ff=d_ff),
            "protein": SharedDiffusionTransformer(config, tokens = 33)#vocab_size=vocab_prot, d_model=d_latent,
                                                  #n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
        })

    def fuse_condition(self, p_seq_lat=None, l_seq_lat=None, p_mask=None, l_mask=None, mode="concat"):
        """
        Build a conditioning sequence in the shared latent space.
        mode="concat": just concatenate [protein; ligand] along sequence dim.
        Alternatives: cross-attn stacks, gated sum, attention pooling, etc.
        """
        if p_seq_lat is None and l_seq_lat is None:
            return None, None
        if p_seq_lat is None:
            return l_seq_lat, l_mask
        if l_seq_lat is None:
            return p_seq_lat, p_mask
        if mode == "concat":
            cond = torch.cat([p_seq_lat, l_seq_lat], dim=1)
            cond_mask = None
            if (p_mask is not None) and (l_mask is not None):
                cond_mask = torch.cat([p_mask, l_mask], dim=1)
            return cond, cond_mask
        raise NotImplementedError

    def encode_to_latent(self, prot_strs=None, smiles_list=None):
        """
        Returns projected per-token sequences in shared latent for protein and ligand.
        """
        p_seq_lat = p_mask = None
        l_seq_lat = l_mask = None

        if prot_strs is not None:
            p_per, _, p_mask = self.prot_enc.encode(prot_strs)
            p_seq_lat = self.projector.forward_protein(p_per)

        if smiles_list is not None:
            l_per, _, l_mask = self.lig_enc.encode(smiles_list)
            l_seq_lat = self.projector.forward_ligand(l_per)

        return p_seq_lat, p_mask, l_seq_lat, l_mask

    def forward(
        self,
        noisy_tokens: torch.Tensor,   # [B, L] ints for the target modality
        timesteps: torch.Tensor,      # [B]
        prot_seq=None,               # List[str] or None
        lig_seq=None,             # List[str] or None
        src_mask=None,                # [B, L] (True=keep) for target tokens
        task:str = "ligand_given_protein"
    ):
        """
        task: "ligand_given_protein" | "protein_given_ligand" | "joint_ligand" | "joint_protein"
              For joint, decide which target you are denoising at this step.
        """
        # 1) Encode & project to shared latent
        p_seq_lat, p_mask, l_seq_lat, l_mask = self.encode_to_latent(prot_seq, lig_seq)

        # 2) Build conditioning for the chosen task
        if task == "ligand_given_protein":
            cond_seq, cond_mask = self.fuse_condition(p_seq_lat, None, p_mask, None, mode="concat")
            logits = self.shared_backbone["ligand"](
                token_ids=noisy_tokens, timesteps=timesteps,
                cond_seq=cond_seq, src_mask=src_mask, cond_mask=cond_mask
            )
            return logits

        elif task == "protein_given_ligand":
            cond_seq, cond_mask = self.fuse_condition(None, l_seq_lat, None, l_mask, mode="concat")
            logits = self.shared_backbone["protein"](
                token_ids=noisy_tokens, timesteps=timesteps,
                cond_seq=cond_seq, src_mask=src_mask, cond_mask=cond_mask
            )
            return logits

        elif task == "joint_ligand":
            # Condition ligand on BOTH protein + ligand context (e.g., masked tokens or teacher forcing)
            cond_seq, cond_mask = self.fuse_condition(p_seq_lat, l_seq_lat, p_mask, l_mask, mode="concat")
            logits = self.shared_backbone["ligand"](
                token_ids=noisy_tokens, timesteps=timesteps,
                cond_seq=cond_seq, src_mask=src_mask, cond_mask=cond_mask
            )
            return logits

        elif task == "joint_protein":
            cond_seq, cond_mask = self.fuse_condition(p_seq_lat, l_seq_lat, p_mask, l_mask, mode="concat")
            logits = self.shared_backbone["protein"](
                token_ids=noisy_tokens, timesteps=timesteps,
                cond_seq=cond_seq, src_mask=src_mask, cond_mask=cond_mask
            )
            return logits

        else:
            raise ValueError(f"Unknown task: {task}")