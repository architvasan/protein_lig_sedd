# pip installs (if needed):
# pip install torch torchvision torchaudio
# pip install fair-esm transformers

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
            padding=True,
            truncation=True,
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
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, x_mask=None, cond=None, cond_mask=None):
        # Self-attention
        x_res = x
        x = self.norm1(x)
        x_sa, _ = self.self_attn(x, x, x, key_padding_mask=(~x_mask) if x_mask is not None else None)
        x = x_res + x_sa

        # Cross-attention (if cond provided)
        if cond is not None:
            x_res = x
            x_norm = self.norm2(x)
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
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, d_model) * 0.01)  # long enough max L
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.to_logits = nn.Linear(d_model, vocab_size)

        # Timestep embedding for diffusion (FiLM-style)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, token_ids, timesteps, cond_seq=None, src_mask=None, cond_mask=None):
        """
        token_ids: [B, L] integer tokens (noisy)
        timesteps: [B] integer or float (diffusion step)
        cond_seq:  [B, Lc, D] (already in same d_model dim)
        src_mask:  [B, L] True for real tokens
        cond_mask: [B, Lc] True for real tokens
        """
        B, L = token_ids.shape
        x = self.token_emb(token_ids)  # [B, L, D]
        x = x + self.pos_emb[:, :L, :]

        # simple sinusoidal or learned t-embedding: map t->R^D and add
        # Here: just embed t via one-hot-ish linear; for real systems use sinusoidal/MLP.
        t = timesteps.float().unsqueeze(-1) / (timesteps.max().float().clamp(min=1.0))
        t = F.pad(t, (0, self.d_model - 1))  # [B, D] very simple placeholder
        t = self.t_mlp(t)                    # [B, D]
        x = x + t.unsqueeze(1)

        h = x
        for blk in self.blocks:
            h = blk(h, x_mask=src_mask, cond=cond_seq, cond_mask=cond_mask)

        logits = self.to_logits(h)  # [B, L, vocab]
        return logits


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
        esm_variant="esm2_t33_650M_UR50D",
        molformer_ckpt="DeepChem/molformer-embeddings",
        d_latent=512,
        vocab_prot=30,    # set your protein vocab size (discrete diffusion tokens)
        vocab_lig=128,    # set your ligand vocab size
        n_layers=6, n_heads=8, d_ff=2048, device="cpu"
    ):
        super().__init__()
        self.device = device

        # Encoders
        self.prot_enc = ProteinEncoderESM(esm_variant, device=device)
        self.lig_enc  = LigandEncoderMolformer(molformer_ckpt, device=device)

        # Infer dims by doing tiny dummy forwards (lazy approach) – or set manually if you prefer.
        # We'll do a tiny dry-run on init with simple inputs to grab dims.
        with torch.no_grad():
            p_per, p_pool, _ = self.prot_enc.encode(["A"])   # single AA; ESM still returns Dp
            l_per, l_pool, _ = self.lig_enc.encode(["C"])    # single char SMILES; tokenizer handles

        d_prot = p_per.size(-1)
        d_lig  = l_per.size(-1)

        # Projectors to shared latent
        self.projector = SharedLatentProjector(d_prot, d_lig, d_latent)

        # Two diffusion heads (one for each vocabulary) sharing the same backbone weights?
        # Option A: single backbone, separate output heads
        self.shared_backbone = nn.ModuleDict({
            "ligand": SharedDiffusionTransformer(vocab_size=vocab_lig, d_model=d_latent,
                                                 n_layers=n_layers, n_heads=n_heads, d_ff=d_ff),
            "protein": SharedDiffusionTransformer(vocab_size=vocab_prot, d_model=d_latent,
                                                  n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
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
        task:str,
        noisy_tokens: torch.Tensor,   # [B, L] ints for the target modality
        timesteps: torch.Tensor,      # [B]
        prot_strs=None,               # List[str] or None
        smiles_list=None,             # List[str] or None
        src_mask=None,                # [B, L] (True=keep) for target tokens
    ):
        """
        task: "ligand_given_protein" | "protein_given_ligand" | "joint_ligand" | "joint_protein"
              For joint, decide which target you are denoising at this step.
        """
        # 1) Encode & project to shared latent
        p_seq_lat, p_mask, l_seq_lat, l_mask = self.encode_to_latent(prot_strs, smiles_list)

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