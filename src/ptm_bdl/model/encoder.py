"""
PTM-BDL Encoder — Typed self-attention encoder for PTM tokens.

Accepts a PTMTypeRegistry and builds all embeddings/buffers from it.
No hardcoded protein IDs, PTM types, or token counts.

Pipeline:
    1) [level, delta, ratio] → value_proj → d_model
    2) type-gated projection: gate ⊙ projected
    3) + type_emb + protein_emb + slot_emb → transformer
    4) residual gate: α·attended + (1-α)·gated_projected
    5) mask-aware mean pool → pooled
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ptm_bdl.registry import PTMTypeRegistry


class PTMBDLEncoder(nn.Module):
    """
    PTM Biological Dynamics Layer encoder.

    Uses a PTMTypeRegistry to dynamically configure:
      - Number of tokens (from registry.n_tokens)
      - Type embedding size (from registry.n_subtypes)
      - Protein embedding size (from registry.n_proteins)
      - Per-protein type_id and pad masks (from registry buffers)

    Accepts a single flat ``ptm_vector`` of size ``n_tokens`` containing
    ALL PTM sites across all modification types.  Per-type differentiation
    is handled by the type embedding and type gate.
    """

    def __init__(self, registry: PTMTypeRegistry, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = registry.n_tokens
        self.d_model = d_model

        # [level, delta, ratio] → d_model
        self.value_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )

        self.type_emb = nn.Embedding(registry.n_subtypes, d_model)
        self.protein_emb = nn.Embedding(registry.n_proteins, d_model)
        self.slot_emb = nn.Embedding(self.n_tokens, d_model)

        # Type-gated projection
        self.type_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid(),
        )

        # Typed self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(d_model)

        # Residual gate
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid(),
        )

        # Register buffers from registry
        self.register_buffer("type_id_table", registry.type_id_table.clone(), persistent=False)
        self.register_buffer("is_real_table", registry.is_real_table.clone(), persistent=False)

    def _stitch(self, levels: torch.Tensor,
                deltas: torch.Tensor) -> torch.Tensor:
        """[level, delta, ratio] per token → (B, n_tokens, 3)."""
        ratios = deltas / (levels.abs() + 1e-6)
        return torch.stack([levels, deltas, ratios], dim=-1)

    def _build_tokens(self, levels: torch.Tensor,
                      deltas: torch.Tensor,
                      protein_id: torch.Tensor):
        """value_proj → type gate → embeddings → pad mask."""
        device = levels.device
        projected = self.value_proj(self._stitch(levels, deltas))
        type_ids_b = self.type_id_table[protein_id]
        is_real_b = self.is_real_table[protein_id]

        # Type-gated projection
        type_emb_b = self.type_emb(type_ids_b)
        gate = self.type_gate(torch.cat([projected, type_emb_b], dim=-1))
        token_emb = gate * projected

        # Add embeddings AFTER gating
        token_emb = token_emb + type_emb_b
        token_emb = token_emb + self.protein_emb(protein_id).unsqueeze(1)
        token_emb = token_emb + self.slot_emb(
            torch.arange(self.n_tokens, device=device)).unsqueeze(0)

        # Zero pad slots
        token_emb = token_emb * is_real_b.unsqueeze(-1).float()

        key_padding_mask = ~is_real_b
        all_pad = key_padding_mask.all(dim=1)
        if all_pad.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False
            is_real_b = is_real_b.clone()
            is_real_b[all_pad, 0] = True

        return token_emb, key_padding_mask, is_real_b, type_ids_b

    def forward(self, ptm_vector, delta_ptm_vector, target_protein) -> dict:
        """
        Forward pass through the PTM-BDL encoder.

        Args:
            ptm_vector: (B, n_tokens) — PTM baseline levels (all types, flat)
            delta_ptm_vector: (B, n_tokens) — drug-induced PTM changes (all types, flat)
            target_protein: (B,) long — protein ID index

        Returns:
            dict with keys: pooled, tokens, mask, type_ids
        """
        protein_id = target_protein.clamp(min=0, max=self.is_real_table.size(0) - 1).long()
        token_emb, kpm, is_real_b, type_ids_b = self._build_tokens(
            ptm_vector, delta_ptm_vector, protein_id,
        )

        pre_attn = token_emb  # save for residual gate

        # Typed self-attention
        x = self.transformer(token_emb, src_key_padding_mask=kpm)
        x = self.out_norm(x)

        # Residual gate — α·attended + (1-α)·pre_attn
        alpha = self.residual_gate(torch.cat([x, pre_attn], dim=-1))
        x = alpha * x + (1 - alpha) * pre_attn

        # Mask-aware mean pool
        mask_f = is_real_b.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return {"pooled": pooled, "tokens": x, "mask": is_real_b,
                "type_ids": type_ids_b}

    @torch.no_grad()
    def compute_attn_weights(self, ptm_vector, delta_ptm_vector,
                             target_protein) -> torch.Tensor:
        """Post-softmax attention from FINAL layer → (B, n_tokens, n_tokens). For XAI."""
        protein_id = target_protein.clamp(min=0, max=self.is_real_table.size(0) - 1).long()
        token_emb, kpm, _, _ = self._build_tokens(
            ptm_vector, delta_ptm_vector, protein_id,
        )
        x = token_emb
        for layer in self.transformer.layers[:-1]:
            x = layer(x, src_key_padding_mask=kpm)
        last = self.transformer.layers[-1]
        x_norm = last.norm1(x)
        _, attn_weights = last.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=kpm, need_weights=True, average_attn_weights=True,
        )
        return attn_weights
