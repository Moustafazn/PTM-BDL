"""
PTM-BDL MLP Ablation — Same inputs as PTMBDLEncoder but MLP instead of self-attention.

Used as the architectural ablation arm to test the value of
inter-token dependencies (typed self-attention) vs independent
per-token processing (MLP).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ptm_bdl.registry import PTMTypeRegistry


def create_ablation_model(cfg: dict, mode: str = "full"):
    """
    Convenience wrapper: build a model from config for a given ablation mode.

    Modes:
      • ``"full"``            — standard model with typed self-attention
      • ``"no_typed_attention"`` — MLP in place of self-attention
      • ``"no_ptm"``          — full architecture (PTM zeroing happens at data level)
      • ``"no_drug"``         — full architecture (drug zeroing at data level)
      • ``"no_structure"``    — full architecture (structure zeroing at data level)
      • any other string      — full architecture (data-level ablation expected)

    Returns a ``MultimodalResistancePredictor`` ready for ``.to(device)``.
    """
    from src.ptm_bdl.training.factory import build_model_from_cfg

    use_typed_attention = (mode != "no_typed_attention")
    return build_model_from_cfg(cfg, use_typed_attention=use_typed_attention)


class PTMBDLMlpAblation(nn.Module):
    """
    MLP-based PTM encoder (no inter-token attention).

    Same input/output contract as PTMBDLEncoder but processes
    each token independently — no cross-token dependencies.
    Uses PTMTypeRegistry for all configuration.
    """

    def __init__(self, registry: PTMTypeRegistry, d_model: int = 64,
                 n_layers: int = 2, dropout: float = 0.1, **_unused):
        super().__init__()
        self.n_tokens = registry.n_tokens
        self.d_model = d_model

        # Same 3-feature input as the full encoder
        self.value_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.type_emb = nn.Embedding(registry.n_subtypes, d_model)
        self.protein_emb = nn.Embedding(registry.n_proteins, d_model)
        self.slot_emb = nn.Embedding(self.n_tokens, d_model)

        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(d_model, d_model * 2), nn.GELU(),
                       nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
                       nn.LayerNorm(d_model)]
        self.token_mlp = nn.Sequential(*layers)

        # Register buffers from registry
        self.register_buffer("type_id_table", registry.type_id_table.clone(), persistent=False)
        self.register_buffer("is_real_table", registry.is_real_table.clone(), persistent=False)

    def forward(self, ptm_vector, delta_ptm_vector, secondary_vector,
                delta_secondary_vector, target_protein):
        protein_id = target_protein.clamp(min=0, max=self.is_real_table.size(0) - 1).long()
        device = ptm_vector.device
        levels = torch.cat([ptm_vector, secondary_vector], dim=1)
        deltas = torch.cat([delta_ptm_vector, delta_secondary_vector], dim=1)
        ratios = deltas / (levels.abs() + 1e-6)
        vals = torch.stack([levels, deltas, ratios], dim=-1)
        x = self.value_proj(vals)
        type_ids_b = self.type_id_table[protein_id]
        is_real_b = self.is_real_table[protein_id]
        x = x + self.type_emb(type_ids_b)
        x = x + self.protein_emb(protein_id).unsqueeze(1)
        x = x + self.slot_emb(torch.arange(self.n_tokens, device=device)).unsqueeze(0)
        x = x * is_real_b.unsqueeze(-1).float()
        x = self.token_mlp(x)
        mask_f = is_real_b.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return {"pooled": pooled, "tokens": x, "mask": is_real_b,
                "type_ids": type_ids_b}
