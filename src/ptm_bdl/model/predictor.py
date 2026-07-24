"""
Multimodal Resistance Predictor — Full two-stage fusion model.

Two-stage fusion architecture:
    Stage 1 (static):  seq + struct + drug → joint attention → S_rep
    Stage 2 (dynamic): PTM-BDL(primary, secondary, type_gate, attn, residual_gate) → P_rep
    Fusion:            S_rep ⊙ P_rep → prediction heads
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ptm_bdl.model.ablation import PTMBDLMlpAblation
from src.ptm_bdl.model.encoder import PTMBDLEncoder
from src.ptm_bdl.model.fusion import BilinearLateFusion
from src.ptm_bdl.model.static import ModalityProjection, StaticJointTransformer, AttentionPooling
from src.ptm_bdl.registry import PTMTypeRegistry


class MultimodalResistancePredictor(nn.Module):
    """
    PTM-BDL multimodal model.

    Two-stage fusion:
      Stage 1 (static): seq + struct + drug → joint attention → S_rep
      Stage 2 (dynamic): PTM-BDL(primary, secondary, ...) → P_rep
      Fusion: S_rep ⊙ P_rep → prediction heads (IC50 regression + resistance classification)

    All PTM configuration (types, subtypes, proteins, pad masks) comes from the
    PTMTypeRegistry — no hardcoded constants.
    """

    def __init__(
            self,
            registry: PTMTypeRegistry,
            seq_dim: int = 1280,
            struct_dim: int = 512,
            drug_dim: int = 384,
            shared_dim: int = 512,
            num_heads: int = 8,
            num_layers: int = 4,
            dropout: float = 0.1,
            ptm_bdl_d_model: int = 64,
            ptm_bdl_n_heads: int = 4,
            ptm_bdl_n_layers: int = 2,
            use_typed_attention: bool = True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.registry = registry

        # ── Static modalities ────────────────────────────────────────────
        self.seq_projection = ModalityProjection(seq_dim, shared_dim, dropout)
        self.struct_projection = ModalityProjection(struct_dim, shared_dim, dropout)
        self.drug_projection = ModalityProjection(drug_dim, shared_dim, dropout)
        self.modality_embeddings = nn.Embedding(3, shared_dim)  # seq/struct/drug

        self.static_transformer = StaticJointTransformer(
            d_model=shared_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
        )
        self.static_pool = AttentionPooling(shared_dim, dropout)

        # ── Dynamic PTM-BDL branch ───────────────────────────────────────
        bdl_cls = PTMBDLEncoder if use_typed_attention else PTMBDLMlpAblation
        self.ptm_bdl = bdl_cls(
            registry=registry,
            d_model=ptm_bdl_d_model,
            n_heads=ptm_bdl_n_heads,
            n_layers=ptm_bdl_n_layers,
            dropout=dropout,
        )

        # ── Late bilinear fusion: S_rep ⊙ P_rep ─────────────────────────
        self.fusion = BilinearLateFusion(
            static_dim=shared_dim,
            ptm_bdl_dim=ptm_bdl_d_model,
            output_dim=shared_dim,
            dropout=dropout,
        )

        # ── Prediction heads ─────────────────────────────────────────────
        self.regression_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(shared_dim // 2, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(shared_dim // 2, 1),
        )

    def forward(
            self, seq_embeddings, struct_embeddings, drug_pooled,
            ptm_vector, delta_ptm_vector,
            secondary_vector=None, delta_secondary_vector=None,
            target_protein=None, drug_embeddings=None,
            return_attention=False, return_ptm_bdl=False,
    ):
        B = seq_embeddings.size(0)
        device = seq_embeddings.device
        L = seq_embeddings.size(1)
        M = struct_embeddings.size(1)

        # Default secondary channel dimension from registry (generic)
        if len(self.registry.ptm_type_order) >= 2:
            secondary_type = self.registry.ptm_type_order[1]
            secondary_dim = self.registry.get_n_sites_per_type(secondary_type)
        else:
            secondary_dim = 0
        if secondary_vector is None:
            secondary_vector = torch.zeros(B, secondary_dim, device=device)
        if delta_secondary_vector is None:
            delta_secondary_vector = torch.zeros(B, secondary_dim, device=device)
        if target_protein is None:
            target_protein = torch.zeros(B, dtype=torch.long, device=device)
        else:
            target_protein = target_protein.long()

        # ── STATIC branch: seq + struct + drug → S_rep ──────────────────
        seq_proj = self.seq_projection(seq_embeddings)
        struct_proj = self.struct_projection(struct_embeddings)

        if drug_embeddings is not None:
            drug_proj = self.drug_projection(drug_embeddings)
            Ndrug = drug_proj.size(1)
        else:
            drug_proj = self.drug_projection(drug_pooled.unsqueeze(1))
            Ndrug = 1

        seq_proj = seq_proj + self.modality_embeddings(
            torch.zeros(L, dtype=torch.long, device=device)).unsqueeze(0)
        struct_proj = struct_proj + self.modality_embeddings(
            torch.ones(M, dtype=torch.long, device=device)).unsqueeze(0)
        drug_proj = drug_proj + self.modality_embeddings(
            torch.full((Ndrug,), 2, dtype=torch.long, device=device)).unsqueeze(0)

        static_tokens = torch.cat([seq_proj, struct_proj, drug_proj], dim=1)
        if return_attention:
            static_out, static_attn = self.static_transformer(
                static_tokens, return_all_attention=True)
        else:
            static_out = self.static_transformer(static_tokens)
            static_attn = None

        static_rep, _ = self.static_pool(static_out)  # (B, shared_dim)

        # ── DYNAMIC PTM-BDL branch → P_rep ──────────────────────────────
        bdl_out = self.ptm_bdl(
            ptm_vector=ptm_vector, delta_ptm_vector=delta_ptm_vector,
            secondary_vector=secondary_vector, delta_secondary_vector=delta_secondary_vector,
            target_protein=target_protein,
        )
        ptm_bdl_pooled = bdl_out["pooled"]  # (B, ptm_bdl_d_model)

        # ── Fusion: S_rep ⊙ P_rep → heads ──────────────────────────────
        fused = self.fusion(static_rep, ptm_bdl_pooled)
        ic50_pred = self.regression_head(fused)
        resistance_logits = self.classification_head(fused)

        if return_attention or return_ptm_bdl:
            extras = {}
            if return_attention:
                extras["static_attention_maps"] = static_attn
            if return_ptm_bdl:
                extras["ptm_bdl"] = bdl_out
            return ic50_pred, resistance_logits, extras
        return ic50_pred, resistance_logits

    def get_static_token_boundaries(self, L, M, Ndrug):
        return {
            "sequence": (0, L),
            "structure": (L, L + M),
            "drug": (L + M, L + M + Ndrug),
        }
