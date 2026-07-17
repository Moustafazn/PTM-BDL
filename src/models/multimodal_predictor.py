"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MULTIMODAL RESISTANCE PREDICTOR — PTM-BDL Architecture (2026-06-28)        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Implements the PTM-Biological-Dynamics-Layer architecture described in     ║
║  PTM_Biological_Dynamics_Layer.md §3, §7, §7.4–§7.7.                        ║
║                                                                              ║
║  SYMMETRIC across:                                                            ║
║    • two PROTEINS  — EGFR and ERBB2 share the PTM-BDL encoder               ║
║    • two PTM TYPES — phospho (intracellular) and glyco_N (extracellular)    ║
║    • six DRUGS     — enter via early static fusion (seq+struct+drug);        ║
║                      PTM-BDL branch is drug-AGNOSTIC so the model           ║
║                      cannot shortcut through drug identity.                  ║
║                                                                              ║
║  Two-stage fusion (proposal §7.1–§7.3):                                      ║
║                                                                              ║
║      STAGE 1 — STATIC (early fusion):                                        ║
║        ESM-2 sequence  ─┐                                                   ║
║        GearNet structure├→ joint self-attention → attention pool → S_rep    ║
║        ChemBERTa drug   ─┘   (drug IS inside S_rep)                          ║
║                                                                              ║
║      STAGE 2 — DYNAMIC (late fusion):                                        ║
║        PTM-BDL encoder: phospho(12) + glyco(12) tokens                      ║
║          → [level, delta, ratio] per token (§7.4)                           ║
║          → type-gated projection (§7.5)                                     ║
║          → typed self-attention (§7.6)                                      ║
║          → residual gate (§7.7)                                             ║
║          → mask-aware mean pool → P_rep                                     ║
║                                                                              ║
║      FUSION: S_rep ⊙ P_rep → prediction heads (§7.2)                       ║
║        "Given the drug CAN bind (S_rep knows), does PTM say it WORKS?"      ║
║                                                                              ║
║  WHY NO SEPARATE DRUG IN LATE FUSION (§7.3):                                ║
║    Drug is already inside S_rep via early joint attention. A separate drug   ║
║    branch in late fusion would let the model learn drug→prediction           ║
║    shortcuts that bypass PTM modulation — exactly the failure mode           ║
║    documented in §1.2 (PTM features become redundant). The delta_ptm        ║
║    already encodes drug-induced changes, so PTM-BDL IS drug-conditioned     ║
║    through its input, not through a fusion shortcut.                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# PTM-BDL CONSTANTS (proposal §3, §7.4)
# ══════════════════════════════════════════════════════════════════════════════

_TYPE_Y = 0   # phospho-tyrosine     — direct TKI target
_TYPE_S = 1   # phospho-serine       — downstream indicator
_TYPE_T = 2   # phospho-threonine    — regulatory feedback
_TYPE_N = 3   # N-glycosylation      — receptor surface biology
N_PTM_TYPES = 4

_TYPE_PHOSPHO_EGFR = [
    _TYPE_Y, _TYPE_S, _TYPE_Y, _TYPE_Y, _TYPE_S, _TYPE_T,
    _TYPE_Y, _TYPE_Y, _TYPE_Y, _TYPE_Y, _TYPE_Y, _TYPE_Y,
]
_TYPE_PHOSPHO_ERBB2 = [
    _TYPE_T, _TYPE_Y, _TYPE_S, _TYPE_T, _TYPE_Y, _TYPE_S,
    _TYPE_Y, _TYPE_Y, _TYPE_Y, _TYPE_Y, _TYPE_Y, _TYPE_Y,
]
_TYPE_GLYCO = [_TYPE_N] * 12

_PAD_EGFR = [False] * 24
_PAD_ERBB2 = (
    [False] * 10 + [True] * 2
    + [False] * 7 + [True] * 5
)

PROTEIN_ID_EGFR = 0
PROTEIN_ID_ERBB2 = 1
PTM_TYPE_NAMES = {0: "phospho_Y", 1: "phospho_S", 2: "phospho_T", 3: "glyco_N"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Modality Projection
# ══════════════════════════════════════════════════════════════════════════════

class ModalityProjection(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )

    def forward(self, x):
        return self.projection(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PTM-BDL Encoder (proposal §7.4–§7.7)
# ══════════════════════════════════════════════════════════════════════════════

class PTMBDLEncoder(nn.Module):
    """
    PTM Biological Dynamics Layer encoder.

    24 typed tokens per sample (12 phospho + 12 glyco).

    Pipeline per proposal §7.4–§7.7:
        1) [level, delta, ratio] → value_proj → d_model          (§7.4)
        2) type-gated projection: gate ⊙ projected                (§7.5)
        3) + type_emb + protein_emb + slot_emb → transformer      (§7.6)
        4) residual gate: α·attended + (1-α)·gated_projected       (§7.7)
        5) mask-aware mean pool → pooled                           (§7.4)
    """

    def __init__(self, phospho_dim=12, glyco_dim=12, d_model=64,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert phospho_dim == 12 and glyco_dim == 12
        self.phospho_dim = phospho_dim
        self.glyco_dim = glyco_dim
        self.n_tokens = 24
        self.d_model = d_model

        # §7.4: [level, delta, ratio] → d_model
        self.value_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )

        self.type_emb = nn.Embedding(N_PTM_TYPES, d_model)
        self.protein_emb = nn.Embedding(2, d_model)
        self.slot_emb = nn.Embedding(self.n_tokens, d_model)

        # §7.5: type-gated projection
        self.type_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid(),
        )

        # §7.6: typed self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(d_model)

        # §7.7: residual gate
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid(),
        )

        # Buffers
        type_ids = torch.zeros(2, 24, dtype=torch.long)
        type_ids[0] = torch.tensor(_TYPE_PHOSPHO_EGFR + _TYPE_GLYCO, dtype=torch.long)
        type_ids[1] = torch.tensor(_TYPE_PHOSPHO_ERBB2 + _TYPE_GLYCO, dtype=torch.long)
        is_real = torch.zeros(2, 24, dtype=torch.bool)
        is_real[0] = ~torch.tensor(_PAD_EGFR, dtype=torch.bool)
        is_real[1] = ~torch.tensor(_PAD_ERBB2, dtype=torch.bool)
        self.register_buffer("type_id_table", type_ids, persistent=False)
        self.register_buffer("is_real_table", is_real, persistent=False)

    def _stitch(self, ptm, dptm, glyco, dglyco):
        """[level, delta, ratio] per token → (B, 24, 3).  §7.4 + §2.4."""
        levels = torch.cat([ptm, glyco], dim=1)
        deltas = torch.cat([dptm, dglyco], dim=1)
        ratios = deltas / (levels.abs() + 1e-6)
        return torch.stack([levels, deltas, ratios], dim=-1)

    def _build_tokens(self, ptm, dptm, glyco, dglyco, protein_id):
        """value_proj → type gate → embeddings → pad mask.  §7.4–§7.5."""
        device = ptm.device
        projected = self.value_proj(self._stitch(ptm, dptm, glyco, dglyco))
        type_ids_b = self.type_id_table[protein_id]
        is_real_b = self.is_real_table[protein_id]

        # §7.5: type-gated projection
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

    def forward(self, ptm_vector, delta_ptm_vector, glyco_vector,
                delta_glyco_vector, target_protein) -> dict:
        protein_id = target_protein.clamp(min=0, max=1).long()
        token_emb, kpm, is_real_b, type_ids_b = self._build_tokens(
            ptm_vector, delta_ptm_vector, glyco_vector,
            delta_glyco_vector, protein_id,
        )

        pre_attn = token_emb  # save for residual gate (§7.7)

        # §7.6: typed self-attention
        x = self.transformer(token_emb, src_key_padding_mask=kpm)
        x = self.out_norm(x)

        # §7.7: residual gate — α·attended + (1-α)·pre_attn
        alpha = self.residual_gate(torch.cat([x, pre_attn], dim=-1))
        x = alpha * x + (1 - alpha) * pre_attn

        # Mask-aware mean pool
        mask_f = is_real_b.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return {"pooled": pooled, "tokens": x, "mask": is_real_b,
                "type_ids": type_ids_b}

    @torch.no_grad()
    def compute_attn_weights(self, ptm_vector, delta_ptm_vector,
                             glyco_vector, delta_glyco_vector,
                             target_protein) -> torch.Tensor:
        """Post-softmax attention from FINAL layer → (B, 24, 24).  For XAI."""
        protein_id = target_protein.clamp(min=0, max=1).long()
        token_emb, kpm, _, _ = self._build_tokens(
            ptm_vector, delta_ptm_vector, glyco_vector,
            delta_glyco_vector, protein_id,
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. PTM-BDL MLP Ablation (no inter-token attention — §9.1)
# ══════════════════════════════════════════════════════════════════════════════

class PTMBDLMlpAblation(nn.Module):
    """Same inputs as PTMBDLEncoder but MLP instead of self-attention."""

    def __init__(self, phospho_dim=12, glyco_dim=12, d_model=64,
                 n_layers=2, dropout=0.1, **_unused):
        super().__init__()
        assert phospho_dim == 12 and glyco_dim == 12
        self.n_tokens = 24
        self.d_model = d_model

        # Same 3-feature input as the full encoder
        self.value_proj = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.type_emb = nn.Embedding(N_PTM_TYPES, d_model)
        self.protein_emb = nn.Embedding(2, d_model)
        self.slot_emb = nn.Embedding(self.n_tokens, d_model)

        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(d_model, d_model * 2), nn.GELU(),
                       nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
                       nn.LayerNorm(d_model)]
        self.token_mlp = nn.Sequential(*layers)

        type_ids = torch.zeros(2, 24, dtype=torch.long)
        type_ids[0] = torch.tensor(_TYPE_PHOSPHO_EGFR + _TYPE_GLYCO, dtype=torch.long)
        type_ids[1] = torch.tensor(_TYPE_PHOSPHO_ERBB2 + _TYPE_GLYCO, dtype=torch.long)
        is_real = torch.zeros(2, 24, dtype=torch.bool)
        is_real[0] = ~torch.tensor(_PAD_EGFR, dtype=torch.bool)
        is_real[1] = ~torch.tensor(_PAD_ERBB2, dtype=torch.bool)
        self.register_buffer("type_id_table", type_ids, persistent=False)
        self.register_buffer("is_real_table", is_real, persistent=False)

    def forward(self, ptm_vector, delta_ptm_vector, glyco_vector,
                delta_glyco_vector, target_protein):
        protein_id = target_protein.clamp(min=0, max=1).long()
        device = ptm_vector.device
        levels = torch.cat([ptm_vector, glyco_vector], dim=1)
        deltas = torch.cat([delta_ptm_vector, delta_glyco_vector], dim=1)
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


# ══════════════════════════════════════════════════════════════════════════════
# 4. Static (early-fusion) Joint Transformer — seq + struct + drug
# ══════════════════════════════════════════════════════════════════════════════

class JointSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=False):
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return (x, attn_weights) if return_attention else x


class StaticJointTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            JointSelfAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, return_all_attention=False):
        all_attn = []
        for layer in self.layers:
            if return_all_attention:
                x, attn = layer(x, return_attention=True)
                all_attn.append(attn)
            else:
                x = layer(x)
        x = self.final_norm(x)
        return (x, all_attn) if return_all_attention else x


# ══════════════════════════════════════════════════════════════════════════════
# 5. Attention Pooling (Ilse et al., ICML 2018)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model // 4, 1),
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Late Bilinear Fusion — static_rep ⊙ ptm_rep  (proposal §7.2)
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY NOT TRIPLE FUSION WITH DRUG:
#   static_rep ALREADY contains drug identity — drug tokens went through
#   4 layers of joint self-attention with sequence + structure tokens in the
#   early fusion stage.  Adding a separate drug branch in late fusion would
#   create a shortcut: drug → prediction that bypasses PTM modulation.
#   This is exactly the failure mode described in §1.2 — the model learns
#   drug_id → response directly instead of through the PTM signaling code.
#
#   The delta_ptm input to PTM-BDL already encodes drug-induced phospho
#   changes (drug-conditioned), so PTM-BDL IS drug-aware through its
#   INPUT FEATURES, not through a fusion shortcut.
#
#   Proposal §7.2 diagram: "static_rep ⊙ ptm_rep → fused"
#   "Given the drug CAN bind (static_rep knows), does the PTM state say
#    it actually WORKS?"

class BilinearLateFusion(nn.Module):
    """
    Two-way late fusion: static_rep ⊙ ptm_bdl_rep.

    static_rep already contains drug context from early joint attention.
    PTM-BDL rep carries the dynamic biological state (drug-conditioned
    through delta_ptm features, NOT through drug identity shortcut).
    """

    def __init__(self, static_dim: int, ptm_bdl_dim: int,
                 output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, output_dim)
        self.ptm_proj = nn.Linear(ptm_bdl_dim, output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, static_rep, ptm_rep):
        s = torch.tanh(self.static_proj(static_rep))
        p = torch.tanh(self.ptm_proj(ptm_rep))
        return self.output_proj(s * p)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Full model
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalResistancePredictor(nn.Module):
    """
    PTM-BDL multimodal model (revised 2026-06-28).

    Two-stage fusion:
      Stage 1 (static): seq + struct + drug → joint attention → S_rep
      Stage 2 (dynamic): PTM-BDL(phospho, glyco, type_gate, attn, residual_gate) → P_rep
      Fusion: S_rep ⊙ P_rep → prediction heads
    """

    def __init__(
        self,
        seq_dim=1280, struct_dim=512, drug_dim=384,
        ptm_dim=12, glyco_dim=12,
        shared_dim=512, num_heads=8, num_layers=4, dropout=0.1,
        ptm_bdl_d_model=64, ptm_bdl_n_heads=4, ptm_bdl_n_layers=2,
        use_typed_attention=True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.ptm_dim = ptm_dim
        self.glyco_dim = glyco_dim

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
            phospho_dim=ptm_dim, glyco_dim=glyco_dim,
            d_model=ptm_bdl_d_model, n_heads=ptm_bdl_n_heads,
            n_layers=ptm_bdl_n_layers, dropout=dropout,
        )

        # ── Late bilinear fusion: S_rep ⊙ P_rep (§7.2) ──────────────────
        # NO separate drug encoder — drug is already in S_rep.
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
        glyco_vector=None, delta_glyco_vector=None,
        target_protein=None, drug_embeddings=None,
        return_attention=False, return_ptm_bdl=False,
    ):
        B = seq_embeddings.size(0)
        device = seq_embeddings.device
        L = seq_embeddings.size(1)
        M = struct_embeddings.size(1)

        if glyco_vector is None:
            glyco_vector = torch.zeros(B, self.glyco_dim, device=device)
        if delta_glyco_vector is None:
            delta_glyco_vector = torch.zeros(B, self.glyco_dim, device=device)
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
            glyco_vector=glyco_vector, delta_glyco_vector=delta_glyco_vector,
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
            "sequence":  (0, L),
            "structure": (L, L + M),
            "drug":      (L + M, L + M + Ndrug),
        }
