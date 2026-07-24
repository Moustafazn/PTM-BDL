"""
PTM-BDL Model Package — Config-driven multimodal architecture.

Components:
    PTMBDLEncoder              — Typed self-attention encoder for PTM tokens
    PTMBDLMlpAblation          — MLP ablation variant (no inter-token attention)
    StaticJointTransformer     — Cross-modal self-attention for seq+struct+drug
    AttentionPooling           — Ilse et al. ICML 2018 attention pooling
    ModalityProjection         — Per-modality linear projection
    BilinearLateFusion         — S_rep ⊙ P_rep bilinear fusion
    MultimodalResistancePredictor — Full two-stage fusion model
"""

from src.ptm_bdl.model.ablation import PTMBDLMlpAblation
from src.ptm_bdl.model.encoder import PTMBDLEncoder
from src.ptm_bdl.model.fusion import BilinearLateFusion
from src.ptm_bdl.model.predictor import MultimodalResistancePredictor
from src.ptm_bdl.model.static import StaticJointTransformer, AttentionPooling, ModalityProjection
