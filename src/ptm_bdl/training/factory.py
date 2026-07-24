"""Model factory — builds MultimodalResistancePredictor from config + registry."""

from __future__ import annotations

from src.ptm_bdl.model.predictor import MultimodalResistancePredictor
from src.ptm_bdl.registry import PTMTypeRegistry


def build_model_from_cfg(cfg, use_typed_attention: bool = True) -> MultimodalResistancePredictor:
    """
    Single source of truth for instantiating the PTM-BDL multimodal model.

    Builds a PTMTypeRegistry from the config and passes it to the model.
    Pass `use_typed_attention=False` for the MLP architectural ablation.
    """
    registry = PTMTypeRegistry.from_config(cfg)
    bdl_cfg = cfg.get("ptm_bdl", {}) or {}

    return MultimodalResistancePredictor(
        registry=registry,
        seq_dim=1280,
        struct_dim=512,
        drug_dim=384,
        shared_dim=cfg["model"]["shared_dim"],
        num_heads=cfg["model"]["num_attention_heads"],
        num_layers=cfg["model"]["num_joint_attention_layers"],
        dropout=cfg["model"]["dropout"],
        ptm_bdl_d_model=int(bdl_cfg.get("d_model", 64)),
        ptm_bdl_n_heads=int(bdl_cfg.get("n_heads", 4)),
        ptm_bdl_n_layers=int(bdl_cfg.get("n_layers", 2)),
        use_typed_attention=use_typed_attention,
    )
