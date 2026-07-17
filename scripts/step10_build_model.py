#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 10 — Build & Verify PTM-BDL Model Architecture                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Instantiates MultimodalResistancePredictor (PTM-BDL architecture)           ║
║  and runs a dummy forward pass to verify all tensor shapes BEFORE training. ║
║                                                                              ║
║  Per-sample input contract (proposal §6 — quick sanity snapshot):            ║
║    ptm_vector          : float32 (12,)  → ptm_Y869..ptm_Y1197                ║
║    delta_ptm_vector    : float32 (12,)  → delta_ptm_Y869..delta_ptm_Y1197    ║
║    glyco_vector        : float32 (12,)  → glyco_slot00..glyco_slot11         ║
║    delta_glyco_vector  : float32 (12,)  → delta_glyco_slot00..delta_glyco_…  ║
║    target_protein      : long    ()     → 0=EGFR, 1=ERBB2                    ║
║                                                                              ║
║  Per-token feature built by PTMBDLEncoder:                                   ║
║    [level, delta, type_id, protein_id]                                          ║
║      tokens 0..11  : phospho (type_id from EGFR/ERBB2 site list)             ║
║      tokens 12..23 : glyco_N (type_id = 3)                                   ║
║      pads: level=delta=0 AND attention-masked                               ║
║                                                                              ║
║  INPUT:  src/models/multimodal_predictor.py + config/config.yaml             ║
║          + data/processed/glyco_slot_schema.json                             ║
║  OUTPUT: data/models/architecture_info.json                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
GLYCO_SCHEMA_PATH = PROJECT_ROOT / "data" / "processed" / "glyco_slot_schema.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal_predictor import (  # noqa: E402
    MultimodalResistancePredictor,
    PTM_TYPE_NAMES,
    PROTEIN_ID_EGFR,
    PROTEIN_ID_ERBB2,
)

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Build
# ══════════════════════════════════════════════════════════════════════════════

def build_model() -> MultimodalResistancePredictor:
    """Create the PTM-BDL multimodal model from config hyperparameters."""
    print("\n" + "=" * 70)
    print("STEP 10.1: Building Multimodal Resistance Predictor (PTM-BDL)")
    print("=" * 70)

    bdl_cfg = cfg.get("ptm_bdl", {})
    model = MultimodalResistancePredictor(
        seq_dim=1280,
        struct_dim=512,
        drug_dim=384,
        ptm_dim=int(cfg["ptm"]["ptm_dim"]),         # 12 phospho slots
        glyco_dim=int(cfg["ptm"]["glyco_dim"]),      # 12 glyco slots
        shared_dim=cfg["model"]["shared_dim"],
        num_heads=cfg["model"]["num_attention_heads"],
        num_layers=cfg["model"]["num_joint_attention_layers"],
        dropout=cfg["model"]["dropout"],
        ptm_bdl_d_model=int(bdl_cfg.get("d_model", 64)),
        ptm_bdl_n_heads=int(bdl_cfg.get("n_heads", 4)),
        ptm_bdl_n_layers=int(bdl_cfg.get("n_layers", 2)),
        use_typed_attention=bool(bdl_cfg.get("use_typed_attention", True)),
    )

    print("\n  Parameter breakdown:")
    print("  " + "-" * 55)
    components = {
        "Sequence Projection (ESM-2 → shared)":      model.seq_projection,
        "Structure Projection (GearNet → shared)":   model.struct_projection,
        "Drug Projection (ChemBERTa → shared)":      model.drug_projection,
        "Modality Type Embeddings":                  model.modality_embeddings,
        "Static Joint Self-Attention":               model.static_transformer,
        "Static Attention Pool":                     model.static_pool,
        "PTM-BDL Encoder (24 typed tokens)":         model.ptm_bdl,
        "Bilinear Late Fusion (static⊙ptm)":        model.fusion,
        "Regression Head (IC50)":                    model.regression_head,
        "Classification Head (Resistance)":          model.classification_head,
    }
    total = 0
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters())
        total += n
        print(f"    {name:48s}: {n:>10,} params")
    print("  " + "-" * 55)
    print(f"    {'TOTAL':48s}: {total:>10,} params")
    print(f"    {'Memory (fp32)':48s}: {total * 4 / 1e6:>10.1f} MB")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2. Shape verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_shapes(model: MultimodalResistancePredictor):
    """Dummy forward pass through the PTM-BDL model."""
    print("\n" + "=" * 70)
    print("STEP 10.2: Shape Verification (PTM-BDL Forward Pass)")
    print("=" * 70)

    model.eval()
    torch.manual_seed(42)

    B = 2
    L, M, Ndrug = 100, 200, 25

    seq_emb = torch.randn(B, L, 1280)
    struct_emb = torch.randn(B, M, 512)
    drug_emb = torch.randn(B, Ndrug, 384)
    drug_pooled = torch.randn(B, 384)

    # Sample 0: EGFR L858R+osimertinib
    # Sample 1: ERBB2 BT-474+lapatinib (slots 10-11 phospho pad + slots 7-11 glyco pad)
    ptm_vec = torch.tensor([
        [2.5, 1.3, 2.0, 1.8, 1.2, 1.5, 0.6, 4.0, 2.5, 2.0, 2.0, 3.5],
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0],
    ], dtype=torch.float32)
    delta_ptm_vec = torch.tensor([
        [-1.2, -2.6, -0.5, -2.5, -0.1, -2.5, -2.5, -5.6, -3.7, -2.5, -3.2, -3.9],
        [-0.8, -0.3, -0.5, -0.3, -0.4, -0.3, -0.8, -2.4, -1.6, -2.1, 0.0, 0.0],
    ], dtype=torch.float32)
    glyco_vec = torch.tensor([
        [1.0, 1.2, 0.9, 1.5, 1.0, 0.8, 1.1, 0.7, 1.3, 1.0, 0.9, 1.0],
        [1.0, 1.2, 0.9, 1.5, 1.0, 0.8, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    delta_glyco_vec = torch.zeros_like(glyco_vec)
    target_protein = torch.tensor([PROTEIN_ID_EGFR, PROTEIN_ID_ERBB2], dtype=torch.long)

    print(f"\n  Input shapes:")
    print(f"    seq_embeddings    : {list(seq_emb.shape)}")
    print(f"    struct_embeddings : {list(struct_emb.shape)}")
    print(f"    drug_embeddings   : {list(drug_emb.shape)}")
    print(f"    drug_pooled       : {list(drug_pooled.shape)}")
    print(f"    ptm_vector        : {list(ptm_vec.shape)}")
    print(f"    delta_ptm_vector  : {list(delta_ptm_vec.shape)}")
    print(f"    glyco_vector      : {list(glyco_vec.shape)}")
    print(f"    delta_glyco_vector: {list(delta_glyco_vec.shape)}")
    print(f"    target_protein    : {list(target_protein.shape)} {target_protein.tolist()}")

    # ── Forward, no attention ────────────────────────────────────────────
    print("\n  Running forward pass (no attention)...")
    with torch.no_grad():
        ic50_pred, resist_logits = model(
            seq_embeddings=seq_emb,
            struct_embeddings=struct_emb,
            drug_pooled=drug_pooled,
            drug_embeddings=drug_emb,
            ptm_vector=ptm_vec,
            delta_ptm_vector=delta_ptm_vec,
            glyco_vector=glyco_vec,
            delta_glyco_vector=delta_glyco_vec,
            target_protein=target_protein,
        )

    print(f"\n  Output shapes:")
    print(f"    ic50_pred         : {list(ic50_pred.shape)}  (expected [{B}, 1])")
    print(f"    resistance_logits : {list(resist_logits.shape)}  (expected [{B}, 1])")
    assert ic50_pred.shape == (B, 1), "IC50 head output shape mismatch"
    assert resist_logits.shape == (B, 1), "Resistance head output shape mismatch"

    print(f"\n  Sample predictions:")
    for i, (tp, label) in enumerate(zip(target_protein.tolist(),
                                         ["EGFR L858R", "ERBB2 amp"])):
        protein = "EGFR" if tp == PROTEIN_ID_EGFR else "ERBB2"
        print(f"    Sample {i} ({protein} / {label}): "
              f"IC50={ic50_pred[i].item():+.4f}, "
              f"P(resist)={torch.sigmoid(resist_logits[i]).item():.4f}")

    # ── Forward with PTM-BDL details (for XAI) ───────────────────────────
    print("\n  Running forward pass (return_ptm_bdl=True, return_attention=True)...")
    with torch.no_grad():
        ic50_pred, resist_logits, extras = model(
            seq_embeddings=seq_emb,
            struct_embeddings=struct_emb,
            drug_pooled=drug_pooled,
            drug_embeddings=drug_emb,
            ptm_vector=ptm_vec,
            delta_ptm_vector=delta_ptm_vec,
            glyco_vector=glyco_vec,
            delta_glyco_vector=delta_glyco_vec,
            target_protein=target_protein,
            return_attention=True,
            return_ptm_bdl=True,
        )

    bdl = extras["ptm_bdl"]
    print(f"\n  PTM-BDL outputs:")
    print(f"    pooled   : {list(bdl['pooled'].shape)}  (expected [{B}, ptm_bdl_d_model])")
    print(f"    tokens   : {list(bdl['tokens'].shape)}  (expected [{B}, 24, ptm_bdl_d_model])")
    print(f"    mask     : {list(bdl['mask'].shape)}    (expected [{B}, 24] bool)")
    print(f"    type_ids : {list(bdl['type_ids'].shape)}    (expected [{B}, 24] long)")

    n_real_egfr = int(bdl["mask"][0].sum().item())
    n_real_erbb2 = int(bdl["mask"][1].sum().item())
    print(f"\n  Real-token counts (mask-aware):")
    print(f"    EGFR  : {n_real_egfr} real / 24 total  (expect 24 — all sites real)")
    print(f"    ERBB2 : {n_real_erbb2} real / 24 total  (expect 17 — 10 phospho + 7 glyco)")
    assert n_real_egfr == 24, f"EGFR real-token count mismatch: {n_real_egfr}"
    assert n_real_erbb2 == 17, f"ERBB2 real-token count mismatch: {n_real_erbb2}"

    # Sanity: per-slot type ids should match _TYPE_PHOSPHO_<GENE> + glyco_N
    egfr_types = bdl["type_ids"][0].tolist()
    erbb2_types = bdl["type_ids"][1].tolist()
    print(f"\n  Per-slot type_id:")
    print(f"    EGFR  : {egfr_types}")
    print(f"    ERBB2 : {erbb2_types}")
    print(f"    (mapping: {PTM_TYPE_NAMES})")

    # Static attention shapes
    static_attn = extras["static_attention_maps"]
    if static_attn is not None:
        T_static = L + M + Ndrug
        print(f"\n  Static joint attention:")
        print(f"    Layers : {len(static_attn)}")
        for i, attn in enumerate(static_attn):
            shape = list(attn.shape)
            print(f"      Layer {i}: {shape}  "
                  f"(expect [{B}, {T_static}, {T_static}])")

    # ── Typed cross-attention extraction (for step13) ────────────────────
    print("\n  Computing PTM-BDL typed cross-attention weights...")
    attn_weights = model.ptm_bdl.compute_attn_weights(
        ptm_vec, delta_ptm_vec, glyco_vec, delta_glyco_vec, target_protein,
    )
    print(f"    attn_weights : {list(attn_weights.shape)}  (expected [{B}, 24, 24])")
    assert attn_weights.shape == (B, 24, 24)

    # Cross-type attention: how much do phospho tokens attend to glyco tokens?
    # Slots 0-11 = phospho, 12-23 = glyco.
    for i, protein in enumerate(["EGFR", "ERBB2"]):
        mask_i = bdl["mask"][i]
        phospho_to_glyco = attn_weights[i, :12, 12:24][mask_i[:12]][:, mask_i[12:24]].mean().item()
        glyco_to_phospho = attn_weights[i, 12:24, :12][mask_i[12:24]][:, mask_i[:12]].mean().item()
        print(f"    {protein}  phospho→glyco mean attn: {phospho_to_glyco:.4f}")
        print(f"    {protein}  glyco→phospho mean attn: {glyco_to_phospho:.4f}")

    print("\n  ✓ All shapes verified! PTM-BDL model is ready for training.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Save architecture info JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_architecture_info(model: MultimodalResistancePredictor):
    """Write a self-describing JSON about the PTM-BDL architecture."""
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    glyco_schema_rel = str(GLYCO_SCHEMA_PATH.relative_to(PROJECT_ROOT))

    info = {
        "model_name": "MultimodalResistancePredictor (PTM-BDL)",
        "architecture_version": "ptm_bdl_v1_2026-06-28",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        "hyperparameters": {
            "shared_dim": cfg["model"]["shared_dim"],
            "num_joint_attention_layers": cfg["model"]["num_joint_attention_layers"],
            "num_attention_heads": cfg["model"]["num_attention_heads"],
            "dropout": cfg["model"]["dropout"],
            "esm2_dim": 1280,
            "gearnet_dim": 512,
            "chemberta_dim": 384,
            "ptm_dim": int(cfg["ptm"]["ptm_dim"]),
            "glyco_dim": int(cfg["ptm"]["glyco_dim"]),
        },
        "ptm_bdl_block": {
            "d_model": int(cfg.get("ptm_bdl", {}).get("d_model", 64)),
            "n_heads": int(cfg.get("ptm_bdl", {}).get("n_heads", 4)),
            "n_layers": int(cfg.get("ptm_bdl", {}).get("n_layers", 2)),
            "use_typed_attention": bool(cfg.get("ptm_bdl", {})
                                         .get("use_typed_attention", True)),
            "pool": "mask_aware_mean",
            "n_tokens": 24,
            "token_layout": (
                "tokens 0..11 = phospho sites (protein-specific); "
                "tokens 12..23 = glyco_N sites (protein-specific). "
                "EGFR: 24 real slots. ERBB2: 10 phospho real + 2 pad, "
                "7 glyco real + 5 pad — pads are attention-masked + pool-excluded."
            ),
            "type_id_legend": PTM_TYPE_NAMES,
            "protein_id_legend": {"0": "EGFR", "1": "ERBB2"},
            "glyco_slot_schema_path": glyco_schema_rel,
        },
        "multi_protein_support": {
            "target_proteins": ["EGFR", "ERBB2"],
            "egfr_phospho_real_slots": 12,
            "erbb2_phospho_real_slots": 10,
            "egfr_glyco_real_slots": 12,
            "erbb2_glyco_real_slots": 7,
            "note": (
                "PTM-BDL keeps a single 24-token tensor shape across both "
                "proteins; the type_id_table + is_real_table buffers select "
                "the protein-specific type vocabulary and pad mask per sample."
            ),
        },
        "components": [
            "ModalityProjection (seq, struct, drug → shared)",
            "StaticJointTransformer (early fusion of seq/struct/drug)",
            "AttentionPooling (static branch)",
            "PTMBDLEncoder (typed 24-token self-attention, type gate, residual gate, mask-aware pool)",
            "BilinearLateFusion (static_rep ⊙ ptm_rep — no separate drug branch)",
            "RegressionHead + ClassificationHead",
        ],
        "rationale": (
            "Replaces the 2026-06-24 phospho-only PTMTokenEncoder + "
            "PTMFeatureModulator (which failed the randomised PTM control, "
            "collapsed mutation groups, and ignored the new glyco channel) "
            "with a typed phospho⊕glyco PTM-BDL block (proposal §3, §7.4). "
            "Symmetric across the two proteins (protein_id + pad mask), the two "
            "PTM types (type_id ∈ {pY, pS, pT, glyco_N}), and the six drugs "
            "(drugs enter only via late bilinear fusion)."
        ),
    }

    info_path = models_dir / "architecture_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  ✓ Architecture info saved: {info_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 10: Build & Verify PTM-BDL Architecture              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not GLYCO_SCHEMA_PATH.exists():
        print(f"\n  ⚠ Warning: glyco slot schema not found at {GLYCO_SCHEMA_PATH}")
        print(f"    Step 06 must be re-run if the glyco channel was added recently.")

    model = build_model()
    verify_shapes(model)
    save_architecture_info(model)

    print("\n✓ Step 10 complete! PTM-BDL architecture verified.")
    print("  Next: Step 11 (training).")
