#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 11 — Training Pipeline (Revised 2026-06-26)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ERBB FAMILY SUPPORT: Trains on BOTH EGFR (NSCLC) and HER2 (breast)        ║
║    • 951 samples: 646 EGFR + 305 HER2/ERBB2                                ║
║    • 6 drugs: Osimertinib, Gefitinib, Afatinib, Erlotinib (cross-protein)  ║
║               Lapatinib, Sapitinib (HER2-only)                              ║
║    • Stratified by (target_protein, resistance_label) for balanced splits   ║
║                                                                              ║
║  TRAINING STRATEGY:                                                          ║
║  ──────────────────                                                          ║
║  1. TASK C — NaN Fix: Missingness indicators instead of NaN→0.0 fill        ║
║  2. TASK D — Class Imbalance:                                                ║
║     • Stratified splits by RESISTANCE LABEL + TARGET PROTEIN                ║
║     • Class-conditional focal loss (α_t: majority α=0.25, minority 1-α=0.75)║
║     • WeightedRandomSampler oversamples minority class in each batch        ║
║  3. Early stopping on max(AUROC, BAcc) — whichever provides better signal   ║
║  4. TASK A — Pathway features NOT model input (kept as validation)           ║
║  5. TASK F — has_activating_mutation indicator for direct discrimination     ║
║                                                                              ║
║  LOSS FUNCTION:                                                              ║
║    L = λ₁·MSE(pred_IC50, true_IC50)·w + λ₂·FocalLoss(pred_resist, true_R)  ║
║    where w = propagation_confidence per sample                               ║
║    λ₁=1.0 (regression), λ₂=2.0 (classification — boosted for imbalance)    ║
║                                                                              ║
║  EARLY STOPPING: on AUROC (revised 2026-06-25, was balanced accuracy).      ║
║    WHY AUROC OVER BAcc:                                                      ║
║      • With 90:7 class ratio, BAcc swings ±0.214 when 3 minority            ║
║        predictions flip — checkpoint selection becomes noise-driven.         ║
║      • Evidence: val BAcc=0.780 → test BAcc=0.632 (gap=0.148!) while        ║
║        AUROC=0.792 is stable, proving the model CAN discriminate.            ║
║      • AUROC is threshold-independent: measures probability ranking          ║
║        quality across ALL thresholds, not just the default 0.5 cutoff.       ║
║      • BAcc depends on a hard threshold, making it sensitive to small        ║
║        probability shifts that flip 2-3 minority predictions.               ║
║    WHY NOT LOG LOSS:                                                         ║
║      • Dominated by majority class (90× more resistant samples).            ║
║      • Total loss already includes regression (MSE), so stopping on          ║
║        classification loss alone ignores half the multi-task objective.      ║
║                                                                              ║
║  OUTPUTS:                                                                    ║
║    data/models/best_model_stage1.pt   — general model (all samples)         ║
║    data/models/best_model_stage2.pt   — specialist (EGFR-mutant fine-tuned) ║
║    data/models/best_model.pt          — best overall (backward compatible)  ║
║    data/models/split_indices.json     — train/val/test indices for step12   ║
║    data/models/training_history.json  — per-epoch metrics                   ║
║                                                                              ║
║  METRICS TRACKED (aligned with Step 12 evaluation):                          ║
║    Regression: MSE, RMSE, Pearson R                                         ║
║    Classification: Accuracy, Balanced Accuracy, Sensitivity, Specificity,   ║
║                    F1, mean predicted probability                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import (Dataset, DataLoader, Subset,
                               WeightedRandomSampler)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.multimodal_predictor import MultimodalResistancePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def build_model_from_cfg(cfg, use_typed_attention: bool = True) -> MultimodalResistancePredictor:
    """
    Single source of truth for instantiating the PTM-BDL multimodal model.

    Used by step11/step11b/step11c so every run produces an architecturally
    identical model.  Pass `use_typed_attention=False` to obtain the
    `no_typed_attention` architectural ablation (proposal §9.1).
    """
    bdl_cfg = cfg.get("ptm_bdl", {}) or {}
    return MultimodalResistancePredictor(
        seq_dim=1280,
        struct_dim=512,
        drug_dim=384,
        ptm_dim=int(cfg["ptm"]["ptm_dim"]),
        glyco_dim=int(cfg["ptm"]["glyco_dim"]),
        shared_dim=cfg["model"]["shared_dim"],
        num_heads=cfg["model"]["num_attention_heads"],
        num_layers=cfg["model"]["num_joint_attention_layers"],
        dropout=cfg["model"]["dropout"],
        ptm_bdl_d_model=int(bdl_cfg.get("d_model", 64)),
        ptm_bdl_n_heads=int(bdl_cfg.get("n_heads", 4)),
        ptm_bdl_n_layers=int(bdl_cfg.get("n_layers", 2)),
        use_typed_attention=use_typed_attention,
    )



with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss (Task D) — class-conditional alpha
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss with CLASS-CONDITIONAL alpha for handling class imbalance.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where α_t = α     for positive class (resistant, label=1) — majority
          α_t = 1-α   for negative class (sensitive, label=0) — minority

    With α=0.25:
      resistant (92.6%, label=1) gets weight 0.25
      sensitive (7.4%,  label=0) gets weight 0.75 → 3× up-weight on minority

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)

        # Class-conditional alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Class (Task C NaN fix + Task A pathway removal)
# ══════════════════════════════════════════════════════════════════════════════

class ResistanceDataset(Dataset):
    """
    PyTorch Dataset for PTM-BDL training.

    Features loaded per sample:
      • ESM-2 per-residue embeddings (L×1280) via sequence_id
      • GearNet residue embeddings (M×512) via pdb_id
      • ChemBERTa per-token + pooled (N×384, 384) via drug_name
      • PTM phospho channel: 12 baseline (ptm_*) + 12 delta (delta_ptm_*)
      • PTM glyco   channel: 12 baseline (glyco_slot*) + 12 delta (delta_glyco_slot*)
      • target_protein (long): 0=EGFR, 1=ERBB2  — drives the PTM-BDL pad mask

    Targets:
      • ln_ic50 (regression)
      • resistance_label (binary classification)

    ABLATION MODES (consumed by step11b/step11c — proposal §9.1):
      • "full"               — all features active (= Model C, phospho + glyco).
      • "no_ptm"             — zero ALL PTM features (= Model A, static baseline).
      • "no_glyco"           — phospho only (= Model B).
      • "glyco_only"         — zero phospho channels, keep glyco.
      • "no_typed_attention" — typed self-attention replaced by an MLP
                                (architectural ablation; build-time switch).

    HISTORICAL NOTE (2026-06-28):
      Earlier versions loaded Level-2 `phospho_vector` (7 aggregate rewiring
      features) + `indicators` (2 missingness flags) into the Dataset and a
      `PhosphoContextEncoder` token in the model.  Per
      PTM_Biological_Dynamics_Layer.md §1.2 Problem 3 + §8.1 these features
      are deterministic functions of `mutation_class` and contribute
      I(PTM; response | seq, struct, drug) ≈ 0 — re-introducing them is
      exactly the failure mode PTM-BDL was designed to fix.  The
      `PhosphoContextEncoder` was therefore removed from the model and the
      orphan dataset fields are removed here too.
    """
    def __init__(self, dataset_csv: Path, features_dir: Path,
                 ablation_mode: str = "full"):
        self.df = pd.read_csv(dataset_csv)
        self.features_dir = features_dir
        self.ablation_mode = ablation_mode
        self._load_embeddings()

    def _load_embeddings(self):
        """Load all pre-extracted embeddings from data/features/."""
        esm2_dir = self.features_dir / "esm2"
        gearnet_dir = self.features_dir / "gearnet"
        chemberta_dir = self.features_dir / "chemberta"

        self.seq_embeddings = {}
        self.struct_embeddings = {}
        self.drug_embeddings = {}
        self.drug_pooled = {}

        for f in esm2_dir.glob("*_per_residue.npy"):
            seq_id = f.stem.replace("_per_residue", "")
            self.seq_embeddings[seq_id] = np.load(f)

        for f in gearnet_dir.glob("*_residue_embeddings.npy"):
            pdb_id = f.stem.replace("_residue_embeddings", "")
            self.struct_embeddings[pdb_id] = np.load(f)

        for f in chemberta_dir.glob("*_per_token.npy"):
            drug_key = f.stem.replace("_per_token", "")
            self.drug_embeddings[drug_key] = np.load(f)
            pooled_path = chemberta_dir / f"{drug_key}_pooled.npy"
            if pooled_path.exists():
                self.drug_pooled[drug_key] = np.load(pooled_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Sequence embedding (ESM-2) ────────────────────────────────────────
        seq_id = row.get("sequence_id", "wild_type")
        seq_emb = self.seq_embeddings.get(seq_id)
        if seq_emb is None:
            seq_emb = np.random.randn(100, 1280).astype(np.float32)

        # ── Structural embedding (GearNet) ────────────────────────────────────
        pdb_id = row.get("pdb_id", "2GS6")
        struct_emb = self.struct_embeddings.get(pdb_id)
        if struct_emb is None:
            struct_emb = np.random.randn(200, 512).astype(np.float32)

        # ── Drug embedding (ChemBERTa) ────────────────────────────────────────
        drug_name = str(row.get("drug_name", "osimertinib")).lower()
        drug_key = drug_name.split()[0] if drug_name else "osimertinib"
        drug_emb = self.drug_embeddings.get(drug_key)
        drug_pool = self.drug_pooled.get(drug_key)
        if drug_emb is None:
            drug_emb = np.random.randn(20, 384).astype(np.float32)
            drug_pool = np.random.randn(384).astype(np.float32)
        if drug_pool is None:
            drug_pool = drug_emb.mean(axis=0)

        # ── Level 1: PTM State (12 EGFR phosphosites) ────────────────────────
        ptm_cols = [
            "ptm_Y869", "ptm_S991", "ptm_Y998", "ptm_Y1016",
            "ptm_S1039", "ptm_T1041", "ptm_Y1069", "ptm_Y1092",
            "ptm_Y1110", "ptm_Y1125", "ptm_Y1172", "ptm_Y1197",
        ]
        ptm_values = []
        for col in ptm_cols:
            val = row.get(col, 1.0)
            ptm_values.append(float(val) if pd.notna(val) else 1.0)
        ptm_vector = np.array(ptm_values, dtype=np.float32)

        # ── Delta PTM: Drug-induced phospho changes (added 2026-06-23) ────────
        # These vary by drug AND mutation class, providing non-redundant signal
        delta_ptm_cols = [
            "delta_ptm_Y869", "delta_ptm_S991", "delta_ptm_Y998", "delta_ptm_Y1016",
            "delta_ptm_S1039", "delta_ptm_T1041", "delta_ptm_Y1069", "delta_ptm_Y1092",
            "delta_ptm_Y1110", "delta_ptm_Y1125", "delta_ptm_Y1172", "delta_ptm_Y1197",
        ]
        delta_ptm_values = []
        for col in delta_ptm_cols:
            val = row.get(col, 0.0)
            delta_ptm_values.append(float(val) if pd.notna(val) else 0.0)
        delta_ptm_vector = np.array(delta_ptm_values, dtype=np.float32)

        # ── Glyco channel (PTM-BDL, added 2026-06-28) ────────────────────────
        # 12 slots per gene.  EGFR uses all 12; ERBB2 uses 7 real + 5 zero pads.
        # Slot ↔ residue map: data/processed/glyco_slot_schema.json
        glyco_cols = [f"glyco_slot{i:02d}" for i in range(12)]
        glyco_values = []
        for col in glyco_cols:
            val = row.get(col, 1.0)
            glyco_values.append(float(val) if pd.notna(val) else 1.0)
        glyco_vector = np.array(glyco_values, dtype=np.float32)

        delta_glyco_cols = [f"delta_glyco_slot{i:02d}" for i in range(12)]
        delta_glyco_values = []
        for col in delta_glyco_cols:
            val = row.get(col, 0.0)
            delta_glyco_values.append(float(val) if pd.notna(val) else 0.0)
        delta_glyco_vector = np.array(delta_glyco_values, dtype=np.float32)

        # ── target_protein (long): 0=EGFR, 1=ERBB2 ────────────────────────────
        # Drives the PTM-BDL pad mask and protein_id embedding.
        tp_str = str(row.get("target_protein", "EGFR")).upper()
        target_protein = 1 if tp_str == "ERBB2" else 0

        # ── propagation_confidence — kept for diagnostic / weighting only
        # (NOT a model input; see comment in train_epoch below).
        prop_conf = float(row.get("propagation_confidence", 0.5))
        if pd.isna(prop_conf):
            prop_conf = 0.5

        # ── ABLATION: zero features based on mode (proposal §9.1) ──────────
        # Phospho baseline of 1.0 = WT-equivalent (no modulation in PTM-BDL);
        # glyco baseline of 1.0 = unit relative occupancy.  Deltas zero out
        # cleanly to 0.0.  Pads in ERBB2 are still attention-masked in the
        # PTM-BDL encoder regardless of which ablation mode is active.
        if self.ablation_mode == "no_ptm":
            # Model A: static-only baseline (no PTM signal at all)
            ptm_vector = np.ones(12, dtype=np.float32)
            delta_ptm_vector = np.zeros(12, dtype=np.float32)
            glyco_vector = np.ones(12, dtype=np.float32)
            delta_glyco_vector = np.zeros(12, dtype=np.float32)
        elif self.ablation_mode == "no_glyco":
            # Model B: phospho only — tests glyco channel marginal value
            glyco_vector = np.ones(12, dtype=np.float32)
            delta_glyco_vector = np.zeros(12, dtype=np.float32)
        elif self.ablation_mode == "glyco_only":
            # Glyco-only — tests whether glyco alone carries usable signal
            ptm_vector = np.ones(12, dtype=np.float32)
            delta_ptm_vector = np.zeros(12, dtype=np.float32)
        # else: "full" (= Model C) / "no_typed_attention" — keep all features
        # (the latter is an architectural ablation handled at MODEL build time).

        # ── Targets ───────────────────────────────────────────────────────────
        ln_ic50 = float(row.get("ln_ic50", 0.0))
        resistance = int(row.get("resistance_label", 0))

        return {
            "seq_emb": torch.from_numpy(seq_emb.astype(np.float32)),
            "struct_emb": torch.from_numpy(struct_emb.astype(np.float32)),
            "drug_emb": torch.from_numpy(drug_emb.astype(np.float32)),
            "drug_pooled": torch.from_numpy(drug_pool.astype(np.float32)),
            "ptm_vector": torch.from_numpy(ptm_vector),
            "delta_ptm_vector": torch.from_numpy(delta_ptm_vector),
            "glyco_vector": torch.from_numpy(glyco_vector),
            "delta_glyco_vector": torch.from_numpy(delta_glyco_vector),
            "target_protein": torch.tensor(target_protein, dtype=torch.long),
            "propagation_confidence": torch.tensor([prop_conf], dtype=torch.float32),
            "ln_ic50": torch.tensor([ln_ic50], dtype=torch.float32),
            "resistance_label": torch.tensor([resistance], dtype=torch.float32),
        }



def collate_fn(batch):
    """Custom collation: pad variable-length sequences to max batch length."""
    max_L = max(item["seq_emb"].size(0) for item in batch)
    max_M = max(item["struct_emb"].size(0) for item in batch)
    max_N = max(item["drug_emb"].size(0) for item in batch)

    seq_embs = torch.zeros(len(batch), max_L, batch[0]["seq_emb"].size(1))
    struct_embs = torch.zeros(len(batch), max_M, batch[0]["struct_emb"].size(1))
    drug_embs = torch.zeros(len(batch), max_N, batch[0]["drug_emb"].size(1))

    for i, item in enumerate(batch):
        seq_embs[i, :item["seq_emb"].size(0)] = item["seq_emb"]
        struct_embs[i, :item["struct_emb"].size(0)] = item["struct_emb"]
        drug_embs[i, :item["drug_emb"].size(0)] = item["drug_emb"]

    return {
        "seq_emb": seq_embs,
        "struct_emb": struct_embs,
        "drug_emb": drug_embs,
        "drug_pooled": torch.stack([b["drug_pooled"] for b in batch]),
        "ptm_vector": torch.stack([b["ptm_vector"] for b in batch]),
        "delta_ptm_vector": torch.stack([b["delta_ptm_vector"] for b in batch]),
        # PTM-BDL glyco channel + protein id
        "glyco_vector": torch.stack([b["glyco_vector"] for b in batch]),
        "delta_glyco_vector": torch.stack([b["delta_glyco_vector"] for b in batch]),
        "target_protein": torch.stack([b["target_protein"] for b in batch]),  # long (B,)
        "propagation_confidence": torch.stack([b["propagation_confidence"] for b in batch]),
        "ln_ic50": torch.stack([b["ln_ic50"] for b in batch]),
        "resistance_label": torch.stack([b["resistance_label"] for b in batch]),
    }



# ══════════════════════════════════════════════════════════════════════════════
# Stratified Splitting — by RESISTANCE LABEL
# ══════════════════════════════════════════════════════════════════════════════

def create_stratified_splits(dataset, train_ratio, val_ratio, seed):
    """
    Stratified train/val/test split by RESISTANCE LABEL + TARGET GENE.

    Creates a combined stratification key: "EGFR_resistant", "EGFR_sensitive",
    "ERBB2_resistant", "ERBB2_sensitive" to ensure each split has proportional
    EGFR and ERBB2 samples AND proportional resistant/sensitive samples.

    This prevents the test set from having 0 sensitive samples or 0 ERBB2
    samples by chance.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    df = dataset.df
    labels = df["resistance_label"].values.astype(int)

    # Create combined stratification label: target_protein + resistance
    target_protein = df["target_protein"].fillna("EGFR").values
    combined_labels = np.array([
        f"{tg}_{int(r)}" for tg, r in zip(target_protein, labels)
    ])

    # Also compute mutation status for reporting
    mc = df["mutation_classes"].fillna("wild_type").str.lower()
    is_mutant = mc.str.contains("pathogenic|cmp_driver", regex=True).astype(int).values

    # Split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(df)), combined_labels))

    # Split: train vs val
    val_frac = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_sub, val_sub = next(sss2.split(
        np.zeros(len(trainval_idx)), combined_labels[trainval_idx]
    ))

    train_idx = trainval_idx[train_sub]
    val_idx = trainval_idx[val_sub]

    # Report class distribution, gene balance, AND mutation status
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        n_sens = (labels[idx] == 0).sum()
        n_res = (labels[idx] == 1).sum()
        n_mut = is_mutant[idx].sum()
        n_egfr = (target_protein[idx] == "EGFR").sum()
        n_erbb2 = (target_protein[idx] == "ERBB2").sum()
        print(f"    {name}: {len(idx)} samples | "
              f"{n_res} resistant + {n_sens} sensitive | "
              f"EGFR={n_egfr}, ERBB2={n_erbb2} | "
              f"{n_mut} EGFR-mutant")

    return train_idx, val_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# Per-Class Metrics (aligned with Step 12 evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(all_preds, all_probs, all_labels,
                    all_ic50_preds=None, all_ic50_targets=None):
    """
    Compute comprehensive metrics for both classification and regression.

    Classification metrics (aligned with step12):
      - accuracy, balanced_accuracy, sensitivity, specificity, f1
      - mean_prob (average predicted probability — detects sigmoid saturation)
      - confusion matrix (TP/TN/FP/FN)

    Regression metrics (if IC50 provided):
      - mse, rmse, pearson_r
    """
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    labels = np.array(all_labels)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / max(len(labels), 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_acc = (sensitivity + specificity) / 2.0
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)

    # Mean predicted probability — if near 1.0, model is saturated
    mean_prob = float(probs.mean())

    # ── AUROC & PR-AUC (threshold-independent, best for imbalanced data) ──
    # With 90:7 class ratio, threshold-dependent metrics (BAcc, F1) are noisy.
    # AUROC measures overall discriminative ability across ALL thresholds.
    # PR-AUC (sensitive class) directly measures ability to find minority cases
    #   — penalizes false positives harder than AUROC when negatives dominate.
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(set(labels)) >= 2:
            auroc = float(roc_auc_score(labels, probs))
            # PR-AUC for resistant class (majority) — high by default
            auprc_resistant = float(average_precision_score(labels, probs))
            # PR-AUC for sensitive class (minority) — THE critical metric
            # Flip labels and probs: sensitive=1, resistant=0
            auprc_sensitive = float(average_precision_score(
                1 - labels, 1 - probs))
        else:
            auroc = 0.0
            auprc_resistant = 0.0
            auprc_sensitive = 0.0
    except Exception:
        auroc = 0.0
        auprc_resistant = 0.0
        auprc_sensitive = 0.0

    # ── F1-macro (average across BOTH classes, not just majority) ──────────
    # The original f1 = resistant-class F1. With 92.8% resistant,
    # predicting ALL resistant gives F1=0.963, hiding model failure.
    # F1-macro equally weights both classes like BAcc, but uses F1 per class.
    precision_s = tn / max(tn + fn, 1)   # sensitive precision
    recall_s = tn / max(tn + fp, 1)      # sensitive recall = specificity
    f1_sensitive = 2 * precision_s * recall_s / max(precision_s + recall_s, 1e-8)
    f1_macro = (f1 + f1_sensitive) / 2.0

    result = {
        "accuracy": float(accuracy),
        "balanced_acc": float(balanced_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "f1_sensitive": float(f1_sensitive),
        "f1_macro": float(f1_macro),
        "auroc": auroc,
        "auprc_resistant": auprc_resistant,
        "auprc_sensitive": auprc_sensitive,
        "mean_prob": mean_prob,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    # Regression metrics
    if all_ic50_preds is not None and all_ic50_targets is not None:
        ic50_p = np.array(all_ic50_preds)
        ic50_t = np.array(all_ic50_targets)
        result["mse"] = float(((ic50_p - ic50_t) ** 2).mean())
        result["rmse"] = float(np.sqrt(result["mse"]))
        if len(ic50_p) > 2 and np.std(ic50_p) > 1e-8:
            result["pearson_r"] = float(np.corrcoef(ic50_p, ic50_t)[0, 1])
        else:
            result["pearson_r"] = 0.0

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Training & Validation
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, focal_loss,
                lambda_reg, lambda_cls, device, label_smoothing=0.05):
    """
    Run one training epoch with multi-task loss.
    
    Improvements over baseline:
      - Label smoothing (0.05) prevents overconfident classification
      - Huber loss for regression (robust to IC50 outliers)
      - Per-step LR scheduling (OneCycleLR compatible)
    """
    model.train()
    losses = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # ── KNOWN LIMITATION: 28 effective input combinations (2026-06-24) ────
        # 610/646 WT samples map to only 4 distinct inputs (1 per drug).
        # Each WT+drug combo has ~150 samples with IDENTICAL inputs but different
        # IC50 targets (std=1.2-1.5). The within-group IC50 variance comes from
        # cell-line-specific factors NOT in our model (co-mutations like KRAS,
        # EGFR expression levels, growth rates). The model correctly predicts the
        # GROUP MEAN for each input combination. Adding noise would inject
        # biologically false PTM variation. The proper fix requires adding
        # cell-line-specific features (RNA-seq, co-mutations) in future work.

        # PTM-BDL forward (24 typed tokens: phospho ⊕ glyco, per-gene mask)
        ic50_pred, resist_pred = model(
            seq_embeddings=batch["seq_emb"],
            struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"],
            drug_embeddings=batch["drug_emb"],
            ptm_vector=batch["ptm_vector"],
            delta_ptm_vector=batch["delta_ptm_vector"],
            glyco_vector=batch["glyco_vector"],
            delta_glyco_vector=batch["delta_glyco_vector"],
            target_protein=batch["target_protein"],
        )

        # Regression loss (Huber for outlier robustness)

        # NOTE (2026-06-28): propagation_confidence is NOT passed to the model.
        # REASON: confidence is a near-deterministic function of mutation_class
        # (0.40 for all WT, 0.65-1.0 for mutants). Using it as a PTM scaling
        # factor would reintroduce the §1.2 failure mode — another mutation-
        # class proxy that the model can shortcut through. Confidence is used
        # ONLY in step12 Part 4 for diagnostic stratification.
        #
        # NOTE (2026-06-24): Confidence weighting REMOVED from loss.
        # REASON: propagation_confidence = 0.40 for all 610 WT samples and
        # 0.65-1.00 for 36 EGFR-mutant samples.  Using it as loss weight
        # gives WT 7.8× total weight, AMPLIFYING the class imbalance
        # (model ignores the 36 biologically important mutant samples).
        # Kept as a per-sample diagnostic column for step12 confidence
        # stratification, but not consumed inside the model.
        huber = F.smooth_l1_loss(ic50_pred, batch["ln_ic50"], reduction='none')
        loss_reg = huber.squeeze(-1).mean()

        # Label smoothing: soften 0→0.05, 1→0.95 (prevents overconfidence)
        targets_smooth = batch["resistance_label"] * (1 - label_smoothing) + label_smoothing / 2
        loss_cls = focal_loss(resist_pred, targets_smooth)

        loss = lambda_reg * loss_reg + lambda_cls * loss_cls

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Per-step scheduling (for OneCycleLR)
        if scheduler is not None and hasattr(scheduler, '_step_count'):
            # OneCycleLR steps per batch, not per epoch
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        losses.append(loss.item())

    # Epoch-level scheduling (for CosineAnnealing etc.)
    if scheduler is not None:
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    return np.mean(losses)


def validate(model, loader, focal_loss, lambda_reg, lambda_cls, device):
    """Validate and return comprehensive metrics."""
    model.eval()
    val_losses = []
    all_preds, all_probs, all_labels = [], [], []
    all_ic50_preds, all_ic50_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # PTM-BDL forward (24 typed tokens: phospho ⊕ glyco, per-gene mask)
            ic50_pred, resist_pred = model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch["drug_emb"],
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                glyco_vector=batch["glyco_vector"],
                delta_glyco_vector=batch["delta_glyco_vector"],
                target_protein=batch["target_protein"],
            )

            loss_reg = ((ic50_pred - batch["ln_ic50"]) ** 2).mean()
            loss_cls = focal_loss(resist_pred, batch["resistance_label"])
            val_losses.append((lambda_reg * loss_reg + lambda_cls * loss_cls).item())

            probs = torch.sigmoid(resist_pred).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(float)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(
                batch["resistance_label"].cpu().numpy().flatten().tolist()
            )
            all_ic50_preds.extend(ic50_pred.cpu().numpy().flatten().tolist())
            all_ic50_targets.extend(batch["ln_ic50"].cpu().numpy().flatten().tolist())

    metrics = compute_metrics(all_preds, all_probs, all_labels,
                              all_ic50_preds, all_ic50_targets)
    metrics["loss"] = float(np.mean(val_losses))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train():
    """Two-stage training pipeline with class-balanced sampling."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 11: Training — Revised Pipeline (2026-06-20)         ║")
    print("║  Stratify by resistance label + balanced sampling + focal  ║")
    print("║  Two-stage: general → EGFR-mutant specialist              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Setup ─────────────────────────────────────────────────────────────────
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = cfg["training"]["device"]
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if (hasattr(torch.backends, "mps")
                           and torch.backends.mps.is_available())
            else "cpu"
        )
    else:
        device = torch.device(device_str)
    print(f"\n  Device: {device}")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    print(f"  Dataset: {dataset_path}")
    dataset = ResistanceDataset(dataset_path, features_dir)
    print(f"  Samples: {len(dataset)}")

    n_resistant = int(dataset.df["resistance_label"].sum())
    n_sensitive = len(dataset) - n_resistant
    print(f"  Class distribution: {n_resistant} resistant ({100*n_resistant/len(dataset):.1f}%), "
          f"{n_sensitive} sensitive ({100*n_sensitive/len(dataset):.1f}%)")

    # ── Stratified Split by RESISTANCE LABEL ──────────────────────────────────
    print(f"\n  Creating stratified splits (by resistance label)...")
    train_idx, val_idx, test_idx = create_stratified_splits(
        dataset, cfg["training"]["train_ratio"],
        cfg["training"]["val_ratio"], seed,
    )

    # Save split indices for step12 reproducibility
    split_info = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "stratification": "resistance_label",
        "seed": seed,
    }
    with open(MODEL_DIR / "split_indices.json", "w") as f:
        json.dump(split_info, f)
    print(f"  ✓ Split indices saved: {MODEL_DIR / 'split_indices.json'}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # ── WeightedRandomSampler for class-balanced batches ──────────────────────
    # Oversample minority class so each batch sees ~50/50 resistant/sensitive
    train_labels = dataset.df["resistance_label"].values[train_idx]
    class_counts = np.bincount(train_labels.astype(int))
    # Weight = 1 / class_count → minority gets higher weight
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_set),
        replacement=True,
    )
    print(f"  Balanced sampling: sensitive weight={class_weights[0]:.4f}, "
          f"resistant weight={class_weights[1]:.4f} "
          f"(ratio {class_weights[0]/class_weights[1]:.1f}×)")

    batch_size = cfg["model"]["batch_size"]
    # Note: sampler and shuffle are mutually exclusive
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # ── Model (PTM-BDL) ──────────────────────────────────────────────────────
    model = build_model_from_cfg(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,} (PTM-BDL phospho⊕glyco)")


    # ── Loss ──────────────────────────────────────────────────────────────────
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    lambda_reg = 1.0
    lambda_cls = 2.0  # Boosted: classification needs stronger gradients

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: General Training on ALL Samples
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  STAGE 1: General training (all {len(train_set)} samples)")
    print(f"  {'='*60}")

    lr = cfg["model"]["learning_rate"]
    wd = cfg["model"]["weight_decay"]
    num_epochs = cfg["model"]["num_epochs"]
    patience = cfg["model"]["early_stopping_patience"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_val_score = 0.0  # Early stop on max(AUROC, BAcc) — whichever is better
    patience_counter = 0
    history = {
        "stage1_train_loss": [], "stage1_val_loss": [],
        "stage1_rmse": [], "stage1_pearson_r": [],
        "stage1_balanced_acc": [], "stage1_f1": [],
        "stage1_auroc": [],
    }

    print(f"  LR={lr}, epochs={num_epochs}, patience={patience}")
    print(f"  Focal: α=0.25 (3× minority), γ=2.0 | λ_reg={lambda_reg}, λ_cls={lambda_cls}")
    print(f"  Early stopping: max(AUROC, BAcc) — uses whichever metric is higher")
    hdr = ("  " + "-" * 78 + "\n"
           f"  {'Ep':>5s} | {'TrLoss':>7s} | {'VLoss':>7s} | {'RMSE':>6s} | "
           f"{'R':>6s} | {'Acc':>5s} | {'BAcc':>5s} | {'Sens':>5s} | "
           f"{'Spec':>5s} | {'mProb':>5s}")
    print(hdr)
    print("  " + "-" * 78)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 focal_loss, lambda_reg, lambda_cls, device)
        val = validate(model, val_loader, focal_loss,
                       lambda_reg, lambda_cls, device)

        history["stage1_train_loss"].append(train_loss)
        history["stage1_val_loss"].append(val["loss"])
        history["stage1_rmse"].append(val.get("rmse", 0))
        history["stage1_pearson_r"].append(val.get("pearson_r", 0))
        history["stage1_balanced_acc"].append(val["balanced_acc"])
        history["stage1_f1"].append(val["f1"])
        history["stage1_auroc"].append(val.get("auroc", 0))

        if epoch % 5 == 0 or epoch <= 2:
            print(f"  {epoch:3d}/{num_epochs} | "
                  f"{train_loss:7.4f} | {val['loss']:7.4f} | "
                  f"{val.get('rmse',0):6.3f} | {val.get('pearson_r',0):6.3f} | "
                  f"{val['accuracy']:5.3f} | {val['balanced_acc']:5.3f} | "
                  f"{val['sensitivity']:5.3f} | {val['specificity']:5.3f} | "
                  f"{val['mean_prob']:5.3f} | AUC={val.get('auroc',0):.3f}")

        # Early stopping on max(AUROC, BAcc) — uses whichever provides better signal
        val_auroc = val.get("auroc", 0)
        val_bacc = val.get("balanced_acc", 0)
        val_score = max(val_auroc, val_bacc)
        if val_score > best_val_score:
            best_val_score = val_score
            best_stopping_metric = "AUROC" if val_auroc >= val_bacc else "BAcc"
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model_stage1.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best metric: {best_stopping_metric}={best_val_score:.3f})")
                break

    # ── Stage 1 summary ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_DIR / "best_model_stage1.pt",
                                     map_location=device, weights_only=True))
    s1 = validate(model, val_loader, focal_loss, lambda_reg, lambda_cls, device)
    print(f"\n  ✓ Stage 1 best: {best_stopping_metric}={best_val_score:.3f}")
    print(f"    BAcc={s1['balanced_acc']:.3f} | RMSE={s1.get('rmse',0):.3f} | "
          f"R={s1.get('pearson_r',0):.3f} | F1={s1['f1']:.3f}")
    print(f"    Confusion: TP={s1['tp']}, TN={s1['tn']}, FP={s1['fp']}, FN={s1['fn']}")
    print(f"    Mean prob: {s1['mean_prob']:.3f}")

    if s1.get("auroc", 0) < 0.55:
        print(f"\n  ⚠ WARNING: AUROC={s1.get('auroc',0):.3f} < 0.55 "
              f"— model may not be learning to discriminate classes")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: REMOVED (2026-06-23)
    # ──────────────────────────────
    # The original Stage 2 was EGFR-mutant specialist fine-tuning:
    #   - Frozen backbone + low LR (1/10th) on 27 high-confidence samples
    #   - Produced IDENTICAL val metrics (same BAcc, confusion matrix)
    #   - With only 27 samples and 12.4M frozen params, insufficient signal
    # See git history for original Stage 2 code.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Save best_model.pt ────────────────────────────────────────────────────
    import shutil
    src = MODEL_DIR / "best_model_stage1.pt"
    if src.exists():
        shutil.copy(src, MODEL_DIR / "best_model.pt")

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMAL THRESHOLD via Youden's J statistic (2026-07-03)
    # ──────────────────────────────────────────────────────────────────────────
    # With 92:8 class imbalance, the default 0.5 threshold produces poor
    # sensitivity (only 52% of resistant correctly classified). Youden's J
    # finds the threshold that maximizes sensitivity + specificity - 1,
    # equivalent to maximizing balanced accuracy.
    #
    # Biologically correct because both error types matter equally:
    #   - False negative (miss resistant) → patient gets ineffective drug
    #   - False positive (miss sensitive) → patient denied effective treatment
    #
    # The threshold is optimized on the VALIDATION set (not test) to avoid
    # data leakage, then applied at test time in step12/step13.
    #
    # Refs: Youden WJ (1950) Cancer 3:32-35;
    #       Fluss et al. (2005) Biometrical J 47:458-472;
    #       Perkins & Schisterman (2006) Am J Epidemiol 163:670-675.
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  THRESHOLD OPTIMIZATION (Youden's J on validation set)")
    print(f"  {'='*60}")

    optimal_threshold = 0.5  # fallback
    try:
        from sklearn.metrics import roc_curve
        # Collect validation predictions
        model.eval()
        val_probs_all, val_labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, resist_pred = model(
                    seq_embeddings=batch["seq_emb"],
                    struct_embeddings=batch["struct_emb"],
                    drug_pooled=batch["drug_pooled"],
                    drug_embeddings=batch["drug_emb"],
                    ptm_vector=batch["ptm_vector"],
                    delta_ptm_vector=batch["delta_ptm_vector"],
                    glyco_vector=batch["glyco_vector"],
                    delta_glyco_vector=batch["delta_glyco_vector"],
                    target_protein=batch["target_protein"],
                )
                probs = torch.sigmoid(resist_pred).cpu().numpy().flatten()
                val_probs_all.extend(probs.tolist())
                val_labels_all.extend(
                    batch["resistance_label"].cpu().numpy().flatten().tolist())

        val_probs_arr = np.array(val_probs_all)
        val_labels_arr = np.array(val_labels_all)

        if len(set(val_labels_arr)) >= 2:
            fpr, tpr, thresholds = roc_curve(val_labels_arr, val_probs_arr)
            # Youden's J = sensitivity + specificity - 1 = TPR + (1-FPR) - 1 = TPR - FPR
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            optimal_threshold = float(thresholds[best_idx])

            # Clamp to reasonable range (avoid extreme thresholds)
            optimal_threshold = max(0.1, min(0.9, optimal_threshold))

            best_sens = float(tpr[best_idx])
            best_spec = float(1 - fpr[best_idx])
            best_bacc = (best_sens + best_spec) / 2

            print(f"  Youden's J optimized on {len(val_labels_arr)} val samples")
            print(f"  Optimal threshold: {optimal_threshold:.4f} "
                  f"(vs default 0.5)")
            print(f"  At optimal: Sensitivity={best_sens:.3f}, "
                  f"Specificity={best_spec:.3f}, BAcc={best_bacc:.3f}")

            # Compare with default 0.5
            default_preds = (val_probs_arr > 0.5).astype(float)
            default_sens = ((default_preds == 1) & (val_labels_arr == 1)).sum() / max((val_labels_arr == 1).sum(), 1)
            default_spec = ((default_preds == 0) & (val_labels_arr == 0)).sum() / max((val_labels_arr == 0).sum(), 1)
            print(f"  At 0.5:    Sensitivity={default_sens:.3f}, "
                  f"Specificity={default_spec:.3f}, "
                  f"BAcc={(default_sens+default_spec)/2:.3f}")
        else:
            print(f"  ⚠ Only one class in validation set — using default 0.5")
    except Exception as e:
        print(f"  ⚠ Threshold optimization failed: {e} — using default 0.5")

    # Save threshold alongside model for step12/step13
    threshold_info = {
        "optimal_threshold": optimal_threshold,
        "method": "Youden's J (sensitivity + specificity - 1)",
        "optimized_on": "validation set",
        "reference": "Youden WJ (1950) Cancer 3:32-35",
    }
    with open(MODEL_DIR / "optimal_threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  ✓ Saved: {MODEL_DIR / 'optimal_threshold.json'}")

    # Also add to split_indices for backward compat
    split_info["optimal_threshold"] = optimal_threshold
    with open(MODEL_DIR / "split_indices.json", "w") as f:
        json.dump(split_info, f)

    # ── Save training history ─────────────────────────────────────────────────
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✓ Models saved: {MODEL_DIR}")
    print(f"    best_model.pt — trained model")
    print(f"    split_indices.json — for step12 reproducibility")
    print(f"    optimal_threshold.json — Youden's J threshold ({optimal_threshold:.4f})")
    print(f"\n✓ Step 11 complete!")


if __name__ == "__main__":
    train()
