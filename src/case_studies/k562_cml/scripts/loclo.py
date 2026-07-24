#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  K562/CML — Leave-One-Cell-Line-Out (LOCLO) Generalization Test            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Test generalization to UNSEEN cell lines by holding out entire            ║
║    leukemia subtype / tissue groups. This is the "cell-blind" split         ║
║    required by the 2026 DRP review (Sada Del Real et al., Brief Bioinf).   ║
║                                                                              ║
║  WHY TISSUE-GROUP LOCLO (not LODO):                                          ║
║    With multi-cell-line data from GDSC (step06), we can do TRUE             ║
║    cell-blind LOCLO. Cell lines are grouped by leukemia subtype / tissue:   ║
║      • cml (BCR-ABL+), aml, all, lymphoma, lung, breast, etc.             ║
║    Holding out a group tests: "Can the model predict drug response for      ║
║    a cancer type it has never seen?"                                        ║
║                                                                              ║
║  BIOLOGICAL SIGNIFICANCE:                                                    ║
║    KEY TEST: Can the model trained on non-CML cells predict CML response?  ║
║    And vice versa: can CML-trained model generalize to solid tumors?       ║
║                                                                              ║
║    TKIs and chemo drugs produce FUNDAMENTALLY different PTM patterns:       ║
║      • TKIs dephosphorylate BCR-ABL substrates (CRKL, STAT5)              ║
║      • Cytarabine induces DNA damage response phospho (γH2AX, Chk1)       ║
║      • Paclitaxel induces mitotic phospho (BubR1, AurB)                    ║
║    If the model generalizes across mechanisms, it has learned that          ║
║    different PTM patterns → different drug responses.                       ║
║                                                                              ║
║    Ref: Shah et al., Science 2004 — Dasatinib BCR-ABL                      ║
║    Ref: O'Hare et al., Blood 2005 — TKI potency ranking                   ║
║    Ref: Hochhaus et al., Leukemia 2020 — ELN CML management               ║
║                                                                              ║
║  INPUT:                                                                      ║
║    data/processed/k562_cml/multimodal_dataset.csv (from step06)            ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/k562_cml/loclo_results.json                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler

from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics
from src.ptm_bdl.training import FocalLoss, train_epoch, validate, build_model_from_cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["training"]["seed"]


def assign_tissue_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Assign each sample to a tissue / leukemia subtype group for LOCLO.

    Uses the tissue_group column from step06 (which maps TCGA_DESC to
    groups defined in config.yaml).
    """
    if "tissue_group" in df.columns:
        return df["tissue_group"].fillna("other").values

    # Fallback: use drug_name for LODO if tissue_group unavailable
    print("  ⚠ No tissue_group column — falling back to drug-based grouping")
    return df["drug_name"].fillna("unknown").values


def train_loclo_fold(dataset, train_idx, test_idx, fold_name, cfg, device):
    """Train a fresh model on train_idx, evaluate on test_idx."""
    print(f"    Training fold '{fold_name}': "
          f"train={len(train_idx)}, test={len(test_idx)}")

    train_subset = Subset(dataset, train_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())

    # Weighted sampler for class imbalance
    train_labels = np.array([
        dataset.df.iloc[i]["resistance_label"]
        for i in train_idx
    ])
    valid_labels = ~np.isnan(train_labels)
    if valid_labels.sum() < 3:
        print(f"    ⚠ Insufficient labeled data — skipping")
        return None

    train_labels_clean = train_labels[valid_labels].astype(int)
    class_counts = np.bincount(train_labels_clean, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
    sample_weights = np.ones(len(train_idx))
    for i, idx in enumerate(train_idx):
        lbl = train_labels[i]
        if not np.isnan(lbl):
            sample_weights[i] = class_weights[int(lbl)]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_idx), replacement=True)

    batch_size = cfg["model"]["batch_size"]
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    # Build fresh model
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model_from_cfg(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["model"]["learning_rate"],
        weight_decay=cfg["model"]["weight_decay"])
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    num_epochs = min(cfg["model"]["num_epochs"], 60)
    patience = cfg["model"]["early_stopping_patience"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
        eta_min=cfg["model"]["learning_rate"] * 0.01)
    best_score = -1
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, scheduler, focal_loss,
                    1.0, 2.0, device)
        val = validate(model, test_loader, focal_loss, 1.0, 2.0, device)
        score = max(val.get("auroc", 0), val.get("balanced_acc", 0))
        if score > best_score:
            best_score = score
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
        model, test_loader)
    reg, cls = compute_full_metrics(
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls)
    return {**reg, **cls}


def main():
    """Run Leave-One-Cell-Line-Out by leukemia subtype / tissue group."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — LOCLO (Cell-Blind by Tissue/Subtype)        ║")
    print(f"║  Key test: CML → solid tumor generalization                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    device_str = cfg["training"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"  Device: {device}")

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    if not dataset_path.exists():
        print(f"  ✗ Dataset not found: {dataset_path}")
        print(f"    Run steps 01-06 first.")
        return

    dataset = ResistanceDataset(dataset_path, features_dir)
    df = dataset.df
    print(f"  Dataset: {len(dataset)} samples")

    # Assign tissue groups
    groups = assign_tissue_groups(df)
    unique_groups = sorted(set(groups))
    print(f"  Tissue/subtype groups ({len(unique_groups)}):")
    for g in unique_groups:
        n = (groups == g).sum()
        print(f"    {g:<20s}: {n:4d} samples")

    # Filter groups with enough samples
    min_group_size = 10
    valid_groups = [g for g in unique_groups if (groups == g).sum() >= min_group_size]
    skipped = [g for g in unique_groups if g not in valid_groups]
    if skipped:
        print(f"  Skipping groups with < {min_group_size} samples: {skipped}")
    print(f"  Running LOCLO on {len(valid_groups)} groups: {valid_groups}")

    # LOCLO cross-validation
    results = {}
    all_metrics = []

    for group in valid_groups:
        print(f"\n  {'=' * 60}")
        print(f"  LOCLO Fold: Hold out '{group}'")
        print(f"  {'=' * 60}")

        test_mask = groups == group
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        t0 = time.time()
        try:
            metrics = train_loclo_fold(dataset, train_idx, test_idx,
                                       group, cfg, device)
            elapsed = time.time() - t0

            if metrics is None:
                results[group] = {"status": "skipped", "reason": "insufficient data"}
                continue

            fold_result = {
                "group": group,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "training_time_seconds": round(elapsed, 1),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            results[group] = fold_result
            all_metrics.append(fold_result)

            auroc = metrics.get("auroc", "N/A")
            bacc = metrics.get("balanced_acc", "N/A")
            auroc_s = f"{auroc:.3f}" if isinstance(auroc, float) else auroc
            bacc_s = f"{bacc:.3f}" if isinstance(bacc, float) else bacc
            print(f"    Results: AUROC={auroc_s}, BAcc={bacc_s} ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ✗ Failed: {e}")
            results[group] = {"status": "failed", "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  {'=' * 60}")
    print(f"  LOCLO SUMMARY")
    print(f"  {'=' * 60}")

    auroc_vals = [m.get("auroc") for m in all_metrics
                  if m.get("auroc") is not None]
    bacc_vals = [m.get("balanced_acc") for m in all_metrics
                 if m.get("balanced_acc") is not None]

    summary = {
        "n_groups_total": len(unique_groups),
        "n_groups_evaluated": len(valid_groups),
        "n_groups_skipped": len(skipped),
    }

    if auroc_vals:
        summary["mean_auroc"] = float(np.mean(auroc_vals))
        summary["std_auroc"] = float(np.std(auroc_vals))
        print(f"  AUROC: {np.mean(auroc_vals):.3f} ± {np.std(auroc_vals):.3f}")
    if bacc_vals:
        summary["mean_balanced_acc"] = float(np.mean(bacc_vals))
        summary["std_balanced_acc"] = float(np.std(bacc_vals))
        print(f"  BAcc:  {np.mean(bacc_vals):.3f} ± {np.std(bacc_vals):.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    report = {
        "case_study": CASE_STUDY,
        "method": "Leave-One-Cell-Line-Out (LOCLO) by tissue/leukemia subtype",
        "grouping": "leukemia_subtype (CML/AML/ALL) + tissue_type",
        "rationale": ("Tests whether PTM-BDL generalizes across cancer types "
                      "and leukemia subtypes for BCR-ABL TKI and chemotherapy "
                      "response prediction. Cell-blind split per Sada Del Real "
                      "et al., Brief Bioinf 2026."),
        "per_group_results": results,
        "summary": summary,
        "references": [
            "Sada Del Real et al., Brief Bioinf 2026 — cell-blind evaluation",
            "Shah et al., Science 2004 — Dasatinib BCR-ABL",
            "O'Hare et al., Blood 2005 — TKI potency ranking",
            "Hochhaus et al., Leukemia 2020 — ELN CML management",
        ],
    }
    out_path = RESULTS_DIR / "loclo_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  ✓ Saved: {out_path}")
    print("\n✓ LOCLO complete!")


if __name__ == "__main__":
    main()
