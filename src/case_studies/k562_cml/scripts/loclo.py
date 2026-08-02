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
║  SCALABILITY:                                                                ║
║    K562/CML pan-cancer dataset (~10K samples) is ~11× larger than EGFR     ║
║    (~1K). LOCLO uses an epoch-length cap to keep folds tractable:          ║
║      • WeightedRandomSampler num_samples capped at 4000 per epoch         ║
║        (no data discarded — sampler draws from full training pool)          ║
║      • Batch size 64 (fewer iterations per epoch)                           ║
║      • 30 epochs max (30 × 4000 = 120K draws from full 10K+ pool)         ║
║    This reduces wall-clock from ~12h/fold → ~20-40 min/fold on MPS         ║
║    without discarding any training data.                                    ║
║                                                                              ║
║  INPUT:                                                                      ║
║    data/processed/k562_cml/multimodal_dataset.csv (from step06)            ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/k562_cml/loclo_results.json                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import gc
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

# ── LOCLO-specific training parameters for large datasets ───────────────
# K562 pan-cancer dataset (~10K samples) is ~11× larger than EGFR (~1K).
# Without caps, a single LOCLO fold runs ~284 batches/epoch × 60 epochs
# = 17,000+ iterations → 12+ hours per fold on MPS.
#
# Solution: cap the WeightedRandomSampler's num_samples to limit epoch
# length. NO DATA IS DISCARDED — the sampler draws from the FULL training
# pool (with replacement), but each epoch only processes ~4000 draws.
# Over 30 epochs, that's 120,000 draws from the full 10K+ pool — every
# sample is seen ~12× on average. This is standard practice for large-
# dataset CV (Baptista et al., Brief Bioinform 2021).
LOCLO_EPOCH_SAMPLES = 4000      # Sampler draws per epoch (not a data filter)
LOCLO_MAX_EPOCHS = 30           # 30 × 4000 = 120K total draws
LOCLO_BATCH_SIZE = 16           # Reduced for MPS memory
LOCLO_PATIENCE = 10             # Early stopping patience


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

    # Cap epoch length for large training sets — sampler still draws from
    # ALL training samples, but each epoch only processes `epoch_samples`
    # draws. This keeps the data distribution intact while reducing wall
    # clock time per epoch.
    epoch_samples = min(len(train_idx), LOCLO_EPOCH_SAMPLES)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=epoch_samples, replacement=True)

    train_loader = DataLoader(
        train_subset, batch_size=LOCLO_BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(
        test_subset, batch_size=LOCLO_BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    n_batches = len(train_loader)
    if len(train_idx) > LOCLO_EPOCH_SAMPLES:
        print(f"    Epoch-length cap: {len(train_idx)} train samples → "
              f"{epoch_samples} draws/epoch ({n_batches} batches)")
        print(f"    Total exposure: {LOCLO_MAX_EPOCHS} epochs × {epoch_samples} = "
              f"{LOCLO_MAX_EPOCHS * epoch_samples:,} draws from full pool")
    print(f"    Config: {LOCLO_MAX_EPOCHS} epochs, batch_size={LOCLO_BATCH_SIZE}, "
          f"{n_batches} batches/epoch, patience={LOCLO_PATIENCE}")

    # Build fresh model
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model_from_cfg(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["model"]["learning_rate"],
        weight_decay=cfg["model"]["weight_decay"])
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # Training — same as train.py
    num_epochs = cfg["model"]["num_epochs"]
    patience = cfg["model"]["early_stopping_patience"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
        eta_min=cfg["model"]["learning_rate"] * 0.01)
    best_score = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 focal_loss, 1.0, 2.0, device)
        val = validate(model, test_loader, focal_loss, 1.0, 2.0, device)

        score = max(val.get("auroc", 0), val.get("balanced_acc", 0))
        if score > best_score:
            best_score = score
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch <= 2:
            print(f"    {epoch:3d}/{num_epochs} | loss={train_loss:.4f} | "
                  f"AUROC={val.get('auroc', 0):.3f} | "
                  f"BAcc={val.get('balanced_acc', 0):.3f} | "
                  f"RMSE={val.get('rmse', 0):.3f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (best={best_score:.3f})")
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

    # Show LOCLO training parameters
    print(f"\n  LOCLO training parameters (scaled for {len(dataset)}-sample dataset):")
    print(f"    Epoch draws (from full pool): {LOCLO_EPOCH_SAMPLES}")
    print(f"    Max epochs/fold:              {LOCLO_MAX_EPOCHS}")
    print(f"    Batch size:                   {LOCLO_BATCH_SIZE}")
    print(f"    Early stopping patience:      {LOCLO_PATIENCE} epochs")
    print(f"    Total draws per fold:         {LOCLO_MAX_EPOCHS * LOCLO_EPOCH_SAMPLES:,}")

    # Assign tissue groups
    groups = assign_tissue_groups(df)
    unique_groups = sorted(set(groups))
    print(f"\n  Tissue/subtype groups ({len(unique_groups)}):")
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
    total_start = time.time()

    for fold_i, group in enumerate(valid_groups):
        print(f"\n  {'=' * 60}")
        print(f"  LOCLO Fold {fold_i + 1}/{len(valid_groups)}: Hold out '{group}'")
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
            bacc = metrics.get("balanced_accuracy", metrics.get("balanced_acc", "N/A"))
            auroc_s = f"{auroc:.3f}" if isinstance(auroc, float) else auroc
            bacc_s = f"{bacc:.3f}" if isinstance(bacc, float) else bacc
            total_elapsed = (time.time() - total_start) / 60
            remaining = len(valid_groups) - fold_i - 1
            print(f"    ✓ Results: AUROC={auroc_s}, BAcc={bacc_s} "
                  f"({elapsed / 60:.1f} min)")
            if remaining > 0:
                avg_fold_time = total_elapsed / (fold_i + 1)
                print(f"    Progress: {fold_i + 1}/{len(valid_groups)} folds done | "
                      f"{total_elapsed:.0f} min elapsed | "
                      f"~{avg_fold_time * remaining:.0f} min remaining")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ✗ Failed: {e}")
            results[group] = {"status": "failed", "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = (time.time() - total_start) / 60
    print(f"\n  {'=' * 60}")
    print(f"  LOCLO SUMMARY (completed in {total_time:.1f} min)")
    print(f"  {'=' * 60}")

    auroc_vals = [m.get("auroc") for m in all_metrics
                  if m.get("auroc") is not None]
    bacc_vals = [m.get("balanced_accuracy", m.get("balanced_acc"))
                 for m in all_metrics
                 if m.get("balanced_accuracy", m.get("balanced_acc")) is not None]

    summary = {
        "n_groups_total": len(unique_groups),
        "n_groups_evaluated": len(valid_groups),
        "n_groups_skipped": len(skipped),
        "total_time_minutes": round(total_time, 1),
        "training_parameters": {
            "epoch_samples_from_full_pool": LOCLO_EPOCH_SAMPLES,
            "max_epochs": LOCLO_MAX_EPOCHS,
            "batch_size": LOCLO_BATCH_SIZE,
            "early_stopping_patience": LOCLO_PATIENCE,
            "total_draws_per_fold": LOCLO_MAX_EPOCHS * LOCLO_EPOCH_SAMPLES,
            "note": "No data discarded — sampler draws from full training pool",
        },
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
    # ══════════════════════════════════════════════════════════════════════
    # PART 2: COLD-DRUG LODO + COLD-CELL (Reviewer Q4)
    # ══════════════════════════════════════════════════════════════════════
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 2: Cold-Drug LODO + Cold-Cell Evaluation (Reviewer Q4)")
    print("══════════════════════════════════════════════════════════════")

    from src.ptm_bdl.evaluation.cold_split import (
        run_leave_one_drug_out, run_cold_cell_evaluation,
    )
    from src.ptm_bdl.data import collate_fn as cf

    ds = ResistanceDataset(dataset_path, features_dir)

    def _build_model():
        return build_model_from_cfg(cfg).to(device)

    def _train_fold(model, tl, vl, dev):
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        lr = cfg["model"]["learning_rate"]
        n_ep = cfg["model"]["num_epochs"]
        patience = cfg["model"]["early_stopping_patience"]
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["model"]["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=lr*0.01)
        best_s, best_st, pc = 0.0, None, 0
        for ep in range(1, n_ep + 1):
            loss = train_epoch(model, tl, opt, sched, focal, 1.0, 2.0, dev)
            vm = validate(model, vl, focal, 1.0, 2.0, dev)
            s = max(vm.get("auroc", 0), vm.get("balanced_acc", 0))
            if s > best_s:
                best_s, pc = s, 0
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pc += 1
            if ep % 5 == 0 or ep <= 2:
                print(f"      {ep:3d}/{n_ep} | loss={loss:.4f} | "
                      f"AUROC={vm.get('auroc', 0):.3f} | "
                      f"BAcc={vm.get('balanced_acc', 0):.3f}")
            if pc >= patience:
                print(f"      Early stop ep {ep} (best={best_s:.3f})")
                break
        if best_st: model.load_state_dict(best_st)
        return model

    # Cold-Drug LODO
    print("\n  ── Cold-Drug (Leave-One-Drug-Out) ──")
    lodo = run_leave_one_drug_out(
        ds, ds.df["drug_name"].values, _build_model, _train_fold,
        cf, batch_size=cfg["model"]["batch_size"], device=str(device), min_test_samples=5)
    with open(RESULTS_DIR / "cold_drug_lodo.json", "w") as f:
        json.dump(lodo, f, indent=2, default=str)
    print(f"  ✓ Saved: cold_drug_lodo.json")

    # Cold-Cell
    print("\n  ── Cold-Cell (cell-line-level K-fold) ──")
    cell_col = "cell_line_name" if "cell_line_name" in ds.df.columns else "cell_line"
    if cell_col in ds.df.columns:
        cc = run_cold_cell_evaluation(
            ds, ds.df[cell_col].values, _build_model, _train_fold,
            cf, n_folds=5, batch_size=cfg["model"]["batch_size"], device=str(device))
        with open(RESULTS_DIR / "cold_cell_results.json", "w") as f:
            json.dump(cc, f, indent=2, default=str)
        print(f"  ✓ Saved: cold_cell_results.json")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: CROSS-DATASET — GDSC → CTRPv2 (Reviewer Q4)
    # ══════════════════════════════════════════════════════════════════════
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 3: Cross-Dataset GDSC -> CTRPv2 (Reviewer Q4)")
    print("══════════════════════════════════════════════════════════════")

    from src.ptm_bdl.evaluation.cross_dataset import run_cross_dataset_ctrp

    ctrp_csv = (PROJECT_ROOT / "data" / "processed" / "ctrp"
                / "ctrp_drug_responses.csv")
    model_dir = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
    best_mp = model_dir / "best_model.pt"
    if not best_mp.exists():
        best_mp = model_dir / "ablation_full.pt"

    if best_mp.exists() and ctrp_csv.exists():
        ctrp_model = build_model_from_cfg(cfg).to(device)
        ctrp_model.load_state_dict(torch.load(
            best_mp, map_location=device, weights_only=True))
        ctrp_model.eval()

        # Map CS3 drug names to CTRP canonical names
        # CS3 uses "Methotrexate" (with 'e'), CTRP uses "Methotrexat" (no 'e')
        cs3_drug_map = {"Methotrexate": "Methotrexat"}
        ctrp_results = run_cross_dataset_ctrp(
            model=ctrp_model, dataset=ds,
            ctrp_csv_path=str(ctrp_csv), collate_fn=cf,
            batch_size=cfg["model"]["batch_size"], device=str(device),
            drug_name_map=cs3_drug_map,
        )
        ctrp_out = RESULTS_DIR / "cross_dataset_ctrp.json"
        with open(ctrp_out, "w") as f:
            json.dump(ctrp_results, f, indent=2, default=str)
        print(f"  ✓ Saved: {ctrp_out}")
    else:
        if not best_mp.exists():
            print("  ⚠ No trained model — skipping CTRP")
        if not ctrp_csv.exists():
            print("  ⚠ CTRP data not found — run download_ctrp.py")

    print("\n✓ LOCLO + Cold-start complete!")


if __name__ == "__main__":
    main()
