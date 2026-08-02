#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 14d — Leave-One-Cell-Line-Out (LOCLO) Generalization Test             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Test generalization to UNSEEN cell lines by holding out entire cell       ║
║    line groups (grouped by mutation class). This is the "cell-blind" split   ║
║    required by the 2026 DRP review (Sada Del Real et al., Brief Bioinf):    ║
║                                                                              ║
║      "cell-blind... is particularly valuable for drug repositioning and,    ║
║       more importantly, for precision medicine"                              ║
║                                                                              ║
║      "models must be assessed under at least one additional cross-          ║
║       validation regime [beyond random splitting]"                           ║
║                                                                              ║
║  WHY MUTATION-CLASS GROUPING (not individual cell lines):                    ║
║    With only ~15–30 cell lines per mutation class, holding out a SINGLE     ║
║    cell line (true LOCLO) gives very noisy results because each cell line   ║
║    has only 4–6 drug measurements. Instead, we group cell lines by EGFR     ║
║    mutation class (WT, L858R, exon19del, T790M, L858R/T790M) and hold      ║
║    out each group. This tests: "Can the model predict drug response for     ║
║    a mutation class it has never seen?"                                      ║
║                                                                              ║
║  MUTATION CLASSES:                                                           ║
║    • wild_type     — No EGFR activating mutation (largest group)            ║
║    • L858R         — Single activating point mutation                        ║
║    • exon19del     — Exon 19 deletion (ELREA)                               ║
║    • T790M         — Gatekeeper mutation (alone)                             ║
║    • L858R_T790M   — Double mutant (acquired resistance)                    ║
║    • other         — Rare/compound mutations                                 ║
║    • HER2_WT       — ERBB2 wild-type (breast cancer cell lines)            ║
║    • HER2_amp      — ERBB2 amplified                                        ║
║                                                                              ║
║  APPROACH:                                                                   ║
║    For each mutation class group:                                            ║
║      1. Hold out ALL samples from that mutation class                       ║
║      2. Train on remaining samples (using our PTM-BDL model)               ║
║      3. Predict held-out samples                                             ║
║      4. Compute Tier A metrics (PCC, RMSE, AUROC, AUPRC-sens)             ║
║      5. Compare to random-split baseline                                    ║
║                                                                              ║
║  METRICS:                                                                    ║
║    Same Tier A metrics as step14a/14b for consistency:                       ║
║    • PCC (Pearson R), RMSE — regression                                     ║
║    • AUROC, AUPRC-sensitive — classification                                ║
║    • Per-drug PCC where group size allows                                   ║
║                                                                              ║
║  INPUT:                                                                      ║
║    data/processed/multimodal_dataset.csv                                     ║
║    data/features/* (ESM-2, GearNet, ChemBERTa embeddings)                   ║
║    config/config.yaml                                                        ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/loclo_results.json                                                ║
║                                                                              ║
║  BENCHMARKING_PLAN.md §5.2, §6 Axis 4, §8 Step 14d                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics
from src.ptm_bdl.training import FocalLoss, train_epoch, validate, build_model_from_cfg

from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["training"]["seed"]


def _clear_mps_cache():
    """Free MPS cached memory to prevent accumulation between folds."""
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Mutation class grouping
# ══════════════════════════════════════════════════════════════════════════════

def assign_mutation_groups(df):
    """
    Assign each sample to a mutation class group for LOCLO.

    Groups are based on the mutation_class column (from step06) and
    target_protein. This creates biologically meaningful hold-out groups.
    """
    groups = []
    for _, row in df.iterrows():
        protein = row.get("target_protein", "EGFR")
        mut_class = str(row.get("mutation_class", "wild_type")).lower()

        if protein == "ERBB2":
            # HER2 cell lines: group by amplification status
            cell_line = str(row.get("cell_line_name", "")).upper()
            her2_high = cfg.get("ptm_modulators", {}).get(
                "her2_amp_tiers", {}).get("high", [])
            her2_high_upper = [c.upper() for c in her2_high]
            if cell_line in her2_high_upper:
                groups.append("HER2_amplified")
            else:
                groups.append("HER2_other")
        elif "l858r" in mut_class and "t790m" in mut_class:
            groups.append("L858R_T790M")
        elif "c797s" in mut_class:
            groups.append("C797S_triple")
        elif "t790m" in mut_class:
            groups.append("T790M")
        elif "l858r" in mut_class:
            groups.append("L858R")
        elif "exon19" in mut_class or "del" in mut_class:
            groups.append("exon19del")
        elif mut_class in ("wild_type", "wt", "none", "nan", ""):
            groups.append("wild_type")
        else:
            groups.append("other_EGFR")

    return np.array(groups)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop for a single LOCLO fold
# ══════════════════════════════════════════════════════════════════════════════

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
    lr = cfg["model"]["learning_rate"]
    wd = cfg["model"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # Training config from config.yaml
    num_epochs = cfg["model"]["num_epochs"]
    patience = cfg["model"]["early_stopping_patience"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    print(f"    Training: max {num_epochs} epochs, patience={patience}, lr={lr}")

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

        print(f"    {epoch:3d}/{num_epochs} | loss={train_loss:.4f} | "
              f"AUROC={val.get('auroc', 0):.3f} | "
              f"BAcc={val.get('balanced_acc', 0):.3f} | "
              f"RMSE={val.get('rmse', 0):.3f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs, best={best_score:.3f})")
            break

    # Load best and free training memory before inference
    if best_state:
        model.load_state_dict(best_state)
    del optimizer, scheduler, focal_loss, best_state
    _clear_mps_cache()

    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
        model, test_loader)
    reg, cls = compute_full_metrics(
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls)

    # Free model memory after collecting predictions
    del model
    _clear_mps_cache()

    return {**reg, **cls}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 14d: Leave-One-Cell-Line-Out (LOCLO) Generalization   ║")
    print("║  Cell-blind split by mutation class group                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    # ── Load dataset ───────────────────────────────────────────────────────
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    if not dataset_path.exists():
        print(f"  ✗ Dataset not found: {dataset_path}")
        print(f"    Run the pipeline (steps 01-06) first.")
        sys.exit(1)

    dataset = ResistanceDataset(
        dataset_csv=dataset_path,
        features_dir=features_dir,
        ablation_mode="full",
    )
    df = dataset.df
    print(f"  Dataset: {len(dataset)} samples")

    # ── Assign mutation groups ─────────────────────────────────────────────
    groups = assign_mutation_groups(df)
    unique_groups = sorted(set(groups))
    print(f"  Mutation class groups ({len(unique_groups)}):")
    for g in unique_groups:
        n = (groups == g).sum()
        n_sens = int((df["resistance_label"].values[groups == g] == 0).sum())
        n_res = int((df["resistance_label"].values[groups == g] == 1).sum())
        print(f"    {g:<20s}: {n:4d} samples "
              f"({n_res} resistant, {n_sens} sensitive)")

    # ── Filter groups with enough samples ──────────────────────────────────
    min_group_size = 10  # Need at least 10 samples to evaluate
    valid_groups = [g for g in unique_groups if (groups == g).sum() >= min_group_size]
    skipped = [g for g in unique_groups if g not in valid_groups]

    if skipped:
        print(f"\n  Skipping groups with < {min_group_size} samples: "
              f"{skipped}")

    print(f"  Running LOCLO on {len(valid_groups)} groups: {valid_groups}")

    # ── LOCLO cross-validation ─────────────────────────────────────────────
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

        # Check class balance in test set
        test_labels = df["resistance_label"].values[test_idx]
        n_cls = len(set(test_labels))
        if n_cls < 2:
            print(f"    ⚠ Only {n_cls} class in held-out group — "
                  f"classification metrics will be None")

        _clear_mps_cache()  # Free cached MPS memory before each fold

        t0 = time.time()
        try:
            metrics = train_loclo_fold(
                dataset, train_idx, test_idx, group, cfg, device)
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
            print(f"    Results: AUROC={auroc_s}, BAcc={bacc_s} ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ✗ Failed: {e}")
            results[group] = {
                "group": group,
                "status": "failed",
                "error": str(e),
                "training_time_seconds": round(elapsed, 1),
            }

    # ── Aggregate results ──────────────────────────────────────────────────
    print(f"\n  {'=' * 60}")
    print(f"  LOCLO SUMMARY")
    print(f"  {'=' * 60}")

    # Summary statistics across folds
    pcc_vals = [m.get("pearson_r") for m in all_metrics
                if m.get("pearson_r") is not None]
    rmse_vals = [m.get("rmse") for m in all_metrics
                 if m.get("rmse") is not None]
    auroc_vals = [m.get("auroc") for m in all_metrics
                  if m.get("auroc") is not None]
    bacc_vals = [m.get("balanced_accuracy", m.get("balanced_acc"))
                 for m in all_metrics
                 if m.get("balanced_accuracy", m.get("balanced_acc")) is not None]

    summary = {
        "n_groups_total": len(unique_groups),
        "n_groups_evaluated": len(valid_groups),
        "n_groups_skipped": len(skipped),
        "skipped_groups": skipped,
    }

    if pcc_vals:
        summary["mean_pearson_r"] = float(np.mean(pcc_vals))
        summary["std_pearson_r"] = float(np.std(pcc_vals))
        summary["min_pearson_r"] = float(np.min(pcc_vals))
        summary["max_pearson_r"] = float(np.max(pcc_vals))
        print(f"  Pearson R: {np.mean(pcc_vals):.3f} ± {np.std(pcc_vals):.3f} "
              f"(range: {np.min(pcc_vals):.3f}–{np.max(pcc_vals):.3f})")

    if rmse_vals:
        summary["mean_rmse"] = float(np.mean(rmse_vals))
        summary["std_rmse"] = float(np.std(rmse_vals))
        print(f"  RMSE:      {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}")

    if auroc_vals:
        summary["mean_auroc"] = float(np.mean(auroc_vals))
        summary["std_auroc"] = float(np.std(auroc_vals))
        summary["n_folds_with_auroc"] = len(auroc_vals)
        print(f"  AUROC:     {np.mean(auroc_vals):.3f} ± {np.std(auroc_vals):.3f} "
              f"({len(auroc_vals)}/{len(valid_groups)} folds)")

    if bacc_vals:
        summary["mean_balanced_acc"] = float(np.mean(bacc_vals))
        summary["std_balanced_acc"] = float(np.std(bacc_vals))
        print(f"  BAcc:      {np.mean(bacc_vals):.3f} ± {np.std(bacc_vals):.3f}")

    # ── Compare to random split ────────────────────────────────────────────
    eval_path = RESULTS_DIR / "evaluation_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        random_pcc = eval_report.get("regression", {}).get("pearson_r", 0)
        random_auroc = eval_report.get("classification", {}).get("auroc", 0)
        summary["random_split_pearson_r"] = random_pcc
        summary["random_split_auroc"] = random_auroc

        if pcc_vals:
            summary["generalization_gap_pcc"] = float(
                random_pcc - np.mean(pcc_vals))
        if auroc_vals:
            summary["generalization_gap_auroc"] = float(
                random_auroc - np.mean(auroc_vals))

        print(f"\n  Generalization gap (random split → cell-blind LOCLO):")
        if pcc_vals:
            gap_pcc = random_pcc - np.mean(pcc_vals)
            print(f"    PCC:   {random_pcc:.3f} → {np.mean(pcc_vals):.3f} "
                  f"(Δ = {gap_pcc:+.3f})")
        if auroc_vals:
            gap_auroc = random_auroc - np.mean(auroc_vals)
            print(f"    AUROC: {random_auroc:.3f} → {np.mean(auroc_vals):.3f} "
                  f"(Δ = {gap_auroc:+.3f})")

    # ── Save ───────────────────────────────────────────────────────────────
    output = {
        "method": "Leave-One-Cell-Line-Out (LOCLO) by mutation class",
        "reference": "Sada Del Real et al., Brief Bioinf 2026 — "
                     "'cell-blind split for precision medicine'",
        "grouping": "mutation_class (EGFR) / amplification_status (HER2)",
        "min_group_size": min_group_size,
        "per_group_results": results,
        "summary": summary,
        "interpretation": {
            "purpose": ("Tests whether PTM-BDL generalizes to mutation "
                        "classes not seen during training. A large "
                        "generalization gap (>0.1 PCC) would indicate the "
                        "model memorizes mutation-specific patterns rather "
                        "than learning generalizable PTM→resistance mapping."),
            "expected_finding": ("Some performance drop is expected and "
                                 "acceptable. The key question is whether "
                                 "PTM-BDL drops less than ML baselines, "
                                 "suggesting PTM features provide "
                                 "transferable biological signal."),
        },
    }

    out_path = RESULTS_DIR / "loclo_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  ✓ Saved: {out_path}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: COLD-DRUG LODO + COLD-CELL (Reviewer Q4)
    # ══════════════════════════════════════════════════════════════════════
    # Uses cold_split.py which retrains from scratch for each fold.
    # Ref: Sada Del Real et al., Brief Bioinf 2026 — "cold-drug and
    #      cold-cell splits are essential for assessing clinical translatability"
    # ══════════════════════════════════════════════════════════════════════
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 2: Cold-Drug LODO + Cold-Cell Evaluation (Reviewer Q4)")
    print("══════════════════════════════════════════════════════════════")

    from src.ptm_bdl.evaluation.cold_split import (
        run_leave_one_drug_out, run_cold_cell_evaluation,
    )

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
        print(f"      Training: max {n_ep} epochs, patience={patience}, lr={lr}")
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
            print(f"      {ep:3d}/{n_ep} | loss={loss:.4f} | "
                  f"AUROC={vm.get('auroc', 0):.3f} | "
                  f"BAcc={vm.get('balanced_acc', 0):.3f} | "
                  f"RMSE={vm.get('rmse', 0):.3f}")
            if pc >= patience:
                print(f"      Early stopping at epoch {ep} "
                      f"(no improvement for {patience} epochs, best={best_s:.3f})")
                break
        if best_st: model.load_state_dict(best_st)
        return model

    # ── Cold-Drug LODO ──────────────────────────────────────────────────
    print("\n  ── Cold-Drug (Leave-One-Drug-Out) ──")
    drug_labels = ds.df["drug_name"].values
    lodo_results = run_leave_one_drug_out(
        ds, drug_labels, _build_model, _train_fold,
        collate_fn, batch_size=cfg["model"]["batch_size"],
        device=str(device), min_test_samples=5,
    )
    lodo_path = RESULTS_DIR / "cold_drug_lodo.json"
    with open(lodo_path, "w") as f:
        json.dump(lodo_results, f, indent=2, default=str)
    print(f"  ✓ Saved: {lodo_path}")

    # ── Cold-Cell (cell-line-level split) ────────────────────────────────
    print("\n  ── Cold-Cell (cell-line-level K-fold) ──")
    cell_col = "cell_line_name" if "cell_line_name" in ds.df.columns else "cell_line"
    if cell_col in ds.df.columns:
        cell_labels = ds.df[cell_col].values
        cold_cell_results = run_cold_cell_evaluation(
            ds, cell_labels, _build_model, _train_fold,
            collate_fn, n_folds=5, batch_size=cfg["model"]["batch_size"],
            device=str(device), seed=SEED,
        )
        cold_cell_path = RESULTS_DIR / "cold_cell_results.json"
        with open(cold_cell_path, "w") as f:
            json.dump(cold_cell_results, f, indent=2, default=str)
        print(f"  ✓ Saved: {cold_cell_path}")
    else:
        print("  ⚠ No cell_line column found — skipping cold-cell")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: CROSS-DATASET — GDSC → CTRPv2 (Reviewer Q4)
    # ══════════════════════════════════════════════════════════════════════
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 3: Cross-Dataset GDSC -> CTRPv2 (Reviewer Q4)")
    print("══════════════════════════════════════════════════════════════")

    from src.ptm_bdl.evaluation.cross_dataset import run_cross_dataset_ctrp

    ctrp_csv = (PROJECT_ROOT / "data" / "processed" / "ctrp"
                / "ctrp_drug_responses.csv")
    best_model_path = MODEL_DIR / "best_model.pt"
    if not best_model_path.exists():
        best_model_path = MODEL_DIR / "ablation_full.pt"

    if best_model_path.exists() and ctrp_csv.exists():
        ctrp_model = build_model_from_cfg(cfg).to(device)
        ctrp_model.load_state_dict(torch.load(
            best_model_path, map_location=device, weights_only=True))
        ctrp_model.eval()

        ctrp_results = run_cross_dataset_ctrp(
            model=ctrp_model, dataset=ds,
            ctrp_csv_path=str(ctrp_csv), collate_fn=collate_fn,
            batch_size=cfg["model"]["batch_size"], device=str(device),
        )
        ctrp_out = RESULTS_DIR / "cross_dataset_ctrp.json"
        with open(ctrp_out, "w") as f:
            json.dump(ctrp_results, f, indent=2, default=str)
        print(f"  ✓ Saved: {ctrp_out}")
    else:
        if not best_model_path.exists():
            print("  ⚠ No trained model — skipping CTRP")
        if not ctrp_csv.exists():
            print("  ⚠ CTRP data not found — run download_ctrp.py")

    print("\n✓ Step 14d complete!")


if __name__ == "__main__":
    main()
