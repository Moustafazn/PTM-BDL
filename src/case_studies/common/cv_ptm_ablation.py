#!/usr/bin/env python3
"""
Cross-Validation PTM Ablation — Generic script for any case study.

Runs stratified K-fold CV training FULL vs NO_PTM models per fold,
computes paired ΔAUROC and ΔBAcc with paired t-test.

This validates whether the PTM branch contribution is robust across
data splits, complementing the single-split ablation in each case study.

Usage:
    python -m src.case_studies.common.cv_ptm_ablation --case k562_cml
    python -m src.case_studies.common.cv_ptm_ablation --case hela_hdac
    python -m src.case_studies.common.cv_ptm_ablation --case egfr_erbb2_tki
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from src.ptm_bdl.config import load_config
from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.training import (
    FocalLoss, train_epoch, validate, build_model_from_cfg,
    resolve_device,
)


def _train_one_fold(cfg, dataset_path, features_dir, train_idx, val_idx,
                    device, ablation_mode="full", seed=42, max_epochs=30):
    """Train one fold in one ablation mode. Returns model + epochs trained."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = ResistanceDataset(dataset_path, features_dir,
                           ablation_mode=ablation_mode)
    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)

    train_labels = ds.df["resistance_label"].values[train_idx].astype(int)
    cc = np.bincount(train_labels)
    weights = (1.0 / np.maximum(cc, 1)).astype(np.float32)
    sw = weights[train_labels]
    sampler = WeightedRandomSampler(torch.from_numpy(sw),
                                    num_samples=len(train_set),
                                    replacement=True)

    bs = cfg["model"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, sampler=sampler,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            collate_fn=collate_fn)

    model = build_model_from_cfg(cfg).to(device)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    lr = cfg["model"]["learning_rate"]
    wd = cfg["model"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01)

    best_score = 0.0
    patience = cfg["model"]["early_stopping_patience"]
    counter = 0
    best_state = None
    epoch_done = 0

    for epoch in range(1, max_epochs + 1):
        train_epoch(model, train_loader, optimizer, scheduler,
                    focal_loss, 1.0, 2.0, device)
        vm = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
        score = max(vm.get("auroc", 0), vm.get("balanced_acc", 0))

        if score > best_score:
            best_score = score
            counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                epoch_done = epoch
                break
        epoch_done = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epoch_done


def _evaluate_model(cfg, model, dataset, test_idx, device):
    """Evaluate a trained model on the test fold."""
    test_set = Subset(dataset, test_idx)
    loader = DataLoader(test_set, batch_size=cfg["model"]["batch_size"],
                        shuffle=False, collate_fn=collate_fn)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    return validate(model, loader, focal_loss, 1.0, 2.0, device)


def run_cv_ablation(case_study: str, n_folds: int = 5, max_epochs: int = 30):
    """Run CV ablation: full vs no_ptm per fold with paired t-test."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  CV PTM Ablation — {case_study:<39s} ║")
    print(f"║  {n_folds}-fold stratified, full vs no_ptm per fold              ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    cfg = load_config(case_study=case_study)
    device = resolve_device(cfg)
    seed = cfg["training"]["seed"]
    print(f"  Device: {device}")

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / case_study / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    results_dir = PROJECT_ROOT / cfg["paths"]["results"] / case_study
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = ResistanceDataset(dataset_path, features_dir)
    df = dataset.df
    n_total = len(dataset)
    labels = df["resistance_label"].values.astype(int)
    print(f"  Dataset: {n_total} samples")

    if "target_protein" in df.columns:
        strat = np.array([f"{g}_{r}" for g, r in
                          zip(df["target_protein"].fillna("UNK").values, labels)])
    else:
        strat = labels

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    full_aurocs, full_baccs = [], []
    noptm_aurocs, noptm_baccs = [], []
    fold_details = []
    t0_total = time.time()

    # ── Check for partial results from a previous interrupted run ──
    partial_path = results_dir / "cv_ptm_ablation_partial.json"
    start_fold = 0
    if partial_path.exists():
        try:
            with open(partial_path) as f:
                partial = json.load(f)
            if partial.get("case_study") == case_study:
                fold_details = partial.get("fold_details", [])
                for fd in fold_details:
                    full_aurocs.append(fd["full_auroc"])
                    full_baccs.append(fd["full_bacc"])
                    noptm_aurocs.append(fd["noptm_auroc"])
                    noptm_baccs.append(fd["noptm_bacc"])
                start_fold = len(fold_details)
                print(f"  ✓ Resuming from fold {start_fold + 1} "
                      f"({start_fold} folds completed previously)")
        except (json.JSONDecodeError, KeyError):
            pass

    for fold_i, (trval_idx, test_idx) in enumerate(
            kfold.split(np.zeros(n_total), strat)):
        if fold_i < start_fold:
            continue  # Skip already-completed folds

        print(f"\n  ── Fold {fold_i + 1}/{n_folds} ──")

        trval_labels = labels[trval_idx]
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                        random_state=seed + fold_i)
            tr_sub, va_sub = next(sss.split(np.zeros(len(trval_idx)),
                                            trval_labels))
        except ValueError:
            perm = np.random.permutation(len(trval_idx))
            v = max(1, int(0.15 * len(trval_idx)))
            va_sub, tr_sub = perm[:v], perm[v:]

        train_idx = trval_idx[tr_sub]
        val_idx = trval_idx[va_sub]
        print(f"    Train: {len(train_idx)} | Val: {len(val_idx)} | "
              f"Test: {len(test_idx)}")

        try:
            # ── Full model ──
            print(f"    Training FULL model...")
            t0 = time.time()
            full_model, full_ep = _train_one_fold(
                cfg, dataset_path, features_dir, train_idx, val_idx,
                device, ablation_mode="full", seed=seed, max_epochs=max_epochs)
            full_m = _evaluate_model(cfg, full_model, dataset, test_idx, device)
            print(f"    Full: AUROC={full_m.get('auroc', 0):.3f}, "
                  f"BAcc={full_m['balanced_acc']:.3f} "
                  f"({time.time()-t0:.0f}s, {full_ep} ep)")

            # ── No PTM model ──
            print(f"    Training NO_PTM model...")
            t0 = time.time()
            noptm_model, noptm_ep = _train_one_fold(
                cfg, dataset_path, features_dir, train_idx, val_idx,
                device, ablation_mode="no_ptm", seed=seed, max_epochs=max_epochs)
            noptm_m = _evaluate_model(cfg, noptm_model, dataset, test_idx, device)
            print(f"    NoPTM: AUROC={noptm_m.get('auroc', 0):.3f}, "
                  f"BAcc={noptm_m['balanced_acc']:.3f} "
                  f"({time.time()-t0:.0f}s, {noptm_ep} ep)")

            d_auroc = full_m.get("auroc", 0) - noptm_m.get("auroc", 0)
            d_bacc = full_m["balanced_acc"] - noptm_m["balanced_acc"]
            print(f"    Δ AUROC={d_auroc:+.4f}, Δ BAcc={d_bacc:+.4f}")

            full_aurocs.append(full_m.get("auroc", 0))
            full_baccs.append(full_m["balanced_acc"])
            noptm_aurocs.append(noptm_m.get("auroc", 0))
            noptm_baccs.append(noptm_m["balanced_acc"])
            fold_details.append({
                "fold": fold_i + 1,
                "full_auroc": full_m.get("auroc", 0),
                "full_bacc": full_m["balanced_acc"],
                "noptm_auroc": noptm_m.get("auroc", 0),
                "noptm_bacc": noptm_m["balanced_acc"],
                "delta_auroc": d_auroc, "delta_bacc": d_bacc,
            })

            # ── Save partial results after EACH fold ──
            partial_save = {
                "case_study": case_study, "n_folds": n_folds,
                "max_epochs": max_epochs, "status": "in_progress",
                "folds_completed": len(fold_details),
                "fold_details": fold_details,
            }
            with open(partial_path, "w") as f:
                json.dump(partial_save, f, indent=2)
            print(f"    ✓ Partial results saved ({len(fold_details)}/{n_folds} folds)")

            # Free GPU memory
            del full_model, noptm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ✗ Fold {fold_i + 1} FAILED: {e}")
            print(f"    Saving partial results and continuing...")
            partial_save = {
                "case_study": case_study, "n_folds": n_folds,
                "max_epochs": max_epochs, "status": "partial_error",
                "folds_completed": len(fold_details),
                "error_fold": fold_i + 1, "error": str(e),
                "fold_details": fold_details,
            }
            with open(partial_path, "w") as f:
                json.dump(partial_save, f, indent=2)
            continue

    if len(fold_details) == 0:
        print("\n  ✗ No folds completed successfully. Cannot compute summary.")
        return

    # ── Aggregate ──
    total_time = time.time() - t0_total
    delta_aurocs = [f["delta_auroc"] for f in fold_details]
    delta_baccs = [f["delta_bacc"] for f in fold_details]
    t_auroc, p_auroc = scipy_stats.ttest_rel(full_aurocs, noptm_aurocs)
    t_bacc, p_bacc = scipy_stats.ttest_rel(full_baccs, noptm_baccs)

    summary = {
        "case_study": case_study, "n_folds": n_folds,
        "max_epochs": max_epochs,
        "total_time_minutes": round(total_time / 60, 1),
        "full_auroc": {"mean": round(float(np.mean(full_aurocs)), 4),
                       "std": round(float(np.std(full_aurocs)), 4)},
        "noptm_auroc": {"mean": round(float(np.mean(noptm_aurocs)), 4),
                        "std": round(float(np.std(noptm_aurocs)), 4)},
        "delta_auroc": {
            "mean": round(float(np.mean(delta_aurocs)), 4),
            "std": round(float(np.std(delta_aurocs)), 4),
            "t_statistic": round(float(t_auroc), 4),
            "p_value": round(float(p_auroc), 6),
            "n_positive_folds": sum(1 for d in delta_aurocs if d > 0),
        },
        "delta_bacc": {
            "mean": round(float(np.mean(delta_baccs)), 4),
            "std": round(float(np.std(delta_baccs)), 4),
            "t_statistic": round(float(t_bacc), 4),
            "p_value": round(float(p_bacc), 6),
            "n_positive_folds": sum(1 for d in delta_baccs if d > 0),
        },
        "fold_details": fold_details,
    }

    print(f"\n  {'='*60}")
    print(f"  CV PTM ABLATION SUMMARY — {case_study}")
    print(f"  {'='*60}")
    print(f"  Full AUROC:  {summary['full_auroc']['mean']:.4f} ± "
          f"{summary['full_auroc']['std']:.4f}")
    print(f"  NoPTM AUROC: {summary['noptm_auroc']['mean']:.4f} ± "
          f"{summary['noptm_auroc']['std']:.4f}")
    print(f"  Δ AUROC:     {summary['delta_auroc']['mean']:+.4f} ± "
          f"{summary['delta_auroc']['std']:.4f}  "
          f"(p={summary['delta_auroc']['p_value']:.6f})")
    print(f"  Δ BAcc:      {summary['delta_bacc']['mean']:+.4f} ± "
          f"{summary['delta_bacc']['std']:.4f}  "
          f"(p={summary['delta_bacc']['p_value']:.6f})")
    print(f"  Folds PTM helps AUROC: "
          f"{summary['delta_auroc']['n_positive_folds']}/{n_folds}")
    print(f"  Folds PTM helps BAcc:  "
          f"{summary['delta_bacc']['n_positive_folds']}/{n_folds}")
    print(f"  Total time: {summary['total_time_minutes']:.1f} min")

    out_path = results_dir / "cv_ptm_ablation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CV PTM Ablation — full vs no_ptm per fold")
    parser.add_argument("--case", required=True,
                        choices=["k562_cml", "hela_hdac", "egfr_erbb2_tki"],
                        help="Case study to run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()
    run_cv_ablation(args.case, args.folds, args.max_epochs)


if __name__ == "__main__":
    main()
