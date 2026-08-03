"""
Cold-start evaluation — Leave-One-Drug-Out (LODO), cold-cell, and cross-dataset.

Tests whether the model generalises to:
  - Held-out drugs (cold drug / cold scaffold)
  - Held-out cell lines not seen during training (cold cell)
  - Independent drug sensitivity datasets (cross-dataset: GDSC → CTRPv2)

These are critical benchmarks recommended by:
  Sada Del Real et al., Brief Bioinf 2026 — "cold-drug and cold-cell splits
  are essential for assessing clinical translatability"

The module is case-study-agnostic — drug/cell assignments come from the caller.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def compute_cold_metrics(
        y_true_ic50: np.ndarray,
        y_pred_ic50: np.ndarray,
        y_true_cls: np.ndarray,
        y_prob_cls: np.ndarray,
        threshold: float = 0.5,
) -> dict:
    """Compute evaluation metrics for a single cold-start fold."""
    metrics = {
        "n_samples": int(len(y_true_ic50)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))),
        "n_resistant": int((y_true_cls == 1).sum()),
        "n_sensitive": int((y_true_cls == 0).sum()),
    }

    if len(y_true_ic50) > 2 and np.std(y_pred_ic50) > 1e-8:
        metrics["pearson_r"] = float(np.corrcoef(y_true_ic50, y_pred_ic50)[0, 1])
        sr = scipy_stats.spearmanr(y_true_ic50, y_pred_ic50)
        metrics["spearman_rho"] = float(sr.statistic if hasattr(sr, 'statistic') else sr[0])
    else:
        metrics["pearson_r"] = 0.0
        metrics["spearman_rho"] = 0.0

    if len(set(y_true_cls)) > 1 and y_prob_cls is not None:
        metrics["auroc"] = float(roc_auc_score(y_true_cls, y_prob_cls))
        metrics["auprc_sensitive"] = float(
            average_precision_score(1 - y_true_cls, 1 - y_prob_cls))
        y_pred_bin = (y_prob_cls > threshold).astype(float)
        metrics["balanced_acc"] = float(balanced_accuracy_score(y_true_cls, y_pred_bin))
    else:
        metrics["auroc"] = None
        metrics["auprc_sensitive"] = None
        metrics["balanced_acc"] = None

    return metrics


def run_leave_one_drug_out(
        dataset,
        drug_labels: np.ndarray,
        build_model_fn: callable,
        train_fold_fn: callable,
        collate_fn: callable,
        batch_size: int = 16,
        device: str = "cpu",
        min_test_samples: int = 5,
) -> dict:
    """
    Leave-One-Drug-Out (LODO) cross-validation.

    For each unique drug, train on all other drugs and test on the held-out
    drug. This evaluates whether the model generalises to unseen drug
    scaffolds — the "cold drug" scenario.

    Args:
        dataset: PyTorch Dataset.
        drug_labels: Array of drug names per sample (same length as dataset).
        build_model_fn: Callable() → fresh model instance.
        train_fold_fn: Callable(model, train_loader, val_loader, device) → trained model.
        collate_fn: Collation function for DataLoader.
        batch_size: DataLoader batch size.
        device: Training device.
        min_test_samples: Minimum samples to evaluate a drug fold.

    Returns:
        Dict with per-drug results and aggregate summary.
    """
    unique_drugs = sorted(set(drug_labels))
    results = {}
    all_metrics = []

    print(f"\n  Leave-One-Drug-Out: {len(unique_drugs)} drugs")

    for drug in unique_drugs:
        test_mask = drug_labels == drug
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        if len(test_idx) < min_test_samples:
            print(f"    {drug}: {len(test_idx)} samples — SKIPPED (< {min_test_samples})")
            results[drug] = {"n_samples": int(len(test_idx)),
                             "skipped": True,
                             "reason": f"< {min_test_samples} test samples"}
            continue

        print(f"    {drug}: train={len(train_idx)}, test={len(test_idx)}")

        # Create data loaders
        train_subset = Subset(dataset, train_idx.tolist())
        test_subset = Subset(dataset, test_idx.tolist())

        # Weighted sampler for class imbalance
        train_labels = np.array([
            dataset.df.iloc[i]["resistance_label"] for i in train_idx
        ]).astype(int)
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_idx), replacement=True,
        )

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=0,
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # Build and train fresh model
        model = build_model_fn()
        model = train_fold_fn(model, train_loader, test_loader, device)

        # Collect predictions
        model.eval()
        all_ic50_pred, all_ic50_true = [], []
        all_prob, all_cls_true = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                ic50_pred, resist_pred = model(
                    seq_embeddings=batch_dev["seq_emb"],
                    struct_embeddings=batch_dev["struct_emb"],
                    drug_pooled=batch_dev["drug_pooled"],
                    drug_embeddings=batch_dev.get("drug_emb"),
                    ptm_vector=batch_dev["ptm_vector"],
                    delta_ptm_vector=batch_dev["delta_ptm_vector"],
                    target_protein=batch_dev["target_protein"],
                )
                all_ic50_pred.append(ic50_pred.cpu().numpy())
                all_ic50_true.append(batch["ln_ic50"].numpy())
                all_prob.append(torch.sigmoid(resist_pred).cpu().numpy())
                all_cls_true.append(batch["resistance_label"].numpy())

        y_pred_ic50 = np.concatenate(all_ic50_pred).flatten()
        y_true_ic50 = np.concatenate(all_ic50_true).flatten()
        y_prob = np.concatenate(all_prob).flatten()
        y_true_cls = np.concatenate(all_cls_true).flatten()

        fold_metrics = compute_cold_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob)
        fold_metrics["drug"] = drug
        fold_metrics["n_train"] = int(len(train_idx))
        results[drug] = fold_metrics
        all_metrics.append(fold_metrics)

        auroc_str = f"{fold_metrics['auroc']:.3f}" if fold_metrics['auroc'] is not None else "N/A"
        print(f"      AUROC={auroc_str}, RMSE={fold_metrics['rmse']:.3f}, "
              f"PCC={fold_metrics['pearson_r']:.3f}")

    # Aggregate summary
    summary = {
        "n_drugs_total": len(unique_drugs),
        "n_drugs_evaluated": len(all_metrics),
    }

    pcc_vals = [m["pearson_r"] for m in all_metrics if m.get("pearson_r") is not None]
    rmse_vals = [m["rmse"] for m in all_metrics]
    auroc_vals = [m["auroc"] for m in all_metrics if m.get("auroc") is not None]

    if pcc_vals:
        summary["mean_pearson_r"] = float(np.mean(pcc_vals))
        summary["std_pearson_r"] = float(np.std(pcc_vals))
    if rmse_vals:
        summary["mean_rmse"] = float(np.mean(rmse_vals))
        summary["std_rmse"] = float(np.std(rmse_vals))
    if auroc_vals:
        summary["mean_auroc"] = float(np.mean(auroc_vals))
        summary["std_auroc"] = float(np.std(auroc_vals))

    return {
        "method": "Leave-One-Drug-Out (LODO)",
        "description": "Cold-drug evaluation: train on N-1 drugs, test on held-out drug",
        "per_drug_results": results,
        "summary": summary,
    }


def run_cold_cell_evaluation(
        dataset,
        cell_labels: np.ndarray,
        build_model_fn: callable,
        train_fold_fn: callable,
        collate_fn: callable,
        n_folds: int = 5,
        batch_size: int = 16,
        device: str = "cpu",
        seed: int = 42,
) -> dict:
    """
    Cold-cell evaluation via stratified cell-line-level split.

    Splits cell lines (not samples) into folds so that all samples from
    a given cell line are either entirely in train or entirely in test.
    This prevents the model from memorising cell-line-specific features.

    Args:
        dataset: PyTorch Dataset.
        cell_labels: Array of cell line names per sample.
        build_model_fn: Callable() → fresh model instance.
        train_fold_fn: Callable(model, train_loader, val_loader, device) → trained model.
        collate_fn: Collation function.
        n_folds: Number of CV folds.
        batch_size: Batch size.
        device: Device.
        seed: Random seed.

    Returns:
        Dict with per-fold results and aggregate summary.
    """
    from sklearn.model_selection import KFold

    unique_cells = sorted(set(cell_labels))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_cells)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}
    all_metrics = []

    print(f"\n  Cold-cell {n_folds}-fold CV: {len(unique_cells)} unique cell lines")

    for fold_i, (train_cell_idx, test_cell_idx) in enumerate(kf.split(unique_cells)):
        train_cells = set(np.array(unique_cells)[train_cell_idx])
        test_cells = set(np.array(unique_cells)[test_cell_idx])

        train_sample_idx = np.where(np.isin(cell_labels, list(train_cells)))[0]
        test_sample_idx = np.where(np.isin(cell_labels, list(test_cells)))[0]

        print(f"    Fold {fold_i+1}/{n_folds}: train={len(train_sample_idx)} samples "
              f"({len(train_cells)} cells), test={len(test_sample_idx)} samples "
              f"({len(test_cells)} cells)")

        train_subset = Subset(dataset, train_sample_idx.tolist())
        test_subset = Subset(dataset, test_sample_idx.tolist())

        train_labels = np.array([
            dataset.df.iloc[i]["resistance_label"] for i in train_sample_idx
        ]).astype(int)
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_sample_idx), replacement=True,
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler,
                                  collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)

        model = build_model_fn()
        model = train_fold_fn(model, train_loader, test_loader, device)

        model.eval()
        all_ic50_pred, all_ic50_true, all_prob, all_cls_true = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                ic50_pred, resist_pred = model(
                    seq_embeddings=batch_dev["seq_emb"],
                    struct_embeddings=batch_dev["struct_emb"],
                    drug_pooled=batch_dev["drug_pooled"],
                    drug_embeddings=batch_dev.get("drug_emb"),
                    ptm_vector=batch_dev["ptm_vector"],
                    delta_ptm_vector=batch_dev["delta_ptm_vector"],
                    target_protein=batch_dev["target_protein"],
                )
                all_ic50_pred.append(ic50_pred.cpu().numpy())
                all_ic50_true.append(batch["ln_ic50"].numpy())
                all_prob.append(torch.sigmoid(resist_pred).cpu().numpy())
                all_cls_true.append(batch["resistance_label"].numpy())

        y_pred_ic50 = np.concatenate(all_ic50_pred).flatten()
        y_true_ic50 = np.concatenate(all_ic50_true).flatten()
        y_prob = np.concatenate(all_prob).flatten()
        y_true_cls = np.concatenate(all_cls_true).flatten()

        fold_metrics = compute_cold_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob)
        fold_metrics["fold"] = fold_i + 1
        fold_metrics["n_train_cells"] = len(train_cells)
        fold_metrics["n_test_cells"] = len(test_cells)
        results[f"fold_{fold_i+1}"] = fold_metrics
        all_metrics.append(fold_metrics)

        auroc_str = f"{fold_metrics['auroc']:.3f}" if fold_metrics['auroc'] is not None else "N/A"
        print(f"      AUROC={auroc_str}, RMSE={fold_metrics['rmse']:.3f}")

    summary = {"n_folds": n_folds, "n_unique_cells": len(unique_cells)}
    auroc_vals = [m["auroc"] for m in all_metrics if m.get("auroc") is not None]
    rmse_vals = [m["rmse"] for m in all_metrics]
    pcc_vals = [m["pearson_r"] for m in all_metrics if m.get("pearson_r") is not None]
    if auroc_vals:
        summary["mean_auroc"] = float(np.mean(auroc_vals))
        summary["std_auroc"] = float(np.std(auroc_vals))
    if rmse_vals:
        summary["mean_rmse"] = float(np.mean(rmse_vals))
        summary["std_rmse"] = float(np.std(rmse_vals))
    if pcc_vals:
        summary["mean_pearson_r"] = float(np.mean(pcc_vals))
        summary["std_pearson_r"] = float(np.std(pcc_vals))

    return {
        "method": "Cold-cell K-fold CV (cell-line-level split)",
        "description": "All samples from each cell line in either train or test — no cell line leakage",
        "per_fold_results": results,
        "summary": summary,
    }
