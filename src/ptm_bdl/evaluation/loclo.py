"""
Leave-One-Class-Line-Out (LOCLO) generalization framework.

Tests whether the model generalizes to held-out groups of samples (e.g.,
cell lines grouped by mutation class, amplification status, or tissue type).

This is the "cell-blind" split recommended by:
  Sada Del Real et al., Brief Bioinf 2026 — "cell-blind... is particularly
  valuable for drug repositioning and, more importantly, for precision medicine"

The module is case-study-agnostic — group assignments are provided by the caller.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def compute_loclo_metrics(
        y_true_ic50: np.ndarray,
        y_pred_ic50: np.ndarray,
        y_true_cls: np.ndarray,
        y_prob_cls: np.ndarray,
        threshold: float = 0.5,
) -> dict:
    """
    Compute evaluation metrics for a single LOCLO fold.

    Returns dict with regression (rmse, pearson_r, spearman_rho) and
    classification (auroc, auprc_sensitive, balanced_acc) metrics.
    """
    metrics = {
        "n_samples": int(len(y_true_ic50)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))),
        "n_resistant": int((y_true_cls == 1).sum()),
        "n_sensitive": int((y_true_cls == 0).sum()),
    }

    if len(y_true_ic50) > 2 and np.std(y_pred_ic50) > 1e-8:
        metrics["pearson_r"] = float(np.corrcoef(y_true_ic50, y_pred_ic50)[0, 1])
        sr = stats.spearmanr(y_true_ic50, y_pred_ic50)
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


def run_loclo(
        dataset,
        group_assignments: np.ndarray,
        build_model_fn: callable,
        train_fold_fn: callable,
        collate_fn: callable,
        min_group_size: int = 10,
        batch_size: int = 16,
        device: str = "cpu",
) -> dict:
    """
    Run Leave-One-Class-Line-Out cross-validation.

    Args:
        dataset: PyTorch Dataset.
        group_assignments: Array of group names per sample (same length as dataset).
        build_model_fn: Callable() → fresh model instance.
        train_fold_fn: Callable(model, train_loader, val_loader, device) → trained model.
                       The caller provides the training loop implementation.
        collate_fn: Collation function for DataLoader.
        min_group_size: Minimum samples to evaluate a group.
        batch_size: DataLoader batch size.
        device: Training device.

    Returns:
        Dict with per-group results and aggregate summary.
    """
    unique_groups = sorted(set(group_assignments))
    valid_groups = [g for g in unique_groups if (group_assignments == g).sum() >= min_group_size]

    results = {}
    all_metrics = []

    for group in valid_groups:
        test_mask = group_assignments == group
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

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
                    secondary_vector=batch_dev.get("secondary_vector"),
                    delta_secondary_vector=batch_dev.get("delta_secondary_vector"),
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

        fold_metrics = compute_loclo_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob)
        fold_metrics["group"] = group
        fold_metrics["n_train"] = int(len(train_idx))
        results[group] = fold_metrics
        all_metrics.append(fold_metrics)

    # Aggregate summary
    summary = {
        "n_groups_total": len(unique_groups),
        "n_groups_evaluated": len(valid_groups),
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
        "per_group_results": results,
        "summary": summary,
    }
