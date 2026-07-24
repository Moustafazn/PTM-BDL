"""Full evaluation — collect predictions, compute metrics, and threshold management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)


def collect_predictions(model, loader):
    """Run the model on a dataloader and collect all predictions."""
    all_ic50_pred, all_ic50_true = [], []
    all_resist_prob, all_resist_true = [], []

    # Detect model device so we can move batch tensors there
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in loader:
            # Move all tensors to model device
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            ic50_pred, resist_pred = model(
                seq_embeddings=batch_dev["seq_emb"],
                struct_embeddings=batch_dev["struct_emb"],
                drug_pooled=batch_dev["drug_pooled"],
                drug_embeddings=batch_dev.get("drug_emb"),
                ptm_vector=batch_dev["ptm_vector"],
                delta_ptm_vector=batch_dev["delta_ptm_vector"],
                secondary_vector=batch_dev["secondary_vector"],
                delta_secondary_vector=batch_dev["delta_secondary_vector"],
                target_protein=batch_dev["target_protein"],
            )
            all_ic50_pred.extend(ic50_pred.squeeze(-1).cpu().numpy().tolist())
            all_ic50_true.extend(batch["ln_ic50"].squeeze(-1).cpu().numpy().tolist())
            all_resist_prob.extend(
                torch.sigmoid(resist_pred).squeeze(-1).cpu().numpy().tolist()
            )
            all_resist_true.extend(
                batch["resistance_label"].squeeze(-1).cpu().numpy().tolist()
            )

    return (np.array(all_ic50_true), np.array(all_ic50_pred),
            np.array(all_resist_true), np.array(all_resist_prob))


def compute_full_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
                         threshold: float = 0.5, return_per_class: bool = True):
    """Compute comprehensive regression and classification metrics."""
    y_pred_cls = (y_prob_cls > threshold).astype(float)
    has_both = len(set(y_true_cls)) > 1

    regression = {
        "mse": float(mean_squared_error(y_true_ic50, y_pred_ic50)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))),
        "r2": float(r2_score(y_true_ic50, y_pred_ic50))
        if len(set(y_true_ic50)) > 1 else 0.0,
    }
    if len(y_true_ic50) > 2:
        pr = stats.pearsonr(y_true_ic50, y_pred_ic50)
        sr = stats.spearmanr(y_true_ic50, y_pred_ic50)
        regression.update({
            "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
        })

    cm = confusion_matrix(y_true_cls, y_pred_cls).tolist() if has_both else [[0]]
    classification = {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_cls, y_pred_cls))
        if has_both else 0.0,
        "f1_score": float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
        "auroc": float(roc_auc_score(y_true_cls, y_prob_cls)) if has_both else 0.0,
        "auprc": float(average_precision_score(y_true_cls, y_prob_cls)) if has_both else 0.0,
        "confusion_matrix": cm,
        "mean_predicted_probability": float(y_prob_cls.mean()),
    }

    if has_both:
        report = classification_report(
            y_true_cls, y_pred_cls,
            target_names=["sensitive", "resistant"],
            output_dict=True, zero_division=0
        )
        classification["per_class"] = {
            "sensitive": report.get("sensitive", {}),
            "resistant": report.get("resistant", {}),
        }

    return regression, classification


def load_threshold(model_dir: str | Path, default: float = 0.5) -> float:
    """
    Load the optimal classification threshold from a model directory.

    Loads `optimal_threshold.json` saved by `compute_optimal_threshold()` during
    training. Falls back to `default` (0.5) if the file doesn't exist.

    Args:
        model_dir: Path to the model directory containing optimal_threshold.json.
        default: Fallback threshold if file not found.

    Returns:
        Optimal threshold (float).
    """
    thr_path = Path(model_dir) / "optimal_threshold.json"
    if thr_path.exists():
        with open(thr_path) as f:
            info = json.load(f)
        return float(info.get("optimal_threshold", default))
    return default


def make_eval_loader(dataset, indices, batch_size: int, collate_fn):
    """
    Create a DataLoader from a dataset and index array for evaluation.

    Standardized helper used across evaluate, crossval, loclo, and explain
    scripts. Ensures consistent DataLoader construction across all case studies.

    Args:
        dataset: PyTorch Dataset.
        indices: Array/list of sample indices.
        batch_size: Batch size.
        collate_fn: Collation function for variable-length sequences.

    Returns:
        DataLoader for the specified subset.
    """
    subset = torch.utils.data.Subset(
        dataset, indices.tolist() if hasattr(indices, 'tolist') else list(indices))
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
