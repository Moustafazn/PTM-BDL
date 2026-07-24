"""Comprehensive metric computation for classification and regression tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(all_preds, all_probs, all_labels,
                    all_ic50_preds=None, all_ic50_targets=None):
    """
    Compute comprehensive metrics for both classification and regression.

    Classification: accuracy, balanced_accuracy, sensitivity, specificity,
                    f1, auroc, auprc, f1_macro
    Regression: mse, rmse, pearson_r (if IC50 provided)
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
    mean_prob = float(probs.mean())

    # AUROC & PR-AUC
    try:
        if len(set(labels)) >= 2:
            auroc = float(roc_auc_score(labels, probs))
            auprc_resistant = float(average_precision_score(labels, probs))
            auprc_sensitive = float(average_precision_score(1 - labels, 1 - probs))
        else:
            auroc = auprc_resistant = auprc_sensitive = 0.0
    except Exception:
        auroc = auprc_resistant = auprc_sensitive = 0.0

    # F1-macro
    precision_s = tn / max(tn + fn, 1)
    recall_s = tn / max(tn + fp, 1)
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
