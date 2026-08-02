"""Training and validation loops for the PTM-BDL model."""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F

from src.ptm_bdl.training.metrics import compute_metrics


def _clear_mps_cache():
    """Free MPS cached memory to prevent accumulation across epochs/folds."""
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def train_epoch(model, loader, optimizer, scheduler, focal_loss,
                lambda_reg, lambda_cls, device, label_smoothing=0.05):
    """
    Run one training epoch with multi-task loss.

    Uses Huber loss for regression (robust to IC50 outliers) and
    focal loss for classification with label smoothing.
    """
    model.train()
    losses = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        ic50_pred, resist_pred = model(
            seq_embeddings=batch["seq_emb"],
            struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"],
            drug_embeddings=batch["drug_emb"],
            ptm_vector=batch["ptm_vector"],
            delta_ptm_vector=batch["delta_ptm_vector"],
            secondary_vector=batch["secondary_vector"],
            delta_secondary_vector=batch["delta_secondary_vector"],
            target_protein=batch["target_protein"],
        )

        huber = F.smooth_l1_loss(ic50_pred, batch["ln_ic50"], reduction='none')
        loss_reg = huber.squeeze(-1).mean()

        targets_smooth = batch["resistance_label"] * (1 - label_smoothing) + label_smoothing / 2
        loss_cls = focal_loss(resist_pred, targets_smooth)

        loss = lambda_reg * loss_reg + lambda_cls * loss_cls

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None and hasattr(scheduler, '_step_count'):
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        losses.append(loss.item())

    if scheduler is not None:
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    _clear_mps_cache()
    return np.mean(losses)


def validate(model, loader, focal_loss, lambda_reg, lambda_cls, device):
    """Validate and return comprehensive metrics.

    Uses an adaptive threshold (Youden's J) instead of hardcoded 0.5.
    This prevents BAcc from being stuck at 0.500 when the model's predicted
    probabilities cluster in a narrow band (e.g., all ~0.46).
    """
    model.eval()
    val_losses = []
    all_probs, all_labels = [], []
    all_ic50_preds, all_ic50_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            ic50_pred, resist_pred = model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch["drug_emb"],
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                secondary_vector=batch["secondary_vector"],
                delta_secondary_vector=batch["delta_secondary_vector"],
                target_protein=batch["target_protein"],
            )

            loss_reg = ((ic50_pred - batch["ln_ic50"]) ** 2).mean()
            loss_cls = focal_loss(resist_pred, batch["resistance_label"])
            val_losses.append((lambda_reg * loss_reg + lambda_cls * loss_cls).item())

            probs = torch.sigmoid(resist_pred).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(
                batch["resistance_label"].cpu().numpy().flatten().tolist()
            )
            all_ic50_preds.extend(ic50_pred.cpu().numpy().flatten().tolist())
            all_ic50_targets.extend(batch["ln_ic50"].cpu().numpy().flatten().tolist())

    # Adaptive threshold: use Youden's J if both classes present,
    # otherwise fall back to 0.5.  This prevents BAcc from being stuck
    # at 0.500 when all probs are on one side of 0.5.
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    threshold = 0.5
    if len(set(all_labels)) >= 2:
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels_arr, probs_arr)
            J = tpr - fpr
            threshold = float(thresholds[np.argmax(J)])
        except Exception:
            threshold = 0.5

    all_preds = (probs_arr > threshold).astype(float).tolist()

    metrics = compute_metrics(all_preds, all_probs, all_labels,
                              all_ic50_preds, all_ic50_targets)
    metrics["loss"] = float(np.mean(val_losses))
    metrics["threshold"] = threshold

    _clear_mps_cache()
    return metrics


def compute_optimal_threshold(model, val_loader, device):
    """
    Compute the optimal classification threshold using Youden's J statistic.

    Finds the probability threshold that maximizes (sensitivity + specificity - 1)
    on the validation set. This is essential for calibrating the resistance
    classifier — especially when focal loss shifts predicted probabilities away
    from 0.5.

    Ref: Youden WJ (1950) "Index for rating diagnostic tests." Cancer 3:32-35.

    Args:
        model: Trained PTM-BDL model (eval mode).
        val_loader: Validation DataLoader.
        device: torch device.

    Returns:
        dict with keys: optimal_threshold (float), method (str), reference (str)
    """
    from sklearn.metrics import roc_curve

    model.eval()
    all_probs, all_labels = [], []

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
                secondary_vector=batch["secondary_vector"],
                delta_secondary_vector=batch["delta_secondary_vector"],
                target_protein=batch["target_protein"],
            )
            all_probs.extend(
                torch.sigmoid(resist_pred).cpu().numpy().flatten().tolist())
            all_labels.extend(
                batch["resistance_label"].cpu().numpy().flatten().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if len(set(all_labels)) >= 2:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        J = tpr - fpr
        optimal_threshold = float(thresholds[np.argmax(J)])
    else:
        optimal_threshold = 0.5

    return {
        "optimal_threshold": optimal_threshold,
        "method": "Youden_J",
        "reference": "Youden WJ (1950) Cancer 3:32-35",
    }
