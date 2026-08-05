"""Training, validation, and checkpoint utilities for the PTM-BDL model."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from src.ptm_bdl.training.metrics import compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ───────────────────────────────────────────────────────────────────────────────
# Centralised save/load so every consumer (train, evaluate, explain, ablation,
# loclo, cross-dataset) uses weights_only=True and consistent conventions.
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: torch.nn.Module, path: Union[str, Path]) -> Path:
    """Save model ``state_dict`` to *path*.

    Creates parent directories if needed.  Returns the resolved ``Path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> torch.nn.Module:
    """Load a ``state_dict`` into *model* from *path*.

    Always uses ``weights_only=True`` (PyTorch ≥ 2.6 default) to prevent
    arbitrary code execution from untrusted checkpoint files.

    Args:
        model:  An already-constructed model (e.g. from ``build_model_from_cfg``).
        path:   Path to the ``.pt`` checkpoint file.
        device: Map location for ``torch.load``.

    Returns:
        The model with loaded weights, in ``eval`` mode.
    """
    path = Path(path)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Device resolution
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_device(cfg: dict) -> torch.device:
    """Resolve the compute device from config ``training.device``.

    Accepts ``"auto"`` (tries CUDA → MPS → CPU), ``"cuda"``, ``"mps"``,
    or ``"cpu"``.  This avoids repeating the 6-line detection block in
    every case study script.
    """
    device_str = cfg.get("training", {}).get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ═══════════════════════════════════════════════════════════════════════════════
# Class-balanced sampling
# ═══════════════════════════════════════════════════════════════════════════════

def create_balanced_sampler(
    dataset,
    train_idx: np.ndarray,
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that balances resistant vs sensitive.

    Drug-resistance datasets are typically highly imbalanced (e.g. 87%
    resistant in EGFR).  Without balancing, the model predicts the
    majority class for everything and appears broken (BAcc ≈ 0.50).

    Args:
        dataset:   A ``ResistanceDataset`` with ``.df`` attribute.
        train_idx: Array of training set indices.

    Returns:
        A ``WeightedRandomSampler`` ready for ``DataLoader(sampler=...)``.
    """
    train_labels = dataset.df["resistance_label"].values[train_idx]
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_idx),
        replacement=True,
    )


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
