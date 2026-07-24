#!/usr/bin/env python3
"""
HeLa/HDAC — K-Fold Cross-Validation.

Runs stratified K-fold CV to assess model robustness.
Reports mean ± std for AUROC, BAcc, Pearson R, RMSE across folds.

Ref: Baptista et al., Brief Bioinform 2021 (PMID 33169146) — DRP CV protocol
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader

from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics
from src.ptm_bdl.training import FocalLoss, train_epoch, validate, build_model_from_cfg, compute_optimal_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_crossval(n_folds: int = 5):
    """Stratified K-fold cross-validation."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — {n_folds}-Fold Cross-Validation            ║")
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
    dataset = ResistanceDataset(dataset_path, features_dir)

    labels = dataset.df["resistance_label"].values.astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=cfg["training"]["seed"])

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n  ── Fold {fold_i + 1}/{n_folds} ──")
        val_size = len(train_idx) // 5
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]

        model = build_model_from_cfg(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["model"]["learning_rate"])
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=cfg["model"]["batch_size"],
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx),
                                batch_size=cfg["model"]["batch_size"],
                                shuffle=False, collate_fn=collate_fn)

        best_score = 0.0
        for epoch in range(1, min(cfg["model"]["num_epochs"], 30) + 1):
            train_epoch(model, train_loader, optimizer, None, focal_loss, 1.0, 2.0, device)
            val = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
            score = max(val.get("auroc", 0), val.get("balanced_acc", 0))
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        # Compute optimal threshold on validation set (Youden's J)
        threshold_info = compute_optimal_threshold(model, val_loader, device)
        fold_threshold = threshold_info["optimal_threshold"]
        print(f"    Optimal threshold (Youden's J): {fold_threshold:.4f}")

        test_loader = DataLoader(Subset(dataset, test_idx),
                                 batch_size=cfg["model"]["batch_size"],
                                 shuffle=False, collate_fn=collate_fn)
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
            model, test_loader)
        reg, cls = compute_full_metrics(
            y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
            threshold=fold_threshold)
        metrics = {**reg, **cls, "threshold": fold_threshold}
        fold_results.append(metrics)
        print(f"    AUROC={metrics.get('auroc', 'N/A'):.3f}, "
              f"BAcc={metrics.get('balanced_accuracy', 'N/A'):.3f}")

    # Summary
    summary = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results if isinstance(f.get(key), (int, float))]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    report = {"case_study": CASE_STUDY, "n_folds": n_folds,
              "fold_results": fold_results, "summary": summary}
    with open(RESULTS_DIR / "crossval_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ {n_folds}-fold CV complete!")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    run_crossval()
