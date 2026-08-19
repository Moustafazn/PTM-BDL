#!/usr/bin/env python3
"""
EGFR/ERBB2 TKI Case Study — Training entry point.

Trains the PTM-BDL model on the EGFR/ERBB2 TKI resistance dataset.
  - 951 samples: 646 EGFR (NSCLC) + 305 ERBB2 (breast cancer)
  - 6 drugs: Gefitinib, Erlotinib, Afatinib, Osimertinib, Lapatinib, Sapitinib
  - 2 PTM types: phospho (Y/S/T subtypes) + glyco (N subtype)
  - 24 PTM tokens per sample (12 phospho + 12 glyco)

Uses the tool packages:
  ptm_bdl.training  — train_epoch, validate, FocalLoss, build_model_from_cfg
  ptm_bdl.data      — ResistanceDataset, collate_fn, create_stratified_splits
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from src.ptm_bdl.data import ResistanceDataset, collate_fn, create_stratified_splits
from src.ptm_bdl.training import (
    FocalLoss, train_epoch, validate, build_model_from_cfg,
    compute_optimal_threshold, save_checkpoint, load_checkpoint,
    resolve_device, create_balanced_sampler,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train():
    """Two-stage training pipeline with class-balanced sampling."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EGFR/ERBB2 TKI Case Study — Training                     ║")
    print("║  Tool: ptm_bdl.training + ptm_bdl.data                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(cfg)
    print(f"\n  Device: {device}")

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    print(f"  Dataset: {dataset_path}")
    dataset = ResistanceDataset(dataset_path, features_dir)
    print(f"  Samples: {len(dataset)}")

    n_resistant = int(dataset.df["resistance_label"].sum())
    n_sensitive = len(dataset) - n_resistant
    print(f"  Class distribution: {n_resistant} resistant ({100 * n_resistant / len(dataset):.1f}%), "
          f"{n_sensitive} sensitive ({100 * n_sensitive / len(dataset):.1f}%)")

    print(f"\n  Creating stratified splits...")
    train_idx, val_idx, test_idx = create_stratified_splits(
        dataset, cfg["training"]["train_ratio"],
        cfg["training"]["val_ratio"], seed,
    )

    split_info = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "stratification": "resistance_label",
        "seed": seed,
    }
    with open(MODEL_DIR / "split_indices.json", "w") as f:
        json.dump(split_info, f)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    sampler = create_balanced_sampler(dataset, train_idx)

    batch_size = cfg["model"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    model = build_model_from_cfg(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    lambda_reg, lambda_cls = 1.0, 2.0

    lr = cfg["model"]["learning_rate"]
    wd = cfg["model"]["weight_decay"]
    num_epochs = cfg["model"]["num_epochs"]
    patience = cfg["model"]["early_stopping_patience"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_val_score = 0.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 focal_loss, lambda_reg, lambda_cls, device)
        val = validate(model, val_loader, focal_loss,
                       lambda_reg, lambda_cls, device)

        if epoch % 5 == 0 or epoch <= 2:
            print(f"  {epoch:3d}/{num_epochs} | "
                  f"loss={train_loss:.4f} | AUROC={val.get('auroc', 0):.3f} | "
                  f"BAcc={val['balanced_acc']:.3f} | RMSE={val.get('rmse', 0):.3f}")

        val_score = max(val.get("auroc", 0), val.get("balanced_acc", 0))
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_checkpoint(model, MODEL_DIR / "best_model_stage1.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (best={best_val_score:.3f})")
                break

    src_path = MODEL_DIR / "best_model_stage1.pt"
    if src_path.exists():
        shutil.copy(src_path, MODEL_DIR / "best_model.pt")

    # ── Compute optimal classification threshold (Youden's J) ──────────────
    print(f"\n  Computing optimal threshold (Youden's J on validation set)...")
    load_checkpoint(model, MODEL_DIR / "best_model.pt", device)
    threshold_info = compute_optimal_threshold(model, val_loader, device)
    with open(MODEL_DIR / "optimal_threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  ✓ Optimal threshold: {threshold_info['optimal_threshold']:.4f}")

    print(f"\n  ✓ Models saved: {MODEL_DIR}")
    print(f"✓ Training complete!")


if __name__ == "__main__":
    train()
