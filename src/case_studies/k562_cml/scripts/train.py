#!/usr/bin/env python3
"""
K562/CML Case Study — Training entry point.

Trains the PTM-BDL model on the K562 CML dataset.
  - Cell line: K562 (BCR-ABL+ CML)
  - 5 drugs: Dasatinib, Imatinib (TKIs) + Cytarabine, Paclitaxel, Methotrexate (chemo)
  - PTM type: phosphorylation (S/T/Y) only — different kinase system (ABL vs EGFR)
  - Proves PTM-BDL generalizes to hematological cancer + mixed drug mechanisms

Uses the same framework packages as EGFR case study — ZERO code changes.
"""
import json
import shutil
import time
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

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train():
    """Training pipeline for K562/CML case study."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Training                                   ║")
    print(f"║  Framework: ptm_bdl.training + ptm_bdl.data                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

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
    print(f"  Class distribution: {n_resistant} resistant, {n_sensitive} sensitive")

    train_idx, val_idx, test_idx = create_stratified_splits(
        dataset, cfg["training"]["train_ratio"],
        cfg["training"]["val_ratio"], seed,
    )

    with open(MODEL_DIR / "split_indices.json", "w") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),
                    "test_idx": test_idx.tolist(), "seed": seed}, f)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    sampler = create_balanced_sampler(dataset, train_idx)

    batch_size = cfg["model"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    model = build_model_from_cfg(cfg).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    lr = cfg["model"]["learning_rate"]
    num_epochs = cfg["model"]["num_epochs"]
    patience = cfg["model"]["early_stopping_patience"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["model"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    best_val_score = 0.0
    patience_counter = 0
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    print(f"  Batches/epoch: {n_train_batches} train + {n_val_batches} val")

    training_start = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 focal_loss, 1.0, 2.0, device)
        val = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
        epoch_time = time.time() - t0

        if epoch % 5 == 0 or epoch <= 2:
            elapsed = (time.time() - training_start) / 60
            print(f"  {epoch:3d}/{num_epochs} | loss={train_loss:.4f} | "
                  f"AUROC={val.get('auroc', 0):.3f} | BAcc={val['balanced_acc']:.3f} | "
                  f"{epoch_time:.0f}s/epoch | {elapsed:.0f}m elapsed")

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

    # Copy to best_model.pt (consistent with EGFR naming)
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
