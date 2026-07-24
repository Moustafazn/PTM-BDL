"""Stratified splitting utilities for PTM-BDL datasets.

Generic implementation — works with ANY case study (EGFR, HeLa, K562, …).
Stratifies by resistance label × target protein (when available).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_stratified_splits(dataset, train_ratio, val_ratio, seed):
    """
    Stratified train/val/test split by resistance label + target protein.

    Creates a combined stratification key to ensure each split has proportional
    representation of all protein × resistance class combinations.
    """

    df = dataset.df
    labels = df["resistance_label"].values.astype(int)

    # Use target_protein for finer stratification when available.
    if "target_protein" in df.columns:
        target_protein = df["target_protein"].fillna("unknown").values
    else:
        target_protein = np.array(["unknown"] * len(df))

    combined_labels = np.array([
        f"{tg}_{int(r)}" for tg, r in zip(target_protein, labels)
    ])

    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(df)), combined_labels))

    val_frac = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_sub, val_sub = next(sss2.split(
        np.zeros(len(trainval_idx)), combined_labels[trainval_idx]
    ))

    train_idx = trainval_idx[train_sub]
    val_idx = trainval_idx[val_sub]

    # ── Generic summary ──────────────────────────────────────────────────
    unique_proteins = sorted(set(target_protein) - {"unknown"})

    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        n_sens = int((labels[idx] == 0).sum())
        n_res = int((labels[idx] == 1).sum())

        parts = [f"{len(idx)} samples",
                 f"{n_res} resistant + {n_sens} sensitive"]

        # Per-protein counts (generic — works for EGFR/ERBB2, HDAC1/EP300, ABL1, …)
        if unique_proteins:
            protein_str = ", ".join(
                f"{p}={int((target_protein[idx] == p).sum())}"
                for p in unique_proteins
            )
            parts.append(protein_str)

        print(f"    {name}: {' | '.join(parts)}")

    return train_idx, val_idx, test_idx
