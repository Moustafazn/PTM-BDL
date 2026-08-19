#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EGFR/ERBB2 TKI — Stratified K-Fold Cross-Validation        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  5-fold stratified CV (resistance_label × target_protein) of the PTM-BDL   ║
║  multimodal model.  Per fold:                                                ║
║    1. Train one full model + one no_ptm model on the same indices.          ║
║    2. Evaluate both on the held-out test fold.                              ║
║    3. Run per-mod-type Integrated Gradients on the PTM-BDL token plane:     ║
║         phospho (slots 0..11), glyco (slots 12..23), all (24-vector).       ║
║       — bucketed per gene (EGFR / ERBB2) so we get parallel rankings.       ║
║                                                                              ║
║  Aggregation across folds:                                                   ║
║    • Mean ± std for BAcc, AUROC, RMSE, Pearson R.                            ║
║    • Paired t-test on the PTM ablation Δ (Full − No PTM).                    ║
║    • Per-mod-type IG rankings:                                              ║
║        Y1068 (EGFR phospho slot 7) ↔ Y1221 (ERBB2 phospho slot 7)           ║
║        N528  (EGFR glyco  slot 8) ↔ N530  (ERBB2 glyco  slot 4)             ║
║      reported as concordant / discordant homology pairs.                    ║
║                                                                              ║
║  OUTPUT: results/crossval_results.json                                       ║
║          results/figures/crossval_*.png                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# ── Import from tool packages ──────────────────────────────────────────
from src.ptm_bdl.data.dataset import ResistanceDataset
from src.ptm_bdl.data.collate import collate_fn
from src.ptm_bdl.training.loss import FocalLoss
from src.ptm_bdl.training.trainer import train_epoch, validate
from src.ptm_bdl.training.factory import build_model_from_cfg

# ── Import from case study biology ──────────────────────────────────────────
from src.case_studies.egfr_erbb2_tki.biology import (
    PHOSPHO_LABELS_EGFR, PHOSPHO_LABELS_ERBB2,
    GLYCO_LABELS_EGFR, GLYCO_LABELS_ERBB2,
    GRB2_PHOSPHO_INDEX, EGFR_N528_INDEX, ERBB2_N530_INDEX,
)

PROTEIN_ID_ERBB2 = 1

from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Site labels, homology indices are imported from case study biology above


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_device():
    device_str = cfg["training"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _get_stratification_labels(df):
    """target_protein × resistance_label combined stratum label."""
    resistance = df["resistance_label"].values.astype(int)
    if "target_protein" in df.columns:
        genes = df["target_protein"].fillna("EGFR").values
        return np.array([f"{g}_{r}" for g, r in zip(genes, resistance)])
    return resistance


def _train_fold(dataset_path, features_dir, train_idx, val_idx, device,
                ablation_mode: str = "full", seed: int = 42):
    """Train one fold (one ablation mode), return trained model + val metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = ResistanceDataset(dataset_path, features_dir, ablation_mode=ablation_mode)
    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)
    train_labels = ds.df["resistance_label"].values[train_idx].astype(int)
    cc = np.bincount(train_labels)
    weights = (1.0 / cc) if cc.min() > 0 else np.ones(len(cc))
    sw = weights[train_labels].astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(train_set),
                                    replacement=True)
    bs = cfg["model"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, sampler=sampler,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            collate_fn=collate_fn)

    model = build_model_from_cfg(cfg).to(device)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    lr = cfg["model"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["model"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["model"]["num_epochs"], eta_min=lr * 0.01)

    best_auroc = 0.0
    patience = cfg["model"]["early_stopping_patience"]
    counter = 0
    best_state = None
    epoch_done = 0
    n_epochs = cfg["model"]["num_epochs"]
    print(f"      Training: max {n_epochs} epochs, patience={patience}")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 focal_loss, 1.0, 2.0, device)
        vm = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
        elapsed = time.time() - t0
        score = vm.get("auroc", 0)
        improved = score > best_auroc
        if epoch <= 2 or epoch % 5 == 0 or improved:
            marker = " ★" if improved else ""
            print(f"      Ep {epoch:3d}/{n_epochs} | "
                  f"loss={train_loss:.4f} | "
                  f"AUROC={score:.3f} | "
                  f"BAcc={vm.get('balanced_acc', 0):.3f} | "
                  f"RMSE={vm.get('rmse', 0):.3f} "
                  f"({elapsed:.1f}s){marker}")
        if improved:
            best_auroc = score
            counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                print(f"      Early stop at epoch {epoch} (best AUROC={best_auroc:.3f})")
                epoch_done = epoch
                break
        epoch_done = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    val_m = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
    return model, val_m, epoch_done


def _evaluate_fold(model, dataset, test_idx, device):
    """Run model on the test fold; return metrics + per-sample probs + ic50."""
    test_set = Subset(dataset, test_idx)
    loader = DataLoader(test_set, batch_size=cfg["model"]["batch_size"],
                        shuffle=False, collate_fn=collate_fn)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    test_m = validate(model, loader, focal_loss, 1.0, 2.0, device)

    model.eval()
    probs, ic50_preds = [], []
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
            probs.extend(torch.sigmoid(resist_pred).cpu().numpy().flatten().tolist())
            ic50_preds.extend(ic50_pred.cpu().numpy().flatten().tolist())
    return test_m, probs, ic50_preds


# ══════════════════════════════════════════════════════════════════════════════
# Per-mod-type IG (PTM-BDL)
# ══════════════════════════════════════════════════════════════════════════════

def _run_ig_per_mod_type(model, dataset, indices, n_steps: int = 20):
    """
    Integrated Gradients on the PTM-BDL 24-token input plane.

    Uses the flat ptm_vector (n_tokens = 12 phospho + 12 glyco) and
    delta_ptm_vector. Integrates along both level and delta channels,
    then slices into phospho (0:12) and glyco (12:24).

    Per-site importance = |grad_level × Δlevel| + |grad_delta × Δdelta|

    Returns dict:
      {
        "EGFR":  {"phospho": (12,), "glyco": (12,), "all": (24,), "n": int},
        "ERBB2": {"phospho": (12,), "glyco": (12,), "all": (24,), "n": int},
      }
    """
    model.train()
    n_tokens = 24  # 12 phospho + 12 glyco
    baseline_level = torch.ones(n_tokens)
    baseline_delta = torch.zeros(n_tokens)

    sums = {
        ("EGFR", "phospho"): np.zeros(12), ("EGFR", "glyco"): np.zeros(12),
        ("ERBB2", "phospho"): np.zeros(12), ("ERBB2", "glyco"): np.zeros(12),
    }
    counts = {"EGFR": 0, "ERBB2": 0}

    for idx in indices:
        sample = dataset[int(idx)]
        actual_level = sample["ptm_vector"]        # (24,) flat
        actual_delta = sample["delta_ptm_vector"]  # (24,) flat
        tp = sample["target_protein"].view(1).long()
        gene = "ERBB2" if tp.item() == PROTEIN_ID_ERBB2 else "EGFR"
        counts[gene] += 1

        seq_e = sample["seq_emb"].unsqueeze(0)
        str_e = sample["struct_emb"].unsqueeze(0)
        drg_e = sample["drug_emb"].unsqueeze(0)
        drg_p = sample["drug_pooled"].unsqueeze(0)

        grads_level = torch.zeros(n_tokens)
        grads_delta = torch.zeros(n_tokens)

        for step in range(n_steps + 1):
            a = step / n_steps
            interp_level = (baseline_level + a * (actual_level - baseline_level)
                           ).unsqueeze(0).requires_grad_(True)
            interp_delta = (baseline_delta + a * (actual_delta - baseline_delta)
                           ).unsqueeze(0).requires_grad_(True)

            _, resist_pred = model(
                seq_embeddings=seq_e,
                struct_embeddings=str_e,
                drug_pooled=drg_p,
                drug_embeddings=drg_e,
                ptm_vector=interp_level,
                delta_ptm_vector=interp_delta,
                target_protein=tp,
            )
            model.zero_grad()
            resist_pred.backward()
            if interp_level.grad is not None:
                grads_level += interp_level.grad.squeeze(0).detach()
            if interp_delta.grad is not None:
                grads_delta += interp_delta.grad.squeeze(0).detach()

        # IG: |grad_level × Δlevel| + |grad_delta × Δdelta| per site
        n_s = n_steps + 1
        d_level = actual_level - baseline_level
        d_delta = actual_delta - baseline_delta
        attr = (np.abs(((grads_level / n_s) * d_level).numpy())
                + np.abs(((grads_delta / n_s) * d_delta).numpy()))
        sums[(gene, "phospho")] += attr[:12]
        sums[(gene, "glyco")] += attr[12:24]

    model.eval()
    out = {}
    for gene in ["EGFR", "ERBB2"]:
        n = max(counts[gene], 1)
        ph_mean = sums[(gene, "phospho")] / n
        gl_mean = sums[(gene, "glyco")] / n
        out[gene] = {
            "phospho": ph_mean,
            "glyco": gl_mean,
            "all": np.concatenate([ph_mean, gl_mean]),
            "n": counts[gene],
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main CV runner
# ══════════════════════════════════════════════════════════════════════════════

def run_crossval(n_folds: int = 5, run_ablation: bool = True,
                 run_ig: bool = True):
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  STEP 11c: PTM-BDL Stratified {n_folds}-fold Cross-Validation        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    device = _get_device()
    print(f"  Device: {device}")

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    dataset = ResistanceDataset(dataset_path, features_dir)
    df = dataset.df
    n_total = len(dataset)
    n_res = int(df["resistance_label"].sum())
    n_sens = n_total - n_res
    print(f"  Dataset: {n_total} samples ({n_res} resistant, {n_sens} sensitive)")
    if "target_protein" in df.columns:
        print(f"  Target proteins: {dict(df['target_protein'].value_counts())}")
    strat = _get_stratification_labels(df)
    seed = cfg["training"]["seed"]
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    all_preds = {}
    ig_per_fold = []  # list of dicts from _run_ig_per_mod_type
    t0 = time.time()

    for fold_i, (trval_idx, test_idx) in enumerate(
            kfold.split(np.zeros(n_total), strat)):
        print(f"\n  {'=' * 60}")
        print(f"  FOLD {fold_i + 1}/{n_folds}")
        print(f"  {'=' * 60}")
        trval_labels = df["resistance_label"].values[trval_idx].astype(int)
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                         random_state=seed + fold_i)
            tr_sub, va_sub = next(sss.split(np.zeros(len(trval_idx)), trval_labels))
        except ValueError:
            np.random.seed(seed + fold_i)
            perm = np.random.permutation(len(trval_idx))
            v = max(1, int(0.15 * len(trval_idx)))
            va_sub, tr_sub = perm[:v], perm[v:]
        train_idx = trval_idx[tr_sub]
        val_idx = trval_idx[va_sub]
        test_labels = df["resistance_label"].values[test_idx]
        n_test_sens = int((test_labels == 0).sum())
        n_test_res = int((test_labels == 1).sum())
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | "
              f"Test: {len(test_idx)} ({n_test_sens} sens, {n_test_res} res)")

        fold = {"fold": fold_i + 1,
                "n_train": len(train_idx), "n_val": len(val_idx),
                "n_test": len(test_idx),
                "n_test_sensitive": n_test_sens,
                "n_test_resistant": n_test_res}

        # ── Full model ──────────────────────────────────────────────────
        print(f"\n  Training FULL PTM-BDL model...")
        full_model, full_val, full_ep = _train_fold(
            dataset_path, features_dir, train_idx, val_idx, device,
            ablation_mode="full", seed=seed)
        full_test, full_probs, full_ic50 = _evaluate_fold(
            full_model, dataset, test_idx, device)
        fold["full"] = {"val_metrics": full_val, "test_metrics": full_test,
                        "epochs": full_ep}
        print(f"    Full: BAcc={full_test['balanced_acc']:.3f}, "
              f"AUROC={full_test.get('auroc', 0):.3f}, "
              f"RMSE={full_test.get('rmse', 0):.3f}")
        for i, idx in enumerate(test_idx):
            all_preds[int(idx)] = {
                "prob": full_probs[i], "ic50_pred": full_ic50[i],
                "label": float(df["resistance_label"].values[idx]),
                "ic50_true": float(df["ln_ic50"].values[idx]),
                "fold": fold_i + 1,
                "drug": str(df.iloc[idx].get("drug_name", "unknown")),
                "target_protein": str(df.iloc[idx].get("target_protein", "EGFR")),
            }

        # ── PTM ablation per fold ───────────────────────────────────────
        if run_ablation:
            print(f"  Training NO_PTM ablation...")
            noptm_model, noptm_val, noptm_ep = _train_fold(
                dataset_path, features_dir, train_idx, val_idx, device,
                ablation_mode="no_ptm", seed=seed)
            noptm_test, _, _ = _evaluate_fold(noptm_model, dataset, test_idx, device)
            fold["no_ptm"] = {"val_metrics": noptm_val, "test_metrics": noptm_test,
                              "epochs": noptm_ep}
            fold["ablation_delta"] = {
                "delta_bacc": full_test["balanced_acc"] - noptm_test["balanced_acc"],
                "delta_auroc": full_test.get("auroc", 0) - noptm_test.get("auroc", 0),
                "delta_rmse": full_test.get("rmse", 0) - noptm_test.get("rmse", 0),
            }
            print(f"    Δ AUROC={fold['ablation_delta']['delta_auroc']:+.3f}, "
                  f"Δ BAcc={fold['ablation_delta']['delta_bacc']:+.3f}")

        # ── Per-mod-type IG ────────────────────────────────────────────
        if run_ig:
            print(f"  Running per-mod-type IG (PTM-BDL, max 30 test samples)...")
            ig_dict = _run_ig_per_mod_type(
                full_model, dataset, test_idx.tolist()[:30], n_steps=20)
            ig_per_fold.append(ig_dict)
            for gene in ["EGFR", "ERBB2"]:
                if ig_dict[gene]["n"] > 0:
                    labels_ph = (PHOSPHO_LABELS_EGFR if gene == "EGFR"
                                 else PHOSPHO_LABELS_ERBB2)
                    labels_gl = (GLYCO_LABELS_EGFR if gene == "EGFR"
                                 else GLYCO_LABELS_ERBB2)
                    ph_top = int(np.argmax(ig_dict[gene]["phospho"]))
                    gl_top = int(np.argmax(ig_dict[gene]["glyco"]))
                    print(f"    {gene} (n={ig_dict[gene]['n']}): "
                          f"phospho top={labels_ph[ph_top]}, "
                          f"glyco top={labels_gl[gl_top]}")
            fold["ig"] = {
                gene: {
                    "phospho": ig_dict[gene]["phospho"].tolist(),
                    "glyco": ig_dict[gene]["glyco"].tolist(),
                    "all": ig_dict[gene]["all"].tolist(),
                    "n": ig_dict[gene]["n"],
                } for gene in ["EGFR", "ERBB2"]
            }
        fold_results.append(fold)

    total_elapsed = time.time() - t0
    print(f"\n  Total CV time: {total_elapsed / 60:.1f} min")

    # ══════════════════════════════════════════════════════════════════════
    # AGGREGATE
    # ══════════════════════════════════════════════════════════════════════

    print("\n  " + "=" * 60)
    print(f"  CV AGGREGATE — {n_folds} folds")
    print("  " + "=" * 60)
    full_baccs = [r["full"]["test_metrics"]["balanced_acc"] for r in fold_results]
    full_aurocs = [r["full"]["test_metrics"].get("auroc", 0) for r in fold_results]
    full_rmses = [r["full"]["test_metrics"].get("rmse", 0) for r in fold_results]
    full_rs = [r["full"]["test_metrics"].get("pearson_r", 0) for r in fold_results]
    metrics_summary = {}
    for name, vals in [("BAcc", full_baccs), ("AUROC", full_aurocs),
                       ("RMSE", full_rmses), ("Pearson_R", full_rs)]:
        m = float(np.mean(vals))
        s = float(np.std(vals))
        ci = 1.96 * s / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
        metrics_summary[name] = {"mean": round(m, 4), "std": round(s, 4),
                                 "ci95": round(ci, 4),
                                 "per_fold": [round(v, 4) for v in vals]}
        print(f"    {name:10s}: {m:.4f} ± {s:.4f}  "
              f"(95% CI: [{m - ci:.4f}, {m + ci:.4f}])")

    ablation_summary = {}
    if run_ablation:
        delta_baccs = [r["ablation_delta"]["delta_bacc"] for r in fold_results]
        delta_aurocs = [r["ablation_delta"]["delta_auroc"] for r in fold_results]
        noptm_baccs = [r["no_ptm"]["test_metrics"]["balanced_acc"]
                       for r in fold_results]
        noptm_aurocs = [r["no_ptm"]["test_metrics"].get("auroc", 0)
                        for r in fold_results]
        t_b, p_b = scipy_stats.ttest_rel(full_baccs, noptm_baccs) if n_folds >= 2 else (0, 1)
        t_a, p_a = scipy_stats.ttest_rel(full_aurocs, noptm_aurocs) if n_folds >= 2 else (0, 1)
        ablation_summary = {
            "delta_bacc": {"mean": round(float(np.mean(delta_baccs)), 4),
                           "std": round(float(np.std(delta_baccs)), 4),
                           "p_value": round(float(p_b), 4),
                           "n_positive_folds": sum(1 for d in delta_baccs if d > 0)},
            "delta_auroc": {"mean": round(float(np.mean(delta_aurocs)), 4),
                            "std": round(float(np.std(delta_aurocs)), 4),
                            "p_value": round(float(p_a), 4),
                            "n_positive_folds": sum(1 for d in delta_aurocs if d > 0)},
        }
        print(f"\n  PTM ablation across folds:")
        print(f"    Δ BAcc  : {ablation_summary['delta_bacc']['mean']:+.4f} ± "
              f"{ablation_summary['delta_bacc']['std']:.4f}  "
              f"(p={ablation_summary['delta_bacc']['p_value']:.4f}, paired t)")
        print(f"    Δ AUROC : {ablation_summary['delta_auroc']['mean']:+.4f} ± "
              f"{ablation_summary['delta_auroc']['std']:.4f}  "
              f"(p={ablation_summary['delta_auroc']['p_value']:.4f}, paired t)")
        print(f"    Folds where PTM helps BAcc:  "
              f"{ablation_summary['delta_bacc']['n_positive_folds']}/{n_folds}")
        print(f"    Folds where PTM helps AUROC: "
              f"{ablation_summary['delta_auroc']['n_positive_folds']}/{n_folds}")

    # ── Per-mod-type IG aggregation ────────────────────────────────────
    ig_summary = {}
    if run_ig and ig_per_fold:
        def _stack_means(gene, mod):
            arrs = [f[gene][mod] for f in ig_per_fold if f[gene]["n"] > 0]
            return np.mean(arrs, axis=0) if arrs else np.zeros(
                12 if mod != "all" else 24)

        egfr_ph = _stack_means("EGFR", "phospho")
        egfr_gl = _stack_means("EGFR", "glyco")
        erbb2_ph = _stack_means("ERBB2", "phospho")
        erbb2_gl = _stack_means("ERBB2", "glyco")

        egfr_ph_top = int(np.argmax(egfr_ph))
        erbb2_ph_top = int(np.argmax(erbb2_ph))
        egfr_gl_top = int(np.argmax(egfr_gl))
        erbb2_gl_masked = erbb2_gl.copy()
        erbb2_gl_masked[7:] = -np.inf
        erbb2_gl_top = int(np.argmax(erbb2_gl_masked)) if np.isfinite(
            erbb2_gl_masked.max()) else int(np.argmax(erbb2_gl))

        homology_phospho = (egfr_ph_top == GRB2_PHOSPHO_INDEX
                            and erbb2_ph_top == GRB2_PHOSPHO_INDEX)
        homology_glyco = (egfr_gl_top == EGFR_N528_INDEX
                          and erbb2_gl_top == ERBB2_N530_INDEX)
        print(f"\n  Per-mod-type IG homology:")
        print(f"    Phospho (slot {GRB2_PHOSPHO_INDEX}): "
              f"EGFR={PHOSPHO_LABELS_EGFR[egfr_ph_top]}, "
              f"ERBB2={PHOSPHO_LABELS_ERBB2[erbb2_ph_top]}  "
              f"{'✓' if homology_phospho else '✗'}")
        print(f"    Glyco   (N528↔N530): "
              f"EGFR={GLYCO_LABELS_EGFR[egfr_gl_top]}, "
              f"ERBB2={GLYCO_LABELS_ERBB2[erbb2_gl_top]}  "
              f"{'✓' if homology_glyco else '✗'}")

        ig_summary = {
            "egfr": {
                "phospho_mean_importance": egfr_ph.tolist(),
                "glyco_mean_importance": egfr_gl.tolist(),
                "phospho_top_site": PHOSPHO_LABELS_EGFR[egfr_ph_top],
                "glyco_top_site": GLYCO_LABELS_EGFR[egfr_gl_top],
            },
            "erbb2": {
                "phospho_mean_importance": erbb2_ph.tolist(),
                "glyco_mean_importance": erbb2_gl.tolist(),
                "phospho_top_site": PHOSPHO_LABELS_ERBB2[erbb2_ph_top],
                "glyco_top_site": GLYCO_LABELS_ERBB2[erbb2_gl_top],
            },
            "homology_phospho_concordant": bool(homology_phospho),
            "homology_glyco_concordant": bool(homology_glyco),
            "phospho_labels_egfr": PHOSPHO_LABELS_EGFR,
            "phospho_labels_erbb2": PHOSPHO_LABELS_ERBB2,
            "glyco_labels_egfr": GLYCO_LABELS_EGFR,
            "glyco_labels_erbb2": GLYCO_LABELS_ERBB2,
        }

    # ── Per-drug & per-gene aggregation ────────────────────────────────
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    drug_summary = {}
    gene_summary = {}
    by_drug = defaultdict(lambda: {"probs": [], "labels": [],
                                   "ic50_true": [], "ic50_pred": []})
    by_gene = defaultdict(lambda: {"probs": [], "labels": [],
                                   "ic50_true": [], "ic50_pred": []})
    for idx, p in all_preds.items():
        by_drug[p["drug"]]["probs"].append(p["prob"])
        by_drug[p["drug"]]["labels"].append(p["label"])
        by_drug[p["drug"]]["ic50_true"].append(p["ic50_true"])
        by_drug[p["drug"]]["ic50_pred"].append(p["ic50_pred"])
        by_gene[p["target_protein"]]["probs"].append(p["prob"])
        by_gene[p["target_protein"]]["labels"].append(p["label"])
        by_gene[p["target_protein"]]["ic50_true"].append(p["ic50_true"])
        by_gene[p["target_protein"]]["ic50_pred"].append(p["ic50_pred"])

    # Load optimal threshold (fallback to 0.5 if not available)
    _thr_path = MODEL_DIR / "optimal_threshold.json"
    if _thr_path.exists():
        with open(_thr_path) as _f:
            _resist_thr = float(json.load(_f).get("optimal_threshold", 0.5))
    else:
        _resist_thr = 0.5

    def _summarize(d):
        probs = np.array(d["probs"])
        labels = np.array(d["labels"])
        preds = (probs > _resist_thr).astype(float)
        ic50_t = np.array(d["ic50_true"])
        ic50_p = np.array(d["ic50_pred"])
        out = {"n": len(probs), "n_sensitive": int((labels == 0).sum())}
        if len(set(labels)) >= 2:
            out["bacc"] = round(float(balanced_accuracy_score(labels, preds)), 4)
            try:
                out["auroc"] = round(float(roc_auc_score(labels, probs)), 4)
            except Exception:
                out["auroc"] = 0.0
            try:
                from sklearn.metrics import average_precision_score
                out["auprc_sensitive"] = round(float(
                    average_precision_score(1 - labels, 1 - probs)), 4)
            except Exception:
                out["auprc_sensitive"] = 0.0
        if len(ic50_t) > 2:
            out["rmse"] = round(float(np.sqrt(((ic50_t - ic50_p) ** 2).mean())), 4)
        return out

    for drug in sorted(by_drug.keys()):
        drug_summary[drug] = _summarize(by_drug[drug])
    for gene in sorted(by_gene.keys()):
        gene_summary[gene] = _summarize(by_gene[gene])

    if drug_summary:
        print("\n  Per-drug (aggregated across folds):")
        for d, m in drug_summary.items():
            print(f"    {d:15s}: n={m['n']}, "
                  f"BAcc={m.get('bacc', 0):.3f}, "
                  f"AUROC={m.get('auroc', 0):.3f}, "
                  f"RMSE={m.get('rmse', 0):.3f}")
    if gene_summary:
        print("\n  Per-target-protein (aggregated across folds):")
        for g, m in gene_summary.items():
            print(f"    {g:6s}: n={m['n']}, "
                  f"BAcc={m.get('bacc', 0):.3f}, "
                  f"AUROC={m.get('auroc', 0):.3f}, "
                  f"RMSE={m.get('rmse', 0):.3f}")

    # ── Save ───────────────────────────────────────────────────────────
    cv_results = {
        "n_folds": n_folds, "n_samples": n_total,
        "n_sensitive": n_sens, "n_resistant": n_res,
        "seed": seed, "total_time_minutes": round(total_elapsed / 60, 1),
        "metrics_summary": metrics_summary,
        "ablation_summary": ablation_summary,
        "ig_summary": ig_summary,
        "drug_summary": drug_summary,
        "gene_summary": gene_summary,
        "per_fold": fold_results,
    }
    out_path = RESULTS_DIR / "crossval_results.json"
    with open(out_path, "w") as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"\n  ✓ Results saved: {out_path}")

    # ── Figure: per-mod-type IG bar plots ──────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if ig_summary:
            fig, axes = plt.subplots(2, 2, figsize=(15, 9))
            for ax, (gene, mod, vec, labels) in zip(
                    axes.flat,
                    [("EGFR", "phospho", ig_summary["egfr"]["phospho_mean_importance"],
                      PHOSPHO_LABELS_EGFR),
                     ("EGFR", "glyco", ig_summary["egfr"]["glyco_mean_importance"],
                      GLYCO_LABELS_EGFR),
                     ("ERBB2", "phospho", ig_summary["erbb2"]["phospho_mean_importance"],
                      PHOSPHO_LABELS_ERBB2),
                     ("ERBB2", "glyco", ig_summary["erbb2"]["glyco_mean_importance"],
                      GLYCO_LABELS_ERBB2)],
            ):
                vec = np.asarray(vec)
                order = np.argsort(-vec)
                ax.bar(range(12), vec[order], color=(
                    "tab:blue" if mod == "phospho" else "tab:orange"))
                ax.set_xticks(range(12))
                ax.set_xticklabels([labels[i].split("(")[0] for i in order],
                                   rotation=45, ha="right", fontsize=8)
                ax.set_title(f"{gene} {mod} (mean over folds)")
                ax.set_ylabel("|attribution|")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "crossval_ig_per_mod_type.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Figure saved: "
                  f"{FIGURES_DIR / 'crossval_ig_per_mod_type.png'}")
    except Exception as e:
        print(f"  ⚠ Could not generate figure: {e}")

    return cv_results


if __name__ == "__main__":
    run_crossval(n_folds=5, run_ablation=True, run_ig=True)
