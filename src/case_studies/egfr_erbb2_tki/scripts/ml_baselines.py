#!/usr/bin/env python3
"""
EGFR/ERBB2 TKI Case Study — ML Baseline Benchmarking (FAIR PROTOCOL v2).

═══════════════════════════════════════════════════════════════════════════
FAIRNESS FIXES APPLIED (7 issues resolved):
═══════════════════════════════════════════════════════════════════════════

  FIX 1: RF max_depth CAPPED at 15 (was: unlimited → memorization)
  FIX 2: GridSearchCV for ALL methods including RF & XGBoost
  FIX 3: Train on train_idx ONLY (was: train+val combined = 15% more data)
  FIX 4: Each method computes its OWN Youden's J threshold on val set
  FIX 5: Same features as DL model (pooled embeddings + PTM)
  FIX 6: Separate classifiers per method (no shared LogisticRegression)
  FIX 7: K-fold CV evaluation alongside single-split

INPUT FEATURES (concatenated, flat vector per sample):
  • ESM-2 pooled:     1280-d (protein sequence)
  • GearNet pooled:    512-d (3D structure)
  • ChemBERTa pooled:  384-d (drug chemistry)
  • PTM features:      all ptm_*/delta_ptm_*/glyco_slot*/delta_glyco_slot*

BASELINES:
  1. Random Forest (GridSearchCV: n_estimators, max_depth, min_samples_leaf)
  2. XGBoost (GridSearchCV: n_estimators, max_depth, learning_rate)
  3. Ridge Regression + Logistic Regression (GridSearchCV: alpha / C)
  4. Elastic Net + L1-Logistic Regression (GridSearchCV: alpha, l1_ratio / C)

REFERENCES:
  Baptista et al., Brief Bioinform 2021 (PMID 33169146) — DRP baselines
  Chen & Guestrin, KDD 2016 — XGBoost
  Yang et al., Brief Bioinform 2024 — GraTransDRP & DRP benchmarks
"""
# ── Fork-safety: MUST be set before any sklearn/joblib/xgboost imports ──────
# On macOS (especially Apple Silicon), the default loky "fork" start method
# inherits corrupted OpenMP state from the parent, causing XGBoost segfaults.
# "loky_init_main" uses fork+exec which re-initialises the child safely.
# OMP_NUM_THREADS=1 prevents OpenMP from spawning a thread pool that gets
# corrupted when loky forks workers for n_jobs=-1 estimators (e.g. RF).
# Without this, XGBoost (which links libomp) segfaults even with n_jobs=1.
import os
os.environ.setdefault("LOKY_START_METHOD", "loky_init_main")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config
from src.ptm_bdl.evaluation.baselines import (
    load_pooled_features, train_and_evaluate_baseline,
    run_kfold_baselines, compute_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_per_drug_metrics(df_test, y_pred_ic50, y_prob_cls, threshold=0.5):
    """Compute per-drug PCC and AUROC."""
    per_drug = {}
    for drug in sorted(df_test["drug_name"].unique()):
        mask = df_test["drug_name"].values == drug
        if mask.sum() < 3:
            continue
        y_t = df_test["ln_ic50"].values[mask]
        y_p = y_pred_ic50[mask]
        y_cls = df_test["resistance_label"].values[mask]
        y_prb = y_prob_cls[mask] if y_prob_cls is not None else None

        drug_met = {"n_samples": int(mask.sum())}
        if np.std(y_p) > 1e-8:
            drug_met["pearson_r"] = float(np.corrcoef(y_t, y_p)[0, 1])
        else:
            drug_met["pearson_r"] = 0.0
        drug_met["rmse"] = float(np.sqrt(mean_squared_error(y_t, y_p)))

        if len(set(y_cls)) > 1 and y_prb is not None:
            drug_met["auroc"] = float(roc_auc_score(y_cls, y_prb))
        else:
            drug_met["auroc"] = 0.0

        per_drug[drug] = drug_met
    return per_drug


def main():
    """Train ML baselines under FAIR protocol and compare."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EGFR/ERBB2 TKI — ML Baselines (FAIR PROTOCOL v2)         ║")
    print("║  All 7 fairness fixes applied                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    seed = cfg["training"]["seed"]
    np.random.seed(seed)

    # ── Load split indices (SAME as PTM-BDL training) ─────────────────────
    split_path = MODEL_DIR / "split_indices.json"
    if not split_path.exists():
        print(f"  ✗ split_indices.json not found at {split_path}")
        print(f"    Run train.py first to create the split.")
        sys.exit(1)

    with open(split_path) as f:
        split = json.load(f)
    train_idx = np.array(split["train_idx"])
    val_idx = np.array(split["val_idx"])
    test_idx = np.array(split["test_idx"])

    # FIX 3: Train on train_idx ONLY (not train+val)
    print(f"  FIX 3: Train={len(train_idx)}, Val={len(val_idx)}, "
          f"Test={len(test_idx)} (train ONLY, no val merging)")

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    if not dataset_path.exists():
        print(f"  ✗ Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"  Dataset: {len(df)} samples")

    # ── FIX 5: Build pooled feature matrix (same as DL model) ─────────────
    print(f"\n  FIX 5: Building pooled feature matrix (same as DL model)...")
    t0 = time.time()
    X = load_pooled_features(df, features_dir)
    elapsed = time.time() - t0
    print(f"  ✓ Feature matrix: {X.shape} ({elapsed:.1f}s)")

    y_ic50 = df["ln_ic50"].values.astype(np.float32)
    y_cls = df["resistance_label"].values.astype(np.int32)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    y_train_ic50 = y_ic50[train_idx]
    y_val_cls = y_cls[val_idx]
    y_test_ic50 = y_ic50[test_idx]
    y_train_cls = y_cls[train_idx]
    y_test_cls = y_cls[test_idx]

    n_sens_test = int((y_test_cls == 0).sum())
    n_res_test = int((y_test_cls == 1).sum())
    print(f"  Test set: {len(test_idx)} samples "
          f"({n_res_test} resistant, {n_sens_test} sensitive)")

    df_test = df.iloc[test_idx].reset_index(drop=True)

    # ── Train all baselines (single-split) ────────────────────────────────
    methods = ["random_forest", "xgboost", "ridge", "elastic_net"]
    results = {}

    for method in methods:
        print(f"\n  ── {method.upper()} ──")
        print(f"    FIX 1+2: GridSearchCV with capped hyperparameters...")
        t0 = time.time()

        result = train_and_evaluate_baseline(
            X_train, y_train_ic50, y_train_cls,
            X_val, y_val_cls,
            X_test, y_test_ic50, y_test_cls,
            method=method, random_state=seed,
        )
        elapsed = time.time() - t0

        if "error" in result:
            print(f"    ⚠ {result['error']}")
            continue

        # FIX 4: Report method-specific threshold
        print(f"    FIX 4: Threshold (Youden's J on val): "
              f"{result.get('threshold_used', 0.5):.4f}")
        print(f"    Best reg params: {result.get('best_reg_params', {})}")
        print(f"    Best cls params: {result.get('best_cls_params', {})}")

        # Per-drug metrics
        preds = result.get("predictions", {})
        per_drug = {}
        if preds:
            y_pred_reg = np.array(preds["y_pred_reg"])
            y_prob_cls_arr = np.array(preds["y_prob_cls"])
            per_drug = compute_per_drug_metrics(
                df_test, y_pred_reg, y_prob_cls_arr,
                result.get("threshold_used", 0.5))

        print(f"    PCC={result.get('pearson_r', 0):.3f} | "
              f"RMSE={result.get('rmse', 0):.3f} | "
              f"AUROC={result.get('auroc', 0):.3f} | "
              f"AUPRC-s={result.get('auprc_sensitive', 0):.3f} | "
              f"BAcc={result.get('balanced_acc', 0):.3f} | "
              f"({elapsed:.1f}s)")

        # Store results
        result_save = {k: v for k, v in result.items() if k != "predictions"}
        result_save["per_drug"] = per_drug
        result_save["feature_dim"] = int(X.shape[1])
        result_save["n_train"] = int(len(train_idx))
        result_save["n_test"] = int(len(test_idx))
        result_save["training_time_seconds"] = round(elapsed, 2)
        results[method] = result_save

        # Cache predictions for statistical tests
        pred_dir = RESULTS_DIR / "baseline_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        if preds:
            np.savez(pred_dir / f"{method}.npz",
                     y_pred_ic50=y_pred_reg,
                     y_prob_cls=y_prob_cls_arr)

    # ── Load our model's test metrics for comparison ──────────────────────
    eval_path = RESULTS_DIR / "evaluation_report.json"
    pred_cache_path = RESULTS_DIR / "test_predictions.npz"
    our_metrics = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        reg = eval_report.get("regression", {})
        cls_rep = eval_report.get("classification", {})

        our_auprc_sens = 0.0
        if pred_cache_path.exists():
            from sklearn.metrics import average_precision_score
            cached = np.load(pred_cache_path)
            c_true = cached["y_true_cls"]
            c_prob = cached["y_prob_cls"]
            if len(set(c_true)) > 1:
                our_auprc_sens = float(
                    average_precision_score(1 - c_true, 1 - c_prob))

        our_metrics = {
            "pearson_r": reg.get("pearson_r", 0),
            "rmse": reg.get("rmse", 0),
            "auroc": cls_rep.get("auroc", 0),
            "auprc_sensitive": our_auprc_sens,
            "balanced_acc": cls_rep.get("balanced_accuracy", 0),
        }
        print(f"\n  Our model: PCC={our_metrics['pearson_r']:.3f}, "
              f"AUROC={our_metrics['auroc']:.3f}")

    # ── FIX 7: K-fold CV evaluation ───────────────────────────────────────
    print(f"\n  ── FIX 7: 5-Fold Cross-Validation (identical to DL CV) ──")
    t0 = time.time()
    cv_results = run_kfold_baselines(
        X, y_ic50, y_cls, n_folds=5, methods=methods, random_state=seed)
    cv_elapsed = time.time() - t0
    print(f"  ✓ CV complete ({cv_elapsed:.0f}s)")

    print(f"\n  {'='*85}")
    print(f"  {'Method':<16s} | {'PCC':>12s} | {'RMSE':>12s} | "
          f"{'AUROC':>12s} | {'AUPRC-s':>12s} | {'BAcc':>12s}")
    print(f"  {'-'*85}")
    for method in methods:
        if method not in cv_results or "error" in cv_results[method]:
            continue
        m = cv_results[method]
        pcc = m.get("pearson_r", {})
        rmse = m.get("rmse", {})
        auroc = m.get("auroc", {})
        auprc = m.get("auprc_sensitive", {})
        bacc = m.get("balanced_acc", {})
        print(f"  {method:<16s} | "
              f"{pcc.get('mean',0):5.3f}±{pcc.get('std',0):.3f} | "
              f"{rmse.get('mean',0):5.3f}±{rmse.get('std',0):.3f} | "
              f"{auroc.get('mean',0):5.3f}±{auroc.get('std',0):.3f} | "
              f"{auprc.get('mean',0):5.3f}±{auprc.get('std',0):.3f} | "
              f"{bacc.get('mean',0):5.3f}±{bacc.get('std',0):.3f}")
    print(f"  {'='*85}")

    # ── Single-split comparison table ─────────────────────────────────────
    print(f"\n  Single-split results:")
    print(f"  {'='*85}")
    print(f"  {'Method':<16s} | {'PCC':>6s} | {'RMSE':>6s} | "
          f"{'AUROC':>6s} | {'AUPRC-s':>7s} | {'BAcc':>6s} | {'Threshold':>9s}")
    print(f"  {'-'*85}")
    for name, res in results.items():
        print(f"  {name:<16s} | "
              f"{res.get('pearson_r', 0):6.3f} | {res.get('rmse', 0):6.3f} | "
              f"{res.get('auroc', 0):6.3f} | "
              f"{res.get('auprc_sensitive', 0):7.3f} | "
              f"{res.get('balanced_acc', 0):6.3f} | "
              f"{res.get('threshold_used', 0.5):9.4f}")
    if our_metrics:
        print(f"  {'Our PTM-BDL':<16s} | "
              f"{our_metrics['pearson_r']:6.3f} | {our_metrics['rmse']:6.3f} | "
              f"{our_metrics['auroc']:6.3f} | "
              f"{our_metrics.get('auprc_sensitive', 0):7.3f} | "
              f"{our_metrics['balanced_acc']:6.3f} | {'(model)':>9s}")
    print(f"  {'='*85}")

    # ── Save ──────────────────────────────────────────────────────────────
    report = {
        "case_study": CASE_STUDY,
        "fairness_protocol": "v2 — all 7 issues fixed",
        "fixes_applied": [
            "FIX 1: RF max_depth capped at 15 (no unlimited depth)",
            "FIX 2: GridSearchCV for ALL methods (RF, XGBoost, Ridge, ElasticNet)",
            "FIX 3: Train on train_idx only (not train+val combined)",
            "FIX 4: Each method computes own Youden's J threshold on val set",
            "FIX 5: Same pooled embedding features as DL model",
            "FIX 6: Separate classifiers per method",
            "FIX 7: 5-fold CV evaluation",
        ],
        "single_split": results,
        "cross_validation": {k: {kk: vv for kk, vv in v.items()
                                  if kk != "fold_details"}
                             for k, v in cv_results.items()},
        "feature_dim": int(X.shape[1]),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
    }
    if our_metrics:
        report["our_model_metrics"] = our_metrics

    out_path = RESULTS_DIR / "ml_baselines.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(RESULTS_DIR / "ml_baselines_cv_details.json", "w") as f:
        json.dump(cv_results, f, indent=2, default=str)

    print(f"\n  ✓ Saved: {out_path}")
    print(f"  ✓ Saved: {RESULTS_DIR / 'ml_baselines_cv_details.json'}")
    print(f"✓ Fair benchmarking complete!")


if __name__ == "__main__":
    main()
