#!/usr/bin/env python3
"""
K562/CML Case Study — ML Baseline Benchmarking (FAIR PROTOCOL v2).

═══════════════════════════════════════════════════════════════════════════
FAIRNESS FIXES APPLIED (7 issues resolved):
═══════════════════════════════════════════════════════════════════════════

  FIX 1: RF max_depth CAPPED at 15 (was: unlimited → memorization)
  FIX 2: GridSearchCV for ALL methods including RF & XGBoost
  FIX 3: Train on train_idx ONLY (was: train+val combined = 15% more data)
  FIX 4: Each method computes its OWN Youden's J threshold on val set
  FIX 5: Same features as DL model (pooled embeddings + PTM, not just PTM)
  FIX 6: Separate classifiers per method (no shared LogisticRegression)
  FIX 7: K-fold CV evaluation alongside single-split

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
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config
from src.ptm_bdl.evaluation.baselines import (
    load_pooled_features, train_and_evaluate_baseline,
    run_kfold_baselines, compute_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Published external method performances (for comparison table only)
EXTERNAL_BENCHMARKS = {
    "DIPK": {
        "pearson_r": 0.72, "rmse": None,
        "ref": "Liu et al., Brief Bioinform 2024 (PMID 38189543)",
        "features": "kinase activity + drug fingerprints",
    },
    "GraphDRP": {
        "pearson_r": 0.85, "rmse": 1.20,
        "ref": "Nguyen et al., Bioinformatics 2022 (PMID 34601570)",
        "features": "drug GNN + cell line gene expression",
    },
    "HiDRA": {
        "pearson_r": 0.89, "rmse": None,
        "ref": "Jin et al., PNAS 2021 (PMID 33658380)",
        "features": "hierarchical attention on gene expression",
    },
    "GraTransDRP": {
        "pearson_r": 0.91, "rmse": 0.98,
        "ref": "Yang et al., Brief Bioinform 2024",
        "features": "Transformer + GNN (drug graph + gene expression)",
    },
}


def main():
    """Train ML baselines under FAIR protocol and compare."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — ML Baselines (FAIR PROTOCOL v2)            ║")
    print(f"║  All 7 fairness fixes applied                              ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    seed = cfg["training"]["seed"]
    np.random.seed(seed)

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    df = pd.read_csv(dataset_path)
    print(f"  Dataset: {len(df)} samples")

    # ── Load split indices (SAME split as PTM-BDL) ────────────────────────
    with open(MODEL_DIR / "split_indices.json") as f:
        split = json.load(f)

    train_idx = np.array(split["train_idx"])
    val_idx = np.array(split.get("val_idx", []))
    test_idx = np.array(split["test_idx"])

    # FIX 3: Train on train_idx ONLY (not train+val)
    # PTM-BDL trains on train_idx and uses val_idx for early stopping.
    # ML baselines now use val_idx for threshold computation only.
    print(f"  FIX 3: Train={len(train_idx)}, Val={len(val_idx)}, "
          f"Test={len(test_idx)} (train ONLY, no val merging)")

    # ── FIX 5: Load same features as DL model ────────────────────────────
    print(f"\n  FIX 5: Building pooled feature matrix (same as DL model)...")
    t0 = time.time()
    X = load_pooled_features(df, features_dir)
    print(f"  ✓ Feature matrix: {X.shape} ({time.time()-t0:.1f}s)")

    y_ic50 = df["ln_ic50"].values.astype(np.float32)
    y_cls = df["resistance_label"].values.astype(np.int32)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx]) if len(val_idx) > 0 else np.empty((0, X.shape[1]))
    X_test = scaler.transform(X[test_idx])

    y_train_ic50 = y_ic50[train_idx]
    y_val_cls = y_cls[val_idx] if len(val_idx) > 0 else np.array([])
    y_test_ic50 = y_ic50[test_idx]
    y_train_cls = y_cls[train_idx]
    y_test_cls = y_cls[test_idx]

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
        print(f"    PCC={result.get('pearson_r', 0):.3f} | "
              f"RMSE={result.get('rmse', 0):.3f} | "
              f"AUROC={result.get('auroc', 0):.3f} | "
              f"BAcc={result.get('balanced_acc', 0):.3f} | "
              f"({elapsed:.1f}s)")

        # Remove predictions for JSON serialization
        result_save = {k: v for k, v in result.items() if k != "predictions"}
        results[method] = result_save

        # Cache predictions for statistical tests
        pred_dir = RESULTS_DIR / "baseline_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        preds = result.get("predictions", {})
        if preds:
            np.savez(pred_dir / f"{method}.npz",
                     y_pred_ic50=np.array(preds["y_pred_reg"]),
                     y_prob_cls=np.array(preds["y_prob_cls"]))

    # ── FIX 7: K-fold CV evaluation ───────────────────────────────────────
    print(f"\n  ── FIX 7: 5-Fold Cross-Validation (identical to DL CV) ──")
    t0 = time.time()
    cv_results = run_kfold_baselines(
        X, y_ic50, y_cls, n_folds=5, methods=methods, random_state=seed)
    cv_elapsed = time.time() - t0
    print(f"  ✓ CV complete ({cv_elapsed:.0f}s)")

    print(f"\n  {'='*80}")
    print(f"  {'Method':<16s} | {'PCC':>12s} | {'RMSE':>12s} | "
          f"{'AUROC':>12s} | {'BAcc':>12s}")
    print(f"  {'-'*80}")
    for method in methods:
        if method not in cv_results or "error" in cv_results[method]:
            continue
        m = cv_results[method]
        pcc = m.get("pearson_r", {})
        rmse = m.get("rmse", {})
        auroc = m.get("auroc", {})
        bacc = m.get("balanced_acc", {})
        print(f"  {method:<16s} | "
              f"{pcc.get('mean',0):5.3f}±{pcc.get('std',0):.3f} | "
              f"{rmse.get('mean',0):5.3f}±{rmse.get('std',0):.3f} | "
              f"{auroc.get('mean',0):5.3f}±{auroc.get('std',0):.3f} | "
              f"{bacc.get('mean',0):5.3f}±{bacc.get('std',0):.3f}")
    print(f"  {'='*80}")

    # ── Single-split comparison table ─────────────────────────────────────
    print(f"\n  Single-split results (for reference):")
    print(f"  {'='*70}")
    print(f"  {'Method':<16s} | {'PCC':>6s} | {'RMSE':>6s} | "
          f"{'AUROC':>6s} | {'BAcc':>6s} | {'Threshold':>9s}")
    print(f"  {'-'*70}")
    for name, m in results.items():
        print(f"  {name:<16s} | {m.get('pearson_r', 0):6.3f} | "
              f"{m.get('rmse', 0):6.3f} | {m.get('auroc', 0):6.3f} | "
              f"{m.get('balanced_acc', 0):6.3f} | "
              f"{m.get('threshold_used', 0.5):9.4f}")
    print(f"  {'='*70}")

    # ── External benchmark comparison ─────────────────────────────────────
    print("\n  External method comparison (published performances):")
    print("  " + "-" * 60)
    print(f"  {'Method':15s} {'Pearson R':>10s} {'RMSE':>8s} {'Reference'}")
    print("  " + "-" * 60)
    for name, info in EXTERNAL_BENCHMARKS.items():
        r_str = f"{info['pearson_r']:.3f}" if info['pearson_r'] else "N/A"
        rmse_str = f"{info['rmse']:.3f}" if info['rmse'] else "N/A"
        print(f"  {name:15s} {r_str:>10s} {rmse_str:>8s}  {info['ref']}")

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
        "external_benchmarks": EXTERNAL_BENCHMARKS,
        "feature_dim": int(X.shape[1]),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
    }
    with open(RESULTS_DIR / "ml_baselines.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save full CV fold details separately
    with open(RESULTS_DIR / "ml_baselines_cv_details.json", "w") as f:
        json.dump(cv_results, f, indent=2, default=str)

    print(f"\n  ✓ Saved: {RESULTS_DIR / 'ml_baselines.json'}")
    print(f"  ✓ Saved: {RESULTS_DIR / 'ml_baselines_cv_details.json'}")
    print(f"✓ Fair benchmarking complete!")


if __name__ == "__main__":
    main()
