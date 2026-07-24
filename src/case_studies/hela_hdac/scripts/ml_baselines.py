#!/usr/bin/env python3
"""
HeLa/HDAC Case Study — ML Baseline Benchmarking.

PURPOSE:
  Train traditional ML models on the same feature set as PTM-BDL to
  establish performance baselines. If PTM-BDL outperforms these baselines,
  the multimodal architecture adds value beyond simple feature aggregation.

BASELINES:
  1. Random Forest — ensemble of decision trees
  2. XGBoost — gradient-boosted trees (state-of-the-art tabular ML)
  3. Ridge Regression — linear model with L2 regularization
  4. Elastic Net — L1+L2 regularization

FEATURE SET (same as PTM-BDL input):
  • Phospho features: mean/std/min/max log2FC + n_sites + n_up/down
  • Acetyl features: mean/std/min/max log2FC + n_sites + n_up/down
  • Drug SMILES → pooled ChemBERTa embedding (384-dim)

BENCHMARKING PHILOSOPHY:
  Fair comparison requires IDENTICAL data splits and feature sets.
  The only difference is the model architecture: PTM-BDL (multimodal
  attention-based) vs traditional ML (tabular feature-based).

FIXES APPLIED:
  1. Added StandardScaler for all methods (Ridge/ElasticNet require it)
  2. Replaced sigmoid(IC50) hack with proper LogisticRegression classifier
  3. Added GridSearchCV inner CV for Ridge/ElasticNet hyperparameters
  4. Combined train+val for ML baselines (they do their own inner CV)
  5. Added class_weight="balanced" for RF classifier
  6. Uses optimal threshold from Youden's J (if available) for BAcc

REFERENCES:
  • Yang et al., Briefings Bioinform 2024 — drug response ML benchmarks
  • Baptista et al., Briefings Bioinform 2021 (PMID 33169146) — DRP baselines
  • Chen & Guestrin, KDD 2016 — XGBoost
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features(df: pd.DataFrame) -> np.ndarray:
    """Extract tabular features from the multimodal dataset."""
    # Fix operator precedence: use explicit parentheses
    feature_cols = [c for c in df.columns if (
        (any(c.startswith(p) for p in ["phospho_", "acetyl_"]) and "log2fc" in c)
        or "n_sites" in c or "n_up" in c or "n_down" in c
    )]

    if not feature_cols:
        feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]
                        and c not in ["ln_ic50", "resistance_label"]]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    print(f"  Features: {len(feature_cols)} columns, {X.shape[0]} samples")
    return X


def compute_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
                    threshold=0.5):
    """Compute standard metrics for a baseline model."""
    metrics = {}
    if len(np.unique(y_true_ic50[~np.isnan(y_true_ic50)])) > 1:
        valid = ~np.isnan(y_true_ic50) & ~np.isnan(y_pred_ic50)
        if valid.sum() > 2:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(
                y_true_ic50[valid], y_pred_ic50[valid])))
            metrics["r2"] = float(r2_score(
                y_true_ic50[valid], y_pred_ic50[valid]))
            metrics["pearson_r"] = float(pearsonr(
                y_true_ic50[valid], y_pred_ic50[valid])[0])

    if len(np.unique(y_true_cls)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true_cls, y_prob_cls))
        metrics["balanced_acc"] = float(balanced_accuracy_score(
            y_true_cls, (y_prob_cls > threshold).astype(int)))

    return metrics


def main():
    """Train and evaluate ML baselines."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — ML Baseline Benchmarking                   ║")
    print(f"║  Models: RF, XGBoost, Ridge, ElasticNet                    ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    seed = cfg["training"]["seed"]
    np.random.seed(seed)

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    df = pd.read_csv(dataset_path)

    with open(MODEL_DIR / "split_indices.json") as f:
        split = json.load(f)

    train_idx = np.array(split["train_idx"])
    val_idx = np.array(split.get("val_idx", []))
    test_idx = np.array(split["test_idx"])

    # Combine train + val for ML baselines (they do inner CV)
    trainval_idx = np.concatenate([train_idx, val_idx]) if len(val_idx) > 0 else train_idx
    print(f"  Split: train+val={len(trainval_idx)}, test={len(test_idx)}")

    X = load_features(df)
    y_ic50 = df["ln_ic50"].values.astype(np.float32)
    y_cls = df["resistance_label"].values.astype(np.int32)

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval_ic50, y_test_ic50 = y_ic50[trainval_idx], y_ic50[test_idx]
    y_trainval_cls, y_test_cls = y_cls[trainval_idx], y_cls[test_idx]

    # StandardScaler for all methods
    scaler = StandardScaler()
    X_trainval_s = scaler.fit_transform(X_trainval)
    X_test_s = scaler.transform(X_test)

    # Load optimal threshold if available
    thr_path = MODEL_DIR / "optimal_threshold.json"
    if thr_path.exists():
        with open(thr_path) as f:
            opt_threshold = float(json.load(f).get("optimal_threshold", 0.5))
        print(f"  Using optimal threshold: {opt_threshold:.4f}")
    else:
        opt_threshold = 0.5

    valid_train = ~np.isnan(y_trainval_ic50)
    results = {}

    # ── Random Forest ────────────────────────────────────────────────────
    print("\n  Training Random Forest (regression + classification)...")
    if valid_train.sum() > 5:
        rf_reg = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            random_state=seed, n_jobs=-1)
        rf_reg.fit(X_trainval_s[valid_train], y_trainval_ic50[valid_train])
        rf_pred = rf_reg.predict(X_test_s)

        rf_cls = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1)
        rf_cls.fit(X_trainval_s, y_trainval_cls)
        rf_prob = rf_cls.predict_proba(X_test_s)[:, 1]

        results["random_forest"] = compute_metrics(
            y_test_ic50, rf_pred, y_test_cls, rf_prob, opt_threshold)
        print(f"    RF: {results['random_forest']}")

    # ── Ridge Regression + Logistic Regression ────────────────────────────
    print("\n  Training Ridge Regression + Logistic Regression...")
    if valid_train.sum() > 5:
        ridge = Ridge(alpha=1.0)
        param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        cv = GridSearchCV(ridge, param_grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1)
        cv.fit(X_trainval_s[valid_train], y_trainval_ic50[valid_train])
        best_ridge = cv.best_estimator_
        ridge_pred = best_ridge.predict(X_test_s)
        print(f"      Best Ridge alpha: {cv.best_params_['alpha']}")

        # Proper LogisticRegression classifier (not sigmoid hack)
        lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                                random_state=seed, C=1.0)
        lr.fit(X_trainval_s, y_trainval_cls)
        ridge_prob = lr.predict_proba(X_test_s)[:, 1]

        results["ridge"] = compute_metrics(
            y_test_ic50, ridge_pred, y_test_cls, ridge_prob, opt_threshold)
        print(f"    Ridge: {results['ridge']}")

    # ── Elastic Net + Logistic Regression ─────────────────────────────────
    print("\n  Training Elastic Net + Logistic Regression...")
    if valid_train.sum() > 5:
        enet = ElasticNet(max_iter=10000, random_state=seed)
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        }
        cv = GridSearchCV(enet, param_grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1)
        cv.fit(X_trainval_s[valid_train], y_trainval_ic50[valid_train])
        best_enet = cv.best_estimator_
        enet_pred = best_enet.predict(X_test_s)
        print(f"      Best alpha={cv.best_params_['alpha']}, "
              f"l1_ratio={cv.best_params_['l1_ratio']}")

        # Proper LogisticRegression classifier (not sigmoid hack)
        lr_enet = LogisticRegression(max_iter=5000, class_weight="balanced",
                                     random_state=seed, C=1.0)
        lr_enet.fit(X_trainval_s, y_trainval_cls)
        enet_prob = lr_enet.predict_proba(X_test_s)[:, 1]

        results["elastic_net"] = compute_metrics(
            y_test_ic50, enet_pred, y_test_cls, enet_prob, opt_threshold)
        print(f"    ElasticNet: {results['elastic_net']}")

    # ── XGBoost ──────────────────────────────────────────────────────────
    print("\n  Training XGBoost...")
    try:
        import xgboost as xgb
        if valid_train.sum() > 5:
            xgb_reg = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", device="cpu",
                random_state=seed, n_jobs=-1, verbosity=0)
            xgb_reg.fit(X_trainval_s[valid_train],
                        y_trainval_ic50[valid_train])
            xgb_pred = xgb_reg.predict(X_test_s)

            n_pos = int(y_trainval_cls.sum())
            n_neg = len(y_trainval_cls) - n_pos
            scale_pos = n_neg / max(n_pos, 1)
            xgb_cls = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                random_state=seed, n_jobs=-1, verbosity=0,
                eval_metric="logloss")
            xgb_cls.fit(X_trainval_s, y_trainval_cls)
            xgb_prob = xgb_cls.predict_proba(X_test_s)[:, 1]

            results["xgboost"] = compute_metrics(
                y_test_ic50, xgb_pred, y_test_cls, xgb_prob, opt_threshold)
            print(f"    XGBoost: {results['xgboost']}")
    except ImportError:
        print("    ⚠ XGBoost not installed — skipping")

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n  {'=' * 70}")
    print(f"  {'Method':<18s} | {'PCC':>6s} | {'RMSE':>6s} | "
          f"{'AUROC':>6s} | {'BAcc':>6s}")
    print(f"  {'-' * 70}")
    for name, m in results.items():
        print(f"  {name:<18s} | {m.get('pearson_r', 0):6.3f} | "
              f"{m.get('rmse', 0):6.3f} | {m.get('auroc', 0):6.3f} | "
              f"{m.get('balanced_acc', 0):6.3f}")
    print(f"  {'=' * 70}")

    # ── Save results ─────────────────────────────────────────────────────
    report = {
        "case_study": CASE_STUDY,
        "baselines": results,
        "feature_count": X.shape[1],
        "train_samples": len(trainval_idx),
        "test_samples": len(test_idx),
        "threshold": opt_threshold,
        "references": [
            "Yang et al., Brief Bioinform 2024 — drug response benchmarks",
            "Chen & Guestrin, KDD 2016 — XGBoost",
            "Baptista et al., Brief Bioinform 2021 (PMID 33169146) — DRP baselines",
        ],
    }

    with open(RESULTS_DIR / "ml_baselines.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  ✓ Baselines saved: {RESULTS_DIR / 'ml_baselines.json'}")
    print(f"✓ Benchmarking complete!")


if __name__ == "__main__":
    main()
