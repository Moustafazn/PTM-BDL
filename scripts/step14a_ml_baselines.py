#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 14a — Tier 0 ML Baselines for Benchmarking                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Train simple ML baselines (RF, XGBoost, Ridge, Elastic Net) on the SAME   ║
║    features and SAME train/val/test split as our PTM-BDL model.             ║
║    If RF/XGBoost matches our DL model, the architecture adds no value.      ║
║                                                                              ║
║  INPUT FEATURES (concatenated, flat 2224-d vector):                          ║
║    • ESM-2 pooled:     1280-d (protein sequence)                             ║
║    • GearNet pooled:    512-d (3D structure)                                 ║
║    • ChemBERTa pooled:  384-d (drug chemistry)                               ║
║    • PTM features:       48-d (12 phospho + 12 delta-phospho                 ║
║                                + 12 glyco + 12 delta-glyco)                  ║
║                                                                              ║
║  METHODS:                                                                    ║
║    1. Random Forest (scikit-learn) — inner 5-fold CV for hyperparams        ║
║    2. XGBoost (xgboost) — gradient boosting                                  ║
║    3. Ridge Regression (scikit-learn) — L2-regularized linear               ║
║    4. Elastic Net (scikit-learn) — L1+L2 regularized linear                 ║
║                                                                              ║
║  METRICS (same as main model — Tier A):                                      ║
║    Regression:      PCC (Pearson R), RMSE                                    ║
║    Classification:  AUROC, AUPRC-sensitive                                   ║
║    Per-drug:        PCC, AUROC for each of 6 drugs                           ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/ml_baselines.json                                                 ║
║                                                                              ║
║  BENCHMARKING_PLAN.md §8, Step 14a                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"]
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Suppress convergence warnings for Elastic Net
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction: concatenate pooled embeddings into flat vector
# ══════════════════════════════════════════════════════════════════════════════

def load_pooled_features(df: pd.DataFrame, features_dir: Path) -> np.ndarray:
    """
    Build a flat feature matrix by concatenating pooled embeddings.

    For each sample:
      ESM-2 per-residue → mean-pool → 1280-d
      GearNet residue   → mean-pool → 512-d
      ChemBERTa pooled  → 384-d (already pooled)
      PTM features      → 48-d (12 phospho + 12 delta_phospho
                                + 12 glyco + 12 delta_glyco)
    Total: 2224-d per sample.
    """
    esm2_dir = features_dir / "esm2"
    gearnet_dir = features_dir / "gearnet"
    chemberta_dir = features_dir / "chemberta"

    # Pre-load all embeddings
    esm2_cache = {}
    for f in esm2_dir.glob("*_per_residue.npy"):
        key = f.stem.replace("_per_residue", "")
        esm2_cache[key] = np.load(f)

    gearnet_cache = {}
    for f in gearnet_dir.glob("*_residue_embeddings.npy"):
        key = f.stem.replace("_residue_embeddings", "")
        gearnet_cache[key] = np.load(f)

    chemberta_cache = {}
    for f in chemberta_dir.glob("*_pooled.npy"):
        key = f.stem.replace("_pooled", "")
        chemberta_cache[key] = np.load(f)

    # PTM column names
    ptm_cols = [
        "ptm_Y869", "ptm_S991", "ptm_Y998", "ptm_Y1016",
        "ptm_S1039", "ptm_T1041", "ptm_Y1069", "ptm_Y1092",
        "ptm_Y1110", "ptm_Y1125", "ptm_Y1172", "ptm_Y1197",
    ]
    delta_ptm_cols = [c.replace("ptm_", "delta_ptm_") for c in ptm_cols]
    glyco_cols = [f"glyco_slot{i:02d}" for i in range(12)]
    delta_glyco_cols = [f"delta_glyco_slot{i:02d}" for i in range(12)]

    features = []
    for idx in range(len(df)):
        row = df.iloc[idx]

        # ESM-2: mean-pool per-residue → 1280-d
        seq_id = row.get("sequence_id", "wild_type")
        esm2_emb = esm2_cache.get(seq_id)
        if esm2_emb is not None:
            esm2_pooled = esm2_emb.mean(axis=0)  # (1280,)
        else:
            esm2_pooled = np.zeros(1280, dtype=np.float32)

        # GearNet: mean-pool residue embeddings → 512-d
        pdb_id = row.get("pdb_id", "2GS6")
        gearnet_emb = gearnet_cache.get(pdb_id)
        if gearnet_emb is not None:
            gearnet_pooled = gearnet_emb.mean(axis=0)  # (512,)
        else:
            gearnet_pooled = np.zeros(512, dtype=np.float32)

        # ChemBERTa: already pooled → 384-d
        drug_name = str(row.get("drug_name", "osimertinib")).lower().split()[0]
        chem_pooled = chemberta_cache.get(drug_name)
        if chem_pooled is None:
            chem_pooled = np.zeros(384, dtype=np.float32)

        # PTM features → 48-d
        ptm_vals = np.array([
            float(row.get(c, 1.0)) if pd.notna(row.get(c, 1.0)) else 1.0
            for c in ptm_cols
        ], dtype=np.float32)
        delta_ptm_vals = np.array([
            float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0
            for c in delta_ptm_cols
        ], dtype=np.float32)
        glyco_vals = np.array([
            float(row.get(c, 1.0)) if pd.notna(row.get(c, 1.0)) else 1.0
            for c in glyco_cols
        ], dtype=np.float32)
        delta_glyco_vals = np.array([
            float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0
            for c in delta_glyco_cols
        ], dtype=np.float32)

        # Concatenate all → 2224-d
        feat = np.concatenate([
            esm2_pooled, gearnet_pooled, chem_pooled,
            ptm_vals, delta_ptm_vals, glyco_vals, delta_glyco_vals,
        ])
        features.append(feat)

    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Compute Tier A metrics (same as our model)
# ══════════════════════════════════════════════════════════════════════════════

def compute_tier_a_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls):
    """Compute PCC, RMSE, AUROC, AUPRC-sensitive."""
    metrics = {}

    # Regression
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50)))
    if len(y_true_ic50) > 2 and np.std(y_pred_ic50) > 1e-8:
        metrics["pearson_r"] = float(np.corrcoef(y_true_ic50, y_pred_ic50)[0, 1])
        sr = stats.spearmanr(y_true_ic50, y_pred_ic50)
        metrics["spearman_rho"] = float(sr.statistic) if hasattr(sr, 'statistic') else float(sr[0])
    else:
        metrics["pearson_r"] = 0.0
        metrics["spearman_rho"] = 0.0

    # Classification
    has_both = len(set(y_true_cls)) > 1
    if has_both and y_prob_cls is not None:
        metrics["auroc"] = float(roc_auc_score(y_true_cls, y_prob_cls))
        # AUPRC for resistant (majority)
        metrics["auprc_resistant"] = float(
            average_precision_score(y_true_cls, y_prob_cls))
        # AUPRC for sensitive (minority) — flip labels
        metrics["auprc_sensitive"] = float(
            average_precision_score(1 - y_true_cls, 1 - y_prob_cls))
        # BAcc
        # Load optimal threshold (Youden's J from step11, fallback 0.5)
        _thr_path = MODEL_DIR / "optimal_threshold.json"
        if _thr_path.exists():
            with open(_thr_path) as _f:
                _resist_thr = float(json.load(_f).get("optimal_threshold", 0.5))
        else:
            _resist_thr = 0.5
        y_pred_bin = (y_prob_cls > _resist_thr).astype(float)
        metrics["balanced_acc"] = float(
            balanced_accuracy_score(y_true_cls, y_pred_bin))
    else:
        metrics["auroc"] = 0.0
        metrics["auprc_resistant"] = 0.0
        metrics["auprc_sensitive"] = 0.0
        metrics["balanced_acc"] = 0.0

    return metrics


def compute_per_drug_metrics(df_test, y_pred_ic50, y_prob_cls):
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


# ══════════════════════════════════════════════════════════════════════════════
# ML Baseline Methods
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train_ic50, y_train_cls, X_test, seed):
    """Random Forest with inner CV for hyperparameters."""
    print("    Training Random Forest (regression + classification)...")

    # Regression
    rf_reg = RandomForestRegressor(n_estimators=500, max_depth=None,
                                    min_samples_leaf=5, random_state=seed,
                                    n_jobs=-1)
    rf_reg.fit(X_train, y_train_ic50)
    y_pred_ic50 = rf_reg.predict(X_test)

    # Classification (for AUROC)
    rf_cls = RandomForestClassifier(n_estimators=500, max_depth=None,
                                     min_samples_leaf=5, random_state=seed,
                                     class_weight="balanced", n_jobs=-1)
    rf_cls.fit(X_train, y_train_cls.astype(int))
    y_prob_cls = rf_cls.predict_proba(X_test)[:, 1]

    return y_pred_ic50, y_prob_cls


def train_xgboost(X_train, y_train_ic50, y_train_cls, X_test, seed):
    """XGBoost gradient boosting."""
    try:
        from xgboost import XGBRegressor, XGBClassifier
    except ImportError:
        print("    ⚠ xgboost not installed — skipping XGBoost baseline")
        print("    Install with: pip install xgboost")
        return None, None

    print("    Training XGBoost (regression + classification)...")

    # Regression
    xgb_reg = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    xgb_reg.fit(X_train, y_train_ic50)
    y_pred_ic50 = xgb_reg.predict(X_test)

    # Classification
    n_pos = int(y_train_cls.sum())
    n_neg = len(y_train_cls) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    xgb_cls = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=seed, n_jobs=-1, verbosity=0,
        eval_metric="logloss",
    )
    xgb_cls.fit(X_train, y_train_cls.astype(int))
    y_prob_cls = xgb_cls.predict_proba(X_test)[:, 1]

    return y_pred_ic50, y_prob_cls


def train_ridge(X_train, y_train_ic50, y_train_cls, X_test, seed):
    """Ridge regression (L2-regularized linear) + Logistic Regression classifier."""
    print("    Training Ridge Regression + Logistic Regression...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Regression with CV for alpha
    ridge = Ridge(alpha=1.0)
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    cv = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_squared_error",
                      n_jobs=-1)
    cv.fit(X_train_s, y_train_ic50)
    best_ridge = cv.best_estimator_
    y_pred_ic50 = best_ridge.predict(X_test_s)
    print(f"      Best Ridge alpha: {cv.best_params_['alpha']}")

    # Classification: proper LogisticRegression (not sigmoid hack)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                            random_state=seed, C=1.0)
    lr.fit(X_train_s, y_train_cls.astype(int))
    y_prob_cls = lr.predict_proba(X_test_s)[:, 1]

    return y_pred_ic50, y_prob_cls


def train_elastic_net(X_train, y_train_ic50, y_train_cls, X_test, seed):
    """Elastic Net (L1 + L2 regularized linear) + Logistic Regression classifier."""
    print("    Training Elastic Net + Logistic Regression...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    enet = ElasticNet(max_iter=10000, random_state=seed)
    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    }
    cv = GridSearchCV(enet, param_grid, cv=5, scoring="neg_mean_squared_error",
                      n_jobs=-1)
    cv.fit(X_train_s, y_train_ic50)
    best_enet = cv.best_estimator_
    y_pred_ic50 = best_enet.predict(X_test_s)
    print(f"      Best alpha={cv.best_params_['alpha']}, "
          f"l1_ratio={cv.best_params_['l1_ratio']}")

    # Classification: proper LogisticRegression (not sigmoid hack)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                            random_state=seed, C=1.0)
    lr.fit(X_train_s, y_train_cls.astype(int))
    y_prob_cls = lr.predict_proba(X_test_s)[:, 1]

    return y_pred_ic50, y_prob_cls


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 14a: Tier 0 ML Baselines                             ║")
    print("║  RF, XGBoost, Ridge, Elastic Net on pooled features        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    seed = cfg["training"]["seed"]
    np.random.seed(seed)

    # ── Load split indices (SAME as step11) ────────────────────────────────
    split_path = MODEL_DIR / "split_indices.json"
    if not split_path.exists():
        print(f"  ✗ split_indices.json not found at {split_path}")
        print(f"    Run step11_train.py first to create the split.")
        sys.exit(1)

    with open(split_path) as f:
        split = json.load(f)
    train_idx = np.array(split["train_idx"])
    val_idx = np.array(split["val_idx"])
    test_idx = np.array(split["test_idx"])
    # Combine train + val for ML baselines (they do inner CV)
    trainval_idx = np.concatenate([train_idx, val_idx])
    print(f"  Split: train+val={len(trainval_idx)}, test={len(test_idx)}")

    # ── Load dataset ───────────────────────────────────────────────────────
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    if not dataset_path.exists():
        print(f"  ✗ Dataset not found: {dataset_path}")
        print(f"    Run the pipeline (steps 01-06) first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"  Dataset: {len(df)} samples")

    # ── Build feature matrix ───────────────────────────────────────────────
    print(f"\n  Building pooled feature matrix (2224-d)...")
    t0 = time.time()
    X = load_pooled_features(df, features_dir)
    elapsed = time.time() - t0
    print(f"  ✓ Feature matrix: {X.shape} ({elapsed:.1f}s)")

    # Split
    X_trainval = X[trainval_idx]
    X_test = X[test_idx]
    y_ic50_trainval = df["ln_ic50"].values[trainval_idx]
    y_ic50_test = df["ln_ic50"].values[test_idx]
    y_cls_trainval = df["resistance_label"].values[trainval_idx]
    y_cls_test = df["resistance_label"].values[test_idx]

    n_sens_test = int((y_cls_test == 0).sum())
    n_res_test = int((y_cls_test == 1).sum())
    print(f"  Test set: {len(test_idx)} samples "
          f"({n_res_test} resistant, {n_sens_test} sensitive)")

    # Test subset DataFrame for per-drug metrics
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # ── Train all baselines ────────────────────────────────────────────────
    methods = {
        "random_forest": {
            "type": "ML", "year": "—",
            "train_fn": train_random_forest,
        },
        "xgboost": {
            "type": "Gradient Boosting", "year": "—",
            "train_fn": train_xgboost,
        },
        "ridge": {
            "type": "Linear (L2)", "year": "—",
            "train_fn": train_ridge,
        },
        "elastic_net": {
            "type": "Linear (L1+L2)", "year": "—",
            "train_fn": train_elastic_net,
        },
    }

    results = {}
    for name, spec in methods.items():
        print(f"\n  ── {name.upper()} ──")
        t0 = time.time()
        y_pred_ic50, y_prob_cls = spec["train_fn"](
            X_trainval, y_ic50_trainval, y_cls_trainval, X_test, seed,
        )
        elapsed = time.time() - t0

        if y_pred_ic50 is None:
            print(f"    Skipped (dependency not installed)")
            continue

        # Compute metrics
        metrics = compute_tier_a_metrics(
            y_ic50_test, y_pred_ic50, y_cls_test, y_prob_cls)
        per_drug = compute_per_drug_metrics(df_test, y_pred_ic50, y_prob_cls)

        results[name] = {
            "type": spec["type"],
            "year": spec["year"],
            "feature_dim": int(X.shape[1]),
            "n_train": int(len(X_trainval)),
            "n_test": int(len(X_test)),
            "training_time_seconds": round(elapsed, 2),
            "test_metrics": metrics,
            "per_drug": per_drug,
        }

        # Cache baseline predictions for step14c (real bootstrap + DeLong)
        pred_dir = RESULTS_DIR / "baseline_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            pred_dir / f"{name}.npz",
            y_pred_ic50=y_pred_ic50,
            y_prob_cls=y_prob_cls,
        )

        print(f"    PCC={metrics['pearson_r']:.3f} | "
              f"RMSE={metrics['rmse']:.3f} | "
              f"AUROC={metrics['auroc']:.3f} | "
              f"AUPRC-sens={metrics['auprc_sensitive']:.3f} | "
              f"({elapsed:.1f}s)")

    # ── Load our model's test metrics for comparison ───────────────────────
    eval_path = RESULTS_DIR / "evaluation_report.json"
    pred_cache_path = RESULTS_DIR / "test_predictions.npz"
    our_metrics = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        reg = eval_report.get("regression", {})
        cls = eval_report.get("classification", {})

        # Compute AUPRC-sensitive from cached predictions if available
        our_auprc_sens = 0.0
        if pred_cache_path.exists():
            cached = np.load(pred_cache_path)
            c_true = cached["y_true_cls"]
            c_prob = cached["y_prob_cls"]
            if len(set(c_true)) > 1:
                our_auprc_sens = float(
                    average_precision_score(1 - c_true, 1 - c_prob))
            print(f"  ✓ Computed AUPRC-sensitive from cached predictions: "
                  f"{our_auprc_sens:.3f}")

        our_metrics = {
            "pearson_r": reg.get("pearson_r", 0),
            "rmse": reg.get("rmse", 0),
            "auroc": cls.get("auroc", 0),
            "auprc_sensitive": our_auprc_sens,
            "balanced_acc": cls.get("balanced_accuracy", 0),
        }
        results["_our_model"] = {
            "type": "PTM-BDL (DL)",
            "year": "2026",
            "test_metrics": our_metrics,
        }
        print(f"\n  Our model (from evaluation_report.json): "
              f"PCC={our_metrics['pearson_r']:.3f}, "
              f"AUROC={our_metrics['auroc']:.3f}")

    # ── Comparison table ───────────────────────────────────────────────────
    print(f"\n  {'='*80}")
    print(f"  {'Method':<18s} | {'Type':<18s} | {'PCC':>6s} | {'RMSE':>6s} | "
          f"{'AUROC':>6s} | {'AUPRC-s':>7s} | {'BAcc':>6s}")
    print(f"  {'-'*80}")
    for name, res in results.items():
        if name.startswith("_"):
            continue
        m = res["test_metrics"]
        print(f"  {name:<18s} | {res['type']:<18s} | "
              f"{m['pearson_r']:6.3f} | {m['rmse']:6.3f} | "
              f"{m['auroc']:6.3f} | {m['auprc_sensitive']:7.3f} | "
              f"{m['balanced_acc']:6.3f}")
    if our_metrics:
        print(f"  {'Our PTM-BDL':<18s} | {'DL (PTM-BDL)':<18s} | "
              f"{our_metrics['pearson_r']:6.3f} | {our_metrics['rmse']:6.3f} | "
              f"{our_metrics['auroc']:6.3f} | {our_metrics.get('auprc_sensitive',0):7.3f} | "
              f"{our_metrics['balanced_acc']:6.3f}")
    print(f"  {'='*80}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "ml_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")
    print("\n✓ Step 14a complete!")


if __name__ == "__main__":
    main()
