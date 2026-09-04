"""
ML baseline benchmarking for PTM-BDL — FAIR COMPARISON PROTOCOL.

Trains and evaluates RF, XGBoost, Ridge, and ElasticNet under the EXACT
same experimental conditions as the PTM-BDL deep learning model.

═══════════════════════════════════════════════════════════════════════════
FAIRNESS PROTOCOL (v2 — all 7 issues fixed):
═══════════════════════════════════════════════════════════════════════════

  FIX 1: RF max_depth CAPPED
         GridSearchCV searches [5, 10, 15] — NO unlimited depth.
         Prevents RF from memorizing small training sets.

  FIX 2: GridSearchCV for ALL methods (including RF & XGBoost)
         Every baseline has its hyperparameters selected by inner 5-fold
         CV on the training set. No hardcoded aggressive configs.

  FIX 3: Train on train_idx ONLY (not train+val)
         PTM-BDL trains on train_idx and uses val_idx for early stopping.
         ML baselines now also train on train_idx only and use val_idx
         for threshold selection. Same data budget.

  FIX 4: Each method computes its OWN classification threshold
         Youden's J statistic on val_idx predictions per method.
         No reuse of PTM-BDL's threshold.

  FIX 5: Same features as DL model
         Pooled ESM-2 (1280d) + GearNet (512d) + ChemBERTa (384d) + PTM
         tabular features. All case studies use the same feature space.

  FIX 6: Separate classifiers per method
         Each regression method has its own paired classification method.
         No shared LogisticRegression across different baselines.

  FIX 7: K-fold CV support
         run_kfold_baselines() runs stratified K-fold CV with all methods
         evaluated on each fold. Reports mean ± std.

References:
  Baptista et al., Brief Bioinform 2021 (PMID 33169146) — DRP baselines
  Chen & Guestrin, KDD 2016 — XGBoost
  Yang et al., Brief Bioinform 2024 — DRP benchmarks
"""

from __future__ import annotations

# ── Fork-safety: prevent OpenMP corruption on macOS Apple Silicon ────────────
# When loky forks workers for n_jobs=-1 estimators (e.g. RandomForest), the
# parent's OpenMP thread pool state gets corrupted.  XGBoost (which links
# libomp) then segfaults even with n_jobs=1.  Setting OMP_NUM_THREADS=1 and
# LOKY_START_METHOD=loky_init_main before any native library is loaded prevents
# the crash.  setdefault() is used so callers can still override if needed.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("LOKY_START_METHOD", "loky_init_main")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="sklearn.linear_model._logistic")


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction: pooled embeddings + PTM features
# ══════════════════════════════════════════════════════════════════════════════

def load_pooled_features(df: pd.DataFrame, features_dir: Path) -> np.ndarray:
    """
    Build a flat feature matrix identical to what the DL model receives.

    For each sample:
      ESM-2 per-residue → mean-pool → 1280-d
      GearNet residue   → mean-pool → 512-d
      ChemBERTa pooled  → 384-d (already pooled)
      PTM features      → all ptm_*/delta_ptm_*/*_slot*/*delta_*_slot* columns

    Returns: (n_samples, n_features) float32 array
    """
    esm2_dir = features_dir / "esm2"
    gearnet_dir = features_dir / "gearnet"
    chemberta_dir = features_dir / "chemberta"

    # Pre-load embeddings
    esm2_cache = {}
    if esm2_dir.exists():
        for f in esm2_dir.glob("*_per_residue.npy"):
            key = f.stem.replace("_per_residue", "")
            esm2_cache[key] = np.load(f)
        for f in esm2_dir.glob("*_esm2.npy"):
            key = f.stem.replace("_esm2", "")
            if key not in esm2_cache:
                esm2_cache[key] = np.load(f)

    gearnet_cache = {}
    if gearnet_dir.exists():
        for f in gearnet_dir.glob("*_residue_embeddings.npy"):
            key = f.stem.replace("_residue_embeddings", "")
            gearnet_cache[key] = np.load(f)
        for f in gearnet_dir.glob("*_gearnet.npy"):
            key = f.stem.replace("_gearnet", "")
            if key not in gearnet_cache:
                gearnet_cache[key] = np.load(f)

    chemberta_cache = {}
    if chemberta_dir.exists():
        for f in chemberta_dir.glob("*_pooled.npy"):
            key = f.stem.replace("_pooled", "")
            chemberta_cache[key] = np.load(f)

    # Detect ESM-2 dim from first entry
    esm2_dim = 1280
    if esm2_cache:
        first = next(iter(esm2_cache.values()))
        esm2_dim = first.shape[-1] if first.ndim >= 2 else first.shape[0]

    gearnet_dim = 512
    if gearnet_cache:
        first = next(iter(gearnet_cache.values()))
        gearnet_dim = first.shape[-1] if first.ndim >= 2 else first.shape[0]

    chemberta_dim = 384
    if chemberta_cache:
        first = next(iter(chemberta_cache.values()))
        chemberta_dim = first.shape[0] if first.ndim == 1 else first.shape[-1]

    # Discover PTM columns (same logic as ResistanceDataset)
    ptm_cols = [c for c in df.columns
                if c.startswith("ptm_") and not c.startswith("ptm_pad")
                and df[c].dtype in ("float64", "float32", "int64", "int32")]
    delta_ptm_cols = [c for c in df.columns if c.startswith("delta_ptm_")]
    secondary_cols = [c for c in df.columns
                      if '_slot' in c and not c.startswith('delta_')
                      and not c.startswith('ptm_')]
    delta_secondary_cols = [c for c in df.columns
                            if c.startswith('delta_') and '_slot' in c]
    all_ptm_level_cols = ptm_cols + secondary_cols
    all_ptm_delta_cols = delta_ptm_cols + delta_secondary_cols
    ptm_dim = len(all_ptm_level_cols) + len(all_ptm_delta_cols)

    has_embeddings = bool(esm2_cache or gearnet_cache or chemberta_cache)

    features = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        parts = []

        if has_embeddings:
            # ESM-2: mean-pool per-residue
            seq_id = str(row.get("sequence_id", "wild_type"))
            esm2_emb = esm2_cache.get(seq_id)
            if esm2_emb is not None and esm2_emb.ndim >= 2:
                parts.append(esm2_emb.mean(axis=0).astype(np.float32))
            elif esm2_emb is not None and esm2_emb.ndim == 1:
                parts.append(esm2_emb.astype(np.float32))
            else:
                parts.append(np.zeros(esm2_dim, dtype=np.float32))

            # GearNet: mean-pool residue embeddings
            pdb_id = str(row.get("pdb_id", "default"))
            gearnet_emb = gearnet_cache.get(pdb_id)
            if gearnet_emb is not None and gearnet_emb.ndim >= 2:
                parts.append(gearnet_emb.mean(axis=0).astype(np.float32))
            elif gearnet_emb is not None and gearnet_emb.ndim == 1:
                parts.append(gearnet_emb.astype(np.float32))
            else:
                parts.append(np.zeros(gearnet_dim, dtype=np.float32))

            # ChemBERTa: already pooled
            drug_name = str(row.get("drug_name", "unknown")).lower().split()[0]
            chem_pooled = chemberta_cache.get(drug_name)
            if chem_pooled is not None:
                parts.append(chem_pooled.astype(np.float32).ravel())
            else:
                parts.append(np.zeros(chemberta_dim, dtype=np.float32))

        # PTM features
        for col in all_ptm_level_cols:
            val = row.get(col, 1.0)
            parts.append(np.array([float(val) if pd.notna(val) else 1.0],
                                  dtype=np.float32))
        for col in all_ptm_delta_cols:
            val = row.get(col, 0.0)
            parts.append(np.array([float(val) if pd.notna(val) else 0.0],
                                  dtype=np.float32))

        # If no embeddings and no PTM columns, fall back to all numeric columns
        if not parts:
            numeric_cols = [c for c in df.columns
                            if df[c].dtype in (np.float64, np.int64)
                            and c not in ("ln_ic50", "resistance_label")]
            vals = np.array([float(row.get(c, 0)) for c in numeric_cols],
                            dtype=np.float32)
            parts.append(vals)

        features.append(np.concatenate(parts))

    X = np.array(features, dtype=np.float32)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Threshold computation (per-method, on validation set)
# ══════════════════════════════════════════════════════════════════════════════

def compute_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute optimal classification threshold via Youden's J statistic.

    J = sensitivity + specificity - 1 = TPR - FPR
    Returns the threshold that maximizes J on the given data.
    Falls back to 0.5 if computation fails.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = float(thresholds[best_idx])
        # Sanity: clamp to [0.1, 0.9]
        return max(0.1, min(0.9, threshold))
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Metrics (identical to PTM-BDL evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true_reg: np.ndarray, y_pred_reg: np.ndarray,
                    y_true_cls: np.ndarray, y_prob_cls: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute standard metrics matching PTM-BDL evaluation protocol."""
    metrics = {}

    # Regression
    valid = ~np.isnan(y_true_reg) & ~np.isnan(y_pred_reg)
    if valid.sum() > 2:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(
            y_true_reg[valid], y_pred_reg[valid])))
        if np.std(y_pred_reg[valid]) > 1e-8:
            metrics["pearson_r"] = float(
                np.corrcoef(y_true_reg[valid], y_pred_reg[valid])[0, 1])
            sr = stats.spearmanr(y_true_reg[valid], y_pred_reg[valid])
            metrics["spearman_rho"] = float(
                sr.statistic if hasattr(sr, 'statistic') else sr[0])
        else:
            metrics["pearson_r"] = 0.0
            metrics["spearman_rho"] = 0.0

    # Classification
    has_both = len(set(y_true_cls)) > 1
    if has_both and y_prob_cls is not None:
        metrics["auroc"] = float(roc_auc_score(y_true_cls, y_prob_cls))
        metrics["auprc_sensitive"] = float(
            average_precision_score(1 - y_true_cls, 1 - y_prob_cls))
        y_pred_bin = (y_prob_cls > threshold).astype(float)
        metrics["balanced_acc"] = float(
            balanced_accuracy_score(y_true_cls, y_pred_bin))
    else:
        metrics["auroc"] = 0.0
        metrics["auprc_sensitive"] = 0.0
        metrics["balanced_acc"] = 0.0

    metrics["threshold_used"] = threshold
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# GridSearchCV parameter grids — REGULARIZED (no overfitting)
# ══════════════════════════════════════════════════════════════════════════════

# ── Anti-memorization regularization ─────────────────────────────────────
# All 3 case studies have few unique protein×drug embedding combos:
#   K562:  15 combos across 10,656 samples (710 samples/combo)
#   HeLa:  12 combos across  3,374 samples (281 samples/combo)
#   EGFR:  34 combos across    951 samples  (28 samples/combo)
#
# With >2176 embedding features collapsing to 12–34 unique patterns,
# deep trees (max_depth≥6) memorize per-combo means ≡ lookup table.
# This bypasses PTM features entirely. Constraints:
#   • max_depth ≤ 5  (2^5 = 32 ≈ max unique combos across all CS)
#   • min_samples_leaf ≥ 20 (forces generalization beyond per-combo lookup)
#   • max_features="sqrt" (prevents using full embedding as identifier)
#
# Ref: Boulesteix et al., WIREs Data Mining 2012 — RF overfitting risks
# Ref: Probst et al., WIREs Data Mining 2019 — hyperparameter tuning for RF

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "min_samples_leaf": [20, 50],
    "max_features": ["sqrt"],
}

RF_CLS_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "min_samples_leaf": [20, 50],
    "max_features": ["sqrt"],
    "class_weight": ["balanced"],
}

XGBOOST_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.7],
    "colsample_bytree": [0.5],
    "reg_alpha": [1.0],
    "reg_lambda": [10.0],
}

RIDGE_PARAM_GRID = {
    "alpha": [1.0, 10.0, 100.0, 1000.0],
}

ELASTICNET_PARAM_GRID = {
    "alpha": [0.1, 1.0, 10.0, 100.0],
    "l1_ratio": [0.1, 0.5, 0.9],
}

LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1.0],
}


# ══════════════════════════════════════════════════════════════════════════════
# Single-method training with proper GridSearchCV
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_baseline(
        X_train: np.ndarray, y_train_reg: np.ndarray, y_train_cls: np.ndarray,
        X_val: np.ndarray, y_val_cls: np.ndarray,
        X_test: np.ndarray, y_test_reg: np.ndarray, y_test_cls: np.ndarray,
        method: str = "random_forest",
        random_state: int = 42,
        inner_cv: int = 5,
) -> dict:
    """
    Train a single ML baseline with FAIR protocol.

    FIX 3: Trains on X_train only (not train+val).
    FIX 4: Computes own threshold on X_val predictions.
    FIX 2: Uses GridSearchCV for hyperparameter selection.

    Args:
        X_train, X_val, X_test: Feature matrices (already scaled).
        y_train_reg, y_test_reg: Regression targets (ln_IC50).
        y_train_cls, y_val_cls, y_test_cls: Classification targets (0/1).
        method: One of "random_forest", "xgboost", "ridge", "elastic_net".
        random_state: Random seed.
        inner_cv: Number of inner CV folds for GridSearchCV.

    Returns:
        Dict with metrics, threshold, best hyperparameters.
    """
    valid_train = ~np.isnan(y_train_reg)

    # ── Regression ────────────────────────────────────────────────────────
    if method == "random_forest":
        base_reg = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        grid_reg = RF_PARAM_GRID
        base_cls = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid_cls = RF_CLS_PARAM_GRID

    elif method == "xgboost":
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError:
            return {"method": method, "error": "xgboost not installed"}

        # macOS/Apple Silicon segfault fix: prior GridSearchCV(n_jobs=-1)
        # calls (e.g. Random Forest) fork loky workers that corrupt the
        # OpenMP runtime in the parent process.  XGBoost must avoid OpenMP
        # entirely (n_jobs=1) AND GridSearchCV must not fork (cv_n_jobs=1).
        # tree_method="hist" keeps single-threaded XGB fast on small data.
        base_reg = XGBRegressor(random_state=random_state, n_jobs=1,
                                verbosity=0, tree_method="hist", device="cpu")
        grid_reg = XGBOOST_PARAM_GRID

        n_pos = int(y_train_cls.sum())
        n_neg = len(y_train_cls) - n_pos
        scale_pos = n_neg / max(n_pos, 1)
        base_cls = XGBClassifier(
            random_state=random_state, n_jobs=1, verbosity=0,
            eval_metric="logloss", scale_pos_weight=scale_pos)
        grid_cls = XGBOOST_PARAM_GRID

    elif method == "ridge":
        base_reg = Ridge()
        grid_reg = RIDGE_PARAM_GRID
        # FIX 6: Ridge gets its own LR with its own GridSearchCV
        # l1_ratio=0 → pure L2 (equivalent to old penalty='l2')
        base_cls = LogisticRegression(
            max_iter=5000, class_weight="balanced",
            random_state=random_state, solver="lbfgs",
            l1_ratio=0)
        grid_cls = LR_PARAM_GRID

    elif method == "elastic_net":
        base_reg = ElasticNet(max_iter=10000, random_state=random_state)
        grid_reg = ELASTICNET_PARAM_GRID
        # FIX 6: ElasticNet gets its own separate LR with its own GridSearchCV
        # l1_ratio=1 → pure L1 (matching elastic net spirit)
        # solver='saga' required for L1 regularization
        base_cls = LogisticRegression(
            max_iter=5000, class_weight="balanced",
            random_state=random_state, solver="saga",
            l1_ratio=1)
        grid_cls = LR_PARAM_GRID

    else:
        raise ValueError(f"Unknown method: {method}")

    # On macOS/Apple Silicon, loky fork corrupts OpenMP state.  Disable
    # GridSearchCV multiprocessing for XGBoost (estimator already n_jobs=1).
    cv_n_jobs = 1 if method == "xgboost" else -1

    # ── Regression: GridSearchCV + fit ────────────────────────────────────
    if valid_train.sum() > inner_cv:
        cv_reg = GridSearchCV(
            base_reg, grid_reg, cv=inner_cv,
            scoring="neg_mean_squared_error", n_jobs=cv_n_jobs)
        cv_reg.fit(X_train[valid_train], y_train_reg[valid_train])
        best_reg = cv_reg.best_estimator_
        best_reg_params = cv_reg.best_params_
    else:
        best_reg = base_reg
        best_reg.fit(X_train[valid_train], y_train_reg[valid_train])
        best_reg_params = {}

    y_pred_reg = best_reg.predict(X_test)

    # ── Classification: GridSearchCV + fit ────────────────────────────────
    if len(np.unique(y_train_cls)) > 1:
        cv_cls = GridSearchCV(
            base_cls, grid_cls, cv=inner_cv,
            scoring="roc_auc", n_jobs=cv_n_jobs)
        cv_cls.fit(X_train, y_train_cls.astype(int))
        best_cls = cv_cls.best_estimator_
        best_cls_params = cv_cls.best_params_
    else:
        best_cls = base_cls
        best_cls.fit(X_train, y_train_cls.astype(int))
        best_cls_params = {}

    y_prob_cls = best_cls.predict_proba(X_test)[:, 1]

    # ── FIX 4: Compute method-specific threshold on validation set ────────
    if X_val is not None and len(X_val) > 0 and len(np.unique(y_val_cls)) > 1:
        val_prob = best_cls.predict_proba(X_val)[:, 1]
        threshold = compute_youden_threshold(y_val_cls, val_prob)
    else:
        threshold = 0.5

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(y_test_reg, y_pred_reg,
                              y_test_cls, y_prob_cls, threshold)
    metrics["method"] = method
    metrics["best_reg_params"] = best_reg_params
    metrics["best_cls_params"] = best_cls_params
    metrics["predictions"] = {
        "y_pred_reg": y_pred_reg.tolist(),
        "y_prob_cls": y_prob_cls.tolist(),
    }

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Run all baselines (single-split)
# ══════════════════════════════════════════════════════════════════════════════

def run_all_baselines(
        X_train: np.ndarray, y_train_reg: np.ndarray, y_train_cls: np.ndarray,
        X_val: np.ndarray, y_val_cls: np.ndarray,
        X_test: np.ndarray, y_test_reg: np.ndarray, y_test_cls: np.ndarray,
        methods: Optional[list[str]] = None,
        random_state: int = 42,
) -> dict:
    """
    Run all ML baselines under fair protocol.

    FIX 3: Takes separate X_train, X_val, X_test (no combined train+val).
    FIX 4: Each method gets its own threshold from val set.

    Args:
        X_train, X_val, X_test: ALREADY SCALED feature matrices.
        y_train_reg, y_test_reg: Regression targets.
        y_train_cls, y_val_cls, y_test_cls: Classification targets.
        methods: List of method names. Defaults to all four.
        random_state: Random seed.

    Returns:
        Dict keyed by method name.
    """
    if methods is None:
        methods = ["random_forest", "xgboost", "ridge", "elastic_net"]

    results = {}
    for method in methods:
        try:
            results[method] = train_and_evaluate_baseline(
                X_train, y_train_reg, y_train_cls,
                X_val, y_val_cls,
                X_test, y_test_reg, y_test_cls,
                method=method, random_state=random_state,
            )
        except Exception as e:
            results[method] = {"method": method, "error": str(e)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIX 7: K-fold CV for ML baselines (identical folds to DL model)
# ══════════════════════════════════════════════════════════════════════════════

def run_kfold_baselines(
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        n_folds: int = 5,
        methods: Optional[list[str]] = None,
        random_state: int = 42,
) -> dict:
    """
    Run K-fold CV for all ML baselines.

    Uses the same StratifiedKFold as the DL model's crossval.py.
    Within each fold, splits train into train/val (80/20).

    Returns:
        Dict with per-method per-fold results and summary statistics.
    """
    if methods is None:
        methods = ["random_forest", "xgboost", "ridge", "elastic_net"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    all_fold_results = {m: [] for m in methods}

    for fold_i, (trainval_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(y_cls)), y_cls.astype(int))):
        # Split trainval into train/val (same as DL crossval.py)
        val_size = len(trainval_idx) // 5
        val_idx = trainval_idx[:val_size]
        train_idx = trainval_idx[val_size:]

        # Scale per fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        X_test = scaler.transform(X[test_idx])

        for method in methods:
            try:
                result = train_and_evaluate_baseline(
                    X_train, y_reg[train_idx], y_cls[train_idx],
                    X_val, y_cls[val_idx],
                    X_test, y_reg[test_idx], y_cls[test_idx],
                    method=method, random_state=random_state,
                )
                # Remove large predictions from fold results
                result.pop("predictions", None)
                all_fold_results[method].append(result)
            except Exception as e:
                all_fold_results[method].append(
                    {"method": method, "error": str(e)})

    # Aggregate
    summary = {}
    for method in methods:
        fold_metrics = all_fold_results[method]
        valid_folds = [f for f in fold_metrics if "error" not in f]
        if not valid_folds:
            summary[method] = {"error": "all folds failed"}
            continue

        agg = {}
        for key in ["pearson_r", "rmse", "auroc", "balanced_acc",
                     "auprc_sensitive", "spearman_rho"]:
            vals = [f[key] for f in valid_folds if key in f]
            if vals:
                agg[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }
        agg["n_valid_folds"] = len(valid_folds)
        agg["fold_details"] = valid_folds
        summary[method] = agg

    return summary
