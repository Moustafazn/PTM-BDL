"""
ML baseline tool for benchmarking PTM-BDL against traditional methods.

Trains and evaluates RF, XGBoost, Ridge, and ElasticNet on the same features
and splits as the PTM-BDL model. Uses concatenated pooled embeddings as input
(ESM-2 + GearNet + ChemBERTa + PTM features).

This module is case-study-agnostic — it operates on generic feature matrices.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_baseline(
        X_train: np.ndarray,
        y_train_reg: np.ndarray,
        y_train_cls: np.ndarray,
        X_test: np.ndarray,
        y_test_reg: np.ndarray,
        y_test_cls: np.ndarray,
        method: str = "random_forest",
        threshold: float = 0.5,
        random_state: int = 42,
) -> dict:
    """
    Train a single ML baseline and evaluate on test set.

    Args:
        X_train, X_test: Feature matrices (n_samples × n_features).
        y_train_reg, y_test_reg: Regression targets (ln_IC50).
        y_train_cls, y_test_cls: Classification targets (0/1 resistance).
        method: One of "random_forest", "xgboost", "ridge", "elastic_net".
        threshold: Classification threshold for binary predictions.
        random_state: Random seed.

    Returns:
        Dict with regression + classification metrics.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train regression model
    if method == "random_forest":
        reg_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )
        cls_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )
    elif method == "xgboost":
        try:
            from xgboost import XGBRegressor, XGBClassifier
            reg_model = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=random_state, verbosity=0,
            )
            cls_model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=random_state, verbosity=0,
                use_label_encoder=False, eval_metric="logloss",
            )
        except ImportError:
            raise ImportError("xgboost is required for XGBoost baseline")
    elif method == "ridge":
        reg_model = Ridge(alpha=1.0, random_state=random_state)
        cls_model = LogisticRegression(
            max_iter=5000, class_weight="balanced",
            random_state=random_state, C=1.0,
        )
    elif method == "elastic_net":
        reg_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)
        cls_model = LogisticRegression(
            max_iter=5000, class_weight="balanced",
            random_state=random_state, C=1.0,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Regression
    reg_model.fit(X_train_s, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_s)

    metrics = {
        "method": method,
        "rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))),
    }
    if len(y_test_reg) > 2 and np.std(y_pred_reg) > 1e-8:
        metrics["pearson_r"] = float(np.corrcoef(y_test_reg, y_pred_reg)[0, 1])
        sr = stats.spearmanr(y_test_reg, y_pred_reg)
        metrics["spearman_rho"] = float(sr.statistic if hasattr(sr, 'statistic') else sr[0])
    else:
        metrics["pearson_r"] = 0.0
        metrics["spearman_rho"] = 0.0

    # Classification
    has_both = len(set(y_test_cls)) > 1
    cls_model.fit(X_train_s, y_train_cls.astype(int))
    y_prob = cls_model.predict_proba(X_test_s)[:, 1]

    y_pred_cls_binary = (y_prob > threshold).astype(float)

    if has_both:
        metrics["auroc"] = float(roc_auc_score(y_test_cls, y_prob))
        metrics["auprc_sensitive"] = float(
            average_precision_score(1 - y_test_cls, 1 - y_prob))
        metrics["balanced_acc"] = float(
            balanced_accuracy_score(y_test_cls, y_pred_cls_binary))
    else:
        metrics["auroc"] = 0.0
        metrics["auprc_sensitive"] = 0.0
        metrics["balanced_acc"] = 0.0

    metrics["predictions"] = {
        "y_pred_reg": y_pred_reg.tolist(),
        "y_prob_cls": y_prob.tolist(),
    }

    return metrics


def run_all_baselines(
        X_train: np.ndarray,
        y_train_reg: np.ndarray,
        y_train_cls: np.ndarray,
        X_test: np.ndarray,
        y_test_reg: np.ndarray,
        y_test_cls: np.ndarray,
        methods: Optional[list[str]] = None,
        threshold: float = 0.5,
        random_state: int = 42,
) -> dict:
    """
    Run all ML baselines and return comparative results.

    Args:
        Methods defaults to ["random_forest", "xgboost", "ridge", "elastic_net"].

    Returns:
        Dict keyed by method name, each containing metrics.
    """
    if methods is None:
        methods = ["random_forest", "xgboost", "ridge", "elastic_net"]

    results = {}
    for method in methods:
        try:
            results[method] = train_and_evaluate_baseline(
                X_train, y_train_reg, y_train_cls,
                X_test, y_test_reg, y_test_cls,
                method=method, threshold=threshold,
                random_state=random_state,
            )
        except Exception as e:
            results[method] = {"method": method, "error": str(e)}

    return results
