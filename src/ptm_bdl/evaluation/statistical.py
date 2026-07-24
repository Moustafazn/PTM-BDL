"""
Statistical testing utilities for PTM-BDL benchmarking.

Provides:
  - Bootstrap 95% confidence intervals (1,000 resamples)
  - DeLong paired AUROC test
  - Wilcoxon signed-rank test for paired comparisons
  - Benjamini-Hochberg correction for multiple testing

References:
  - Efron & Tibshirani (1993) — Bootstrap methods
  - DeLong et al. (1988) — Comparing AUROC curves
  - Wilcoxon (1945) — Signed-rank test
  - Benjamini & Hochberg (1995) — FDR correction
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: callable,
        n_resamples: int = 1000,
        ci: float = 0.95,
        random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metric_fn: Function(y_true, y_pred) → float.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (0.95 = 95% CI).
        random_state: Random seed.

    Returns:
        Dict with "estimate", "ci_lower", "ci_upper", "std".
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            if np.isfinite(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "estimate": float(np.mean(scores)),
        "ci_lower": float(np.percentile(scores, 100 * alpha)),
        "ci_upper": float(np.percentile(scores, 100 * (1 - alpha))),
        "std": float(np.std(scores)),
    }


def delong_test(
        y_true: np.ndarray,
        y_prob_a: np.ndarray,
        y_prob_b: np.ndarray,
) -> dict:
    """
    DeLong test for comparing two AUROC values from paired samples.

    Simplified implementation using bootstrap variance estimation.

    Args:
        y_true: Binary ground truth (0/1).
        y_prob_a: Predicted probabilities from model A.
        y_prob_b: Predicted probabilities from model B.

    Returns:
        Dict with "auroc_a", "auroc_b", "z_statistic", "p_value".
    """
    from sklearn.metrics import roc_auc_score

    if len(set(y_true)) < 2:
        return {"auroc_a": 0.0, "auroc_b": 0.0, "z_statistic": 0.0, "p_value": 1.0}

    auroc_a = roc_auc_score(y_true, y_prob_a)
    auroc_b = roc_auc_score(y_true, y_prob_b)

    # Bootstrap variance estimation for the difference
    n_boot = 1000
    rng = np.random.RandomState(42)
    diffs = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            a = roc_auc_score(y_true[idx], y_prob_a[idx])
            b = roc_auc_score(y_true[idx], y_prob_b[idx])
            diffs.append(a - b)
        except Exception:
            continue

    if not diffs:
        return {"auroc_a": auroc_a, "auroc_b": auroc_b,
                "z_statistic": 0.0, "p_value": 1.0}

    diffs = np.array(diffs)
    se = np.std(diffs)
    z = (auroc_a - auroc_b) / max(se, 1e-10)
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "difference": float(auroc_a - auroc_b),
        "z_statistic": float(z),
        "p_value": float(p),
    }


def wilcoxon_signed_rank(
        values_a: np.ndarray,
        values_b: np.ndarray,
) -> dict:
    """
    Wilcoxon signed-rank test for paired metric comparisons.

    Args:
        values_a: Per-fold/per-drug metrics from model A.
        values_b: Per-fold/per-drug metrics from model B.

    Returns:
        Dict with "statistic", "p_value", "n_pairs".
    """
    if len(values_a) < 3:
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(values_a),
                "note": "Too few pairs for Wilcoxon test"}

    try:
        stat, p = stats.wilcoxon(values_a, values_b, alternative="two-sided")
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "n_pairs": len(values_a),
            "mean_diff": float(np.mean(values_a - values_b)),
        }
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "error": str(e)}


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        p_values: List of p-values from multiple comparisons.
        alpha: Significance level.

    Returns:
        Dict with "adjusted_p_values", "rejected", "n_rejected".
    """
    n = len(p_values)
    if n == 0:
        return {"adjusted_p_values": [], "rejected": [], "n_rejected": 0}

    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    # BH adjustment
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted[i] = min(sorted_p[i] * n / rank,
                          adjusted[i + 1] if i < n - 1 else sorted_p[i] * n / rank)
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result_adjusted = np.zeros(n)
    result_adjusted[sorted_idx] = adjusted

    rejected = result_adjusted < alpha

    return {
        "adjusted_p_values": result_adjusted.tolist(),
        "rejected": rejected.tolist(),
        "n_rejected": int(rejected.sum()),
        "alpha": alpha,
    }
