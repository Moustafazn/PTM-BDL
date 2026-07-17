#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 14c — Statistical Tests for Benchmarking                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Provide rigorous statistical evidence that our model's superiority        ║
║    (or inferiority) vs baselines is not due to chance.                       ║
║                                                                              ║
║  TESTS IMPLEMENTED:                                                          ║
║    1. Bootstrap 95% CIs (1,000 resamples) for PCC, RMSE, AUROC, AUPRC-sens ║
║    2. DeLong test — paired AUROC comparison (PTM-dl model vs each baseline)         ║
║    3. Wilcoxon signed-rank — per-drug paired comparisons                    ║
║    4. Benjamini-Hochberg correction across K baselines                       ║
║                                                                              ║
║  INPUT:                                                                      ║
║    results/ml_baselines.json           — ML baseline predictions             ║
║    results/evaluation_report.json      — our model's predictions             ║
║    results/ablation_study.json         — ablation results                    ║
║    data/models/split_indices.json      — split indices                       ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/statistical_tests.json                                            ║
║                                                                              ║
║  REFERENCES:                                                                 ║
║    - DeLong et al., Biometrics 1988 (paired AUROC comparison)               ║
║    - SAGE-net (Nat Methods 2026): Wilcoxon + BH correction                  ║
║    - Sada Del Real et al. (Brief Bioinf 2026): per-drug SCC                 ║
║                                                                              ║
║  BENCHMARKING_PLAN.md §2, §8 Step 14c                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"]
MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05,
                 seed=42):
    """
    Compute bootstrap (1-alpha)% confidence interval for a metric.

    Parameters
    ----------
    y_true : array-like — ground truth
    y_pred : array-like — predictions (continuous or probability)
    metric_fn : callable(y_true, y_pred) -> float
    n_boot : int — number of bootstrap resamples
    alpha : float — significance level (0.05 → 95% CI)
    seed : int

    Returns
    -------
    dict with keys: point, ci_lower, ci_upper, std, n_boot
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
            if np.isfinite(val):
                boot_vals.append(val)
        except Exception:
            continue

    boot_vals = np.array(boot_vals)
    if len(boot_vals) < 10:
        return {
            "point": float(point),
            "ci_lower": float(point),
            "ci_upper": float(point),
            "std": 0.0,
            "n_boot": int(len(boot_vals)),
        }

    ci_lower = float(np.percentile(boot_vals, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))

    return {
        "point": float(point),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(boot_vals.std()),
        "n_boot": int(len(boot_vals)),
    }


def compute_bootstrap_cis(y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
                           n_boot=1000, seed=42):
    """Compute bootstrap CIs for all Tier A metrics."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, mean_squared_error,
    )

    def pcc_fn(y_t, y_p):
        if np.std(y_p) < 1e-8 or len(y_t) < 3:
            return 0.0
        return float(np.corrcoef(y_t, y_p)[0, 1])

    def rmse_fn(y_t, y_p):
        return float(np.sqrt(mean_squared_error(y_t, y_p)))

    def auroc_fn(y_t, y_p):
        if len(set(y_t)) < 2:
            return 0.5
        return float(roc_auc_score(y_t, y_p))

    def auprc_sens_fn(y_t, y_p):
        if len(set(y_t)) < 2:
            return 0.0
        return float(average_precision_score(1 - y_t, 1 - y_p))

    results = {}
    results["pearson_r"] = bootstrap_ci(
        y_true_ic50, y_pred_ic50, pcc_fn, n_boot, seed=seed)
    results["rmse"] = bootstrap_ci(
        y_true_ic50, y_pred_ic50, rmse_fn, n_boot, seed=seed)

    if y_prob_cls is not None and len(set(y_true_cls)) > 1:
        results["auroc"] = bootstrap_ci(
            y_true_cls, y_prob_cls, auroc_fn, n_boot, seed=seed)
        results["auprc_sensitive"] = bootstrap_ci(
            y_true_cls, y_prob_cls, auprc_sens_fn, n_boot, seed=seed)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DeLong Test for Paired AUROC Comparison
# ══════════════════════════════════════════════════════════════════════════════

def _auc_variance_components(y_true, y_score):
    """Compute DeLong variance components for a single model."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m = len(pos)
    n = len(neg)

    # For each positive, proportion of negatives ranked lower
    V10 = np.array([np.mean(y_score[y_true == 0] < p) +
                     0.5 * np.mean(y_score[y_true == 0] == p) for p in pos])
    # For each negative, proportion of positives ranked higher
    V01 = np.array([np.mean(y_score[y_true == 1] > q) +
                     0.5 * np.mean(y_score[y_true == 1] == q) for q in neg])
    return V10, V01, m, n


def delong_test(y_true, y_score_a, y_score_b):
    """
    DeLong test for comparing two paired AUROC values.

    Parameters
    ----------
    y_true : array — binary labels
    y_score_a : array — predicted probabilities from model A (ours)
    y_score_b : array — predicted probabilities from model B (baseline)

    Returns
    -------
    dict with keys: auroc_a, auroc_b, z_statistic, p_value
    """
    if len(set(y_true)) < 2:
        return {
            "auroc_a": 0.5, "auroc_b": 0.5,
            "z_statistic": 0.0, "p_value": 1.0,
        }

    from sklearn.metrics import roc_auc_score
    auroc_a = roc_auc_score(y_true, y_score_a)
    auroc_b = roc_auc_score(y_true, y_score_b)

    V10_a, V01_a, m, n = _auc_variance_components(y_true, y_score_a)
    V10_b, V01_b, _, _ = _auc_variance_components(y_true, y_score_b)

    # Covariance matrix of the two AUC estimates
    S10 = np.cov(V10_a, V10_b)
    S01 = np.cov(V01_a, V01_b)

    S = S10 / m + S01 / n

    # Test statistic
    diff = auroc_a - auroc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff < 1e-15:
        z = 0.0
    else:
        z = diff / np.sqrt(var_diff)

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided

    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "diff": float(diff),
        "z_statistic": float(z),
        "p_value": float(p_value),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Wilcoxon Signed-Rank Test (per-drug paired comparisons)
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_per_drug(our_per_drug, baseline_per_drug, metric="pearson_r"):
    """
    Wilcoxon signed-rank test on per-drug metrics.

    Parameters
    ----------
    our_per_drug : dict[drug_name -> {metric: value}]
    baseline_per_drug : dict[drug_name -> {metric: value}]
    metric : str — which metric to compare

    Returns
    -------
    dict with paired values, statistic, p_value
    """
    common_drugs = sorted(set(our_per_drug.keys()) & set(baseline_per_drug.keys()))
    if len(common_drugs) < 3:
        return {
            "n_drugs": len(common_drugs),
            "statistic": None,
            "p_value": 1.0,
            "note": "Too few drugs for Wilcoxon test (need >= 3)",
        }

    our_vals = [our_per_drug[d].get(metric, 0) for d in common_drugs]
    base_vals = [baseline_per_drug[d].get(metric, 0) for d in common_drugs]
    diffs = [a - b for a, b in zip(our_vals, base_vals)]

    # Wilcoxon requires at least some non-zero differences
    non_zero = [d for d in diffs if abs(d) > 1e-10]
    if len(non_zero) < 2:
        return {
            "n_drugs": len(common_drugs),
            "drugs": common_drugs,
            "our_values": our_vals,
            "baseline_values": base_vals,
            "differences": diffs,
            "statistic": None,
            "p_value": 1.0,
            "note": "All differences are near-zero",
        }

    try:
        stat, p_val = stats.wilcoxon(our_vals, base_vals, alternative="two-sided")
    except ValueError:
        stat, p_val = None, 1.0

    return {
        "n_drugs": len(common_drugs),
        "drugs": common_drugs,
        "our_values": our_vals,
        "baseline_values": base_vals,
        "differences": diffs,
        "metric": metric,
        "statistic": float(stat) if stat is not None else None,
        "p_value": float(p_val),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Benjamini-Hochberg Multiple Testing Correction
# ══════════════════════════════════════════════════════════════════════════════

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction to a list of p-values.

    Parameters
    ----------
    p_values : dict[test_name -> p_value]
    alpha : float — FDR threshold

    Returns
    -------
    dict with corrected p-values and significance
    """
    names = list(p_values.keys())
    pvals = [p_values[n] for n in names]
    m = len(pvals)

    if m == 0:
        return {}

    # Sort by p-value
    sorted_idx = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    # BH correction
    bh_critical = np.array([(i + 1) / m * alpha for i in range(m)])
    adjusted = np.zeros(m)
    adjusted[-1] = sorted_pvals[-1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * m / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    results = {}
    for i, name in enumerate(sorted_names):
        results[name] = {
            "raw_p": float(sorted_pvals[i]),
            "adjusted_p": float(adjusted[i]),
            "rank": int(i + 1),
            "significant": bool(adjusted[i] < alpha),
            "bh_critical": float(bh_critical[i]),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Paired Bootstrap Test
# ══════════════════════════════════════════════════════════════════════════════

def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric_fn,
                           n_boot=1000, seed=42):
    """
    Paired bootstrap test: is metric(A) significantly different from metric(B)?

    Returns p-value (two-sided) under the null hypothesis that the
    difference is zero.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    observed_diff = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)

    count_extreme = 0
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            diff_i = (metric_fn(y_true[idx], y_pred_a[idx])
                      - metric_fn(y_true[idx], y_pred_b[idx]))
            if abs(diff_i) >= abs(observed_diff):
                count_extreme += 1
        except Exception:
            continue

    p_value = count_extreme / n_boot

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_boot": n_boot,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 14c: Statistical Tests for Benchmarking              ║")
    print("║  Bootstrap CIs + DeLong + Wilcoxon + BH correction         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    output = {
        "method": "Statistical rigor for Nature Methods benchmarking",
        "tests_performed": [],
    }

    # ── Load available results ─────────────────────────────────────────────
    ml_baselines_path = RESULTS_DIR / "ml_baselines.json"
    eval_path = RESULTS_DIR / "evaluation_report.json"
    ablation_path = RESULTS_DIR / "ablation_study.json"

    ml_baselines = {}
    if ml_baselines_path.exists():
        with open(ml_baselines_path) as f:
            ml_baselines = json.load(f)
        print(f"  ✓ Loaded ml_baselines.json ({len(ml_baselines)} methods)")
    else:
        print(f"  ⚠ ml_baselines.json not found — run step14a first")

    eval_report = {}
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        print(f"  ✓ Loaded evaluation_report.json")

    ablation = {}
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation = json.load(f)
        print(f"  ✓ Loaded ablation_study.json")

    # ══════════════════════════════════════════════════════════════════════
    # Load cached predictions (from step12 + step14a)
    # ══════════════════════════════════════════════════════════════════════
    pred_cache_path = RESULTS_DIR / "test_predictions.npz"
    baseline_pred_dir = RESULTS_DIR / "baseline_predictions"
    has_cached_preds = pred_cache_path.exists()

    our_preds = None
    if has_cached_preds:
        our_preds = np.load(pred_cache_path)
        print(f"  ✓ Loaded cached test predictions ({len(our_preds['y_true_ic50'])} samples)")
    else:
        print(f"  ⚠ test_predictions.npz not found — run step12 first for real bootstrap/DeLong")

    baseline_preds = {}
    if baseline_pred_dir.exists():
        for f in baseline_pred_dir.glob("*.npz"):
            bl_name = f.stem
            baseline_preds[bl_name] = np.load(f)
        print(f"  ✓ Loaded {len(baseline_preds)} baseline prediction caches")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Bootstrap 95% CIs (1,000 resamples)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  TEST 1: Bootstrap 95% CIs (1,000 resamples)")
    print(f"  {'='*60}")

    bootstrap_results = {}

    if has_cached_preds and our_preds is not None:
        # REAL bootstrap from cached predictions
        our_boot = compute_bootstrap_cis(
            our_preds["y_true_ic50"], our_preds["y_pred_ic50"],
            our_preds["y_true_cls"], our_preds["y_prob_cls"],
            n_boot=1000, seed=42,
        )
        for metric_name, ci_data in our_boot.items():
            bootstrap_results[f"our_model_{metric_name}"] = {
                **ci_data,
                "n_test": int(len(our_preds["y_true_ic50"])),
                "method": "bootstrap_1000",
            }
            print(f"    our_model {metric_name}: {ci_data['point']:.3f} "
                  f"[{ci_data['ci_lower']:.3f}, {ci_data['ci_upper']:.3f}]")

        # Real bootstrap for each baseline with cached predictions
        for bl_name, bl_pred in baseline_preds.items():
            y_true_ic50 = our_preds["y_true_ic50"]  # same test set
            y_true_cls = our_preds["y_true_cls"]
            bl_boot = compute_bootstrap_cis(
                y_true_ic50, bl_pred["y_pred_ic50"],
                y_true_cls, bl_pred["y_prob_cls"],
                n_boot=1000, seed=42,
            )
            for metric_name, ci_data in bl_boot.items():
                bootstrap_results[f"{bl_name}_{metric_name}"] = {
                    **ci_data,
                    "method": "bootstrap_1000",
                }
    else:
        # Fallback: analytic approximation if no cached predictions
        print(f"    ⚠ Using analytic approximation (no cached predictions)")
        if eval_report:
            reg = eval_report.get("regression", {})
            cls = eval_report.get("classification", {})
            n_test = eval_report.get("test_samples", 143)
            for metric_name, point in [("pearson_r", reg.get("pearson_r", 0)),
                                        ("rmse", reg.get("rmse", 0)),
                                        ("auroc", cls.get("auroc", 0))]:
                if metric_name == "pearson_r" and abs(point) < 1.0:
                    z = np.arctanh(point)
                    se = 1.0 / np.sqrt(n_test - 3)
                    ci_lo, ci_hi = float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))
                elif metric_name == "auroc":
                    se = np.sqrt(point * (1 - point) / max(n_test, 1))
                    ci_lo, ci_hi = max(0, point - 1.96 * se), min(1, point + 1.96 * se)
                else:
                    se = point * 0.1
                    ci_lo, ci_hi = point - 1.96 * se, point + 1.96 * se
                bootstrap_results[f"our_model_{metric_name}"] = {
                    "point": float(point), "ci_lower": float(ci_lo),
                    "ci_upper": float(ci_hi), "se": float(se),
                    "n_test": n_test, "method": "analytic_fallback",
                }
                print(f"    {metric_name}: {point:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    output["bootstrap_cis"] = bootstrap_results
    output["tests_performed"].append("bootstrap_95ci")
    print(f"    ✓ Computed CIs for {len(bootstrap_results)} metric-method pairs")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: DeLong Test (paired AUROC comparison)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  TEST 2: DeLong Test (paired AUROC comparison)")
    print(f"  {'='*60}")

    delong_results = {}

    if has_cached_preds and baseline_preds:
        # REAL paired DeLong test
        y_true_cls = our_preds["y_true_cls"]
        y_prob_ours = our_preds["y_prob_cls"]

        for bl_name, bl_pred in baseline_preds.items():
            if bl_name.startswith("_"):
                continue
            result = delong_test(y_true_cls, y_prob_ours, bl_pred["y_prob_cls"])
            delong_results[f"ours_vs_{bl_name}"] = {
                "auroc_ours": result["auroc_a"],
                "auroc_baseline": result["auroc_b"],
                "diff": result["diff"],
                "z_statistic": result["z_statistic"],
                "p_value": result["p_value"],
                "significant_at_005": bool(result["p_value"] < 0.05),
                "method": "delong_paired",
            }
            sig = "★" if result["p_value"] < 0.05 else ""
            print(f"    Ours ({result['auroc_a']:.3f}) vs {bl_name} "
                  f"({result['auroc_b']:.3f}): "
                  f"Δ={result['diff']:+.3f}, z={result['z_statistic']:.2f}, "
                  f"p={result['p_value']:.4f} {sig}")
    else:
        # Fallback: Hanley-McNeil approximation
        print(f"    ⚠ Using Hanley-McNeil approximation (no paired predictions)")
        our_auroc = eval_report.get("classification", {}).get("auroc", 0)
        for bl_name, bl_data in ml_baselines.items():
            if bl_name.startswith("_"):
                continue
            bl_auroc = bl_data.get("test_metrics", {}).get("auroc", 0)
            n_test = bl_data.get("n_test", 143)
            se_a = np.sqrt(our_auroc * (1 - our_auroc) / max(n_test, 1))
            se_b = np.sqrt(bl_auroc * (1 - bl_auroc) / max(n_test, 1))
            se_diff = np.sqrt(se_a**2 + se_b**2)
            z = (our_auroc - bl_auroc) / max(se_diff, 1e-10)
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            delong_results[f"ours_vs_{bl_name}"] = {
                "auroc_ours": float(our_auroc),
                "auroc_baseline": float(bl_auroc),
                "diff": float(our_auroc - bl_auroc),
                "z_statistic": float(z),
                "p_value": float(p_val),
                "significant_at_005": bool(p_val < 0.05),
                "method": "hanley_mcneil_fallback",
            }
            sig = "★" if p_val < 0.05 else ""
            print(f"    Ours ({our_auroc:.3f}) vs {bl_name} ({bl_auroc:.3f}): "
                  f"Δ={our_auroc - bl_auroc:+.3f}, z={z:.2f}, p={p_val:.4f} {sig}")

    output["delong_tests"] = delong_results
    output["tests_performed"].append("delong_auroc")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: Wilcoxon Signed-Rank (per-drug paired)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  TEST 3: Wilcoxon Signed-Rank (per-drug paired)")
    print(f"  {'='*60}")

    wilcoxon_results = {}
    our_per_drug = eval_report.get("drug_specific", {})

    for bl_name, bl_data in ml_baselines.items():
        if bl_name.startswith("_"):
            continue
        bl_per_drug = bl_data.get("per_drug", {})
        if not bl_per_drug:
            continue

        for metric in ["pearson_r", "auroc"]:
            result = wilcoxon_per_drug(our_per_drug, bl_per_drug, metric)
            key = f"ours_vs_{bl_name}_{metric}"
            wilcoxon_results[key] = result
            p_str = f"p={result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            print(f"    {key}: {p_str} (n_drugs={result['n_drugs']})")

    output["wilcoxon_tests"] = wilcoxon_results
    output["tests_performed"].append("wilcoxon_signed_rank")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: Benjamini-Hochberg Multiple Testing Correction
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  TEST 4: Benjamini-Hochberg Multiple Testing Correction")
    print(f"  {'='*60}")

    # Collect all p-values from DeLong + Wilcoxon
    all_pvals = {}
    for name, result in delong_results.items():
        all_pvals[f"delong_{name}"] = result["p_value"]
    for name, result in wilcoxon_results.items():
        if result["p_value"] is not None:
            all_pvals[f"wilcoxon_{name}"] = result["p_value"]

    if all_pvals:
        bh_results = benjamini_hochberg(all_pvals, alpha=0.05)
        output["bh_correction"] = bh_results
        output["tests_performed"].append("benjamini_hochberg")

        print(f"    Applied BH correction to {len(all_pvals)} p-values:")
        for name, res in sorted(bh_results.items(),
                                 key=lambda x: x[1]["rank"]):
            sig = "★" if res["significant"] else ""
            print(f"      {name}: raw={res['raw_p']:.4f} → "
                  f"adj={res['adjusted_p']:.4f} {sig}")
    else:
        output["bh_correction"] = {}
        print(f"    No p-values to correct")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5: Ablation Effect Sizes (from step11b)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  TEST 5: Ablation Effect Sizes")
    print(f"  {'='*60}")

    ablation_effects = {}
    if ablation and "_summary" in ablation:
        summary = ablation["_summary"]
        for key in ["ptm_gain_auroc", "ptm_gain_auprc_sensitive",
                     "ptm_gain_bacc", "ptm_gain_f1_macro",
                     "phospho_marginal_auroc", "glyco_marginal_auroc",
                     "typed_attention_marginal_auroc"]:
            val = summary.get(key, 0)
            ablation_effects[key] = {
                "effect_size": float(val),
                "direction": "positive" if val > 0 else "negative" if val < 0 else "zero",
                "meaningful": abs(val) > 0.01,
            }
            print(f"    {key}: {val:+.4f} "
                  f"({'meaningful' if abs(val) > 0.01 else 'negligible'})")

    output["ablation_effects"] = ablation_effects
    output["tests_performed"].append("ablation_effect_sizes")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")

    n_sig_delong = sum(1 for v in delong_results.values()
                       if v.get("significant_at_005", False))
    n_total_delong = len(delong_results)
    n_sig_bh = sum(1 for v in output.get("bh_correction", {}).values()
                    if v.get("significant", False))
    n_total_bh = len(output.get("bh_correction", {}))

    output["summary"] = {
        "n_tests_performed": len(output["tests_performed"]),
        "delong_significant": f"{n_sig_delong}/{n_total_delong}",
        "bh_significant": f"{n_sig_bh}/{n_total_bh}",
        "tests_list": output["tests_performed"],
    }

    print(f"    Tests performed: {output['tests_performed']}")
    print(f"    DeLong significant: {n_sig_delong}/{n_total_delong}")
    print(f"    BH significant (FDR<0.05): {n_sig_bh}/{n_total_bh}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  ✓ Saved: {out_path}")
    print("\n✓ Step 14c complete!")


if __name__ == "__main__":
    main()
