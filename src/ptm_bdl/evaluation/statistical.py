"""
Statistical testing utilities for PTM-BDL benchmarking.

Provides:
  - Bootstrap 95% confidence intervals (1,000 resamples)
  - DeLong paired AUROC test
  - Wilcoxon signed-rank test for paired comparisons
  - Benjamini-Hochberg correction for multiple testing
  - Expected Calibration Error (ECE) and reliability diagram data
  - IG rank stability metrics (Spearman rank correlation across seeds)

References:
  - Efron & Tibshirani (1993) — Bootstrap methods
  - DeLong et al. (1988) — Comparing AUROC curves
  - Wilcoxon (1945) — Signed-rank test
  - Benjamini & Hochberg (1995) — FDR correction
  - Guo et al. (2017) — On Calibration of Modern Neural Networks, ICML
  - Naeini et al. (2015) — Obtaining Well Calibrated Probabilities, AAAI
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


# ══════════════════════════════════════════════════════════════════════════════
# Formal Leakage Analysis (Reviewer Q2)
# Quantifies information overlap between PTM features and IC50 labels.
# ══════════════════════════════════════════════════════════════════════════════

def compute_leakage_analysis(
        df,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        ptm_source_col: str = "data_source",
        cell_col: str = "cell_line_name",
        drug_col: str = "drug_name",
) -> dict:
    """
    Formal leakage analysis between PTM features and IC50 labels.

    Checks:
      1. PTM diversity — how many unique PTM vectors exist vs sample count
      2. Train/test cell-line overlap — shared cell lines across splits
      3. Train/test drug overlap — shared drugs across splits
      4. Temporal/institutional separation — documents data provenance
      5. Measured PTM ↔ IC50 overlap — samples with both measured PTM
         data AND GDSC2 IC50 (highest leakage risk)

    Args:
        df: Full dataset DataFrame.
        train_idx, val_idx, test_idx: Split index arrays.
        ptm_source_col: Column tracking PTM data provenance (if present).
        cell_col: Cell line name column.
        drug_col: Drug name column.

    Returns:
        Comprehensive leakage analysis dict.
    """
    report = {
        "n_samples": len(df),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }

    # ── 1. PTM diversity ──────────────────────────────────────────────────
    ptm_cols = [c for c in df.columns
                if c.startswith("ptm_") and not c.startswith("ptm_pad")
                and df[c].dtype in ("float64", "float32", "int64", "int32")]
    delta_cols = [c for c in df.columns if c.startswith("delta_ptm_")]
    secondary_cols = [c for c in df.columns if '_slot' in c and not c.startswith('delta_')]
    delta_sec_cols = [c for c in df.columns if c.startswith('delta_') and '_slot' in c]

    all_ptm_feature_cols = ptm_cols + delta_cols + secondary_cols + delta_sec_cols

    if ptm_cols:
        n_unique_baseline = int(df[ptm_cols].drop_duplicates().shape[0])
        report["ptm_baseline_unique"] = n_unique_baseline
        report["ptm_baseline_diversity"] = round(n_unique_baseline / len(df), 4)

    if delta_cols:
        n_unique_delta = int(df[delta_cols].drop_duplicates().shape[0])
        report["ptm_delta_unique"] = n_unique_delta
        report["ptm_delta_diversity"] = round(n_unique_delta / len(df), 4)

    if all_ptm_feature_cols:
        n_unique_all = int(df[all_ptm_feature_cols].drop_duplicates().shape[0])
        report["ptm_all_unique"] = n_unique_all
        report["ptm_all_diversity"] = round(n_unique_all / len(df), 4)

    # ── 2. Cell-line overlap across splits ────────────────────────────────
    if cell_col in df.columns:
        train_cells = set(df.iloc[train_idx][cell_col].unique())
        val_cells = set(df.iloc[val_idx][cell_col].unique())
        test_cells = set(df.iloc[test_idx][cell_col].unique())
        report["cell_line_overlap"] = {
            "n_train_cells": len(train_cells),
            "n_val_cells": len(val_cells),
            "n_test_cells": len(test_cells),
            "train_test_overlap": len(train_cells & test_cells),
            "train_val_overlap": len(train_cells & val_cells),
            "note": "Cell-line overlap is expected for drug-wise evaluation; "
                    "cold-cell evaluation (Q4) eliminates this overlap.",
        }

    # ── 3. Drug overlap across splits ─────────────────────────────────────
    if drug_col in df.columns:
        train_drugs = set(df.iloc[train_idx][drug_col].unique())
        val_drugs = set(df.iloc[val_idx][drug_col].unique())
        test_drugs = set(df.iloc[test_idx][drug_col].unique())
        report["drug_overlap"] = {
            "n_train_drugs": len(train_drugs),
            "n_test_drugs": len(test_drugs),
            "train_test_overlap": len(train_drugs & test_drugs),
            "note": "Drug overlap is expected for standard splits; "
                    "cold-drug LODO evaluation (Q4) eliminates this overlap.",
        }

    # ── 4. Propagation confidence distribution ────────────────────────────
    if "propagation_confidence" in df.columns:
        conf = df["propagation_confidence"].values
        measured_mask = conf >= 0.90  # high confidence = directly measured
        n_measured = int(measured_mask.sum())
        n_propagated = int((~measured_mask).sum())
        report["ptm_provenance"] = {
            "n_measured_high_conf": n_measured,
            "n_propagated_lower_conf": n_propagated,
            "frac_measured": round(n_measured / max(len(df), 1), 4),
            "test_measured": int(measured_mask[test_idx].sum()),
            "test_propagated": int((~measured_mask[test_idx]).sum()),
        }

    # ── 5. Institutional separation (documented, not computed) ────────────
    report["institutional_separation"] = {
        "ptm_data_source": "DrugPTM-Bench (Xie lab, phosphoproteomics, mass spectrometry)",
        "ic50_data_source": "GDSC2 (Sanger Institute, pharmacology, viability assays)",
        "modality_independence": True,
        "explanation": "PTM measurements (phosphoproteomics via LC-MS/MS) and IC50 labels "
                       "(dose-response viability assays) are from different experimental "
                       "modalities performed at different institutions. No shared "
                       "measurement apparatus or experimental protocol.",
    }

    # ── 6. Constant-channel detection (Q3/Q10 related) ────────────────────
    constant_channels = []
    for col in all_ptm_feature_cols:
        if df[col].std() < 1e-8:
            constant_channels.append(col)
    report["constant_channels"] = {
        "n_constant": len(constant_channels),
        "columns": constant_channels[:20],  # cap at 20 for readability
        "note": "Constant channels contribute zero IG attribution and should "
                "be documented (e.g., EGFR glyco baseline is uniformly 1.0 "
                "for all samples — see Q3 ablation for their exclusion).",
    }

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Expected Calibration Error (ECE) and Reliability Diagram Data
# Ref: Guo et al. (2017) "On Calibration of Modern Neural Networks", ICML
#      Naeini et al. (2015) "Obtaining Well Calibrated Probabilities", AAAI
# ══════════════════════════════════════════════════════════════════════════════

def compute_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.

    ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|

    where B_b is the set of samples whose predicted probability falls into
    bin b, acc(B_b) is the empirical accuracy in that bin, and conf(B_b)
    is the mean predicted probability.

    Args:
        y_true: Binary ground truth labels (0/1).
        y_prob: Predicted probabilities (0-1 range).
        n_bins: Number of calibration bins.

    Returns:
        Dict with "ece", "mce" (max calibration error),
        "bin_edges", "bin_accs", "bin_confs", "bin_counts"
        (the latter four for plotting reliability diagrams).
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n_in_bin = mask.sum()
        bin_counts[i] = n_in_bin
        if n_in_bin > 0:
            bin_accs[i] = y_true[mask].mean()
            bin_confs[i] = y_prob[mask].mean()

    # ECE = weighted mean of |acc - conf|
    weights = bin_counts / max(bin_counts.sum(), 1)
    gaps = np.abs(bin_accs - bin_confs)
    ece = float((weights * gaps).sum())

    # MCE = max calibration error (worst bin)
    mce = float(gaps[bin_counts > 0].max()) if (bin_counts > 0).any() else 0.0

    return {
        "ece": ece,
        "mce": mce,
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_accs": bin_accs.tolist(),
        "bin_confs": bin_confs.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


def compute_ece_per_drug(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        drug_labels: np.ndarray,
        n_bins: int = 10,
) -> dict:
    """
    Compute ECE separately for each drug (per-drug calibration metrics).

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        drug_labels: Drug name per sample (same length as y_true).
        n_bins: Number of calibration bins.

    Returns:
        Dict mapping drug_name → ECE result dict, plus "overall" key.
    """
    results = {"overall": compute_ece(y_true, y_prob, n_bins)}

    for drug in sorted(set(drug_labels)):
        mask = drug_labels == drug
        n_drug = mask.sum()
        if n_drug < 5:
            results[drug] = {"ece": None, "n_samples": int(n_drug),
                             "note": "Too few samples"}
            continue
        results[drug] = compute_ece(y_true[mask], y_prob[mask], n_bins)
        results[drug]["n_samples"] = int(n_drug)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# IG Rank Stability Metrics
# Ref: Sundararajan et al. (2017) — Integrated Gradients
# ══════════════════════════════════════════════════════════════════════════════

def ig_rank_stability(
        ig_rankings_per_seed: list[np.ndarray],
        site_labels: list[str] | None = None,
) -> dict:
    """
    Compute rank stability metrics for IG site rankings across seeds.

    For each pair of seeds, computes Spearman rank correlation on the
    IG attribution values. Reports mean, min, max Spearman ρ.

    Also reports how often the top-k sites are consistent across seeds
    (top-k overlap fraction).

    Args:
        ig_rankings_per_seed: List of arrays, each (n_sites,) of IG values.
                              One per seed.
        site_labels: Optional list of site names for reporting.

    Returns:
        Dict with "mean_spearman_rho", "min_spearman_rho", "pairwise_rhos",
        "top1_consistent", "top3_overlap", "top5_overlap",
        "per_seed_top_site", "n_seeds".
    """
    n_seeds = len(ig_rankings_per_seed)
    if n_seeds < 2:
        return {"n_seeds": n_seeds, "note": "Need ≥2 seeds for stability analysis"}

    # Pairwise Spearman rank correlations
    pairwise_rhos = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            rho, p = stats.spearmanr(ig_rankings_per_seed[i], ig_rankings_per_seed[j])
            pairwise_rhos.append({
                "seed_pair": f"{i}-{j}",
                "spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
                "p_value": float(p) if np.isfinite(p) else 1.0,
            })

    rho_values = [r["spearman_rho"] for r in pairwise_rhos]

    # Top-k consistency
    top_indices_per_seed = [np.argsort(-arr) for arr in ig_rankings_per_seed]
    top1_per_seed = [int(t[0]) for t in top_indices_per_seed]
    top1_consistent = len(set(top1_per_seed)) == 1

    def _topk_overlap(k):
        sets = [set(t[:k].tolist()) for t in top_indices_per_seed]
        # Mean pairwise Jaccard
        overlaps = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                overlaps.append(inter / max(union, 1))
        return float(np.mean(overlaps)) if overlaps else 0.0

    result = {
        "n_seeds": n_seeds,
        "mean_spearman_rho": float(np.mean(rho_values)),
        "min_spearman_rho": float(np.min(rho_values)),
        "max_spearman_rho": float(np.max(rho_values)),
        "pairwise_rhos": pairwise_rhos,
        "top1_consistent": bool(top1_consistent),
        "top3_jaccard_overlap": _topk_overlap(3),
        "top5_jaccard_overlap": _topk_overlap(5),
        "per_seed_top_site_idx": top1_per_seed,
    }

    if site_labels:
        result["per_seed_top_site"] = [
            site_labels[idx] if idx < len(site_labels) else f"site_{idx}"
            for idx in top1_per_seed
        ]

    return result


def ig_permutation_significance(
        observed_ig: np.ndarray,
        model,
        dataset,
        indices: list[int],
        n_permutations: int = 100,
        n_ig_steps: int = 20,
        target: str = "resistance",
) -> dict:
    """
    Permutation-based significance test for top IG sites.

    Shuffles PTM labels across samples and recomputes IG to build a
    null distribution. Reports p-value for each site's observed IG
    against the null.

    Args:
        observed_ig: (n_sites,) array of observed mean IG attributions.
        model: Trained PTM-BDL model.
        dataset: ResistanceDataset.
        indices: Sample indices for IG computation.
        n_permutations: Number of permutation iterations.
        n_ig_steps: IG interpolation steps per permutation.
        target: "resistance" or "ic50".

    Returns:
        Dict with per-site p-values and overall significance summary.
    """
    from src.ptm_bdl.xai.integrated_gradients import compute_ig_batch

    n_sites = len(observed_ig)
    null_dist = np.zeros((n_permutations, n_sites))

    import copy
    for perm_i in range(n_permutations):
        # Shuffle PTM columns in the dataset
        shuffled_dataset = copy.copy(dataset)
        shuffled_df = dataset.df.copy()
        rng = np.random.RandomState(perm_i)
        perm_idx = rng.permutation(len(shuffled_df))

        # Shuffle all ptm_ and delta_ptm_ columns
        for col in shuffled_df.columns:
            if col.startswith("ptm_") or col.startswith("delta_ptm_") or \
               "_slot" in col:
                shuffled_df[col] = shuffled_df[col].values[perm_idx]
        shuffled_dataset.df = shuffled_df

        # Compute IG on shuffled data
        ig_result = compute_ig_batch(
            model, shuffled_dataset, indices[:10],  # use subset for speed
            n_steps=n_ig_steps, target=target,
        )
        # Average across proteins
        all_attrs = []
        for pid, attrs in ig_result.items():
            for key, vals in attrs.items():
                if key != "n_samples" and isinstance(vals, np.ndarray):
                    if len(vals) == n_sites:
                        all_attrs.append(vals)
        if all_attrs:
            null_dist[perm_i] = np.mean(all_attrs, axis=0)

    # Compute per-site p-values: fraction of permutations with IG ≥ observed
    p_values = np.zeros(n_sites)
    for s in range(n_sites):
        p_values[s] = (null_dist[:, s] >= observed_ig[s]).mean()

    n_significant = int((p_values < 0.05).sum())

    return {
        "per_site_p_values": p_values.tolist(),
        "n_sites": n_sites,
        "n_significant_005": n_significant,
        "n_permutations": n_permutations,
        "method": "Permutation Feature Importance on IG attributions",
    }
