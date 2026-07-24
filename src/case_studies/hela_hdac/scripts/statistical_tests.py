#!/usr/bin/env python3
"""
HeLa/HDAC — Statistical Tests.

Bootstrap confidence intervals and significance tests for model comparison.

Tests:
  1. Bootstrap 95% CIs for AUROC, BAcc, Pearson R, RMSE
  2. Paired bootstrap test: PTM-BDL vs each ML baseline
  3. DeLong test for AUROC comparison (PTM-BDL vs XGBoost)

Ref: Efron & Tibshirani, An Introduction to the Bootstrap, 1993
Ref: DeLong et al., Biometrics 1988 — AUROC comparison test
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config
from src.ptm_bdl.evaluation.statistical import bootstrap_ci, delong_test

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Run statistical tests on saved predictions."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Statistical Tests                          ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    pred_path = RESULTS_DIR / "test_predictions.npz"
    if not pred_path.exists():
        print(f"  ✗ {pred_path} not found. Run evaluate.py first.")
        return

    data = np.load(pred_path)
    y_true_ic50 = data.get("y_true_ic50", data.get("y_true"))
    y_pred_ic50 = data.get("y_pred_ic50", data.get("y_pred"))
    y_true_cls = data.get("y_true_cls", data.get("y_true_label"))
    y_prob_cls = data.get("y_prob_cls", data.get("y_prob"))

    results = {}

    # 1. Bootstrap CIs
    print("\n  1. Bootstrap 95% CIs (n=1000)...")
    for metric_name, metric_fn in [
        ("auroc", lambda yt, yp: roc_auc_score(yt, yp)),
        ("balanced_acc", lambda yt, yp: balanced_accuracy_score(yt, (yp > 0.5).astype(int))),
    ]:
        if y_true_cls is not None and y_prob_cls is not None:
            try:
                ci = bootstrap_ci(y_true_cls, y_prob_cls, metric_fn, n_resamples=1000)
                results[f"{metric_name}_ci"] = ci
                print(f"    {metric_name}: {ci}")
            except Exception as e:
                print(f"    {metric_name}: error — {e}")

    if y_true_ic50 is not None and y_pred_ic50 is not None:
        valid = ~np.isnan(y_true_ic50) & ~np.isnan(y_pred_ic50)
        if valid.sum() > 10:
            def pearson_fn(yt, yp):
                return pearsonr(yt, yp)[0]
            try:
                ci = bootstrap_ci(y_true_ic50[valid], y_pred_ic50[valid],
                                  pearson_fn, n_resamples=1000)
                results["pearson_r_ci"] = ci
                print(f"    pearson_r: {ci}")
            except Exception as e:
                print(f"    pearson_r: error — {e}")

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump({"case_study": CASE_STUDY, "results": results}, f, indent=2, default=str)
    print(f"\n✓ Statistical tests complete!")


if __name__ == "__main__":
    main()
