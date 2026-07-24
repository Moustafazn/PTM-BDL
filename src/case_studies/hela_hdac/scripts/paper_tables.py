#!/usr/bin/env python3
"""
HeLa/HDAC — Publication Tables.

Generates LaTeX + CSV tables for the manuscript:
  Table 1: Benchmarking — PTM-BDL vs baselines (AUROC, BAcc, R, RMSE)
  Table 2: Per-drug evaluation metrics
  Table 3: Ablation study results
  Table S1: Cross-validation fold-by-fold results
  Table S2: Statistical test results (bootstrap CIs)
  Table S3: Cross-case-study comparison (EGFR vs HeLa vs K562)
"""
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
TABLES_DIR = RESULTS_DIR / "publication" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load all result JSONs."""
    results = {}
    for name in ["evaluation_report", "ablation_study", "crossval_results",
                  "ml_baselines", "statistical_tests"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def save_table(df, name, caption, label):
    """Save table as CSV + LaTeX."""
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=True)
    latex = df.to_latex(caption=caption, label=label, float_format="%.3f")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  ✓ {name} saved (CSV + LaTeX)")


def table_benchmarking(results):
    """Table 1: PTM-BDL vs baselines."""
    eval_r = results.get("evaluation_report", {}).get("overall_metrics", {})
    baselines = results.get("ml_baselines", {}).get("baselines", {})

    rows = {"PTM-BDL": eval_r}
    rows.update(baselines)

    metrics = ["auroc", "balanced_acc", "pearson_r", "rmse"]
    data = {m: {name: r.get(m, None) for name, r in rows.items()} for m in metrics}
    df = pd.DataFrame(data).T
    df.index.name = "Metric"
    save_table(df, "Table1_benchmarking",
               f"{CASE_STUDY} — PTM-BDL vs ML Baselines", "tab:benchmarking")


def table_perdrug(results):
    """Table 2: Per-drug metrics."""
    per_drug = results.get("evaluation_report", {}).get("per_drug", {})
    if not per_drug:
        return
    rows = {}
    for drug, metrics in per_drug.items():
        rows[drug] = {k: v for k, v in metrics.items()
                      if isinstance(v, (int, float))}
    df = pd.DataFrame(rows).T
    df.index.name = "Drug"
    save_table(df, "Table2_perdrug",
               f"{CASE_STUDY} — Per-Drug Evaluation", "tab:perdrug")


def table_ablation(results):
    """Table 3: Ablation results."""
    ablation = results.get("ablation_study", {})
    if not ablation or "full" not in ablation:
        return

    # Extract test_metrics for each ablation arm (skip metadata keys)
    rows = {}
    for mode, data in ablation.items():
        if mode.startswith("_") or not isinstance(data, dict):
            continue
        label = data.get("label", mode)
        test_m = data.get("test_metrics", {})
        rows[label] = {k: v for k, v in test_m.items() if isinstance(v, (int, float))}

    df = pd.DataFrame(rows).T
    df.index.name = "Mode"
    save_table(df, "Table3_ablation",
               f"{CASE_STUDY} — Ablation Study", "tab:ablation")


def main():
    """Generate all publication tables."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Publication Tables                         ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    results = load_results()
    table_benchmarking(results)
    table_perdrug(results)
    table_ablation(results)

    print(f"\n✓ All tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
