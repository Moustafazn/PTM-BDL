#!/usr/bin/env python3
"""
K562/CML — Publication Tables.

Generates LaTeX + CSV tables:
  Table 1: Benchmarking — PTM-BDL vs baselines + external methods
  Table 2: Per-drug evaluation (TKI vs chemo)
  Table 3: Ablation results
  Table 4: Published IC50 benchmark comparison
  Table S1: Cross-validation results
  Table S2: Statistical tests
"""
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
TABLES_DIR = RESULTS_DIR / "publication" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    results = {}
    for name in ["evaluation_report", "ablation_study", "crossval_results",
                  "ml_baselines", "statistical_tests"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def save_table(df, name, caption, label):
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=True)
    latex = df.to_latex(caption=caption, label=label, float_format="%.3f")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  ✓ {name} saved")


def table_benchmarking(results):
    eval_r = results.get("evaluation_report", {}).get("overall_metrics", {})
    baselines = results.get("ml_baselines", {}).get("internal_baselines", {})
    external = results.get("ml_baselines", {}).get("external_benchmarks", {})

    rows = {"PTM-BDL": eval_r}
    rows.update(baselines)
    for name, info in external.items():
        rows[f"{name} (published)"] = {
            "pearson_r": info.get("pearson_r"),
            "rmse": info.get("rmse"),
        }

    metrics = ["auroc", "balanced_acc", "pearson_r", "rmse"]
    data = {m: {name: r.get(m, None) for name, r in rows.items()} for m in metrics}
    df = pd.DataFrame(data).T
    save_table(df, "Table1_benchmarking",
               f"{CASE_STUDY} — Benchmarking", "tab:benchmarking")


def table_perdrug(results):
    per_drug = results.get("evaluation_report", {}).get("per_drug", {})
    if not per_drug:
        return
    rows = {drug: {k: v for k, v in m.items() if isinstance(v, (int, float))}
            for drug, m in per_drug.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "Drug"
    save_table(df, "Table2_perdrug",
               f"{CASE_STUDY} — Per-Drug", "tab:perdrug")


def table_ic50_benchmarks(results):
    benchmarks = results.get("evaluation_report", {}).get("published_ic50_benchmarks", {})
    if not benchmarks:
        return
    df = pd.DataFrame(benchmarks).T
    save_table(df, "Table4_ic50_benchmarks",
               f"{CASE_STUDY} — Published IC50 Benchmarks", "tab:ic50bench")


def table_ablation(results):
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
               f"{CASE_STUDY} — Ablation", "tab:ablation")


def main():
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Publication Tables                         ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    results = load_results()
    table_benchmarking(results)
    table_perdrug(results)
    table_ablation(results)
    table_ic50_benchmarks(results)
    print(f"\n✓ All tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
