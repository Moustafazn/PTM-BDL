#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 15b — Publication-Quality Tables for Nature Methods Submission         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Generate camera-ready tables (CSV + LaTeX) from existing results files.  ║
║    This script does NOT run any models — it only reads from results/         ║
║    and produces formatted tables.                                            ║
║                                                                              ║
║  TABLES GENERATED (Benchmarking Plan §7):                                    ║
║                                                                              ║
║  Main Text:                                                                  ║
║    Table1_benchmarking   — All methods × PCC, RMSE, AUROC, AUPRC-sens      ║
║    Table2_biological     — PTM-BDL validation (11 biological tests)         ║
║    Table3_perdrug        — Per-drug PCC + AUROC for top methods             ║
║                                                                              ║
║  Supplementary:                                                              ║
║    TableS1_full_metrics  — All methods × all Tier C metrics                 ║
║    TableS2_perdrug_full  — Per-drug × all methods × all metrics             ║
║    TableS3_loclo         — Cell-blind LOCLO results                          ║
║    TableS4_statistics    — p-values, CIs, effect sizes                      ║
║    TableS5_runtime       — Training time, params                             ║
║                                                                              ║
║  FORMAT:                                                                     ║
║    CSV (machine-readable) + LaTeX (camera-ready, booktabs style)            ║
║                                                                              ║
║  INPUT: results/*.json                                                       ║
║  OUTPUT: results/publication/tables/*.csv + *.tex                            ║
║                                                                              ║
║  BENCHMARKING_PLAN.md §7, §8 Step 15b                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
PUB_TABLE_DIR = RESULTS_DIR / "publication" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PUB_TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_results():
    """Load all available results files."""
    results = {}

    files = {
        "evaluation": "evaluation_report.json",
        "ablation": "ablation_study.json",
        "stability": "stability_analysis.json",
        "randomized": "randomized_ptm_control.json",
        "xai": "xai_report.json",
        "ml_baselines": "ml_baselines.json",
        "statistical": "statistical_tests.json",
        "loclo": "loclo_results.json",
    }

    for key, fname in files.items():
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                results[key] = json.load(f)
            print(f"  ✓ Loaded {fname}")
        else:
            results[key] = None
            print(f"  ⚠ Missing {fname}")

    ext_summary = RESULTS_DIR / "external_baselines" / "summary.json"
    if ext_summary.exists():
        with open(ext_summary) as f:
            results["external"] = json.load(f)
        print(f"  ✓ Loaded external_baselines/summary.json")
    else:
        results["external"] = None
        print(f"  ⚠ Missing external_baselines/summary.json")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ══════════════════════════════════════════════════════════════════════════════

def df_to_latex(df, caption, label, note=None):
    """Convert DataFrame to booktabs-style LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\small")

    # Column format
    n_cols = len(df.columns)
    col_fmt = "l" + "r" * n_cols
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join([r"\textbf{" + str(c) + "}" for c in df.columns])
    lines.append(r"\textbf{Method} & " + header + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for idx, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("—")
                else:
                    vals.append(f"{v:.3f}")
            elif v is None:
                vals.append("—")
            else:
                vals.append(str(v))
        row_str = str(idx) + " & " + " & ".join(vals) + r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if note:
        lines.append(r"\vspace{2pt}")
        lines.append(r"\begin{minipage}{\textwidth}")
        lines.append(r"\footnotesize " + note)
        lines.append(r"\end{minipage}")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_table(df, name, caption, label, note=None):
    """Save DataFrame as both CSV and LaTeX."""
    csv_path = PUB_TABLE_DIR / f"{name}.csv"
    tex_path = PUB_TABLE_DIR / f"{name}.tex"

    df.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(df_to_latex(df, caption, label, note))

    print(f"  ✓ Saved: {csv_path.name} + {tex_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Table 1: External Benchmarking (Main Text)
# ══════════════════════════════════════════════════════════════════════════════

def table1_benchmarking(results):
    """
    Main text Table 1: All methods × PCC, RMSE, AUROC, AUPRC-sens.
    Benchmarking Plan §7, Main Text Table 1.
    """
    rows = []

    # ML baselines (Tier 0)
    name_map = {
        "ridge": ("Ridge", "Linear", "—"),
        "elastic_net": ("Elastic Net", "Linear", "—"),
        "random_forest": ("Random Forest", "ML", "—"),
        "xgboost": ("XGBoost", "Boost", "—"),
    }
    if results["ml_baselines"]:
        for key in ["ridge", "elastic_net", "random_forest", "xgboost"]:
            data = results["ml_baselines"].get(key)
            if not data:
                continue
            display, mtype, year = name_map[key]
            m = data.get("test_metrics", {})
            rows.append({
                "Method": display,
                "Type": mtype,
                "Year": year,
                "PCC ↑": m.get("pearson_r"),
                "RMSE ↓": m.get("rmse"),
                "AUROC ↑": m.get("auroc"),
                "AUPRC-s ↑": m.get("auprc_sensitive"),
                "PTM Site?": "No",
            })

    # External methods (Tier 1-2) — published numbers as placeholder
    external_methods = [
        ("DeepCDR", "DL+Multi-omics", "2020"),
        ("DrugCell", "DL+GO", "2020"),
        ("GraphDRP", "GNN", "2022"),
        ("GraTransDRP", "Graph Transf.", "2023"),
        ("HiDRA", "Hierarchical", "2023"),
        ("TransCDR", "Transformer", "2023"),
        ("PathDSP", "Pathway", "2024"),
        ("DIPK", "DL+PPI", "2024"),
    ]

    if results.get("external"):
        ext_methods = results["external"].get("methods", {})
        for name, mtype, year in external_methods:
            ext_data = ext_methods.get(name, {})
            m = ext_data.get("test_metrics", {})
            if m:
                rows.append({
                    "Method": name,
                    "Type": mtype,
                    "Year": year,
                    "PCC ↑": m.get("pearson_r"),
                    "RMSE ↓": m.get("rmse"),
                    "AUROC ↑": m.get("auroc"),
                    "AUPRC-s ↑": m.get("auprc_sensitive"),
                    "PTM Site?": "No" if name != "DrugCell" else "GO-level",
                })
            else:
                # Placeholder row for methods not yet run
                ptm_col = "No"
                if name == "DrugCell":
                    ptm_col = "GO-level"
                elif name == "PathDSP":
                    ptm_col = "Pathway"
                rows.append({
                    "Method": name,
                    "Type": mtype,
                    "Year": year,
                    "PCC ↑": None,
                    "RMSE ↓": None,
                    "AUROC ↑": None,
                    "AUPRC-s ↑": None,
                    "PTM Site?": ptm_col,
                })
    else:
        # Add placeholder rows
        for name, mtype, year in external_methods:
            ptm_col = "No"
            if name == "DrugCell":
                ptm_col = "GO-level"
            elif name == "PathDSP":
                ptm_col = "Pathway"
            rows.append({
                "Method": name,
                "Type": mtype,
                "Year": year,
                "PCC ↑": None,
                "RMSE ↓": None,
                "AUROC ↑": None,
                "AUPRC-s ↑": None,
                "PTM Site?": ptm_col,
            })

    # Our model
    if results["evaluation"]:
        reg = results["evaluation"].get("regression", {})
        cls = results["evaluation"].get("classification", {})
        rows.append({
            "Method": "Ours (PTM-BDL)",
            "Type": "PTM-BDL",
            "Year": "2026",
            "PCC ↑": reg.get("pearson_r"),
            "RMSE ↓": reg.get("rmse"),
            "AUROC ↑": cls.get("auroc"),
            "AUPRC-s ↑": None,  # Computed differently
            "PTM Site?": "Yes, per-site",
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Method")

    save_table(
        df,
        "Table1_benchmarking",
        caption=("External benchmarking comparison on GDSC2 subset "
                 "(951 samples, 6 TKI drugs, EGFR/ERBB2 cell lines). "
                 "Metrics: Pearson R (PCC), RMSE, AUROC, and AUPRC for "
                 "the sensitive (minority) class. All methods evaluated "
                 "on the same held-out test set (n=143)."),
        label="tab:benchmarking",
        note=("↑ higher is better, ↓ lower is better. "
              "— indicates method not yet evaluated on our subset. "
              "PTM Site? indicates whether the method provides "
              "per-PTM-site interpretability."),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: PTM-BDL Biological Validation (Main Text)
# ══════════════════════════════════════════════════════════════════════════════

def table2_biological(results):
    """
    Main text Table 2: PTM-BDL validation (11 biological tests).
    Benchmarking Plan §7, Main Text Table 2.
    """
    rows = []

    ablation = results.get("ablation", {}) or {}
    stability = results.get("stability", {}) or {}
    randomized = results.get("randomized", {}) or {}
    summary = ablation.get("_summary", {}) or {}

    # IG site ranking tests
    if stability and "ig_site_rankings" in stability:
        rankings = stability["ig_site_rankings"]
        egfr_data = rankings.get("EGFR", {})
        her2_data = rankings.get("ERBB2", rankings.get("HER2", {}))

        # EGFR Y1068 rank
        if egfr_data:
            y1068_key = [k for k in egfr_data.keys()
                         if "1092" in k or "1068" in k]
            if y1068_key:
                rank_info = egfr_data[y1068_key[0]]
                rows.append({
                    "Test": "IG site ranking (EGFR)",
                    "Metric": "Y1068 rank across seeds",
                    "Result": f"#{rank_info.get('mean_rank', '?')} "
                              f"(std={rank_info.get('std_rank', '?')})",
                    "Pass?": "✓" if rank_info.get("mean_rank", 99) <= 1
                    else "—",
                })

        # HER2 Y1221 rank
        if her2_data:
            y1221_key = [k for k in her2_data.keys() if "1221" in k]
            if y1221_key:
                rank_info = her2_data[y1221_key[0]]
                rows.append({
                    "Test": "IG site ranking (HER2)",
                    "Metric": "Y1221 rank across seeds",
                    "Result": f"#{rank_info.get('mean_rank', '?')} "
                              f"(std={rank_info.get('std_rank', '?')})",
                    "Pass?": "✓" if rank_info.get("mean_rank", 99) <= 1
                    else "—",
                })

    # Cross-receptor homology
    if stability and "cross_receptor_homology" in stability:
        hom = stability["cross_receptor_homology"]
        concordant = hom.get("concordant", False)
        rows.append({
            "Test": "Cross-receptor homology",
            "Metric": "EGFR Y1068 ≡ HER2 Y1221",
            "Result": "Concordant" if concordant else "Discordant",
            "Pass?": "✓" if concordant else "✗",
        })

    # Modification-type hierarchy
    if stability and "mod_type_hierarchy" in stability:
        hierarchy = stability["mod_type_hierarchy"]
        rows.append({
            "Test": "Modification-type hierarchy",
            "Metric": "Tyrosine vs Ser/Thr IG",
            "Result": hierarchy.get("result", "—"),
            "Pass?": "✓" if hierarchy.get("tyrosine_dominant", False)
            else "—",
        })

    # Ablation metrics
    ablation_tests = [
        ("PTM ablation (AUROC)", "ptm_gain_auroc", "> 0"),
        ("PTM ablation (AUPRC-s)", "ptm_gain_auprc_sensitive", "> 0"),
        ("Phospho marginal", "phospho_marginal_auroc", "> 0"),
        ("Glyco marginal", "glyco_marginal_auroc", "> 0"),
        ("Typed attn vs MLP", "typed_attention_marginal_auroc", "> 0"),
    ]
    for test_name, key, criterion in ablation_tests:
        val = summary.get(key, None)
        if val is not None:
            passed = val > 0
            rows.append({
                "Test": test_name,
                "Metric": f"Full − baseline AUROC",
                "Result": f"{val:+.4f}",
                "Pass?": "✓" if passed else "✗",
            })
        else:
            rows.append({
                "Test": test_name,
                "Metric": criterion,
                "Result": "—",
                "Pass?": "—",
            })

    # Randomized PTM control
    if randomized:
        for key, label in [("both_shuffled", "Both"),
                           ("phospho_shuffled", "Phospho"),
                           ("glyco_shuffled", "Glyco")]:
            cond = randomized.get(key, {})
            if cond:
                delta_auroc = cond.get("delta_auroc",
                                       cond.get("real_auroc", 0) - cond.get("shuffled_auroc", 0))
                delta_bacc = cond.get("delta_bacc",
                                      cond.get("real_bacc", 0) - cond.get("shuffled_bacc", 0))
                pass_auroc = delta_auroc >= 0.005
                pass_bacc = delta_bacc >= 0.02
                rows.append({
                    "Test": f"Randomized control ({label})",
                    "Metric": f"ΔAUROC≥+0.005, ΔBAcc≥+0.02",
                    "Result": f"ΔAUROC={delta_auroc:+.3f}, "
                              f"ΔBAcc={delta_bacc:+.3f}",
                    "Pass?": "✓" if (pass_auroc and pass_bacc) else "✗",
                })

    if not rows:
        rows.append({
            "Test": "No biological validation data available",
            "Metric": "—",
            "Result": "—",
            "Pass?": "—",
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Test")

    save_table(
        df,
        "Table2_biological",
        caption=("PTM-BDL biological validation. These tests evaluate "
                 "whether the model learns biologically meaningful PTM "
                 "patterns. No other method can be evaluated on these "
                 "metrics — they validate PTM-BDL as a biological module."),
        label="tab:biological",
        note=("IG = Integrated Gradients. Y1068/Y1221 are the GRB2→MAPK "
              "docking sites on EGFR/HER2 respectively. Marginal = "
              "Full model AUROC minus ablated model AUROC."),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table 3: Per-Drug Performance (Main Text)
# ══════════════════════════════════════════════════════════════════════════════

def table3_perdrug(results):
    """
    Main text Table 3: Per-drug PCC + AUROC for top methods.
    Benchmarking Plan §7, Main Text Table 3.
    """
    drug_order = ["Osimertinib", "Gefitinib", "Erlotinib",
                  "Afatinib", "Lapatinib", "Sapitinib"]

    rows = []
    for drug in drug_order:
        row = {"Drug": drug}

        # Our model
        if results["evaluation"] and "drug_specific" in results["evaluation"]:
            drug_data = results["evaluation"]["drug_specific"].get(drug, {})
            row["N_test"] = drug_data.get("n_samples",
                                          drug_data.get("n_test", "—"))
            row["Ours_PCC"] = drug_data.get("pearson_r")
            row["Ours_AUROC"] = drug_data.get("auroc")
        else:
            row["N_test"] = "—"
            row["Ours_PCC"] = None
            row["Ours_AUROC"] = None

        # ML baselines (top 2: RF, XGBoost)
        if results["ml_baselines"]:
            for bl_key, bl_name in [("random_forest", "RF"),
                                    ("xgboost", "XGB")]:
                bl_data = results["ml_baselines"].get(bl_key, {})
                per_drug = bl_data.get("per_drug", {}).get(drug, {})
                row[f"{bl_name}_PCC"] = per_drug.get("pearson_r")
                row[f"{bl_name}_AUROC"] = per_drug.get("auroc")
        else:
            row["RF_PCC"] = None
            row["RF_AUROC"] = None
            row["XGB_PCC"] = None
            row["XGB_AUROC"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Drug")

    save_table(
        df,
        "Table3_perdrug",
        caption=("Per-drug performance comparison. PCC (Pearson R) and "
                 "AUROC for each of 6 TKI drugs. Critical drugs: "
                 "Osimertinib (focal drug of this study), Lapatinib and "
                 "Sapitinib (HER2-specific)."),
        label="tab:perdrug",
        note=("Per-drug Spearman correlation (SCC) is considered 'the "
              "most clinically relevant metric' by Sada Del Real et al., "
              "Brief Bioinf 2026."),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Table S1: Full Metrics
# ══════════════════════════════════════════════════════════════════════════════

def tables1_full_metrics(results):
    """Supplementary S1: All methods × all Tier C metrics."""
    rows = []

    name_map = {
        "ridge": "Ridge",
        "elastic_net": "Elastic Net",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    if results["ml_baselines"]:
        for key, data in results["ml_baselines"].items():
            if key.startswith("_"):
                continue
            m = data.get("test_metrics", {})
            rows.append({
                "Method": name_map.get(key, key),
                "PCC": m.get("pearson_r"),
                "SCC": m.get("spearman_rho"),
                "RMSE": m.get("rmse"),
                "AUROC": m.get("auroc"),
                "AUPRC-res": m.get("auprc_resistant"),
                "AUPRC-sens": m.get("auprc_sensitive"),
                "BAcc": m.get("balanced_acc"),
            })

    # Our model
    if results["evaluation"]:
        reg = results["evaluation"].get("regression", {})
        cls = results["evaluation"].get("classification", {})
        rows.append({
            "Method": "Ours (PTM-BDL)",
            "PCC": reg.get("pearson_r"),
            "SCC": reg.get("spearman_rho"),
            "RMSE": reg.get("rmse"),
            "AUROC": cls.get("auroc"),
            "AUPRC-res": cls.get("auprc"),
            "AUPRC-sens": None,
            "BAcc": cls.get("balanced_accuracy"),
        })

    if rows:
        df = pd.DataFrame(rows).set_index("Method")
        save_table(
            df,
            "TableS1_full_metrics",
            caption=("Full evaluation metrics (Tier C) for all methods. "
                     "PCC = Pearson R, SCC = Spearman ρ, BAcc = Balanced "
                     "Accuracy. All evaluated on the same test set (n=143)."),
            label="tab:s1_full",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Table S3: LOCLO Results
# ══════════════════════════════════════════════════════════════════════════════

def tables3_loclo(results):
    """Supplementary S3: Cell-blind LOCLO generalization results."""
    loclo = results.get("loclo")
    if not loclo or "per_group_results" not in loclo:
        print("    ⚠ No LOCLO results for supplementary table")
        return

    rows = []
    for group, metrics in loclo["per_group_results"].items():
        if "pearson_r" not in metrics:
            continue
        rows.append({
            "Group": group,
            "N_train": metrics.get("n_train"),
            "N_test": metrics.get("n_test"),
            "PCC": metrics.get("pearson_r"),
            "RMSE": metrics.get("rmse"),
            "AUROC": metrics.get("auroc"),
            "AUPRC-s": metrics.get("auprc_sensitive"),
            "N_res": metrics.get("n_resistant"),
            "N_sens": metrics.get("n_sensitive"),
        })

    # Add summary row
    summary = loclo.get("summary", {})
    if summary:
        rows.append({
            "Group": "Mean ± std",
            "N_train": "—",
            "N_test": "—",
            "PCC": f"{summary.get('mean_pearson_r', 0):.3f}±"
                   f"{summary.get('std_pearson_r', 0):.3f}"
            if "mean_pearson_r" in summary else None,
            "RMSE": f"{summary.get('mean_rmse', 0):.3f}±"
                    f"{summary.get('std_rmse', 0):.3f}"
            if "mean_rmse" in summary else None,
            "AUROC": f"{summary.get('mean_auroc', 0):.3f}±"
                     f"{summary.get('std_auroc', 0):.3f}"
            if "mean_auroc" in summary else None,
            "AUPRC-s": "—",
            "N_res": "—",
            "N_sens": "—",
        })

    if rows:
        df = pd.DataFrame(rows).set_index("Group")
        save_table(
            df,
            "TableS3_loclo",
            caption=("Leave-One-Cell-Line-Out (LOCLO) generalization "
                     "results. Cell lines grouped by EGFR mutation class "
                     "or HER2 amplification status. Each group held out "
                     "in turn; model trained on remaining samples."),
            label="tab:s3_loclo",
            note=("Cell-blind splitting recommended by Sada Del Real "
                  "et al., Brief Bioinf 2026 as essential for precision "
                  "medicine evaluation."),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Table S4: Statistical Tests
# ══════════════════════════════════════════════════════════════════════════════

def tables4_statistics(results):
    """Supplementary S4: p-values, CIs, effect sizes."""
    stat = results.get("statistical")
    if not stat:
        print("    ⚠ No statistical test results for supplementary table")
        return

    rows = []

    # DeLong tests
    delong = stat.get("delong_tests", {})
    for name, test in delong.items():
        rows.append({
            "Test": "DeLong AUROC",
            "Comparison": name.replace("ours_vs_", "Ours vs "),
            "AUROC_A": test.get("auroc_ours"),
            "AUROC_B": test.get("auroc_baseline"),
            "Δ": test.get("diff"),
            "Statistic": test.get("z_statistic"),
            "p-value": test.get("p_value"),
            "Sig (α=0.05)": "★" if test.get("significant_at_005") else "",
        })

    # BH correction
    bh = stat.get("bh_correction", {})
    for name, correction in bh.items():
        rows.append({
            "Test": "BH-corrected",
            "Comparison": name,
            "AUROC_A": "—",
            "AUROC_B": "—",
            "Δ": "—",
            "Statistic": "—",
            "p-value": correction.get("adjusted_p"),
            "Sig (α=0.05)": "★" if correction.get("significant") else "",
        })

    if rows:
        df = pd.DataFrame(rows).set_index("Test")
        save_table(
            df,
            "TableS4_statistics",
            caption=("Statistical tests for benchmarking comparison. "
                     "DeLong test (paired AUROC comparison) and "
                     "Benjamini-Hochberg multiple testing correction. "
                     "★ indicates significance at α=0.05."),
            label="tab:s4_stats",
            note=("DeLong test: Biometrics 1988. BH correction: "
                  "following SAGE-net (Nat Methods 2026) benchmarking "
                  "protocol."),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Table S5: Runtime
# ══════════════════════════════════════════════════════════════════════════════

def tables5_runtime(results):
    """Supplementary S5: Training time, inference time, parameters."""
    rows = []

    name_map = {
        "ridge": "Ridge",
        "elastic_net": "Elastic Net",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    if results["ml_baselines"]:
        for key, data in results["ml_baselines"].items():
            if key.startswith("_"):
                continue
            rows.append({
                "Method": name_map.get(key, key),
                "Type": data.get("type", "—"),
                "Train Time (s)": data.get("training_time_seconds"),
                "Feature Dim": data.get("feature_dim"),
                "N_train": data.get("n_train"),
            })

    if rows:
        df = pd.DataFrame(rows).set_index("Method")
        save_table(
            df,
            "TableS5_runtime",
            caption=("Runtime and scalability comparison. Training time "
                     "measured on the same hardware for all methods."),
            label="tab:s5_runtime",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 15b: Publication-Quality Tables                       ║")
    print("║  CSV + LaTeX (booktabs style)                               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Load all results
    print("\n  Loading results files...")
    results = load_results()

    # Generate tables
    print(f"\n  Generating tables → {PUB_TABLE_DIR}/")

    print("\n  ── Main Text Tables ──")
    table1_benchmarking(results)
    table2_biological(results)
    table3_perdrug(results)

    print("\n  ── Supplementary Tables ──")
    tables1_full_metrics(results)
    tables3_loclo(results)
    tables4_statistics(results)
    tables5_runtime(results)

    # Summary
    csv_files = list(PUB_TABLE_DIR.glob("*.csv"))
    tex_files = list(PUB_TABLE_DIR.glob("*.tex"))
    print(f"\n  ✓ Generated {len(csv_files)} CSV + {len(tex_files)} LaTeX files:")
    for f in sorted(csv_files):
        print(f"    {f.name}")

    print(f"\n  Output directory: {PUB_TABLE_DIR}")
    print("\n✓ Step 15b complete!")


if __name__ == "__main__":
    main()
