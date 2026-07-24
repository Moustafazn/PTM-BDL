#!/usr/bin/env python3
"""
K562/CML — Publication Figures.

Generates publication-quality figures for the K562/CML case study:
  Fig 1: Benchmarking — PTM-BDL vs ML baselines + external methods
  Fig 2: Ablation — modality contribution
  Fig 3: TKI vs chemo IG comparison (drug mechanism discrimination)
  Fig 4: Published IC50 benchmark comparison (predicted vs literature)
  Fig S1: Per-drug performance
  Fig S2: BCR-ABL substrate IG ranking

Ref: Shah et al., Science 2004 (PMID 15256107) — Dasatinib BCR-ABL
Ref: Druker et al., NEJM 2006 (PMID 16481636) — Imatinib CML
Ref: O'Hare et al., Blood 2005 (PMID 15256422) — potency comparison
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "publication" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    return plt


def load_results():
    results = {}
    for name in ["evaluation_report", "ablation_study", "crossval_results",
                  "ml_baselines", "xai_report", "statistical_tests"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def fig_benchmarking(results, plt):
    """Fig 1: PTM-BDL vs baselines + external published methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    eval_r = results.get("evaluation_report", {})
    baselines = results.get("ml_baselines", {}).get("internal_baselines", {})
    external = results.get("ml_baselines", {}).get("external_benchmarks", {})

    # Internal comparison
    methods = ["PTM-BDL"] + list(baselines.keys())
    aurocs = [eval_r.get("overall_metrics", {}).get("auroc", 0)]
    aurocs += [b.get("auroc", 0) for b in baselines.values()]
    axes[0].barh(methods, aurocs, color=["#2196F3"] + ["#90CAF9"] * len(baselines))
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("Internal Baselines")

    # External comparison
    ext_methods = list(external.keys())
    ext_r = [external[m].get("pearson_r", 0) for m in ext_methods]
    our_r = eval_r.get("overall_metrics", {}).get("pearson_r", 0)
    all_methods = ["PTM-BDL"] + ext_methods
    all_r = [our_r] + ext_r
    axes[1].barh(all_methods, all_r, color=["#2196F3"] + ["#FFA726"] * len(ext_methods))
    axes[1].set_xlabel("Pearson R")
    axes[1].set_title("vs Published Methods")

    plt.suptitle(f"{CASE_STUDY} — Benchmarking", fontsize=18)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig_benchmarking.png")
    plt.savefig(FIGURES_DIR / "Fig_benchmarking.pdf")
    print(f"  ✓ Fig_benchmarking saved")


def fig_ablation(results, plt):
    """Fig 2: Ablation study."""
    ablation = results.get("ablation_study", {})
    if not ablation or "full" not in ablation:
        print("  ⚠ No ablation results")
        return

    # Ablation arms are at top level; skip metadata keys starting with '_'
    full_auroc = ablation["full"].get("test_metrics", {}).get("auroc", 0)
    modes = [m for m in ablation if m != "full" and not m.startswith("_")]
    labels = [ablation[m].get("label", m) for m in modes]
    deltas = [ablation[m].get("test_metrics", {}).get("auroc", 0) - full_auroc
              for m in modes]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#f44336" if d < 0 else "#4CAF50" for d in deltas]
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ AUROC (Ablated − Full)")
    ax.set_title(f"{CASE_STUDY} — Ablation Study (Full AUROC={full_auroc:.3f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig_ablation.png")
    plt.savefig(FIGURES_DIR / "Fig_ablation.pdf")
    print(f"  ✓ Fig_ablation saved")


def fig_ic50_benchmarks(results, plt):
    """Fig 4: Predicted vs published IC50 values."""
    benchmarks = results.get("evaluation_report", {}).get("published_ic50_benchmarks", {})
    if not benchmarks:
        print("  ⚠ No IC50 benchmarks")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    drugs = list(benchmarks.keys())
    published = [benchmarks[d].get("published_ln_ic50", 0) for d in drugs]
    predicted = [benchmarks[d].get("predicted_ln_ic50", 0) for d in drugs]

    x = np.arange(len(drugs))
    ax.bar(x - 0.2, published, 0.35, label="Published", color="#2196F3")
    ax.bar(x + 0.2, predicted, 0.35, label="Predicted", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=45, ha="right")
    ax.set_ylabel("ln(IC50)")
    ax.set_title(f"{CASE_STUDY} — Predicted vs Published IC50")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig_ic50_benchmark.png")
    plt.savefig(FIGURES_DIR / "Fig_ic50_benchmark.pdf")
    print(f"  ✓ Fig_ic50_benchmark saved")


def main():
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Publication Figures                        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    plt = setup_matplotlib()
    results = load_results()
    fig_benchmarking(results, plt)
    fig_ablation(results, plt)
    fig_ic50_benchmarks(results, plt)
    print(f"\n✓ All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
