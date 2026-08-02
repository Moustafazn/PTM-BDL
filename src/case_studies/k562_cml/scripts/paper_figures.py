#!/usr/bin/env python3
"""
CS3 (K562/CML) — Publication Figures (Updated).

Generates all main-text and supplementary figures addressing professor feedback:
  - Fig_S_cold_start:      Cold-cell + cold-drug LODO (Q4)
  - Fig_S_cross_dataset:   GDSC→CTRP (Q4)
  - Fig_S_stability:       IG rank stability across seeds (Q5)
  - Fig_S_calibration:     Reliability diagram + ECE (Q9)
  - Fig_S_baseline_ablation: Baseline-only vs delta-only PTM (Q1/Q3)
  - Fig_S_loclo:           LOCLO by leukemia subtype (Q4 — was missing)
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

NM_DPI = 300
SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4

COLORS = {
    "ours": "#0072B2", "ablation_no": "#D55E00", "ablation_full": "#0072B2",
    "phospho": "#56B4E9", "cold_cell": "#CC79A7", "cold_drug": "#E69F00",
    "baseline": "#F0E442",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": NM_DPI, "savefig.dpi": NM_DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.5, "lines.linewidth": 1.0,
    })
    return plt


def load_results():
    results = {}
    for name in ["evaluation_report", "ablation_study", "crossval_results",
                  "ml_baselines", "xai_report", "statistical_tests",
                  "stability_analysis", "loclo_results",
                  "cold_cell_results", "cold_drug_lodo", "cross_dataset_ctrp"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  ✓ Loaded {name}.json")
        else:
            results[name] = None
            print(f"  ⚠ Missing {name}.json")
    return results


def _save(fig, plt, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(FIGURES_DIR / f"{name}{ext}", dpi=NM_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {name}")


# ═══════════════════ Main text figures ═══════════════════

def fig_benchmarking(results, plt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    eval_r = results.get("evaluation_report", {})
    baselines = results.get("ml_baselines", {}).get("internal_baselines",
                results.get("ml_baselines", {}).get("baselines", {}))
    methods = {"PTM-BDL": eval_r.get("overall_metrics", {})}
    methods.update(baselines)

    names = list(methods.keys())
    aurocs = [methods[n].get("auroc", 0) for n in names]
    rmses = [methods[n].get("rmse", 0) for n in names]
    colors = [COLORS["ours"]] + ["#90CAF9"] * (len(names) - 1)

    axes[0].barh(names, aurocs, color=colors, edgecolor="white")
    axes[0].set_xlabel("AUROC"); axes[0].set_title("(a) Classification", fontsize=9, fontweight="bold")
    for i, v in enumerate(aurocs): axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=6)

    axes[1].barh(names, rmses, color=colors, edgecolor="white")
    axes[1].set_xlabel("RMSE"); axes[1].set_title("(b) Regression", fontsize=9, fontweight="bold")
    for i, v in enumerate(rmses): axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=6)

    plt.suptitle("CS3: K562/CML — Benchmarking", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, plt, "Fig_benchmarking")


def fig_ablation(results, plt):
    ablation = results.get("ablation_study", {})
    if not ablation or "full" not in ablation: print("  ⚠ No ablation"); return

    full_auroc = ablation["full"].get("test_metrics", {}).get("auroc", 0)
    modes = [m for m in ablation if m != "full" and not m.startswith("_")]
    labels = [ablation[m].get("label", m) for m in modes]
    deltas = [ablation[m].get("test_metrics", {}).get("auroc", 0) - full_auroc for m in modes]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.35))
    colors = ["#f44336" if d < 0 else "#4CAF50" for d in deltas]
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ AUROC (Ablated − Full)")
    ax.set_title(f"CS3 Ablation (Full AUROC={full_auroc:.3f})", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_ablation")


def fig_ic50_benchmarks(results, plt):
    benchmarks = results.get("evaluation_report", {}).get("published_ic50_benchmarks", {})
    if not benchmarks: print("  ⚠ No IC50 benchmarks"); return
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    drugs = list(benchmarks.keys())
    published = [benchmarks[d].get("published_ln_ic50", 0) for d in drugs]
    predicted = [benchmarks[d].get("predicted_ln_ic50", 0) for d in drugs]
    x = np.arange(len(drugs)); w = 0.35
    ax.bar(x - w/2, published, w, label="Published", color="#2196F3")
    ax.bar(x + w/2, predicted, w, label="Predicted (pan-cancer)", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(drugs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ln(IC50)"); ax.legend(fontsize=7)
    ax.set_title("Predicted vs Published IC50", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_ic50_benchmark")


# ═══════════════════ Supplementary figures ═══════════════════

def fig_s_loclo(results, plt):
    """LOCLO by leukemia subtype — was missing from manuscript."""
    loclo = results.get("loclo_results")
    if not loclo or "per_group_results" not in loclo: print("  ⚠ No LOCLO results"); return
    per_group = loclo["per_group_results"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
    groups, pccs, aurocs, ns = [], [], [], []
    for g, m in per_group.items():
        if "pearson_r" not in m: continue
        groups.append(g); pccs.append(m["pearson_r"]); aurocs.append(m.get("auroc", 0) or 0)
        ns.append(m.get("n_test", 0))
    if not groups: plt.close(fig); return
    x = np.arange(len(groups))
    for panel, vals, ylabel in [(0, pccs, "Pearson R"), (1, aurocs, "AUROC")]:
        ax = axes[panel]
        ax.bar(x, vals, color=COLORS["ours"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)
        for i, (v, n) in enumerate(zip(vals, ns)): ax.text(i, v + 0.01, f"n={n}", ha="center", fontsize=5)
    plt.suptitle("CS3: LOCLO by Leukemia Subtype / Tissue", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, plt, "Fig_S_loclo")


def fig_s_cold_start(results, plt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
    cc = results.get("cold_cell_results")
    if cc and "per_fold_results" in cc:
        ax = axes[0]
        folds = sorted(cc["per_fold_results"].keys())
        aurocs = [cc["per_fold_results"][f]["auroc"] for f in folds]
        x = np.arange(len(folds))
        ax.bar(x, aurocs, color=COLORS["cold_cell"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([f"F{i+1}" for i in range(len(folds))], fontsize=7)
        ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 1.0)
        mean_a = cc["summary"]["mean_auroc"]; std_a = cc["summary"]["std_auroc"]
        ax.axhline(mean_a, color="red", ls="--", lw=0.8, label=f"Mean={mean_a:.3f}±{std_a:.3f}")
        ax.legend(fontsize=6)
        for i, v in enumerate(aurocs): ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6)
    axes[0].set_title("(a) Cold-Cell K-Fold CV", fontsize=9, fontweight="bold")

    cd = results.get("cold_drug_lodo")
    if cd and "per_drug_results" in cd:
        ax = axes[1]
        drugs = sorted(cd["per_drug_results"].keys())
        aurocs = [cd["per_drug_results"][d].get("auroc", 0) or 0 for d in drugs]
        x = np.arange(len(drugs))
        ax.bar(x, aurocs, color=COLORS["cold_drug"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([d[:6] for d in drugs], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.1)
        for i, v in enumerate(aurocs): ax.text(i, max(v, 0) + 0.02, f"{v:.3f}", ha="center", fontsize=6)
    axes[1].set_title("(b) Leave-One-Drug-Out", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_S_cold_start")


def fig_s_cross_dataset(results, plt):
    cd = results.get("cross_dataset_ctrp")
    if not cd or "per_drug" not in cd: print("  ⚠ No cross-dataset"); return
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    drugs = sorted(cd["per_drug"].keys())
    pred_r = [cd["per_drug"][d].get("pred_vs_ctrp_pearson_r", 0) or 0 for d in drugs]
    gdsc_r = [cd["per_drug"][d].get("gdsc_vs_ctrp_pearson_r", 0) or 0 for d in drugs]
    x = np.arange(len(drugs)); w = 0.35
    ax.bar(x - w/2, gdsc_r, w, label="GDSC→CTRP (raw)", color="#009E73")
    ax.bar(x + w/2, pred_r, w, label="PTM-BDL→CTRP", color=COLORS["ours"])
    ax.set_xticks(x); ax.set_xticklabels([d[:6] for d in drugs], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson R"); ax.legend(fontsize=6)
    ax.set_title("Cross-Dataset (GDSC→CTRPv2)", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_S_cross_dataset")


def fig_s_stability(results, plt):
    stability = results.get("stability_analysis")
    if not stability or "per_protein" not in stability: return
    per_protein = stability["per_protein"]
    proteins = sorted(per_protein.keys())
    fig, axes = plt.subplots(1, len(proteins), figsize=(4.5 * len(proteins), 4.5), squeeze=False)
    for col, protein in enumerate(proteins):
        ax = axes[0, col]
        d = per_protein[protein]
        sites = d.get("phospho_site_labels", [])
        vals = d.get("phospho_mean_importance", [])
        if sites and vals:
            active = [(s, v) for s, v in zip(sites, vals) if v > 0]
            if active:
                labs, vs = zip(*sorted(active, key=lambda x: x[1], reverse=True))
                colors = [COLORS["phospho"] if "Y" in s.upper() else "#CC79A7" for s in labs]
                ax.barh(range(len(labs)), vs, color=colors, edgecolor="white")
                ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
                ax.set_xlabel("Mean |IG| (3 seeds)")
        top = d.get("phospho_top_site", "")
        concordant = d.get("cross_seed_top_concordant", False)
        ax.set_title(f"{protein}\nTop: {top} ({'concordant' if concordant else 'varies'})",
                     fontsize=9, fontweight="bold")
    plt.suptitle("CS3: IG Stability Across Seeds", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, plt, "Fig_S_stability")


def fig_s_calibration(results, plt):
    ev = results.get("evaluation_report")
    if not ev or "calibration" not in ev: return
    cal = ev["calibration"]
    overall = cal.get("overall_ece", {})
    per_drug = cal.get("per_drug_ece", {})

    valid_drugs = [k for k in per_drug if k != "overall" and per_drug[k].get("bin_accs")]
    n_panels = min(len(valid_drugs) + 1, 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(DOUBLE_COL, DOUBLE_COL * 0.32), squeeze=False)
    axes = axes[0]

    def _plot_rel(ax, data, title):
        if not data or "bin_accs" not in data: return
        edges = data["bin_edges"]; accs = data["bin_accs"]; counts = data["bin_counts"]
        midpoints = [(edges[i] + edges[i+1]) / 2 for i in range(len(accs))]
        active = [(m, a, n) for m, a, n in zip(midpoints, accs, counts) if n > 0]
        if not active: return
        ms, acs, ns = zip(*active)
        ax.bar(ms, acs, width=0.08, alpha=0.7, color=COLORS["ours"], edgecolor="white")
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Pred Prob", fontsize=6); ax.set_ylabel("Frac Positive", fontsize=6)
        ece = data.get("ece", 0)
        ax.set_title(f"{title}\nECE={ece:.3f}" if ece else title, fontsize=8, fontweight="bold")

    _plot_rel(axes[0], overall, "Overall")
    idx = 1
    for drug in sorted(valid_drugs):
        if idx >= n_panels: break
        _plot_rel(axes[idx], per_drug[drug], drug[:10])
        idx += 1
    plt.suptitle("CS3: Reliability Diagrams", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, plt, "Fig_S_calibration")


def fig_s_baseline_ablation(results, plt):
    """Q1/Q3: Baseline-only vs delta-only vs full PTM ablation."""
    ablation = results.get("ablation_study", {})
    arms_needed = ["no_ptm", "baseline_only", "delta_only", "full"]
    if not all(arm in ablation for arm in arms_needed):
        print("  ⚠ Missing baseline/delta ablation arms"); return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    arm_labels = ["No PTM", "Baseline Only", "Delta Only", "Measured Only", "Full"]
    arm_keys = ["no_ptm", "baseline_only", "delta_only", "measured_only", "full"]
    arm_colors = [COLORS["ablation_no"], COLORS["baseline"], "#E69F00", "#CC79A7", COLORS["ablation_full"]]

    for panel, metric, ylabel in [(0, "auroc", "AUROC"), (1, "balanced_acc", "Balanced Accuracy")]:
        ax = axes[panel]
        vals, valid_labels, valid_colors = [], [], []
        for arm, label, color in zip(arm_keys, arm_labels, arm_colors):
            d = ablation.get(arm, {})
            if not d: continue
            v = d.get("test_metrics", {}).get(metric, 0)
            vals.append(v); valid_labels.append(label); valid_colors.append(color)
        ax.bar(range(len(vals)), vals, color=valid_colors, edgecolor="white")
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)
        for i, v in enumerate(vals): ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6)
        ax.set_title(f"({'a' if panel == 0 else 'b'}) {ylabel}", fontsize=9, fontweight="bold")
    plt.suptitle("CS3: Baseline-Only vs Delta-Only PTM Ablation", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, plt, "Fig_S_baseline_ablation")


# ═══════════════════ Main ═══════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CS3 (K562/CML) — Publication Figures (Updated)            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    plt = setup_matplotlib()
    results = load_results()

    print("\n  ── Main Text Figures ──")
    fig_benchmarking(results, plt)
    fig_ablation(results, plt)
    fig_ic50_benchmarks(results, plt)

    print("\n  ── Supplementary Figures ──")
    fig_s_loclo(results, plt)
    fig_s_cold_start(results, plt)
    fig_s_cross_dataset(results, plt)
    fig_s_stability(results, plt)
    fig_s_calibration(results, plt)
    fig_s_baseline_ablation(results, plt)

    generated = list(FIGURES_DIR.glob("*.pdf"))
    print(f"\n  ✓ Generated {len(generated)} figures in {FIGURES_DIR}")
    print("✓ CS3 figures complete!")


if __name__ == "__main__":
    main()
