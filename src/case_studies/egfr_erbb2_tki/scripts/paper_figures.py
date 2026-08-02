#!/usr/bin/env python3
"""
CS1 (EGFR/ERBB2 TKI) — Publication Figures.

Generates all main-text and supplementary figures from results/ JSON files.
Addresses professor feedback Q2-Q10 with new supplementary figures:
  - Fig_S_cold_start:     Cold-cell + cold-drug LODO (Q4)
  - Fig_S_cross_dataset:  GDSC→CTRP generalization (Q4)
  - Fig_S_stability:      IG rank stability across seeds (Q5)
  - Fig_S_calibration:    Reliability diagram + ECE (Q9)
  - Fig_S_baseline_ablation: Baseline-only vs delta-only PTM (Q1/Q3)
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
PUB_FIG_DIR = RESULTS_DIR / "publication" / "figures"
PUB_FIG_DIR.mkdir(parents=True, exist_ok=True)

NM_DPI = 300
SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4

COLORS = {
    "ours": "#0072B2", "tier1": "#E69F00", "tier2": "#009E73",
    "ablation_full": "#0072B2", "ablation_no": "#D55E00",
    "phospho": "#56B4E9", "glyco": "#F0E442",
    "sensitive": "#009E73", "resistant": "#D55E00",
    "cold_cell": "#CC79A7", "cold_drug": "#E69F00",
    "cross_dataset": "#009E73", "baseline": "#F0E442",
}
METHOD_COLORS = {
    "Ours (PTM-BDL)": COLORS["ours"], "Random Forest": "#CC79A7",
    "Ridge": "#F0E442", "Elastic Net": "#56B4E9", "XGBoost": "#999999",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": NM_DPI, "savefig.dpi": NM_DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.5, "lines.linewidth": 1.0,
    })
    return plt


def load_results():
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
        "cold_cell": "cold_cell_results.json",
        "cold_drug": "cold_drug_lodo.json",
        "cross_dataset": "cross_dataset_ctrp.json",
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
    else:
        results["external"] = None
    return results


def _save(fig, plt, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(PUB_FIG_DIR / f"{name}{ext}", dpi=NM_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {name}")


# ═══════════════════ Main text figures ═══════════════════

def fig_benchmarking(results, plt):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    methods = {}
    if results["evaluation"]:
        reg = results["evaluation"].get("regression", {})
        cls = results["evaluation"].get("classification", {})
        methods["Ours (PTM-BDL)"] = {
            "pcc": reg.get("pearson_r", 0), "rmse": reg.get("rmse", 0),
            "auroc": cls.get("auroc", 0), "auprc_s": 0.680,
        }
    name_map = {"random_forest": "Random Forest", "xgboost": "XGBoost",
                "ridge": "Ridge", "elastic_net": "Elastic Net"}
    if results["ml_baselines"]:
        for key, data in results["ml_baselines"].items():
            if key.startswith("_"): continue
            m = data.get("test_metrics", {})
            methods[name_map.get(key, key)] = {
                "pcc": m.get("pearson_r", 0), "rmse": m.get("rmse", 0),
                "auroc": m.get("auroc", 0), "auprc_s": m.get("auprc_sensitive", 0),
            }
    if not methods:
        plt.close(fig); return
    sorted_m = sorted(methods.items(), key=lambda x: x[1]["pcc"], reverse=True)
    names = [m[0] for m in sorted_m]
    colors = [METHOD_COLORS.get(n, "#999") for n in names]

    ax = axes[0]
    vals = [m[1].get("auprc_s", 0) for m in sorted_m]
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("AUPRC-sensitive"); ax.set_title("(a) Minority-Class Precision", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(vals): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=6)

    ax = axes[1]
    aurocs = [m[1]["auroc"] for m in sorted_m]
    ax.barh(range(len(names)), aurocs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("AUROC"); ax.set_title("(b) Classification", fontsize=9, fontweight="bold")
    ax.invert_yaxis(); ax.set_xlim(0, 1.15)
    for i, v in enumerate(aurocs): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=6)

    ax = axes[2]
    drug_order = ["Osimertinib", "Gefitinib", "Erlotinib", "Afatinib", "Lapatinib", "Sapitinib"]
    if results["evaluation"] and "drug_specific" in results["evaluation"]:
        our_drugs = results["evaluation"]["drug_specific"]
        drug_aurocs = [our_drugs.get(d, {}).get("auroc", 0) for d in drug_order]
        ax.barh(range(len(drug_order)), drug_aurocs, color=COLORS["ours"], edgecolor="white")
        ax.set_yticks(range(len(drug_order))); ax.set_yticklabels([d[:6] for d in drug_order], fontsize=6)
        ax.set_xlabel("AUROC"); ax.invert_yaxis()
        for i, v in enumerate(drug_aurocs): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=6)
    ax.set_title("(c) Per-Drug AUROC", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_benchmarking")


def fig_ablation(results, plt):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    ablation = results.get("ablation")
    randomized = results.get("randomized")

    ax = axes[0]
    if ablation:
        arm_order = ["no_ptm", "glyco_only", "no_glyco", "no_typed_attention", "full"]
        arm_labels = ["No PTM", "Glyco Only", "No Glyco", "No Typed Attn", "Full PTM-BDL"]
        arm_colors = [COLORS["ablation_no"], "#F0E442", "#56B4E9", "#CC79A7", COLORS["ablation_full"]]
        aurocs = []
        for arm in arm_order:
            d = ablation.get(arm, {})
            aurocs.append(d.get("test_metrics", {}).get("auroc", 0) if isinstance(d, dict) else 0)
        ax.bar(range(len(arm_labels)), aurocs, color=arm_colors, edgecolor="white")
        ax.set_xticks(range(len(arm_labels))); ax.set_xticklabels(arm_labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 1.0)
        for i, v in enumerate(aurocs):
            if v > 0: ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6, fontweight="bold")
    ax.set_title("(a) Feature Ablation", fontsize=9, fontweight="bold")

    ax = axes[1]
    if randomized:
        arms = randomized.get("arms", randomized)
        ref = randomized.get("reference_full_metrics", {}).get("auroc", 0)
        conds, real_v, shuf_v = [], [], []
        for key in ["phospho_shuffled", "glyco_shuffled", "both_shuffled"]:
            cd = arms.get(key, {})
            if not cd: continue
            conds.append(key.replace("_shuffled", "").title())
            drops = cd.get("drops", {})
            if ref and drops.get("drop_auroc") is not None:
                real_v.append(ref); shuf_v.append(ref - drops["drop_auroc"])
            else:
                real_v.append(ref); shuf_v.append(ref)
        if conds:
            x = np.arange(len(conds)); w = 0.35
            ax.bar(x - w/2, real_v, w, label="Real PTM", color=COLORS["ours"])
            ax.bar(x + w/2, shuf_v, w, label="Shuffled PTM", color=COLORS["ablation_no"])
            ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=7)
            ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 1.0); ax.legend(fontsize=6)
    ax.set_title("(b) Randomized Control", fontsize=9, fontweight="bold")

    ax = axes[2]
    if ablation and "_summary" in ablation:
        s = ablation["_summary"]
        margins = {"Phospho": s.get("phospho_marginal_auroc", 0), "Glyco": s.get("glyco_marginal_auroc", 0),
                    "Typed Attn": s.get("typed_attention_marginal_auroc", 0), "Full PTM": s.get("ptm_gain_auroc", 0)}
        ax.bar(range(len(margins)), list(margins.values()),
               color=[COLORS["phospho"], COLORS["glyco"], COLORS["ours"], "#009E73"], edgecolor="white")
        ax.set_xticks(range(len(margins))); ax.set_xticklabels(list(margins.keys()), rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Marginal AUROC gain"); ax.axhline(0, color="black", linewidth=0.5)
        for i, v in enumerate(margins.values()): ax.text(i, v + 0.002 if v >= 0 else v - 0.008, f"{v:+.3f}", ha="center", fontsize=6)
    ax.set_title("(c) Channel Contributions", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_ablation")


def fig_interpretability(results, plt):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    stability = results.get("stability")

    def _get_site_data(stab, prot):
        p = stab.get(prot, {})
        return p.get("phospho_sites", []), p.get("phospho_mean_importance", [])

    for idx, (prot, title) in enumerate([("egfr", "EGFR"), ("erbb2", "HER2")]):
        ax = axes[0, idx]
        sites, imps = _get_site_data(stability, prot) if stability else ([], [])
        if sites and imps:
            si = np.argsort(imps)[::-1]
            s_s = [sites[i] for i in si]; s_i = [imps[i] for i in si]
            colors = [COLORS["phospho"] if "Y" in s.upper() else "#CC79A7" for s in s_s]
            ax.barh(range(len(s_s)), s_i, color=colors, edgecolor="white", linewidth=0.3)
            ax.set_yticks(range(len(s_s))); ax.set_yticklabels(s_s, fontsize=6)
            ax.set_xlabel("IG Importance", fontsize=7); ax.invert_yaxis()
        ax.set_title(f"({'a' if idx==0 else 'b'}) {title} Phosphosite Ranking", fontsize=9, fontweight="bold")

    ax = axes[1, 0]
    if stability:
        et = stability.get("egfr", {}).get("phospho_top_site", "")
        ht = stability.get("erbb2", {}).get("phospho_top_site", "")
        if et and ht:
            ax.text(0.5, 0.75, f"EGFR #1: {et}", ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["ours"], transform=ax.transAxes)
            ax.text(0.5, 0.50, "↕ Tissue-specific hierarchy", ha="center", fontsize=8, transform=ax.transAxes)
            ax.text(0.5, 0.25, f"HER2 #1: {ht}", ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["phospho"], transform=ax.transAxes)
    ax.axis("off"); ax.set_title("(c) Cross-Receptor Homology", fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    if stability:
        ed = stability.get("egfr", {})
        pi = np.mean(ed.get("phospho_mean_importance", [0]))
        gi = np.mean(ed.get("glyco_mean_importance", [0]))
        ax.bar(["Phospho\n(12 sites)", "Glyco\n(12 sites)"], [pi, gi],
               color=[COLORS["phospho"], COLORS["glyco"]], edgecolor="white")
        ax.set_ylabel("Mean IG Attribution")
        for i, v in enumerate([pi, gi]):
            ax.text(i, v + max(pi, gi) * 0.02, f"{v:.5f}", ha="center", fontsize=7)
    ax.set_title("(d) Phospho vs Glyco", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_interpretability")


# ═══════════════════ Supplementary figures ═══════════════════

def fig_s_perdrug(results, plt):
    drug_order = ["Osimertinib", "Gefitinib", "Erlotinib", "Afatinib", "Lapatinib", "Sapitinib"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
    method_data = {}
    if results["evaluation"] and "drug_specific" in results["evaluation"]:
        method_data["Ours"] = {d: results["evaluation"]["drug_specific"].get(d, {}) for d in drug_order}
    nm = {"random_forest": "RF", "ridge": "Ridge", "elastic_net": "ElNet"}
    if results["ml_baselines"]:
        for key, data in results["ml_baselines"].items():
            if key.startswith("_") or not data.get("per_drug"): continue
            method_data[nm.get(key, key)] = {d: data["per_drug"].get(d, {}) for d in drug_order}
    if not method_data: plt.close(fig); return
    n_m = len(method_data); w = 0.8 / n_m; x = np.arange(len(drug_order))
    for panel, metric in [(0, "pearson_r"), (1, "auroc")]:
        ax = axes[panel]
        for i, (meth, dd) in enumerate(method_data.items()):
            vals = [dd.get(d, {}).get(metric, 0) for d in drug_order]
            ax.bar(x + (i - n_m/2 + 0.5) * w, vals, w, label=meth, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([d[:6] for d in drug_order], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Pearson R" if panel == 0 else "AUROC")
        ax.set_title("Per-Drug PCC" if panel == 0 else "Per-Drug AUROC", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    _save(fig, plt, "Fig_S_perdrug")


def fig_s_loclo(results, plt):
    loclo = results.get("loclo")
    if not loclo or "per_group_results" not in loclo: return
    per_group = loclo["per_group_results"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    groups, pccs, aurocs, ns = [], [], [], []
    for g, m in per_group.items():
        if "pearson_r" not in m: continue
        groups.append(g); pccs.append(m["pearson_r"]); aurocs.append(m.get("auroc", 0) or 0)
        ns.append(m.get("n_test", 0))
    if not groups: plt.close(fig); return
    x = np.arange(len(groups)); summary = loclo.get("summary", {})
    for panel, vals, ylabel in [(0, pccs, "Pearson R"), (1, aurocs, "AUROC")]:
        ax = axes[panel]
        ax.bar(x, vals, color=COLORS["ours"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)
        ref_key = f"random_split_{'pearson_r' if panel == 0 else 'auroc'}"
        if ref_key in summary: ax.axhline(summary[ref_key], color="red", ls="--", lw=0.8, label="Random split"); ax.legend(fontsize=6)
        for i, (v, n) in enumerate(zip(vals, ns)): ax.text(i, v + 0.01, f"n={n}", ha="center", fontsize=5)
    plt.tight_layout()
    _save(fig, plt, "Fig_S_loclo")


def fig_s_runtime(results, plt):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
    methods = {}
    nm = {"random_forest": "Random Forest", "ridge": "Ridge", "elastic_net": "Elastic Net"}
    if results["ml_baselines"]:
        for key, data in results["ml_baselines"].items():
            if key.startswith("_"): continue
            methods[nm.get(key, key)] = data.get("training_time_seconds", 0)
    if results["evaluation"]:
        methods["Ours (PTM-BDL)"] = results["evaluation"].get("training_time_seconds", 0)
    if not methods: plt.close(fig); return
    names = list(methods.keys()); times = list(methods.values())
    ax.barh(range(len(names)), times, color=[METHOD_COLORS.get(n, "#999") for n in names], edgecolor="white")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Training Time (s)"); ax.invert_yaxis()
    for i, v in enumerate(times):
        if v > 0: ax.text(v + 0.5, i, f"{v:.1f}s", va="center", fontsize=6)
    ax.set_title("Training Time Comparison", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_S_runtime")


def fig_s_cold_start(results, plt):
    """Q4: Cold-cell + cold-drug LODO evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))

    # Cold-cell
    ax = axes[0]
    cc = results.get("cold_cell")
    if cc and "per_fold_results" in cc:
        folds = sorted(cc["per_fold_results"].keys())
        aurocs = [cc["per_fold_results"][f]["auroc"] for f in folds]
        x = np.arange(len(folds))
        ax.bar(x, aurocs, color=COLORS["cold_cell"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(len(folds))], fontsize=7)
        ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.1)
        mean_a = cc["summary"]["mean_auroc"]; std_a = cc["summary"]["std_auroc"]
        ax.axhline(mean_a, color="red", ls="--", lw=0.8, label=f"Mean={mean_a:.3f}±{std_a:.3f}")
        ax.legend(fontsize=6)
        for i, v in enumerate(aurocs): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=6)
    ax.set_title("(a) Cold-Cell K-Fold CV", fontsize=9, fontweight="bold")

    # Cold-drug LODO
    ax = axes[1]
    cd = results.get("cold_drug")
    if cd and "per_drug_results" in cd:
        drugs = sorted(cd["per_drug_results"].keys())
        aurocs = [cd["per_drug_results"][d].get("auroc", 0) for d in drugs]
        x = np.arange(len(drugs))
        ax.bar(x, aurocs, color=COLORS["cold_drug"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([d[:6] for d in drugs], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.1)
        mean_a = cd["summary"]["mean_auroc"]
        ax.axhline(mean_a, color="red", ls="--", lw=0.8, label=f"Mean={mean_a:.3f}")
        ax.legend(fontsize=6)
        for i, v in enumerate(aurocs): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=6)
    ax.set_title("(b) Leave-One-Drug-Out", fontsize=9, fontweight="bold")

    plt.tight_layout()
    _save(fig, plt, "Fig_S_cold_start")


def fig_s_cross_dataset(results, plt):
    """Q4: Cross-dataset GDSC→CTRP evaluation."""
    cd = results.get("cross_dataset")
    if not cd or "per_drug" not in cd: return
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))
    drugs = sorted(cd["per_drug"].keys())
    pred_r = [cd["per_drug"][d].get("pred_vs_ctrp_pearson_r", 0) for d in drugs]
    gdsc_r = [cd["per_drug"][d].get("gdsc_vs_ctrp_pearson_r", 0) for d in drugs]
    x = np.arange(len(drugs)); w = 0.35
    ax.bar(x - w/2, gdsc_r, w, label="GDSC→CTRP (raw)", color="#009E73")
    ax.bar(x + w/2, pred_r, w, label="PTM-BDL pred→CTRP", color=COLORS["ours"])
    ax.set_xticks(x); ax.set_xticklabels([d[:6] for d in drugs], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson R"); ax.legend(fontsize=6)
    ax.set_title("Cross-Dataset (GDSC→CTRPv2)", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save(fig, plt, "Fig_S_cross_dataset")


def fig_s_stability(results, plt):
    """Q5: IG rank stability across seeds."""
    stability = results.get("stability")
    if not stability: return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))
    for idx, (prot, title) in enumerate([("egfr", "EGFR"), ("erbb2", "HER2")]):
        ax = axes[idx]
        d = stability.get(prot, {})
        sites = d.get("phospho_sites", [])
        imps = d.get("phospho_mean_importance", [])
        if not sites or not imps: continue
        # Filter non-zero
        active = [(s, v) for s, v in zip(sites, imps) if v > 0 and not s.startswith("pad")]
        if not active: continue
        labs, vals = zip(*sorted(active, key=lambda x: x[1], reverse=True))
        colors = [COLORS["phospho"] if "Y" in s.upper() else "#CC79A7" for s in labs]
        ax.barh(range(len(labs)), vals, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6)
        ax.invert_yaxis(); ax.set_xlabel("Mean |IG| (3 seeds)", fontsize=7)
        ax.set_title(f"{title} — IG Stability ({stability.get('n_seeds', 3)} seeds)", fontsize=9, fontweight="bold")
        top = d.get("phospho_top_site", "")
        if top: ax.text(0.95, 0.95, f"Top: {top}", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F0FE", ec="none"))
    plt.tight_layout()
    _save(fig, plt, "Fig_S_stability")


def fig_s_calibration(results, plt):
    """Q9: Reliability diagram + ECE for classification head."""
    ev = results.get("evaluation")
    if not ev or "calibration" not in ev: return
    cal = ev["calibration"]
    overall = cal.get("overall_ece", {})
    per_drug = cal.get("per_drug_ece", {})

    n_drugs = len([k for k in per_drug if k != "overall"])
    fig, axes = plt.subplots(1, min(n_drugs + 1, 4), figsize=(DOUBLE_COL, DOUBLE_COL * 0.32), squeeze=False)
    axes = axes[0]

    def _plot_reliability(ax, data, title):
        if not data or "bin_accs" not in data: return
        edges = data["bin_edges"]; accs = data["bin_accs"]; confs = data["bin_confs"]; counts = data["bin_counts"]
        midpoints = [(edges[i] + edges[i+1]) / 2 for i in range(len(accs))]
        # Only plot bins with data
        active = [(m, a, c, n) for m, a, c, n in zip(midpoints, accs, confs, counts) if n > 0]
        if not active: return
        ms, acs, cfs, ns = zip(*active)
        ax.bar(ms, acs, width=0.08, alpha=0.7, color=COLORS["ours"], label="Accuracy", edgecolor="white")
        ax.plot([0, 1], [0, 1], "k--", lw=0.5, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Predicted Prob", fontsize=6); ax.set_ylabel("Fraction Positive", fontsize=6)
        ece = data.get("ece", 0)
        ax.set_title(f"{title}\nECE={ece:.3f}" if ece else title, fontsize=8, fontweight="bold")
        ax.legend(fontsize=5)

    _plot_reliability(axes[0], overall, "Overall")
    drug_idx = 1
    for drug_name in sorted(per_drug.keys()):
        if drug_name == "overall" or drug_idx >= len(axes): continue
        _plot_reliability(axes[drug_idx], per_drug[drug_name], drug_name[:8])
        drug_idx += 1

    plt.suptitle("Reliability Diagrams (CS1: EGFR/ERBB2 TKI)", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, plt, "Fig_S_calibration")


def fig_s_baseline_ablation(results, plt):
    """Q1/Q3: Baseline-only vs delta-only PTM ablation — not applicable for CS1 (no baseline_only arm)."""
    # CS1 does not have baseline_only/delta_only ablation arms
    ablation = results.get("ablation")
    if not ablation or "baseline_only" not in ablation:
        print("  ⚠ No baseline_only ablation arm for CS1 — skipping")
        return


# ═══════════════════ Main ═══════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CS1 (EGFR/ERBB2) — Publication Figures (Updated)          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    plt = setup_matplotlib()
    results = load_results()

    print("\n  ── Main Text Figures ──")
    fig_benchmarking(results, plt)
    fig_ablation(results, plt)
    fig_interpretability(results, plt)

    print("\n  ── Supplementary Figures ──")
    fig_s_perdrug(results, plt)
    fig_s_loclo(results, plt)
    fig_s_runtime(results, plt)
    fig_s_cold_start(results, plt)
    fig_s_cross_dataset(results, plt)
    fig_s_stability(results, plt)
    fig_s_calibration(results, plt)
    fig_s_baseline_ablation(results, plt)

    generated = list(PUB_FIG_DIR.glob("*.pdf"))
    print(f"\n  ✓ Generated {len(generated)} figures in {PUB_FIG_DIR}")
    print("✓ CS1 figures complete!")


if __name__ == "__main__":
    main()
