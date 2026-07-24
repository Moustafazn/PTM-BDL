#!/usr/bin/env python3
"""
HeLa/HDAC — Publication Figures.

Generates publication-quality figures for the HeLa/HDAC case study:
  Fig 1: Benchmarking — PTM-BDL vs ML baselines (bar chart)
  Fig 2: Ablation — modality contribution (ΔPerformance per ablation)
  Fig 3: XAI — phospho vs acetyl IG attributions per drug
  Fig 4: Cross-type attention heatmap (phospho ↔ acetyl crosstalk)
  Fig S1: Per-drug performance breakdown
  Fig S2: A486 (inactive control) discrimination analysis

Ref: Lasko et al., Nat Rev Drug Discov 2024 (PMID 38382638) — HDAC resistance
Ref: Fischle et al., Nature 2003 (PMID 14573844) — H3S10ph-K9ac switch
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "publication" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def setup_matplotlib():
    """Configure matplotlib for publication quality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })
    return plt


def load_results():
    """Load all result JSONs."""
    results = {}
    for name in ["evaluation_report", "ablation_study", "crossval_results",
                  "ml_baselines", "xai_report", "statistical_tests",
                  "stability_analysis"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def fig_benchmarking(results, plt):
    """Fig 1: PTM-BDL vs ML baselines performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    eval_report = results.get("evaluation_report", {})
    baselines = results.get("ml_baselines", {}).get("baselines", {})

    methods = ["PTM-BDL"] + list(baselines.keys())
    aurocs = [eval_report.get("overall_metrics", {}).get("auroc", 0)]
    aurocs += [b.get("auroc", 0) for b in baselines.values()]

    axes[0].barh(methods, aurocs, color=["#2196F3"] + ["#90CAF9"] * len(baselines))
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("Classification Performance")

    rmses = [eval_report.get("overall_metrics", {}).get("rmse", 0)]
    rmses += [b.get("rmse", 0) for b in baselines.values()]
    axes[1].barh(methods, rmses, color=["#2196F3"] + ["#90CAF9"] * len(baselines))
    axes[1].set_xlabel("RMSE")
    axes[1].set_title("Regression Performance")

    plt.suptitle(f"{CASE_STUDY} — Benchmarking", fontsize=18)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig_benchmarking.png")
    plt.savefig(FIGURES_DIR / "Fig_benchmarking.pdf")
    print(f"  ✓ Fig_benchmarking saved")


def fig_ablation(results, plt):
    """Fig 2: Ablation study — modality contributions."""
    ablation = results.get("ablation_study", {})
    if not ablation or "full" not in ablation:
        print("  ⚠ No ablation results — skipping")
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


def _resolve_labels(protein, labels, channel="phospho"):
    """Map generic slot labels to per-protein biological residue names from config.

    The dataset uses shared column names (ptm_S29, secondary_slot00) for ALL
    proteins, but each protein has different actual sites at each slot position.
    This function resolves by position index using the config's per-protein
    site definitions.
    """
    ptm_cfg = cfg.get("ptm", {})
    prot_cfg = ptm_cfg.get(protein, {})
    key = "phospho_sites" if channel == "phospho" else "acetyl_sites"
    sites = prot_cfg.get(key, [])

    resolved = []
    for i, lbl in enumerate(labels):
        if i < len(sites):
            resolved.append(sites[i].get("residue", lbl))
        else:
            resolved.append(f"pad_{i}")
    return resolved


def fig_xai_ptm_types(results, plt):
    """Fig 3: XAI — phospho vs acetyl IG attributions from stability_analysis.

    Reads per-protein IG data (phospho + acetyl) from stability_analysis.json
    (generated by ablation.py Part 2) and produces publication-quality bar
    charts showing per-site attribution rankings for each protein.
    Labels are resolved to biological residue names via config.yaml.
    """
    stability = results.get("stability_analysis", {})
    if not stability or "per_protein" not in stability:
        print("  ⚠ No stability_analysis results — skipping Fig_interpretability")
        return

    per_protein = stability["per_protein"]
    proteins = sorted(per_protein.keys())
    n_proteins = len(proteins)
    if n_proteins == 0:
        print("  ⚠ No per-protein data in stability_analysis — skipping")
        return

    # Check if acetyl data exists with non-zero values
    has_acetyl = any(
        "acetyl_mean_importance" in per_protein[p]
        and len(per_protein[p].get("acetyl_mean_importance", [])) > 0
        and any(v > 0 for v in per_protein[p].get("acetyl_mean_importance", []))
        for p in proteins
    )

    n_rows = 2 if has_acetyl else 1
    fig, axes = plt.subplots(n_rows, n_proteins,
                             figsize=(6 * n_proteins, 4.5 * n_rows),
                             squeeze=False)

    # Residue-type colour palette (Wong colourblind-safe)
    aa_colors = {"Y": "#D55E00", "S": "#56B4E9", "T": "#F0E442", "K": "#009E73"}

    def _residue_color(label, default="#999999"):
        """Pick colour based on residue letter in the label."""
        up = label.upper()
        for aa in ["Y", "S", "T", "K"]:
            if aa in up:
                return aa_colors[aa]
        return default

    for col, protein in enumerate(proteins):
        data = per_protein[protein]

        # ── Phospho panel ────────────────────────────────────────────
        ph_labels_raw = data.get("phospho_site_labels", [])
        ph_values = data.get("phospho_mean_importance", [])
        # Resolve generic labels → biological names from config
        ph_labels = _resolve_labels(protein, ph_labels_raw, "phospho")
        if ph_labels and ph_values:
            # Filter out padded slots (zero importance)
            active = [(lbl, val) for lbl, val in zip(ph_labels, ph_values)
                      if not lbl.startswith("pad_")]
            if active:
                ph_labels_f, ph_values_f = zip(*active)
            else:
                ph_labels_f, ph_values_f = ph_labels, ph_values
            order = sorted(range(len(ph_values_f)),
                           key=lambda i: ph_values_f[i], reverse=True)
            s_labels = [ph_labels_f[i] for i in order]
            s_values = [ph_values_f[i] for i in order]
            colors = [_residue_color(lbl) for lbl in s_labels]

            ax = axes[0, col]
            ax.barh(range(len(s_labels)), s_values,
                    color=colors, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(s_labels)))
            ax.set_yticklabels(s_labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |IG| Attribution", fontsize=9)
            ax.set_title(f"{protein} — Phospho Sites", fontsize=11,
                         fontweight="bold")
            for i, v in enumerate(s_values):
                ax.text(v, i, f" {v:.5f}", va="center", fontsize=6.5,
                        color="#333")

        # ── Acetyl panel ─────────────────────────────────────────────
        if has_acetyl and n_rows > 1:
            ac_labels_raw = data.get("acetyl_site_labels", [])
            ac_values = data.get("acetyl_mean_importance", [])
            # Resolve generic labels → biological names from config
            ac_labels = _resolve_labels(protein, ac_labels_raw, "acetyl")
            if ac_labels and ac_values:
                # Filter out padded slots
                active = [(lbl, val) for lbl, val in zip(ac_labels, ac_values)
                          if not lbl.startswith("pad_")]
                if active:
                    ac_labels_f, ac_values_f = zip(*active)
                else:
                    ac_labels_f, ac_values_f = ac_labels, ac_values
                order = sorted(range(len(ac_values_f)),
                               key=lambda i: ac_values_f[i], reverse=True)
                s_labels = [ac_labels_f[i] for i in order]
                s_values = [ac_values_f[i] for i in order]

                ax = axes[1, col]
                ax.barh(range(len(s_labels)), s_values,
                        color="#009E73", edgecolor="white", linewidth=0.5)
                ax.set_yticks(range(len(s_labels)))
                ax.set_yticklabels(s_labels, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Mean |IG| Attribution", fontsize=9)
                ax.set_title(f"{protein} — Acetyl Sites", fontsize=11,
                             fontweight="bold")
                for i, v in enumerate(s_values):
                    ax.text(v, i, f" {v:.5f}", va="center", fontsize=6.5,
                            color="#333")

    n_seeds = stability.get("n_seeds", "?")
    plt.suptitle(
        f"CS2: HeLa/HDAC — Integrated Gradient Attributions\n"
        f"(phospho + acetyl per protein, {n_seeds} seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "Fig_interpretability.png", dpi=300)
    plt.savefig(FIGURES_DIR / "Fig_interpretability.pdf", dpi=300)
    plt.close()
    print(f"  ✓ Fig_interpretability saved ({n_proteins} proteins, "
          f"{'phospho+acetyl' if has_acetyl else 'phospho only'})")


def main():
    """Generate all publication figures."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Publication Figures                        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    plt = setup_matplotlib()
    results = load_results()

    fig_benchmarking(results, plt)
    fig_ablation(results, plt)
    fig_xai_ptm_types(results, plt)

    print(f"\n✓ All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
