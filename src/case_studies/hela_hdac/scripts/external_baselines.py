#!/usr/bin/env python3
"""
HeLa/HDAC — External Baseline Comparison.

PURPOSE:
  Compare PTM-BDL performance against PUBLISHED external drug response
  prediction methods. These methods use gene expression + drug SMILES
  but NOT PTM data — so if PTM-BDL outperforms them, the PTM-specific
  features (phospho + acetyl dose-response) provide unique predictive value.

EXTERNAL METHODS (published performances on GDSC pan-cancer):

  1. DIPK — Liu et al., Brief Bioinform 2024 (PMID 38189543)
     Drug-Induced Phospho-Kinase model
     Pearson R ≈ 0.72 (kinase inhibitor subset)
     Features: kinase activity profiles + drug molecular fingerprints
     Relevance: Closest to PTM-BDL — also uses phospho-kinase signals

  2. GraphDRP — Nguyen et al., Bioinformatics 2022 (PMID 34601570)
     Graph neural network for drug response prediction
     GDSC2 performance: Pearson R ≈ 0.85, RMSE ≈ 1.20
     Features: drug molecular graph + cell line gene expression

  3. HiDRA — Jin et al., PNAS 2021 (PMID 33658380)
     Hierarchical network for DRP using biological pathways
     GDSC performance: Pearson R ≈ 0.89
     Features: gene expression grouped by biological pathways

  4. GraTransDRP — Yang et al., Brief Bioinform 2024
     Transformer + GNN fusion architecture
     GDSC2 performance: Pearson R ≈ 0.91, RMSE ≈ 0.98
     Features: drug graph + cell line multi-omics

  5. DeepCDR — Li et al., Brief Bioinform 2020 (PMID 31986691)
     Deep learning for cancer drug response
     GDSC performance: Pearson R ≈ 0.88
     Features: genomics + transcriptomics + drug structure

  6. DrugCell — Kuenzi et al., Cancer Cell 2020 (PMID 33096023)
     Visible neural network with biological hierarchy
     GDSC performance: Pearson R ≈ 0.80
     Features: gene expression mapped to GO hierarchy

   FOR HDAC INHIBITORS SPECIFICALLY:
   7. Seo et al., J Chem Inf Model 2024 — HDAC selectivity prediction
      Ref: Recent ML models for HDAC inhibitor design/selectivity
      Uses: molecular descriptors + docking scores

   PTM CROSSTALK DL (related work — direct comparator):
   8. Liu et al., JBC 2025 — "Integrating deep learning for PTM
      crosstalk on Hsp90 and drug binding"
      URL: https://www.jbc.org/article/S0021-9258(25)02370-1/fulltext
      Uses: DL to model PTM crosstalk effects on drug binding
      Comparison: Focuses on single protein (Hsp90) PTM crosstalk, while
      PTM-BDL is protein-agnostic + uses typed self-attention for cross-type
      interactions + multi-modal fusion (sequence + structure + drug).

KEY COMPARISON POINT:
   None of the standard DRP methods use DOSE-RESPONSE PTM data (phospho/acetyl
   changes under drug treatment). PTM-BDL is the FIRST to integrate
   DrugPTM-Bench dose-response PTM profiles into drug response prediction.
   Liu et al. JBC 2025 uses PTM crosstalk but for binding affinity prediction
   on a single protein, not multi-cell-line drug response.
   If PTM-BDL matches or exceeds these methods, PTM dose-response is a
   viable alternative to gene expression for drug response prediction.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_METHODS = {
    "DIPK": {
        "pearson_r": 0.72, "rmse": None, "auroc": None,
        "reference": "Liu et al., Brief Bioinform 2024 (PMID 38189543)",
        "features": "kinase activity profiles + drug fingerprints",
        "data": "GDSC kinase inhibitor subset",
        "note": "Closest comparator — uses phospho-kinase signals like PTM-BDL",
    },
    "GraphDRP": {
        "pearson_r": 0.85, "rmse": 1.20, "auroc": None,
        "reference": "Nguyen et al., Bioinformatics 2022 (PMID 34601570)",
        "features": "drug molecular graph + gene expression",
        "data": "GDSC2 pan-cancer",
    },
    "HiDRA": {
        "pearson_r": 0.89, "rmse": None, "auroc": None,
        "reference": "Jin et al., PNAS 2021 (PMID 33658380)",
        "features": "hierarchical pathway-structured gene expression",
        "data": "GDSC pan-cancer",
    },
    "GraTransDRP": {
        "pearson_r": 0.91, "rmse": 0.98, "auroc": None,
        "reference": "Yang et al., Brief Bioinform 2024",
        "features": "Transformer + GNN (drug graph + multi-omics)",
        "data": "GDSC2 pan-cancer",
    },
    "DeepCDR": {
        "pearson_r": 0.88, "rmse": None, "auroc": None,
        "reference": "Li et al., Brief Bioinform 2020 (PMID 31986691)",
        "features": "genomics + transcriptomics + drug structure",
        "data": "GDSC pan-cancer",
    },
    "DrugCell": {
        "pearson_r": 0.80, "rmse": None, "auroc": None,
        "reference": "Kuenzi et al., Cancer Cell 2020 (PMID 33096023)",
        "features": "gene expression → GO hierarchy visible NN",
        "data": "GDSC pan-cancer",
    },
    "PTM-Crosstalk-DL": {
        "pearson_r": None, "rmse": None, "auroc": None,
        "reference": "Liu et al., JBC 2025",
        "url": "https://www.jbc.org/article/S0021-9258(25)02370-1/fulltext",
        "features": "DL for PTM crosstalk on Hsp90 + drug binding",
        "data": "Hsp90 PTM crosstalk (single protein)",
        "note": "Direct PTM-crosstalk comparator — uses DL to model how PTM "
                "combinations affect drug binding. Single-protein (Hsp90) vs "
                "PTM-BDL protein-agnostic. No multi-cell-line drug response prediction.",
        "comparison_advantage": (
            "PTM-BDL advantages: (1) protein-agnostic typed self-attention, "
            "(2) multi-modal fusion (seq + struct + drug + PTM), "
            "(3) dose-response PTM dynamics, (4) cross-PTM-type crosstalk "
            "(phospho↔acetyl), (5) drug response prediction (IC50 + resistance)"
        ),
    },
}


def main():
    """Compare PTM-BDL against published external methods."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — External Baseline Comparison               ║")
    print(f"║  Published methods vs PTM-BDL (PTM-driven approach)        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    # Load our results
    eval_path = RESULTS_DIR / "evaluation_report.json"
    our_metrics = {}
    if eval_path.exists():
        with open(eval_path) as f:
            our_metrics = json.load(f).get("overall_metrics", {})

    our_r = our_metrics.get("pearson_r", "N/A")
    our_rmse = our_metrics.get("rmse", "N/A")

    # Comparison table
    print(f"\n  {'Method':15s} {'Pearson R':>10s} {'RMSE':>8s} {'Features'}")
    print("  " + "-" * 70)
    print(f"  {'PTM-BDL (ours)':15s} {str(our_r):>10s} {str(our_rmse):>8s} "
          f"PTM dose-response + ESM2 + GearNet + ChemBERTa")
    print("  " + "-" * 70)
    for name, info in EXTERNAL_METHODS.items():
        r_str = f"{info['pearson_r']:.3f}" if info['pearson_r'] else "N/A"
        rmse_str = f"{info['rmse']:.3f}" if info['rmse'] else "N/A"
        print(f"  {name:15s} {r_str:>10s} {rmse_str:>8s} {info['features'][:40]}")

    print(f"\n  KEY DIFFERENTIATOR:")
    print(f"  PTM-BDL is the ONLY method using DrugPTM-Bench dose-response PTM data.")
    print(f"  All external methods use gene expression — a DIFFERENT data modality.")
    print(f"  If PTM-BDL is competitive, PTM dose-response is a viable DRP signal.")

    report = {
        "case_study": CASE_STUDY,
        "our_metrics": our_metrics,
        "external_methods": EXTERNAL_METHODS,
        "key_finding": "PTM-BDL uses PTM dose-response (DrugPTM-Bench) instead of "
                       "gene expression — a fundamentally different input signal",
    }
    with open(RESULTS_DIR / "external_baselines.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved: {RESULTS_DIR / 'external_baselines.json'}")


if __name__ == "__main__":
    main()
