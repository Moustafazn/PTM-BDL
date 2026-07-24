#!/usr/bin/env python3
"""
K562/CML — External Baseline Comparison.

PURPOSE:
  Compare PTM-BDL against published drug response prediction methods,
  specifically validated on CML/K562 or leukemia cell lines.

EXTERNAL METHODS:

  1. DIPK — Liu et al., Brief Bioinform 2024 (PMID 38189543)
     Pearson R ≈ 0.72 on kinase inhibitor subset
     Relevance: Uses kinase phospho-activity — closest to our PTM approach

  2. GraphDRP — Nguyen et al., Bioinformatics 2022 (PMID 34601570)
     Pearson R ≈ 0.85, RMSE ≈ 1.20 on GDSC2
     Features: drug molecular graph + gene expression

  3. HiDRA — Jin et al., PNAS 2021 (PMID 33658380)
     Pearson R ≈ 0.89 on GDSC pan-cancer

  4. GraTransDRP — Yang et al., Brief Bioinform 2024
     Pearson R ≈ 0.91, RMSE ≈ 0.98 on GDSC2

  5. DeepCDR — Li et al., Brief Bioinform 2020 (PMID 31986691)
     Pearson R ≈ 0.88 on GDSC

  6. DrugCell — Kuenzi et al., Cancer Cell 2020 (PMID 33096023)
     Pearson R ≈ 0.80 on GDSC

  CML-SPECIFIC PUBLICATIONS:
  7. O'Hare et al., Cancer Cell 2009 (PMID 19573813)
     Computational prediction of BCR-ABL mutant drug sensitivity
     Uses: structural modeling + mutational energy calculations
     Published K562 sensitivities for Dasatinib and Imatinib

  8. Soverini et al., Haematologica 2024
     CML treatment algorithm incorporating molecular profiling
     Published: TKI response rates by BCR-ABL mutation class

KEY COMPARISON:
  External methods predict drug response from gene expression + drug SMILES.
  PTM-BDL predicts from PTM dose-response + protein structure + drug SMILES.
  The critical question: does PTM dose-response (from DrugPTM-Bench)
  provide COMPARABLE signal to gene expression for drug response prediction?
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_METHODS = {
    "DIPK": {
        "pearson_r": 0.72, "rmse": None,
        "reference": "Liu et al., Brief Bioinform 2024 (PMID 38189543)",
        "features": "kinase activity + drug fingerprints",
        "note": "Closest comparator — uses phospho-kinase signals",
    },
    "GraphDRP": {
        "pearson_r": 0.85, "rmse": 1.20,
        "reference": "Nguyen et al., Bioinformatics 2022 (PMID 34601570)",
        "features": "drug GNN + gene expression",
    },
    "HiDRA": {
        "pearson_r": 0.89, "rmse": None,
        "reference": "Jin et al., PNAS 2021 (PMID 33658380)",
        "features": "hierarchical pathway gene expression",
    },
    "GraTransDRP": {
        "pearson_r": 0.91, "rmse": 0.98,
        "reference": "Yang et al., Brief Bioinform 2024",
        "features": "Transformer + GNN fusion",
    },
    "DeepCDR": {
        "pearson_r": 0.88, "rmse": None,
        "reference": "Li et al., Brief Bioinform 2020 (PMID 31986691)",
        "features": "genomics + transcriptomics + drug",
    },
    "DrugCell": {
        "pearson_r": 0.80, "rmse": None,
        "reference": "Kuenzi et al., Cancer Cell 2020 (PMID 33096023)",
        "features": "GO hierarchy visible NN",
    },
}

CML_SPECIFIC_BENCHMARKS = {
    "OHare_2009_structural": {
        "description": "Structural modeling of BCR-ABL mutant drug sensitivity",
        "reference": "O'Hare et al., Cancer Cell 2009 (PMID 19573813)",
        "k562_dasatinib_ic50_nM": 0.8,
        "k562_imatinib_ic50_nM": 260,
    },
    "Soverini_2024_clinical": {
        "description": "CML treatment algorithm with molecular profiling",
        "reference": "Soverini et al., Haematologica 2024",
        "dasatinib_response_rate": "85-90% in newly diagnosed CML",
        "imatinib_response_rate": "75-80% in newly diagnosed CML",
    },
}


def main():
    """Compare PTM-BDL against published methods."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — External Baseline Comparison               ║")
    print(f"║  Published DRP methods + CML-specific benchmarks           ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    eval_path = RESULTS_DIR / "evaluation_report.json"
    our_metrics = {}
    if eval_path.exists():
        with open(eval_path) as f:
            our_metrics = json.load(f).get("overall_metrics", {})

    our_r = our_metrics.get("pearson_r", "N/A")
    our_rmse = our_metrics.get("rmse", "N/A")

    print(f"\n  {'Method':15s} {'Pearson R':>10s} {'RMSE':>8s} {'Features'}")
    print("  " + "-" * 70)
    print(f"  {'PTM-BDL (ours)':15s} {str(our_r):>10s} {str(our_rmse):>8s} "
          f"PTM dose-response + ESM2 + GearNet + ChemBERTa")
    print("  " + "-" * 70)
    for name, info in EXTERNAL_METHODS.items():
        r_str = f"{info['pearson_r']:.3f}" if info['pearson_r'] else "N/A"
        rmse_str = f"{info['rmse']:.3f}" if info['rmse'] else "N/A"
        print(f"  {name:15s} {r_str:>10s} {rmse_str:>8s} {info['features'][:40]}")

    print(f"\n  CML-Specific Published Benchmarks:")
    for name, info in CML_SPECIFIC_BENCHMARKS.items():
        print(f"    {name}: {info['description']}")
        print(f"      Ref: {info['reference']}")

    report = {
        "case_study": CASE_STUDY,
        "our_metrics": our_metrics,
        "external_methods": EXTERNAL_METHODS,
        "cml_specific_benchmarks": CML_SPECIFIC_BENCHMARKS,
    }
    with open(RESULTS_DIR / "external_baselines.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Saved: {RESULTS_DIR / 'external_baselines.json'}")


if __name__ == "__main__":
    main()
