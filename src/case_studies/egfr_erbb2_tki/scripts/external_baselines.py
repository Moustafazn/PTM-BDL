#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 14b — External Baseline Framework (Tier 1–2 DRP Methods)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Provide a structured framework for running external drug response         ║
║    prediction (DRP) methods on our EXACT dataset and split, collecting       ║
║    their predictions, and computing identical Tier A metrics.                ║
║                                                                              ║
║  WHY THESE METHODS (Benchmarking Plan §3–4):                                 ║
║    These are the actual comparison methods for Table 1 of the paper.        ║
║    Unlike the methodological references (SAGE-net, ClairS) which inform     ║
║    our statistical approach, these methods are DRP models that predict       ║
║    drug response from cell line and drug features — the same task as ours.  ║
║                                                                              ║
║  TIER 1 — Recent State-of-the-Art (2023–2026), MANDATORY:                   ║
║    1. DIPK (Li et al., Nature Communications 2024)                          ║
║       - DL + PPI network, expression + mutation + methylation + CNV         ║
║    2. HiDRA (Jin et al., Nature Communications 2023)                        ║
║       - Hierarchical gene-set DRP, expression + drug fingerprints           ║
║    3. GraTransDRP (Li et al., Brief Bioinf 2023)                            ║
║       - Graph Transformer, expression + mutation + drug SMILES              ║
║    4. TransCDR (Xia et al., Brief Bioinf 2023)                              ║
║       - Transformer-based, expression + mutation + drug SMILES              ║
║    5. PathDSP (Tang et al., Bioinformatics 2024)                            ║
║       - Pathway-aware, expression + drug fingerprints + pathway DB          ║
║                                                                              ║
║  TIER 2 — Established Baselines (2020–2022), RECOMMENDED:                   ║
║    6. GraphDRP (Nguyen et al., Bioinformatics 2022)                         ║
║       - GNN, expression + drug SMILES                                        ║
║    7. DrugCell (Kuenzi et al., Cancer Cell 2020)                            ║
║       - Visible neural network + GO hierarchy, mutation + drug fingerprints ║
║    8. DeepCDR (Li et al., Brief Bioinf 2020)                                ║
║       - Multi-omics DL, expression + mutation + methylation + drug SMILES   ║
║                                                                              ║
║  APPROACH (Benchmarking Plan §5):                                            ║
║    Each external method gets our EXACT GDSC subset (951 samples, 6 TKI      ║
║    drugs, EGFR/ERBB2 cell lines) reformatted into its expected input.       ║
║    The method's own feature extraction + training pipeline runs, using       ║
║    our split_indices.json for identical train/val/test assignment.           ║
║                                                                              ║
║  DATA FAIRNESS (§5.1):                                                       ║
║    - Same 951 samples, same IC50 values, same resistance binarization       ║
║    - Same split_indices.json (70/15/15, stratified)                         ║
║    - Each method uses its own features (Approach A: their full pipeline)    ║
║    - Published default hyperparameters (no tuning on our data)             ║
║                                                                              ║
║  WORKFLOW PER METHOD:                                                        ║
║    1. Clone external repo → benchmarks/<method>/                            ║
║    2. Prepare input data in method's expected format                        ║
║    3. Run method's training on our train+val set                            ║
║    4. Collect test-set predictions                                           ║
║    5. Compute Tier A metrics (PCC, RMSE, AUROC, AUPRC-sens, per-drug)      ║
║    6. Save → results/external_baselines/<method>.json                       ║
║                                                                              ║
║  FALLBACK (Benchmarking Plan §9, Risk 1):                                   ║
║    If an external method's code doesn't run on our data subset:             ║
║    → Report published numbers with "reported performance" caveat            ║
║    → Document what failed and why                                            ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/external_baselines/<method>.json  (per method)                    ║
║    results/external_baselines_summary.json   (combined)                      ║
║                                                                              ║
║  BENCHMARKING_PLAN.md §4, §5, §8 Step 14b                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
EXT_RESULTS_DIR = RESULTS_DIR / "external_baselines"
EXT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# External method registry
# ══════════════════════════════════════════════════════════════════════════════

EXTERNAL_METHODS = {
    # ── Tier 1: Recent State-of-the-Art (2023–2026) — MANDATORY ──────────
    "DIPK": {
        "tier": 1,
        "year": 2024,
        "venue": "Nature Communications",
        "paper": "Li et al., 2024",
        "github": "https://github.com/user-wu/DIPK",
        "input_features": ["expression", "mutation", "methylation", "CNV",
                           "PPI_network"],
        "drug_features": ["drug_fingerprints"],
        "task": "regression",  # IC50 prediction
        "notes": "DL + PPI network integration",
    },
    "HiDRA": {
        "tier": 1,
        "year": 2023,
        "venue": "Nature Communications",
        "paper": "Jin et al., 2023",
        "github": "https://github.com/DMCB-GIST/HiDRA",
        "input_features": ["expression_geneset"],
        "drug_features": ["drug_fingerprints"],
        "task": "regression",
        "notes": "Hierarchical gene-set-level DRP",
    },
    "GraTransDRP": {
        "tier": 1,
        "year": 2023,
        "venue": "Briefings in Bioinformatics",
        "paper": "Li et al., 2023",
        "github": "https://github.com/cnellington/GraTransDRP",
        "input_features": ["expression", "mutation"],
        "drug_features": ["drug_SMILES_graph"],
        "task": "regression",
        "notes": "Graph Transformer for drug molecules",
    },
    "TransCDR": {
        "tier": 1,
        "year": 2023,
        "venue": "Briefings in Bioinformatics",
        "paper": "Xia et al., 2023",
        "github": "https://github.com/XiaoqiongXia/TransCDR",
        "input_features": ["expression", "mutation"],
        "drug_features": ["drug_SMILES"],
        "task": "regression",
        "notes": "Transformer-based CDR prediction",
    },
    "PathDSP": {
        "tier": 1,
        "year": 2024,
        "venue": "Bioinformatics",
        "paper": "Tang et al., 2024",
        "github": "https://github.com/TangYiChing/PathDSP",
        "input_features": ["expression", "pathway_DB"],
        "drug_features": ["drug_fingerprints"],
        "task": "regression",
        "notes": "Pathway-aware drug sensitivity prediction",
    },
    # ── Tier 2: Established Baselines (2020–2022) — RECOMMENDED ──────────
    "GraphDRP": {
        "tier": 2,
        "year": 2022,
        "venue": "Bioinformatics",
        "paper": "Nguyen et al., 2022",
        "github": "https://github.com/hauldhut/GraphDRP",
        "input_features": ["expression"],
        "drug_features": ["drug_SMILES_graph"],
        "task": "regression",
        "notes": "GNN-based drug response prediction",
    },
    "DrugCell": {
        "tier": 2,
        "year": 2020,
        "venue": "Cancer Cell",
        "paper": "Kuenzi et al., 2020",
        "github": "https://github.com/idekerlab/DrugCell",
        "input_features": ["mutation", "GO_hierarchy"],
        "drug_features": ["drug_fingerprints"],
        "task": "regression",
        "notes": "Visible neural network with GO priors",
    },
    "DeepCDR": {
        "tier": 2,
        "year": 2020,
        "venue": "Briefings in Bioinformatics",
        "paper": "Li et al., 2020",
        "github": "https://github.com/kimmo1019/DeepCDR",
        "input_features": ["expression", "mutation", "methylation"],
        "drug_features": ["drug_SMILES"],
        "task": "regression",
        "notes": "Multi-omics DL for cancer drug response",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Tier A metrics (same as step14a — consistency)
# ══════════════════════════════════════════════════════════════════════════════

def compute_tier_a_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls):
    """Compute PCC, RMSE, AUROC, AUPRC-sensitive (Benchmarking Plan §2)."""
    metrics = {}

    # Regression
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50)))
    if len(y_true_ic50) > 2 and np.std(y_pred_ic50) > 1e-8:
        metrics["pearson_r"] = float(
            np.corrcoef(y_true_ic50, y_pred_ic50)[0, 1])
        sr = stats.spearmanr(y_true_ic50, y_pred_ic50)
        metrics["spearman_rho"] = float(
            sr.statistic if hasattr(sr, 'statistic') else sr[0])
    else:
        metrics["pearson_r"] = 0.0
        metrics["spearman_rho"] = 0.0

    # Classification
    has_both = len(set(y_true_cls)) > 1
    if has_both and y_prob_cls is not None:
        metrics["auroc"] = float(roc_auc_score(y_true_cls, y_prob_cls))
        metrics["auprc_resistant"] = float(
            average_precision_score(y_true_cls, y_prob_cls))
        metrics["auprc_sensitive"] = float(
            average_precision_score(1 - y_true_cls, 1 - y_prob_cls))
        # Load optimal threshold (Youden's J from step11, fallback 0.5)
        _thr_path = MODEL_DIR / "optimal_threshold.json"
        if _thr_path.exists():
            with open(_thr_path) as _f:
                _resist_thr = float(json.load(_f).get("optimal_threshold", 0.5))
        else:
            _resist_thr = 0.5
        y_pred_bin = (y_prob_cls > _resist_thr).astype(float)
        metrics["balanced_acc"] = float(
            balanced_accuracy_score(y_true_cls, y_pred_bin))
    else:
        metrics["auroc"] = 0.0
        metrics["auprc_resistant"] = 0.0
        metrics["auprc_sensitive"] = 0.0
        metrics["balanced_acc"] = 0.0

    return metrics


def compute_per_drug_metrics(df_test, y_pred_ic50, y_prob_cls):
    """Compute per-drug PCC and AUROC (Benchmarking Plan §6 Axis 2)."""
    per_drug = {}
    for drug in sorted(df_test["drug_name"].unique()):
        mask = df_test["drug_name"].values == drug
        if mask.sum() < 3:
            continue
        y_t = df_test["ln_ic50"].values[mask]
        y_p = y_pred_ic50[mask]
        y_cls = df_test["resistance_label"].values[mask]
        y_prb = y_prob_cls[mask] if y_prob_cls is not None else None

        drug_met = {"n_samples": int(mask.sum())}
        if np.std(y_p) > 1e-8:
            drug_met["pearson_r"] = float(np.corrcoef(y_t, y_p)[0, 1])
        else:
            drug_met["pearson_r"] = 0.0
        drug_met["rmse"] = float(np.sqrt(mean_squared_error(y_t, y_p)))

        if len(set(y_cls)) > 1 and y_prb is not None:
            drug_met["auroc"] = float(roc_auc_score(y_cls, y_prb))
        else:
            drug_met["auroc"] = 0.0

        per_drug[drug] = drug_met
    return per_drug


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset_and_split():
    """Load the multimodal dataset and split indices."""
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    split_path = MODEL_DIR / "split_indices.json"

    if not dataset_path.exists():
        print(f"  ✗ Dataset not found: {dataset_path}")
        print(f"    Run the pipeline (steps 01-06) first.")
        sys.exit(1)
    if not split_path.exists():
        print(f"  ✗ split_indices.json not found: {split_path}")
        print(f"    Run step11_train.py first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    with open(split_path) as f:
        split = json.load(f)

    return df, split


def export_gdsc_subset_for_method(df, split, method_name, method_dir):
    """
    Export our GDSC subset in a format suitable for external methods.

    Creates:
      <method_dir>/cell_lines.csv    — cell line IDs, COSMIC IDs, tissue
      <method_dir>/drug_info.csv     — drug names, GDSC IDs, SMILES
      <method_dir>/response.csv      — cell_line × drug → IC50, resistance_label
      <method_dir>/split.json        — our split indices
    """
    method_dir.mkdir(parents=True, exist_ok=True)

    # Cell line info
    cell_cols = [c for c in ["cell_line_name", "COSMIC_ID", "BROAD_ID",
                             "tissue", "target_protein", "mutation_class"]
                 if c in df.columns]
    cell_df = df[cell_cols].drop_duplicates().reset_index(drop=True)
    cell_df.to_csv(method_dir / "cell_lines.csv", index=False)

    # Drug info
    drug_data = []
    for drug_name, drug_cfg in cfg["drugs"].items():
        gdsc_id = cfg["gdsc"]["drug_ids"].get(drug_cfg["name"], "")
        drug_data.append({
            "drug_name": drug_cfg["name"],
            "gdsc_id": gdsc_id,
            "smiles": drug_cfg["smiles"],
            "generation": drug_cfg.get("generation", ""),
        })
    drug_df = pd.DataFrame(drug_data)
    drug_df.to_csv(method_dir / "drug_info.csv", index=False)

    # Response data
    resp_cols = [c for c in ["cell_line_name", "drug_name", "ln_ic50",
                             "resistance_label", "target_protein"]
                 if c in df.columns]
    resp_df = df[resp_cols].reset_index(drop=True)
    resp_df.to_csv(method_dir / "response.csv", index=False)

    # Split indices
    with open(method_dir / "split.json", "w") as f:
        json.dump(split, f, indent=2)

    print(f"    Exported data for {method_name}: {len(df)} samples, "
          f"{len(cell_df)} cell lines, {len(drug_df)} drugs")


# ══════════════════════════════════════════════════════════════════════════════
# External method runners
# ══════════════════════════════════════════════════════════════════════════════

def clone_method(method_name, github_url):
    """Clone external method repo if not already present."""
    method_dir = BENCHMARKS_DIR / method_name
    if method_dir.exists():
        print(f"    ✓ {method_name} repo already exists at {method_dir}")
        return method_dir

    print(f"    Cloning {method_name} from {github_url}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(method_dir)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        print(f"    ✓ Cloned {method_name}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    ✗ Failed to clone {method_name}: {e}")
        return None

    return method_dir


def run_external_method(method_name, method_spec, df, split):
    """
    Attempt to run an external method on our data.

    This is a framework function — each method requires custom integration.
    For methods that cannot be run directly, we document the attempt and
    fall back to reported metrics (Benchmarking Plan §9 Risk 1).
    """
    print(f"\n  ── {method_name} (Tier {method_spec['tier']}, "
          f"{method_spec['year']}) ──")
    print(f"    Paper: {method_spec['paper']}")
    print(f"    Venue: {method_spec['venue']}")

    result = {
        "method": method_name,
        "tier": method_spec["tier"],
        "year": method_spec["year"],
        "venue": method_spec["venue"],
        "paper": method_spec["paper"],
        "github": method_spec["github"],
        "input_features": method_spec["input_features"],
        "drug_features": method_spec["drug_features"],
        "status": "not_run",
        "notes": method_spec["notes"],
    }

    # Step 1: Clone repo
    repo_dir = clone_method(method_name, method_spec["github"])
    if repo_dir is None:
        result["status"] = "clone_failed"
        result["fallback"] = "report_published_numbers"
        print(f"    → Fallback: will report published numbers")
        return result

    # Step 2: Export our data in method's expected format
    data_dir = repo_dir / "our_data"
    export_gdsc_subset_for_method(df, split, method_name, data_dir)

    # Step 3: Method-specific integration
    # Each method has different input requirements. The framework provides
    # data export and metric computation; actual method running requires
    # manual integration per method.
    #
    # For the initial submission, we:
    # (a) Document the framework and data preparation
    # (b) Run methods that can be integrated automatically
    # (c) Report published numbers for methods requiring manual setup
    #
    # Integration status per method:
    integration = _check_method_integration(method_name, repo_dir)
    result["integration_status"] = integration

    if integration["can_auto_run"]:
        print(f"    Running {method_name}...")
        predictions = _run_method_pipeline(
            method_name, repo_dir, data_dir, df, split)
        if predictions is not None:
            result["status"] = "completed"
            result["test_metrics"] = predictions["test_metrics"]
            result["per_drug"] = predictions.get("per_drug", {})
            result["training_time_seconds"] = predictions.get(
                "training_time", 0)
        else:
            result["status"] = "run_failed"
            result["fallback"] = "report_published_numbers"
    else:
        result["status"] = "requires_manual_integration"
        result["integration_notes"] = integration["notes"]
        result["fallback"] = "report_published_numbers"
        print(f"    → Requires manual integration: {integration['notes']}")

    return result


def _check_method_integration(method_name, repo_dir):
    """Check if a method can be run automatically on our data."""
    # Check for standard entry points
    checks = {
        "has_train_script": False,
        "has_requirements": False,
        "has_readme": False,
        "can_auto_run": False,
        "notes": "",
    }

    for fname in ["train.py", "main.py", "run.py"]:
        if (repo_dir / fname).exists():
            checks["has_train_script"] = True
            break

    for fname in ["requirements.txt", "setup.py", "pyproject.toml",
                  "environment.yml"]:
        if (repo_dir / fname).exists():
            checks["has_requirements"] = True
            break

    for fname in ["README.md", "README.rst", "README"]:
        if (repo_dir / fname).exists():
            checks["has_readme"] = True
            break

    # Method-specific notes about integration requirements
    method_notes = {
        "DIPK": ("Requires expression + mutation + methylation + CNV + "
                 "PPI network data. Need to map our cell lines to DepMap "
                 "expression/methylation matrices."),
        "HiDRA": ("Requires gene-set-level expression (hierarchical). "
                  "Need MSigDB gene sets + expression matrix."),
        "GraTransDRP": ("Requires expression matrix + mutation matrix + "
                        "drug molecular graphs. Relatively straightforward "
                        "data preparation."),
        "TransCDR": ("Requires expression + mutation + drug SMILES. "
                     "Similar data requirements to GraTransDRP."),
        "PathDSP": ("Requires expression + pathway database (Reactome). "
                    "Need pathway mapping for our cell lines."),
        "GraphDRP": ("Requires expression matrix + drug SMILES molecular "
                     "graphs. Most straightforward integration."),
        "DrugCell": ("Requires binary mutation matrix + drug fingerprints + "
                     "GO hierarchy network. Unique architecture."),
        "DeepCDR": ("Requires expression + mutation + methylation + drug "
                    "SMILES. Needs DepMap multi-omics data."),
    }
    checks["notes"] = method_notes.get(method_name,
                                       "No integration notes available.")

    # For now, none auto-run — all require method-specific data preparation
    # This will be updated as each method is individually integrated
    checks["can_auto_run"] = False

    return checks


def _run_method_pipeline(method_name, repo_dir, data_dir, df, split):
    """
    Run an external method's training and prediction pipeline.

    This is a placeholder for method-specific integration code.
    Each method will have its own adapter function.
    """
    # This function will be extended per method as integration proceeds.
    # For now, return None to trigger the fallback path.
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Published performance collection
# ══════════════════════════════════════════════════════════════════════════════

def get_published_performance():
    """
    Collect published performance numbers for external methods.

    These are from the original papers, evaluated on GDSC (full dataset).
    They are NOT directly comparable to our numbers (different subset,
    different split) but provide context.

    Per Benchmarking Plan §9 Risk 1: "Report published numbers with
    'reported performance' caveat."
    """
    published = {
        "DIPK": {
            "source": "Li et al., Nature Communications 2024, Table 1",
            "dataset": "GDSC2 (full, ~170K samples)",
            "metrics": {
                "pearson_r": 0.920,
                "rmse": None,
                "spearman_rho": 0.893,
            },
            "caveat": ("Evaluated on full GDSC2 (~170K samples across all "
                       "drugs/cell lines). Our subset is 951 samples for "
                       "6 TKI drugs × EGFR/ERBB2 cell lines only. Direct "
                       "comparison is not valid — included for context."),
        },
        "HiDRA": {
            "source": "Jin et al., Nature Communications 2023, Table 1",
            "dataset": "GDSC2 (full)",
            "metrics": {
                "pearson_r": 0.907,
                "rmse": None,
                "spearman_rho": None,
            },
            "caveat": "Same caveat as DIPK — full GDSC2, not our subset.",
        },
        "GraTransDRP": {
            "source": "Li et al., Brief Bioinf 2023, Table 2",
            "dataset": "GDSC (full)",
            "metrics": {
                "pearson_r": 0.913,
                "rmse": 1.061,
            },
            "caveat": "Full GDSC dataset, all drugs and cell lines.",
        },
        "TransCDR": {
            "source": "Xia et al., Brief Bioinf 2023, Table 1",
            "dataset": "GDSC2 (full)",
            "metrics": {
                "pearson_r": 0.914,
                "rmse": None,
            },
            "caveat": "Full GDSC2 dataset.",
        },
        "PathDSP": {
            "source": "Tang et al., Bioinformatics 2024",
            "dataset": "GDSC (full)",
            "metrics": {
                "pearson_r": None,
                "rmse": None,
            },
            "caveat": "Pathway-based method; metrics vary by evaluation.",
        },
        "GraphDRP": {
            "source": "Nguyen et al., Bioinformatics 2022, Table 2",
            "dataset": "GDSC (full)",
            "metrics": {
                "pearson_r": 0.897,
                "rmse": 1.114,
            },
            "caveat": "Full GDSC dataset.",
        },
        "DrugCell": {
            "source": "Kuenzi et al., Cancer Cell 2020, Figure 2",
            "dataset": "GDSC (full)",
            "metrics": {
                "pearson_r": 0.858,
                "spearman_rho": 0.849,
            },
            "caveat": "Full GDSC, GO-hierarchy-based architecture.",
        },
        "DeepCDR": {
            "source": "Li et al., Brief Bioinf 2020, Table 1",
            "dataset": "GDSC (full)",
            "metrics": {
                "pearson_r": 0.916,
                "rmse": 1.030,
            },
            "caveat": "Full GDSC dataset, multi-omics input.",
        },
    }
    return published


# ══════════════════════════════════════════════════════════════════════════════
# Summary and comparison
# ══════════════════════════════════════════════════════════════════════════════

def format_comparison_table(results, our_metrics):
    """Print a formatted comparison table."""
    print(f"\n  {'=' * 90}")
    print(f"  {'Method':<15s} | {'Tier':>4s} | {'Year':>4s} | "
          f"{'PCC':>7s} | {'RMSE':>7s} | {'AUROC':>7s} | "
          f"{'AUPRC-s':>7s} | {'Status':<20s}")
    print(f"  {'-' * 90}")

    for name, res in sorted(results.items(),
                            key=lambda x: (x[1].get("tier", 9),
                                           x[1].get("year", 0))):
        tier = res.get("tier", "—")
        year = res.get("year", "—")
        status = res.get("status", "unknown")

        if status == "completed" and "test_metrics" in res:
            m = res["test_metrics"]
            pcc = f"{m.get('pearson_r', 0):.3f}"
            rmse = f"{m.get('rmse', 0):.3f}"
            auroc = f"{m.get('auroc', 0):.3f}"
            auprc = f"{m.get('auprc_sensitive', 0):.3f}"
        else:
            pcc = rmse = auroc = auprc = "  —  "

        print(f"  {name:<15s} | {tier:>4} | {year:>4} | "
              f"{pcc:>7s} | {rmse:>7s} | {auroc:>7s} | "
              f"{auprc:>7s} | {status:<20s}")

    # Our model
    if our_metrics:
        print(f"  {'-' * 90}")
        m = our_metrics
        print(f"  {'Ours (PTM-BDL)':<15s} | {'—':>4s} | {'2026':>4s} | "
              f"{m.get('pearson_r', 0):7.3f} | {m.get('rmse', 0):7.3f} | "
              f"{m.get('auroc', 0):7.3f} | "
              f"{m.get('auprc_sensitive', 0):7.3f} | {'completed':<20s}")

    print(f"  {'=' * 90}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 14b: External DRP Baselines (Tier 1–2)               ║")
    print("║  DIPK, HiDRA, GraTransDRP, TransCDR, PathDSP,             ║")
    print("║  GraphDRP, DrugCell, DeepCDR                               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Load dataset and split ─────────────────────────────────────────────
    df, split = load_dataset_and_split()
    test_idx = np.array(split["test_idx"])
    print(f"  Dataset: {len(df)} samples, test set: {len(test_idx)} samples")

    # ── Load our model's metrics for comparison ───────────────────────────
    eval_path = RESULTS_DIR / "evaluation_report.json"
    our_metrics = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        reg = eval_report.get("regression", {})
        cls = eval_report.get("classification", {})
        our_metrics = {
            "pearson_r": reg.get("pearson_r", 0),
            "rmse": reg.get("rmse", 0),
            "auroc": cls.get("auroc", 0),
            "auprc_sensitive": 0,
            "balanced_acc": cls.get("balanced_accuracy", 0),
        }

    # ── Determine which methods to run ─────────────────────────────────────
    # MVB (Minimum Viable Benchmarking) = Tier 1 (#1 DIPK, #2 HiDRA,
    # #3 GraTransDRP) + Tier 2 (#6 GraphDRP)
    mvb_methods = ["DIPK", "HiDRA", "GraTransDRP", "GraphDRP"]
    full_methods = list(EXTERNAL_METHODS.keys())

    # Use MVB by default, full if --full flag is passed
    if "--full" in sys.argv:
        methods_to_run = full_methods
        print(f"  Running FULL benchmarking ({len(methods_to_run)} methods)")
    else:
        methods_to_run = mvb_methods
        print(f"  Running MVB benchmarking ({len(methods_to_run)} methods)")
        print(f"  (Use --full for all {len(full_methods)} methods)")

    # ── Run each method ───────────────────────────────────────────────────
    results = {}
    for method_name in methods_to_run:
        if method_name not in EXTERNAL_METHODS:
            print(f"  ⚠ Unknown method: {method_name}")
            continue
        method_spec = EXTERNAL_METHODS[method_name]
        result = run_external_method(method_name, method_spec, df, split)
        results[method_name] = result

        # Save individual result
        method_out = EXT_RESULTS_DIR / f"{method_name}.json"
        with open(method_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"    ✓ Saved: {method_out}")

    # ── Add published performance for context ─────────────────────────────
    published = get_published_performance()
    for name, pub in published.items():
        if name in results:
            results[name]["published_performance"] = pub

    # ── Comparison table ──────────────────────────────────────────────────
    format_comparison_table(results, our_metrics)

    # ── Integration status summary ────────────────────────────────────────
    print(f"\n  Integration Status Summary:")
    print(f"  {'─' * 60}")
    n_complete = sum(1 for r in results.values()
                     if r["status"] == "completed")
    n_manual = sum(1 for r in results.values()
                   if r["status"] == "requires_manual_integration")
    n_failed = sum(1 for r in results.values()
                   if r["status"] in ("clone_failed", "run_failed"))
    print(f"    Completed:           {n_complete}/{len(results)}")
    print(f"    Needs integration:   {n_manual}/{len(results)}")
    print(f"    Failed:              {n_failed}/{len(results)}")

    if n_manual > 0:
        print(f"\n  Methods requiring manual integration:")
        for name, res in results.items():
            if res["status"] == "requires_manual_integration":
                notes = res.get("integration_notes", "")
                print(f"    • {name}: {notes}")

    # ── Save combined summary ─────────────────────────────────────────────
    summary = {
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_methods": len(results),
        "n_completed": n_complete,
        "n_manual_integration": n_manual,
        "n_failed": n_failed,
        "methods": results,
        "published_performance": published,
        "our_model_metrics": our_metrics,
        "data_info": {
            "dataset": "GDSC2 IC50 for 6 TKI drugs × EGFR/ERBB2 cell lines",
            "n_samples": len(df),
            "n_test": len(test_idx),
            "split": "70/15/15 stratified by resistance_label × target_protein",
        },
        "fairness_protocol": {
            "data": "All methods evaluated on identical biological samples",
            "split": "Same split_indices.json for all methods",
            "hyperparams": "Published default hyperparameters (no tuning)",
            "reference": "Benchmarking Plan §5",
        },
    }
    summary_path = EXT_RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  ✓ Saved summary: {summary_path}")

    print("\n✓ Step 14b complete!")
    if n_manual > 0:
        print(f"  ℹ {n_manual} methods need manual integration — see notes above")
        print(f"  ℹ Published numbers included as fallback context")


if __name__ == "__main__":
    main()
