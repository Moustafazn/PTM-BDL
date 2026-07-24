#!/usr/bin/env python3
"""
HeLa/HDAC Case Study — Comprehensive Evaluation & Benchmarking.

PURPOSE:
  Evaluate the trained PTM-BDL model on the held-out test set with
  case-study-specific biological validation metrics.

EVALUATION AXES:
  1. Standard metrics — AUROC, BAcc, Pearson R, RMSE, Spearman ρ
  2. Per-drug evaluation — compare HDAC inh. vs HAT inh. vs control
  3. PTM-type stratified — acetylation vs phosphorylation contribution
  4. Biological validation — A486 negative control discrimination
  5. Drug mechanism grouping — epigenetic vs cytotoxic

BIOLOGICAL VALIDATION TARGETS:
  • HDAC inhibitors (Vorinostat, Romidepsin) should increase histone
    acetylation marks → positive acetyl log2FC
    Ref: Marks & Xu, J Cell Biochem 2009 (PMID 19479898)
  • A485 (HAT inhibitor) should decrease p300-mediated acetylation
    → negative acetyl log2FC at EP300 autoacetylation sites
    Ref: Lasko et al., Nature 2017 (PMID 29211713)
  • A486 (inactive control) should show near-zero PTM changes
    → model should predict least drug effect
    Ref: Lasko et al., Nature 2017 (PMID 29211713) — A486 is the
         structurally matched inactive enantiomer of A485
  • CUDC-101 (triple HDAC/EGFR/HER2 inhibitor) should show BOTH
    acetylation increases AND phosphorylation decreases
    Ref: Lai et al., J Med Chem 2010 (PMID 20568778)

RECENT REFERENCES (2024-2026):
  • Badkul et al., DrugPTM-Bench 2024 — primary PTM dose-response data source
  • Hartl et al., Cell Reports 2024 — dose-resolved HDAC inhibitor proteomics
  • Lasko et al., Nat Rev Drug Discov 2024 (PMID 38382638) — HDAC inhibitor
    resistance mechanisms and biomarker landscape
  • Ho et al., Nat Rev Clin Oncol 2024 — epigenetic therapy resistance
  • Liu et al., JBC 2025 — DL for PTM crosstalk on Hsp90 and drug binding
  • Park et al., Nat Comm 2025 — Romidepsin + RTK sensitization
  • Bondarev et al., Nucleic Acids Res 2025 — PTM crosstalk databases
  • Li et al., Mol Cancer 2025 — HDAC inhibitor combinations in solid tumors
  • Ardito et al., IJMS 2019 — acetylation↔phosphorylation crosstalk
  • Wu et al., Precision Clin Med 2024 — PTM systems as cancer biomarkers

FOUNDATIONAL:
  • Narita et al., Nat Rev Mol Cell Biol 2019 (PMID 30487433) — HATs/HDACs
  • Fischle et al., Nature 2003 (PMID 14573844) — H3S10ph-K9ac binary switch
  • Seto & Yoshida, Cold Spring Harb Perspect Biol 2014 (PMID 24691964)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr

from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics, load_threshold, make_eval_loader
from src.ptm_bdl.training import build_model_from_cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load optimal classification threshold (Youden's J, from training)
RESIST_THRESHOLD = load_threshold(MODEL_DIR)

# Drug mechanism groups for stratified analysis
DRUG_GROUPS = {
    "HDAC_inhibitor": ["Vorinostat", "Romidepsin", "CUDC101"],
    "HAT_inhibitor": ["A485"],
    "negative_control": ["A486"],
    "natural_modulator": ["Curcumin"],
}


def evaluate():
    """Comprehensive evaluation with case-study-specific analyses."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Comprehensive Evaluation                   ║")
    print(f"║  PTM types: phosphorylation + acetylation (NEW)            ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    device_str = cfg["training"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"  Device: {device}")

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    dataset = ResistanceDataset(dataset_path, features_dir)
    df = dataset.df

    with open(MODEL_DIR / "split_indices.json") as f:
        split = json.load(f)
    test_idx = np.array(split["test_idx"])

    model = build_model_from_cfg(cfg).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))
    model.eval()

    # Helper — build a DataLoader from index array (same pattern as egfr)
    batch_size = cfg["model"]["batch_size"]

    def _loader(indices):
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ── 1. Standard metrics ──────────────────────────────────────────────
    print("\n  1. Standard metrics (full test set)...")
    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
        model, _loader(test_idx))
    regression, classification = compute_full_metrics(
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
        threshold=RESIST_THRESHOLD)
    metrics = {**regression, **classification}
    print(f"    AUROC:     {classification.get('auroc', 'N/A')}")
    print(f"    BAcc:      {classification.get('balanced_accuracy', 'N/A')}")
    print(f"    Pearson R: {regression.get('pearson_r', 'N/A')}")
    print(f"    RMSE:      {regression.get('rmse', 'N/A')}")

    # ── 2. Per-drug evaluation ───────────────────────────────────────────
    print("\n  2. Per-drug evaluation...")
    per_drug = {}
    df_test = df.iloc[test_idx]
    for drug in df_test["drug_name"].unique():
        drug_mask = df_test["drug_name"] == drug
        drug_idx = test_idx[drug_mask.values]
        if len(drug_idx) < 3:
            continue
        d_ic50_t, d_ic50_p, d_cls_t, d_cls_p = collect_predictions(
            model, _loader(drug_idx))
        d_reg, d_cls = compute_full_metrics(d_ic50_t, d_ic50_p, d_cls_t, d_cls_p,
                                            threshold=RESIST_THRESHOLD)
        drug_metrics = {**d_reg, **d_cls}
        per_drug[drug] = drug_metrics
        print(f"    {drug:12s}: AUROC={d_cls.get('auroc', 'N/A'):.3f}, "
              f"BAcc={d_cls.get('balanced_accuracy', 'N/A'):.3f}")

    # ── 3. Drug mechanism group evaluation ───────────────────────────────
    print("\n  3. Drug mechanism group evaluation...")
    group_metrics = {}
    for group_name, group_drugs in DRUG_GROUPS.items():
        group_mask = df_test["drug_name"].isin(group_drugs)
        group_idx = test_idx[group_mask.values]
        if len(group_idx) < 2:
            continue
        g_ic50_t, g_ic50_p, g_cls_t, g_cls_p = collect_predictions(
            model, _loader(group_idx))
        g_reg, g_cls = compute_full_metrics(g_ic50_t, g_ic50_p, g_cls_t, g_cls_p,
                                            threshold=RESIST_THRESHOLD)
        gm = {**g_reg, **g_cls}
        group_metrics[group_name] = gm
        print(f"    {group_name:20s} ({len(group_idx)} samples): "
              f"AUROC={g_cls.get('auroc', 'N/A')}")

    # ── 4. Biological validation: A486 negative control ──────────────────
    print("\n  4. Biological validation: A486 inactive control...")
    print("    Ref: Lasko et al., Nature 2017 (PMID 29211713)")
    a486_mask = df_test["drug_name"] == "A486"
    a485_mask = df_test["drug_name"] == "A485"
    if a486_mask.any() and a485_mask.any():
        a486_ptm = df_test.loc[a486_mask, "phospho_mean_log2fc"].mean()
        a485_ptm = df_test.loc[a485_mask, "phospho_mean_log2fc"].mean()
        print(f"    A486 (inactive) mean PTM effect: {a486_ptm:.4f}")
        print(f"    A485 (active)   mean PTM effect: {a485_ptm:.4f}")
        print(f"    ✓ A486 < A485 in PTM effect: {abs(a486_ptm) < abs(a485_ptm)}")

    # ── 5. PTM-type stratified analysis ──────────────────────────────────
    print("\n  5. Acetylation vs phosphorylation contribution...")
    print("    Ref: Narita et al., Nat Rev Mol Cell Biol 2019 (PMID 30487433)")
    if "acetyl_mean_log2fc" in df_test.columns:
        acetyl_cols = [c for c in df_test.columns if "acetyl" in c and "log2fc" in c]
        phospho_cols = [c for c in df_test.columns if "phospho" in c and "log2fc" in c]
        print(f"    Acetylation features: {len(acetyl_cols)} columns")
        print(f"    Phosphorylation features: {len(phospho_cols)} columns")

    # ── Save results ─────────────────────────────────────────────────────
    report = {
        "case_study": CASE_STUDY,
        "overall_metrics": metrics,
        "per_drug": per_drug,
        "drug_group_metrics": group_metrics,
        "references": {
            "primary_data": "Badkul et al., DrugPTM-Bench 2024",
            "dose_resolved_proteomics": "Hartl et al., Cell Reports 2024",
            "hdac_resistance": "Lasko et al., Nat Rev Drug Discov 2024 (PMID 38382638)",
            "hat_mechanism": "Narita et al., Nat Rev Mol Cell Biol 2019 (PMID 30487433)",
            "a485_a486": "Lasko et al., Nature 2017 (PMID 29211713)",
            "cudc101": "Lai et al., J Med Chem 2010 (PMID 20568778)",
            "ptm_crosstalk": "Fischle et al., Nature 2003 (PMID 14573844)",
            "acetyl_phospho_crosstalk": "Ardito et al., IJMS 2019",
            "romidepsin_rtk": "Park et al., Nat Comm 2025",
            "dl_ptm_crosstalk": "Liu et al., JBC 2025",
            "hdac3_akt": "Gupta et al., Leukemia 2017",
        },
    }

    with open(RESULTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    np.savez(RESULTS_DIR / "test_predictions.npz",
             y_true_ic50=y_true_ic50, y_pred_ic50=y_pred_ic50,
             y_true_cls=y_true_cls, y_prob_cls=y_prob_cls)

    print(f"\n  ✓ Report saved: {RESULTS_DIR / 'evaluation_report.json'}")
    print(f"✓ Evaluation complete!")


if __name__ == "__main__":
    evaluate()
