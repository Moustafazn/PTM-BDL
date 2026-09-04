#!/usr/bin/env python3
"""
EGFR/ERBB2 TKI Resistance — Comprehensive Evaluation & Benchmarking.

PURPOSE:
  Evaluate the trained PTM-BDL model on the HELD-OUT test set using the SAME
  stratified split indices saved during training (split_indices.json).
  This prevents data leakage — the test set is never seen during training.

METRICS (comprehensive, both regression and classification):
  Regression:      MSE, RMSE, R², Pearson R, Spearman ρ
  Classification:  Accuracy, Balanced Accuracy, Sensitivity, Specificity,
                   F1, AUROC, AUPRC
  Per-class:       Confusion matrix, classification report

NOVEL ANALYSES:
  1. Per-Protein Evaluation — EGFR (NSCLC) vs ERBB2/HER2 (breast)
  2. Drug-Specific Evaluation — per-drug BAcc, RMSE, R
     • Compare Afatinib vs Osimertinib (same binding site, diff selectivity)
     • Compare 3rd-gen vs 1st-gen TKIs
     • Ref: Zhao et al., Nat Rev Clin Oncol 2026 (PMID 41219394) —
       catalogues generation-specific resistance mechanisms.
  3. Mutation-Stratified Analysis — per-mutation-group predictions
     • EGFR: Run on ALL 36 EGFR-mutant samples
     • ERBB2: HER2-amplification tier groups (Hudis 2007, Citri & Yarden 2006)
  4. Glyco-State Stratified Analysis — mean glyco occupancy tertiles
  5. Confidence-Aware Analysis — measured vs propagated phospho
  6. Baseline Comparisons — majority-class, mean-prediction

INPUTS:
  data/models/best_model.pt          — trained model weights
  data/models/split_indices.json     — train/val/test indices from training
  data/processed/multimodal_dataset.csv + data/features/*

OUTPUTS:
  results/evaluation_report.json     — all metrics
  results/test_predictions.npz       — cached predictions for statistical tests
  results/figures/evaluation_plots.png — evaluation visualizations
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)

from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics
from src.ptm_bdl.training import build_model_from_cfg, load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load optimal classification threshold (Youden's J, from training) ────────
# Falls back to 0.5 if optimal_threshold.json doesn't exist.
# Ref: Youden WJ (1950) Cancer 3:32-35.
_threshold_path = MODEL_DIR / "optimal_threshold.json"
if _threshold_path.exists():
    with open(_threshold_path) as _f:
        _thr_info = json.load(_f)
    RESIST_THRESHOLD = float(_thr_info.get("optimal_threshold", 0.5))
else:
    RESIST_THRESHOLD = 0.5


def load_model(device):
    """Load PTM-BDL multimodal model + weights."""
    model = build_model_from_cfg(cfg).to(device)
    model_path = MODEL_DIR / "best_model.pt"
    if model_path.exists():
        load_checkpoint(model, model_path, device)
        print(f"  ✓ Loaded model: {model_path.name}")
    else:
        print(f"  ⚠ No trained model found! Using random weights for demo.")
        model.eval()
    return model


def evaluate():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EGFR/ERBB2 TKI — Evaluation & Benchmarking               ║")
    print("║  + Per-protein + Drug-specific + Mutation-stratified       ║")
    print("║  + Glyco-state + Confidence-aware                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    device = torch.device("cpu")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "multimodal_dataset.csv"
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    dataset = ResistanceDataset(dataset_path, features_dir)
    df = dataset.df
    print(f"\n  Dataset: {len(dataset)} samples")
    print(f"  Threshold: {RESIST_THRESHOLD:.4f}")

    # ── Load split indices from training ──────────────────────────────────────
    split_path = MODEL_DIR / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_info = json.load(f)
        test_idx = split_info["test_idx"]
        print(f"  ✓ Loaded split indices (stratified by {split_info['stratification']})")
    else:
        print(f"  ⚠ split_indices.json not found — using last 15% as test")
        n_test = max(1, int(len(dataset) * 0.15))
        test_idx = list(range(len(dataset) - n_test, len(dataset)))

    test_set = torch.utils.data.Subset(dataset, test_idx)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg["model"]["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )

    test_labels = df["resistance_label"].values[test_idx]
    n_res = int((test_labels == 1).sum())
    n_sens = int((test_labels == 0).sum())
    print(f"  Test set: {len(test_idx)} samples "
          f"({n_res} resistant, {n_sens} sensitive)")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(device)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: STANDARD EVALUATION ON TEST SET
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 1: Standard Test Set Evaluation")
    print(f"  {'=' * 55}")

    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
        model, test_loader
    )
    y_pred_cls_binary = (y_prob_cls > RESIST_THRESHOLD).astype(float)

    # Cache test predictions for statistical tests (bootstrap CIs, DeLong)
    pred_cache_path = RESULTS_DIR / "test_predictions.npz"
    np.savez(
        pred_cache_path,
        y_true_ic50=y_true_ic50,
        y_pred_ic50=y_pred_ic50,
        y_true_cls=y_true_cls,
        y_prob_cls=y_prob_cls,
    )
    print(f"  ✓ Cached test predictions: {pred_cache_path}")

    regression, classification = compute_full_metrics(
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
        threshold=RESIST_THRESHOLD,
    )

    results = {
        "test_samples": len(y_true_ic50),
        "threshold": RESIST_THRESHOLD,
        "regression": regression,
        "classification": classification,
    }

    print(f"\n  REGRESSION (IC50 prediction):")
    for k, v in regression.items():
        print(f"    {k:15s}: {v:.4f}")

    print(f"\n  CLASSIFICATION (Resistance prediction):")
    for k, v in classification.items():
        if isinstance(v, (int, float)):
            print(f"    {k:30s}: {v:.4f}")
        elif k == "confusion_matrix":
            print(f"    {k:30s}: {v}")

    # ── Baseline comparisons ──────────────────────────────────────────────────
    has_both = len(set(y_true_cls)) > 1
    majority_pred = np.ones_like(y_true_cls)
    majority_acc = float(accuracy_score(y_true_cls, majority_pred))
    majority_bacc = float(balanced_accuracy_score(y_true_cls, majority_pred)) \
        if has_both else 0.0
    mean_ic50 = y_true_ic50.mean()
    mean_mse = float(((y_true_ic50 - mean_ic50) ** 2).mean())

    results["baselines"] = {
        "majority_class_accuracy": majority_acc,
        "majority_class_balanced_accuracy": majority_bacc,
        "mean_prediction_mse": mean_mse,
        "mean_prediction_rmse": float(np.sqrt(mean_mse)),
    }

    model_bacc = classification["balanced_accuracy"]
    model_rmse = regression["rmse"]
    print(f"\n  MODEL vs BASELINES:")
    print(f"    Balanced Acc: {model_bacc:.4f} vs {majority_bacc:.4f} "
          f"({'✓ BETTER' if model_bacc > majority_bacc else '✗ WORSE'})")
    print(f"    RMSE:         {model_rmse:.4f} vs {np.sqrt(mean_mse):.4f} "
          f"({'✓ BETTER' if model_rmse < np.sqrt(mean_mse) else '✗ WORSE'})")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1b: PER-PROTEIN EVALUATION (EGFR vs HER2)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 1b: Per-Protein Evaluation (EGFR vs HER2/ERBB2)")
    print(f"  {'=' * 55}")

    test_df_gene = df.iloc[test_idx].reset_index(drop=True)
    test_df_gene["ic50_pred"] = y_pred_ic50
    test_df_gene["resist_prob"] = y_prob_cls

    protein_results = {}
    if "target_protein" in test_df_gene.columns:
        for gene in sorted(test_df_gene["target_protein"].dropna().unique()):
            gene_mask = test_df_gene["target_protein"] == gene
            gene_sub = test_df_gene[gene_mask]
            if len(gene_sub) < 2:
                continue
            g_ic50_t = gene_sub["ln_ic50"].values
            g_ic50_p = gene_sub["ic50_pred"].values
            g_cls_t = gene_sub["resistance_label"].values
            g_cls_p = gene_sub["resist_prob"].values
            g_cls_bin = (g_cls_p > RESIST_THRESHOLD).astype(float)
            n_sens_g = int((g_cls_t == 0).sum())
            gene_met = {
                "n_samples": len(gene_sub),
                "n_sensitive": n_sens_g,
                "n_resistant": len(gene_sub) - n_sens_g,
                "mean_prob": float(g_cls_p.mean()),
            }
            if len(set(g_cls_t)) > 1:
                gene_met["balanced_accuracy"] = float(balanced_accuracy_score(g_cls_t, g_cls_bin))
                gene_met["auroc"] = float(roc_auc_score(g_cls_t, g_cls_p))
            else:
                gene_met["balanced_accuracy"] = float(accuracy_score(g_cls_t, g_cls_bin))
                gene_met["auroc"] = 0.0
            gene_met["rmse"] = float(np.sqrt(mean_squared_error(g_ic50_t, g_ic50_p)))
            if len(g_ic50_t) > 2 and np.std(g_ic50_p) > 1e-8:
                gene_met["pearson_r"] = float(np.corrcoef(g_ic50_t, g_ic50_p)[0, 1])
            else:
                gene_met["pearson_r"] = 0.0
            protein_results[gene] = gene_met
            print(f"\n  {gene}:")
            print(f"    Samples: {gene_met['n_samples']} "
                  f"({gene_met['n_sensitive']} sensitive, {gene_met['n_resistant']} resistant)")
            print(f"    BAcc: {gene_met['balanced_accuracy']:.3f} | "
                  f"AUROC: {gene_met['auroc']:.3f} | "
                  f"RMSE: {gene_met['rmse']:.3f} | "
                  f"R: {gene_met['pearson_r']:.3f}")

    results["per_protein"] = protein_results

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: DRUG-SPECIFIC EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 2: Drug-Specific Evaluation")
    print(f"  {'=' * 55}")

    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_df["ic50_pred"] = y_pred_ic50
    test_df["resist_prob"] = y_prob_cls
    test_df["resist_pred"] = y_pred_cls_binary

    drug_results = {}
    print(f"\n  {'Drug':<15s} | {'N':>4s} | {'Sens':>4s} | {'BAcc':>6s} | "
          f"{'RMSE':>6s} | {'R':>6s} | {'AUROC':>6s} | {'mProb':>5s}")
    print(f"  {'-' * 65}")

    for drug_name in sorted(test_df["drug_name"].unique()):
        mask = test_df["drug_name"] == drug_name
        drug_sub = test_df[mask]
        y_t_ic50 = drug_sub["ln_ic50"].values
        y_p_ic50 = drug_sub["ic50_pred"].values
        y_t_cls = drug_sub["resistance_label"].values
        y_p_prob = drug_sub["resist_prob"].values
        y_p_cls = drug_sub["resist_pred"].values
        n_sens_d = int((y_t_cls == 0).sum())
        n_total = len(drug_sub)
        drug_metrics = {
            "n_samples": n_total,
            "n_sensitive": n_sens_d,
            "n_resistant": n_total - n_sens_d,
            "mean_prob": float(y_p_prob.mean()),
        }
        if len(set(y_t_cls)) > 1:
            drug_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_t_cls, y_p_cls))
            drug_metrics["auroc"] = float(roc_auc_score(y_t_cls, y_p_prob))
        else:
            drug_metrics["balanced_accuracy"] = float(accuracy_score(y_t_cls, y_p_cls))
            drug_metrics["auroc"] = 0.0
        drug_metrics["rmse"] = float(np.sqrt(mean_squared_error(y_t_ic50, y_p_ic50)))
        if len(y_t_ic50) > 2 and np.std(y_p_ic50) > 1e-8:
            drug_metrics["pearson_r"] = float(np.corrcoef(y_t_ic50, y_p_ic50)[0, 1])
        else:
            drug_metrics["pearson_r"] = 0.0
        drug_results[drug_name] = drug_metrics
        print(f"  {drug_name:<15s} | {n_total:4d} | {n_sens_d:4d} | "
              f"{drug_metrics['balanced_accuracy']:6.3f} | "
              f"{drug_metrics['rmse']:6.3f} | "
              f"{drug_metrics['pearson_r']:6.3f} | "
              f"{drug_metrics['auroc']:6.3f} | "
              f"{drug_metrics['mean_prob']:5.3f}")

    results["drug_specific"] = drug_results

    # ── Drug comparison insight (Afatinib vs Osimertinib) ─────────────────────
    # Both bind C797 covalently but have different selectivity profiles:
    #   Afatinib (2nd-gen, pan-ERBB) — irreversible, targets EGFR + HER2
    #   Osimertinib (3rd-gen, T790M-selective) — covalent, targets T790M
    if "Afatinib" in drug_results and "Osimertinib" in drug_results:
        afa = drug_results["Afatinib"]
        osi = drug_results["Osimertinib"]
        print(f"\n  Drug Comparison (same C797 binding, different selectivity):")
        print(f"    Afatinib (2nd-gen, pan-ERBB):  BAcc={afa['balanced_accuracy']:.3f}, "
              f"RMSE={afa['rmse']:.3f}")
        print(f"    Osimertinib (3rd-gen, T790M):  BAcc={osi['balanced_accuracy']:.3f}, "
              f"RMSE={osi['rmse']:.3f}")

    # ── Cross-protein drug analysis ───────────────────────────────────────────
    # ALL 4 EGFR drugs are tested on BOTH NSCLC (EGFR) and breast (ERBB2) cell
    # lines in GDSC2. Lapatinib and Sapitinib are ERBB2-only.
    # This dual-context approach lets the model learn cross-receptor patterns.
    if "target_protein" in test_df.columns:
        cross_drugs = ["Osimertinib", "Gefitinib", "Afatinib", "Erlotinib",
                       "Lapatinib", "Sapitinib"]
        print(f"\n  Cross-Protein Drug Analysis:")
        print(f"  {'Drug':<15s} | {'Gene':>5s} | {'N':>4s} | {'BAcc':>6s} | {'mProb':>5s} | {'RMSE':>6s}")
        print(f"  {'-' * 55}")
        for drug_name in cross_drugs:
            for gene in ["EGFR", "ERBB2"]:
                mask = (test_df["drug_name"] == drug_name) & (test_df["target_protein"] == gene)
                sub = test_df[mask]
                if len(sub) < 2:
                    continue
                g_cls_t = sub["resistance_label"].values
                g_cls_p = (sub["resist_prob"].values > RESIST_THRESHOLD).astype(float)
                g_ic50_t = sub["ln_ic50"].values
                g_ic50_p = sub["ic50_pred"].values
                bacc = float(balanced_accuracy_score(g_cls_t, g_cls_p)) if len(set(g_cls_t)) > 1 else float(
                    accuracy_score(g_cls_t, g_cls_p))
                rmse = float(np.sqrt(mean_squared_error(g_ic50_t, g_ic50_p)))
                mprob = float(sub["resist_prob"].values.mean())
                print(f"  {drug_name:<15s} | {gene:>5s} | {len(sub):4d} | {bacc:6.3f} | {mprob:5.3f} | {rmse:6.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: MUTATION-STRATIFIED BIOLOGICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    # Two parallel stratifications (gene-aware):
    #   3a) EGFR mutation groups — driven by activating point/del mutations
    #       (L858R, exon19del, T790M, C797S, etc.)
    #   3b) ERBB2 HER2-amplification tier groups — HER2 oncogenicity in
    #       breast cancer is amplification-driven, not point-mutation-driven
    #       (Hudis 2007 PMID 17626692; Citri & Yarden 2006 PMID 16829981)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 3: Mutation-Stratified Analysis (ALL samples)")
    print(f"  {'=' * 55}")

    # Run model on ALL samples for biological analysis
    all_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["model"]["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )
    all_ic50_true, all_ic50_pred, all_cls_true, all_cls_prob = collect_predictions(
        model, all_loader
    )

    mc = df["mutation_classes"].fillna("wild_type").str.lower()
    is_mutant = mc.str.contains("pathogenic|cmp_driver", regex=True)
    has_target = "target_protein" in df.columns
    is_egfr = (df["target_protein"] == "EGFR") if has_target else pd.Series([True] * len(df))
    is_erbb2 = (df["target_protein"] == "ERBB2") if has_target else pd.Series([False] * len(df))

    # ── 3a) EGFR mutation groups ──────────────────────────────────────────
    mutation_groups = defaultdict(list)
    for i in range(len(df)):
        if not is_egfr.iloc[i]:
            continue
        mutations = str(df.iloc[i].get("egfr_mutations", "wild_type"))
        if is_mutant.iloc[i]:
            mutation_groups[mutations].append(i)

    mutation_results = {}
    print(f"\n  EGFR mutation groups:")
    print(f"  {'Mutation Group':<25s} | {'N':>3s} | {'Sens':>4s} | {'Res':>4s} | "
          f"{'mProb':>5s} | {'mIC50_T':>7s} | {'mIC50_P':>7s} | {'Note'}")
    print(f"  {'-' * 80}")

    for mut_name, indices in sorted(mutation_groups.items(), key=lambda x: -len(x[1])):
        idx_arr = np.array(indices)
        probs = all_cls_prob[idx_arr]
        labels = all_cls_true[idx_arr]
        ic50_t = all_ic50_true[idx_arr]
        ic50_p = all_ic50_pred[idx_arr]
        n_sens_m = int((labels == 0).sum())
        n_res_m = int((labels == 1).sum())
        mean_prob = float(probs.mean())
        if n_sens_m > n_res_m:
            note = "mostly sensitive"
        elif n_res_m > 3 * n_sens_m:
            note = "mostly resistant"
        else:
            note = "mixed"
        mutation_results[mut_name] = {
            "n_samples": len(indices), "n_sensitive": n_sens_m,
            "n_resistant": n_res_m, "mean_resist_prob": mean_prob,
            "mean_ic50_true": float(ic50_t.mean()),
            "mean_ic50_pred": float(ic50_p.mean()),
            "sample_indices": indices,
        }
        print(f"  {mut_name:<25s} | {len(indices):3d} | {n_sens_m:4d} | {n_res_m:4d} | "
              f"{mean_prob:5.3f} | {float(ic50_t.mean()):7.2f} | {float(ic50_p.mean()):7.2f} | {note}")

    results["mutation_stratified"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "sample_indices"}
        for k, v in mutation_results.items()
    }

    # ── 3b) ERBB2 amplification-tier groups ─────────────────────────────────
    # HER2 oncogenicity in breast cancer is amplification-driven.
    # Hudis NEJM 2007 (PMID 17626692); Citri & Yarden NRMCB 2006 (PMID 16829981)
    erbb2_groups = defaultdict(list)
    if has_target and is_erbb2.any():
        for i in range(len(df)):
            if not is_erbb2.iloc[i]:
                continue
            tier = str(df.iloc[i].get("mutation_classes", "ERBB2_wild_type"))
            erbb2_groups[tier].append(i)

    erbb2_results = {}
    if erbb2_groups:
        print(f"\n  ERBB2 amplification tiers (HER2 biology — Hudis 2007):")
        print(f"  {'HER2 Group':<25s} | {'N':>3s} | {'Sens':>4s} | {'Res':>4s} | "
              f"{'mProb':>5s} | {'mIC50_T':>7s} | {'mIC50_P':>7s}")
        print(f"  {'-' * 75}")
        for tier_name, indices in sorted(erbb2_groups.items(), key=lambda x: -len(x[1])):
            idx_arr = np.array(indices)
            probs = all_cls_prob[idx_arr]
            labels = all_cls_true[idx_arr]
            ic50_t = all_ic50_true[idx_arr]
            ic50_p = all_ic50_pred[idx_arr]
            n_sens_m = int((labels == 0).sum())
            n_res_m = int((labels == 1).sum())
            erbb2_results[tier_name] = {
                "n_samples": len(indices), "n_sensitive": n_sens_m,
                "n_resistant": n_res_m, "mean_resist_prob": float(probs.mean()),
                "mean_ic50_true": float(ic50_t.mean()),
                "mean_ic50_pred": float(ic50_p.mean()),
            }
            print(f"  {tier_name:<25s} | {len(indices):3d} | {n_sens_m:4d} | "
                  f"{n_res_m:4d} | {float(probs.mean()):5.3f} | "
                  f"{float(ic50_t.mean()):7.2f} | {float(ic50_p.mean()):7.2f}")
        results["erbb2_amp_stratified"] = erbb2_results

    # ── Biological insight: EGFR-mutant vs WT ─────────────────────────────────
    if mutation_results:
        activating_probs = []
        for mut_name, data in mutation_results.items():
            if data["n_sensitive"] > 0:
                activating_probs.extend(
                    all_cls_prob[np.array(data["sample_indices"])].tolist())
        wt_indices = [i for i in range(len(df)) if is_egfr.iloc[i] and not is_mutant.iloc[i]]
        if wt_indices and activating_probs:
            wt_mean_prob = float(all_cls_prob[np.array(wt_indices)].mean())
            mut_mean_prob = float(np.mean(activating_probs))
            print(f"\n  Biological Insight (EGFR):")
            print(f"    EGFR WT/VUS mean resist prob:   {wt_mean_prob:.3f}")
            print(f"    EGFR-mutant mean prob:          {mut_mean_prob:.3f}")
            diff = wt_mean_prob - mut_mean_prob
            print(f"    Difference: {diff:+.3f} "
                  f"({'✓ mutants more sensitive' if diff > 0 else '⚠ unexpected'})")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3b: GLYCO-STATE STRATIFIED EVALUATION (PTM-BDL §2f)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 3b: Glyco-State Stratified Evaluation")
    print(f"  {'=' * 55}")

    glyco_cols = [f"glyco_slot{i:02d}" for i in range(12)]
    have_glyco = all(c in df.columns for c in glyco_cols)
    glyco_state_stratified = {}
    if have_glyco:
        glyco_mean = df[glyco_cols].mean(axis=1).values
        q1 = float(np.nanpercentile(glyco_mean, 33))
        q2 = float(np.nanpercentile(glyco_mean, 66))

        def _bin_of(v):
            if not np.isfinite(v): return "mid"
            if v <= q1: return "low"
            if v >= q2: return "high"
            return "mid"

        glyco_bins = np.array([_bin_of(v) for v in glyco_mean])
        mc_str = df["mutation_classes"].fillna("wild_type").astype(str).values
        print(f"  Glyco-mean tertile cutoffs: q33={q1:.3f}, q66={q2:.3f}")
        for bin_name in ["low", "mid", "high"]:
            for mclass in sorted(set(mc_str)):
                mask = (glyco_bins == bin_name) & (mc_str == mclass)
                if mask.sum() < 1:
                    continue
                idx_arr = np.where(mask)[0]
                probs = all_cls_prob[idx_arr]
                labels = all_cls_true[idx_arr]
                preds = (probs > RESIST_THRESHOLD).astype(float)
                ic50_t = all_ic50_true[idx_arr]
                ic50_p = all_ic50_pred[idx_arr]
                entry = {
                    "n_samples": int(len(idx_arr)),
                    "n_sensitive": int((labels == 0).sum()),
                    "mean_resist_prob": float(probs.mean()),
                }
                if len(set(labels)) > 1:
                    entry["bacc"] = float(balanced_accuracy_score(labels, preds))
                if len(ic50_t) > 0:
                    entry["rmse"] = float(np.sqrt(((ic50_t - ic50_p) ** 2).mean()))
                glyco_state_stratified[f"{bin_name}__{mclass}"] = entry

        bins_present = {k.split("__")[0] for k in glyco_state_stratified.keys()}
        print(f"  Bins observed: {sorted(bins_present)} (≥3 required: {'✓' if len(bins_present) >= 3 else '✗'})")
        results["glyco_state_stratified"] = {
            "tertiles": {"q33": q1, "q66": q2},
            "entries": glyco_state_stratified,
            "n_bins_present": int(len(bins_present)),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3c: FORMAL LEAKAGE ANALYSIS (Reviewer Q2)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 3c: Formal Leakage Analysis (Reviewer Q2)")
    print(f"  {'=' * 55}")

    from src.ptm_bdl.evaluation.statistical import compute_leakage_analysis
    split_path_q2 = MODEL_DIR / "split_indices.json"
    if split_path_q2.exists():
        with open(split_path_q2) as f:
            split_q2 = json.load(f)
        leakage = compute_leakage_analysis(
            df,
            train_idx=np.array(split_q2["train_idx"]),
            val_idx=np.array(split_q2["val_idx"]),
            test_idx=np.array(split_q2["test_idx"]),
        )
        results["leakage_analysis"] = leakage

        print(f"    PTM baseline unique vectors: {leakage.get('ptm_baseline_unique', '?')} "
              f"/ {leakage['n_samples']} (diversity={leakage.get('ptm_baseline_diversity', '?')})")
        print(f"    PTM delta unique vectors:    {leakage.get('ptm_delta_unique', '?')}")
        if "cell_line_overlap" in leakage:
            cl = leakage["cell_line_overlap"]
            print(f"    Cell-line overlap (train↔test): {cl['train_test_overlap']} "
                  f"/ train={cl['n_train_cells']}, test={cl['n_test_cells']}")
        if "ptm_provenance" in leakage:
            prov = leakage["ptm_provenance"]
            print(f"    Measured (conf≥0.90): {prov['n_measured_high_conf']} samples "
                  f"({prov['frac_measured']:.1%})")
            print(f"    Test: {prov['test_measured']} measured, "
                  f"{prov['test_propagated']} propagated")
        if "constant_channels" in leakage:
            cc = leakage["constant_channels"]
            print(f"    Constant PTM channels: {cc['n_constant']}")
            if cc['n_constant'] > 0:
                print(f"      Examples: {', '.join(cc['columns'][:5])}")
        print(f"    Institutional separation: PTM=DrugPTM-Bench (Xie lab, MS) | "
              f"IC50=GDSC2 (Sanger, viability)")
    else:
        print(f"    ⚠ split_indices.json not found")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3d: PER-DRUG EXCLUSION ANALYSIS (Reviewer Q8)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 3d: Per-Drug Exclusion Analysis (Reviewer Q8)")
    print(f"  {'=' * 55}")

    exclusion_report = {}
    # For CS1: exclude HER2-only drugs to test if they dominate metrics
    exclusion_drugs = ["Lapatinib", "Sapitinib"]
    test_drug_names = test_df["drug_name"].values

    for excl_drug in exclusion_drugs:
        excl_mask = test_drug_names != excl_drug
        if excl_mask.sum() < 5:
            continue
        y_excl_ic50_t = y_true_ic50[excl_mask]
        y_excl_ic50_p = y_pred_ic50[excl_mask]
        y_excl_cls_t = y_true_cls[excl_mask]
        y_excl_prob = y_prob_cls[excl_mask]
        entry = {
            "n_samples": int(excl_mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_excl_ic50_t, y_excl_ic50_p))),
        }
        if len(set(y_excl_cls_t)) > 1:
            entry["auroc"] = float(roc_auc_score(y_excl_cls_t, y_excl_prob))
            entry["bacc"] = float(balanced_accuracy_score(
                y_excl_cls_t, (y_excl_prob > RESIST_THRESHOLD).astype(float)))
        if len(y_excl_ic50_t) > 2 and np.std(y_excl_ic50_p) > 1e-8:
            entry["pearson_r"] = float(np.corrcoef(y_excl_ic50_t, y_excl_ic50_p)[0, 1])
        exclusion_report[f"excluding_{excl_drug}"] = entry
        auroc_s = f"{entry.get('auroc', 'N/A'):.3f}" if 'auroc' in entry else "N/A"
        print(f"    Excluding {excl_drug}: n={entry['n_samples']}, "
              f"AUROC={auroc_s}, RMSE={entry['rmse']:.3f}")

    results["per_drug_exclusion"] = exclusion_report

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3e: ECE CALIBRATION ANALYSIS (Reviewer Q9)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 3e: ECE Calibration Analysis (Reviewer Q9)")
    print(f"  {'=' * 55}")

    from src.ptm_bdl.evaluation.statistical import compute_ece, compute_ece_per_drug

    ece_overall = compute_ece(y_true_cls, y_prob_cls, n_bins=10)
    print(f"    Overall ECE: {ece_overall['ece']:.4f}")
    print(f"    Overall MCE: {ece_overall['mce']:.4f}")
    print(f"    Mean predicted probability: {y_prob_cls.mean():.3f}")

    ece_per_drug = compute_ece_per_drug(
        y_true_cls, y_prob_cls, test_df["drug_name"].values, n_bins=10)
    for drug_name, ece_d in ece_per_drug.items():
        if drug_name == "overall" or ece_d.get("ece") is None:
            continue
        print(f"    {drug_name:20s}: ECE={ece_d['ece']:.4f}, n={ece_d.get('n_samples', '?')}")

    results["calibration"] = {
        "overall_ece": ece_overall,
        "per_drug_ece": ece_per_drug,
        "lambda_weights": {
            "lambda_reg": 1.0,
            "lambda_cls": 2.0,
            "justification": "Classification receives 2× weight because resistance "
                           "prediction is the primary clinical question. Huber loss for "
                           "regression provides robustness to IC50 outliers. Focal loss "
                           "(γ=2.0, α=0.25) addresses class imbalance. Ref: Lin et al., "
                           "Focal Loss for Dense Object Detection, ICCV 2017.",
        },
    }

    # Generate reliability diagram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        bin_mids = [(ece_overall["bin_edges"][i] + ece_overall["bin_edges"][i+1]) / 2
                    for i in range(len(ece_overall["bin_accs"]))]
        counts = ece_overall["bin_counts"]
        accs = ece_overall["bin_accs"]
        confs = ece_overall["bin_confs"]

        # Plot bars for accuracy
        width = 0.08
        bars = ax.bar(bin_mids, accs, width=width, alpha=0.7, color="#0072B2",
                      edgecolor="black", label="Accuracy")
        # Plot gap
        for i, (mid, acc, conf, cnt) in enumerate(zip(bin_mids, accs, confs, counts)):
            if cnt > 0:
                ax.plot([mid, mid], [acc, conf], color="red", linewidth=1.5, alpha=0.7)
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.set_title(f"Reliability Diagram (ECE={ece_overall['ece']:.3f})", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.set_aspect("equal")
        plt.tight_layout()
        calib_path = FIGURES_DIR / "reliability_diagram.png"
        plt.savefig(calib_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Reliability diagram saved: {calib_path}")
    except Exception as e:
        print(f"  ⚠ Could not generate reliability diagram: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3f: IG SCALE AUDIT (Reviewer Q10)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 3f: IG Scale Audit (Reviewer Q10)")
    print(f"  {'=' * 55}")

    xai_path = RESULTS_DIR / "xai_report.json"
    ig_audit = {"ig_value_ranges": {}, "issues_found": []}
    if xai_path.exists():
        with open(xai_path) as f:
            xai_data = json.load(f)
        for key in xai_data:
            if not key.startswith("integrated_gradients_"):
                continue
            mod_type = key.replace("integrated_gradients_", "")
            for protein, prot_data in xai_data[key].items():
                if not isinstance(prot_data, dict) or "resist_site_ranking" not in prot_data:
                    continue
                rankings = prot_data["resist_site_ranking"]
                if not rankings:
                    continue
                values = [r["mean_abs_attribution"] for r in rankings]
                min_v, max_v = min(values), max(values)
                ig_audit["ig_value_ranges"][f"{protein}_{mod_type}"] = {
                    "min": float(min_v), "max": float(max_v),
                    "range": float(max_v - min_v), "n_sites": len(values),
                }
                # Check for near-zero (constant input → no signal)
                if max_v < 1e-6:
                    ig_audit["issues_found"].append(
                        f"{protein} {mod_type}: all IG near zero (max={max_v:.2e}) "
                        f"— likely constant input data")
                # Check for near-uniform (no discrimination)
                if max_v > 0 and min_v / max_v > 0.9:
                    ig_audit["issues_found"].append(
                        f"{protein} {mod_type}: near-uniform IG (min/max={min_v/max_v:.3f})")
                # Check for near-binary (Q10: bimodal distribution)
                if len(values) > 2:
                    mid_range = (max_v + min_v) / 2
                    n_near_min = sum(1 for v in values if v < min_v + 0.1 * (max_v - min_v))
                    n_near_max = sum(1 for v in values if v > max_v - 0.1 * (max_v - min_v))
                    if n_near_min + n_near_max > 0.8 * len(values) and max_v - min_v > 1e-4:
                        ig_audit["issues_found"].append(
                            f"{protein} {mod_type}: near-binary IG magnitudes "
                            f"({n_near_min} near-zero + {n_near_max} near-max)")
                print(f"    {protein:8s} {mod_type:12s}: "
                      f"min={min_v:.2e}, max={max_v:.2e}, range={max_v-min_v:.2e}")
        if ig_audit["issues_found"]:
            print(f"    Issues found:")
            for issue in ig_audit["issues_found"]:
                print(f"      ⚠ {issue}")
        else:
            print(f"    ✓ No obvious scale issues detected")
    else:
        print(f"    ⚠ xai_report.json not found — run explain.py first")

    results["ig_scale_audit"] = ig_audit

    # -- Reviewer Q6: Frozen Encoder Documentation --
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["frozen_encoders"] = {
        "trainable_parameters": int(train_p), "total_model_parameters": int(total_p),
        "pretrained_frozen": {"ESM-2": "650M FROZEN", "ChemBERTa": "77M FROZEN", "GearNet": "FROZEN"},
        "note": "All pretrained encoders frozen (pre-extracted). Fine-tuning not tested.",
    }
    print(f"\n  Q6: Trainable={train_p:,} / Total={total_p:,} | Frozen: ESM-2+ChemBERTa+GearNet")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4: CONFIDENCE-AWARE ANALYSIS (Propagation Validation)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  PART 4: Confidence-Aware Analysis")
    print(f"  {'=' * 55}")

    prop_conf = df["propagation_confidence"].values
    confidence_results = {}
    for label, lo, hi in [("high_confidence (≥0.80)", 0.80, 2.0),
                          ("medium_confidence (0.40-0.80)", 0.40, 0.80),
                          ("low_confidence (<0.40)", -1.0, 0.40)]:
        idx_group = np.where((prop_conf >= lo) & (prop_conf < hi))[0]
        if len(idx_group) == 0:
            continue
        probs = all_cls_prob[idx_group]
        labels_g = all_cls_true[idx_group]
        preds_g = (probs > RESIST_THRESHOLD).astype(float)
        n_correct = int((preds_g == labels_g).sum())
        conf_metrics = {
            "n_samples": len(idx_group),
            "n_sensitive": int((labels_g == 0).sum()),
            "n_resistant": int((labels_g == 1).sum()),
            "accuracy": n_correct / len(idx_group),
            "mean_prob": float(probs.mean()),
        }
        if len(set(labels_g)) > 1:
            conf_metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels_g, preds_g))
        confidence_results[label] = conf_metrics
        print(f"\n  {label}: {conf_metrics['n_samples']} samples, "
              f"acc={conf_metrics['accuracy']:.3f}")

    results["confidence_analysis"] = confidence_results

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'=' * 55}")
    print(f"  FINAL SUMMARY")
    print(f"  {'=' * 55}")
    print(f"    Test BAcc:     {model_bacc:.4f}")
    print(f"    Test AUROC:    {classification.get('auroc', 0):.4f}")
    print(f"    Test RMSE:     {model_rmse:.4f}")
    print(f"    Test R:        {regression.get('pearson_r', 0):.4f}")
    print(f"    Drugs analyzed: {len(drug_results)}")
    print(f"    Mutation groups: {len(mutation_results)}")

    # ── Randomized PTM control reference (from ablation study) ──────────────
    rand_path = RESULTS_DIR / "randomized_ptm_control.json"
    if rand_path.exists():
        with open(rand_path) as f:
            rand_ctrl = json.load(f)
        primary_pass = rand_ctrl.get("primary_pass", False)
        drops = rand_ctrl.get("arms", {}).get(
            rand_ctrl.get("primary_arm", "both_shuffled"), {}).get("drops", {})
        results["randomized_ptm_control"] = {
            "primary_pass": primary_pass,
            "drop_auroc": drops.get("drop_auroc", 0),
            "drop_bacc": drops.get("drop_bacc", 0),
        }
        print(f"\n  RANDOMIZED PTM CONTROL:")
        print(f"    ΔAUROC: {drops.get('drop_auroc', 0):+.4f} (positive = PTM helps)")
        print(f"    Pass: {'✓ PASS' if primary_pass else '✗ FAIL'}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✓ Report saved: {report_path}")

    # ── Generate evaluation figures ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: IC50 prediction scatter
        ax = axes[0, 0]
        colors = ["green" if c == 0 else "red" for c in y_true_cls]
        ax.scatter(y_true_ic50, y_pred_ic50, c=colors, alpha=0.6, s=30)
        lims = [min(y_true_ic50.min(), y_pred_ic50.min()) - 0.5,
                max(y_true_ic50.max(), y_pred_ic50.max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("True ln(IC50)")
        ax.set_ylabel("Predicted ln(IC50)")
        ax.set_title(f"IC50 Prediction (R={regression.get('pearson_r', 0):.3f}, RMSE={model_rmse:.3f})")

        # Plot 2: Resistance probability histogram
        ax = axes[0, 1]
        sens_probs = y_prob_cls[y_true_cls == 0]
        res_probs = y_prob_cls[y_true_cls == 1]
        if len(sens_probs) > 0:
            ax.hist(sens_probs, bins=15, alpha=0.7, label=f"Sensitive (n={len(sens_probs)})",
                    color="green", density=True)
        if len(res_probs) > 0:
            ax.hist(res_probs, bins=15, alpha=0.7, label=f"Resistant (n={len(res_probs)})",
                    color="red", density=True)
        ax.axvline(x=RESIST_THRESHOLD, color="black", linestyle="--", label="Threshold")
        ax.set_xlabel("Predicted Resistance Probability")
        ax.set_title(f"Classification (BAcc={model_bacc:.3f})")
        ax.legend()

        # Plot 3: Drug-specific BAcc
        ax = axes[1, 0]
        drug_names = sorted(drug_results.keys())
        drug_baccs = [drug_results[d].get("balanced_accuracy", 0) for d in drug_names]
        ax.bar(drug_names, drug_baccs, alpha=0.85, edgecolor="black")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title("Drug-Specific Performance (Test Set)")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=25)

        # Plot 4: Mutation-group mean predictions
        ax = axes[1, 1]
        if mutation_results:
            mut_names = sorted(mutation_results.keys(),
                               key=lambda x: mutation_results[x]["mean_resist_prob"])
            mut_probs = [mutation_results[m]["mean_resist_prob"] for m in mut_names]
            short_names = [m[:20] + "..." if len(m) > 20 else m for m in mut_names]
            ax.barh(range(len(mut_names)), mut_probs, color="steelblue", alpha=0.7)
            ax.set_yticks(range(len(mut_names)))
            ax.set_yticklabels(short_names, fontsize=8)
            ax.set_xlabel("Mean Resistance Probability")
            ax.set_title("EGFR Mutation Groups (All Samples)")
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)

        plt.suptitle("Model Evaluation — Comprehensive Analysis",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig_path = FIGURES_DIR / "evaluation_plots.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure saved: {fig_path}")
    except Exception as e:
        print(f"  ⚠ Could not generate figure: {e}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    evaluate()
