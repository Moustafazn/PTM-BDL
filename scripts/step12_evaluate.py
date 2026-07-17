#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 12 — Evaluation & Benchmarking (Enhanced 2026-06-20)                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Evaluate the trained model on the HELD-OUT test set using the SAME        ║
║    stratified split indices saved by Step 11 (split_indices.json).           ║
║    This prevents data leakage — the test set is never seen during training.  ║
║                                                                              ║
║  METRICS (comprehensive, both regression and classification):                ║
║    Regression:      MSE, RMSE, R², Pearson R, Spearman ρ                    ║
║    Classification:  Accuracy, Balanced Accuracy, Sensitivity, Specificity,  ║
║                     F1, AUROC, AUPRC                                        ║
║    Per-class:       Confusion matrix, classification report                  ║
║                                                                              ║
║  NOVEL ANALYSES (added 2026-06-20):                                          ║
║    1. Drug-Specific Evaluation — per-drug BAcc, RMSE, R                     ║
║       • Compare Afatinib vs Osimertinib (same binding site, diff selectivity)║
║       • Compare 3rd-gen vs 1st-gen TKIs                                     ║
║       • Ref: Zhao et al., Nat Rev Clin Oncol 2026 (PMID 41219394) —        ║
║         catalogues generation-specific resistance mechanisms.               ║
║    2. Mutation-Stratified Analysis — per-mutation-group predictions          ║
║       • Run on ALL 36 EGFR-mutant samples (qualitative, not test-only)      ║
║       • Report attention patterns + predictions per mutation class          ║
║       • Ref: Zhao 2026 reviews T790M, C797S, exon19del/L858R resistance    ║
║         — our mutation-stratified analysis quantifies model per class.      ║
║    3. Confidence-Aware Analysis — measured vs propagated phospho            ║
║       • Compare 32 high-confidence vs 614 medium-confidence samples         ║
║       • Validates the propagation framework                                 ║
║       • Ref: Zhao 2026 highlights shift to molecular monitoring —          ║
║         PTM confidence analysis demonstrates biomarker feasibility.         ║
║    4. Baseline Comparisons — majority-class, mean-prediction               ║
║                                                                              ║
║  INPUT:                                                                      ║
║    data/models/best_model.pt          — trained model weights                ║
║    data/models/split_indices.json     — train/val/test indices from step11   ║
║    data/processed/multimodal_dataset.csv + data/features/*                   ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    results/evaluation_report.json     — all metrics                          ║
║    results/figures/evaluation_plots.png — evaluation visualizations          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import yaml, json, torch, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
    balanced_accuracy_score
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step11_train import ResistanceDataset, collate_fn, build_model_from_cfg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / cfg["paths"]["figures"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"]

# ── Load optimal classification threshold (Youden's J, from step11) ──────
# Falls back to 0.5 if optimal_threshold.json doesn't exist (backward compat).
# Ref: Youden WJ (1950) Cancer 3:32-35.
_threshold_path = MODEL_DIR / "optimal_threshold.json"
if _threshold_path.exists():
    with open(_threshold_path) as _f:
        _thr_info = json.load(_f)
    RESIST_THRESHOLD = float(_thr_info.get("optimal_threshold", 0.5))
    print(f"  ✓ Loaded optimal threshold: {RESIST_THRESHOLD:.4f} "
          f"(Youden's J, from {_threshold_path.name})")
else:
    RESIST_THRESHOLD = 0.5
    print(f"  ⚠ No optimal_threshold.json — using default threshold 0.5")


def load_model(device):
    """Load PTM-BDL multimodal model + weights (if any)."""
    model = build_model_from_cfg(cfg).to(device)
    model_path = MODEL_DIR / "best_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
        print(f"  ✓ Loaded model: {model_path.name}")
    else:
        print(f"  ⚠ No trained model found! Using random weights for demo.")
    model.eval()
    return model


def collect_predictions(model, loader):
    """Run the PTM-BDL model on a dataloader and collect all predictions."""
    all_ic50_pred, all_ic50_true = [], []
    all_resist_prob, all_resist_true = [], []

    with torch.no_grad():
        for batch in loader:
            ic50_pred, resist_pred = model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch.get("drug_emb"),
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                glyco_vector=batch["glyco_vector"],
                delta_glyco_vector=batch["delta_glyco_vector"],
                target_protein=batch["target_protein"],
            )
            all_ic50_pred.extend(ic50_pred.squeeze(-1).cpu().numpy().tolist())
            all_ic50_true.extend(batch["ln_ic50"].squeeze(-1).cpu().numpy().tolist())
            all_resist_prob.extend(
                torch.sigmoid(resist_pred).squeeze(-1).cpu().numpy().tolist()
            )
            all_resist_true.extend(
                batch["resistance_label"].squeeze(-1).cpu().numpy().tolist()
            )

    return (np.array(all_ic50_true), np.array(all_ic50_pred),
            np.array(all_resist_true), np.array(all_resist_prob))



def compute_full_metrics(y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls):
    """Compute comprehensive metrics for both regression and classification."""
    y_pred_cls = (y_prob_cls > RESIST_THRESHOLD).astype(float)
    has_both = len(set(y_true_cls)) > 1

    # Regression
    regression = {
        "mse": float(mean_squared_error(y_true_ic50, y_pred_ic50)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))),
        "r2": float(r2_score(y_true_ic50, y_pred_ic50))
              if len(set(y_true_ic50)) > 1 else 0.0,
    }
    if len(y_true_ic50) > 2:
        pr = stats.pearsonr(y_true_ic50, y_pred_ic50)
        sr = stats.spearmanr(y_true_ic50, y_pred_ic50)
        regression.update({
            "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
        })

    # Classification
    cm = confusion_matrix(y_true_cls, y_pred_cls).tolist() if has_both else [[0]]
    classification = {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_cls, y_pred_cls))
                             if has_both else 0.0,
        "f1_score": float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
        "auroc": float(roc_auc_score(y_true_cls, y_prob_cls)) if has_both else 0.0,
        "auprc": float(average_precision_score(y_true_cls, y_prob_cls)) if has_both else 0.0,
        "confusion_matrix": cm,
        "mean_predicted_probability": float(y_prob_cls.mean()),
    }

    if has_both:
        report = classification_report(
            y_true_cls, y_pred_cls,
            target_names=["sensitive", "resistant"],
            output_dict=True, zero_division=0
        )
        classification["per_class"] = {
            "sensitive": report.get("sensitive", {}),
            "resistant": report.get("resistant", {}),
        }

    return regression, classification


def evaluate():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 12: Evaluation & Benchmarking (Enhanced)             ║")
    print("║  + Drug-specific + Mutation-stratified + Confidence-aware  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    device = torch.device("cpu")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "multimodal_dataset.csv"
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    dataset = ResistanceDataset(dataset_path, features_dir)
    df = dataset.df
    print(f"\n  Dataset: {len(dataset)} samples")

    # ── Load SAME split indices from step11 ───────────────────────────────────
    split_path = MODEL_DIR / "split_indices.json"
    if split_path.exists():
        with open(split_path) as f:
            split_info = json.load(f)
        test_idx = split_info["test_idx"]
        print(f"  ✓ Loaded split indices from step11 "
              f"(stratified by {split_info['stratification']})")
    else:
        print(f"  ⚠ split_indices.json not found — recreating with seed")
        from step11_train import create_stratified_splits
        _, _, test_idx = create_stratified_splits(
            dataset, cfg["training"]["train_ratio"],
            cfg["training"]["val_ratio"], cfg["training"]["seed"]
        )
        test_idx = test_idx.tolist()

    test_set = torch.utils.data.Subset(dataset, test_idx)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg["model"]["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )

    # Report test set composition
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
    print(f"\n  {'='*55}")
    print(f"  PART 1: Standard Test Set Evaluation")
    print(f"  {'='*55}")

    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
        model, test_loader
    )
    y_pred_cls_binary = (y_prob_cls > RESIST_THRESHOLD).astype(float)

    # ── Cache test predictions for step14c (bootstrap CIs, DeLong) ──────────
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
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls
    )

    results = {
        "test_samples": len(y_true_ic50),
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
    print(f"\n  {'='*55}")
    print(f"  PART 1b: Per-Protein Evaluation (EGFR vs HER2/ERBB2)")
    print(f"  {'='*55}")
    print(f"  NOTE: 'target_protein' column maps to target RECEPTOR PROTEIN.")
    print(f"  EGFR = EGFR protein (NSCLC), ERBB2 = HER2 protein (breast).")

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
                gene_met["balanced_accuracy"] = float(
                    balanced_accuracy_score(g_cls_t, g_cls_bin))
                gene_met["auroc"] = float(roc_auc_score(g_cls_t, g_cls_p))
            else:
                gene_met["balanced_accuracy"] = float(
                    accuracy_score(g_cls_t, g_cls_bin))
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
    print(f"\n  {'='*55}")
    print(f"  PART 2: Drug-Specific Evaluation")
    print(f"  {'='*55}")

    # Collect per-sample predictions with metadata
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_df["ic50_pred"] = y_pred_ic50
    test_df["resist_prob"] = y_prob_cls
    test_df["resist_pred"] = y_pred_cls_binary

    drug_results = {}
    print(f"\n  {'Drug':<15s} | {'N':>4s} | {'Sens':>4s} | {'BAcc':>6s} | "
          f"{'RMSE':>6s} | {'R':>6s} | {'AUROC':>6s} | {'mProb':>5s}")
    print(f"  {'-'*65}")

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

        # Classification (if both classes present)
        if len(set(y_t_cls)) > 1:
            drug_metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(y_t_cls, y_p_cls))
            drug_metrics["auroc"] = float(roc_auc_score(y_t_cls, y_p_prob))
        else:
            drug_metrics["balanced_accuracy"] = float(
                accuracy_score(y_t_cls, y_p_cls))
            drug_metrics["auroc"] = 0.0

        # Regression
        drug_metrics["rmse"] = float(np.sqrt(mean_squared_error(y_t_ic50, y_p_ic50)))
        if len(y_t_ic50) > 2 and np.std(y_p_ic50) > 1e-8:
            drug_metrics["pearson_r"] = float(
                np.corrcoef(y_t_ic50, y_p_ic50)[0, 1])
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

    # Drug comparison insight
    if "Afatinib" in drug_results and "Osimertinib" in drug_results:
        afa = drug_results["Afatinib"]
        osi = drug_results["Osimertinib"]
        print(f"\n  Drug Comparison (same C797 binding, different selectivity):")
        print(f"    Afatinib (2nd-gen, pan-ERBB):  BAcc={afa['balanced_accuracy']:.3f}, "
              f"RMSE={afa['rmse']:.3f}")
        print(f"    Osimertinib (3rd-gen, T790M):  BAcc={osi['balanced_accuracy']:.3f}, "
              f"RMSE={osi['rmse']:.3f}")

    # ── Cross-protein drug insights ──────────────────────────────────────────
    # Drugs like Afatinib target BOTH EGFR and HER2 — compare their performance
    # across target genes to test if the model learns cross-receptor patterns
    if "target_protein" in test_df.columns:
        # All 4 EGFR drugs are tested on BOTH NSCLC (EGFR) and breast (ERBB2)
        # Lapatinib and Sapitinib are ERBB2-only
        cross_drugs = ["Osimertinib", "Gefitinib", "Afatinib", "Erlotinib",
                        "Lapatinib", "Sapitinib"]
        print(f"\n  Cross-Protein Drug Analysis:")
        print(f"  (4 EGFR drugs tested on BOTH EGFR + ERBB2; 2 HER2-only drugs)")
        print(f"  {'Drug':<15s} | {'Gene':>5s} | {'N':>4s} | {'BAcc':>6s} | {'mProb':>5s} | {'RMSE':>6s}")
        print(f"  {'-'*55}")
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
                bacc = float(balanced_accuracy_score(g_cls_t, g_cls_p)) if len(set(g_cls_t)) > 1 else float(accuracy_score(g_cls_t, g_cls_p))
                rmse = float(np.sqrt(mean_squared_error(g_ic50_t, g_ic50_p)))
                mprob = float(sub["resist_prob"].values.mean())
                print(f"  {drug_name:<15s} | {gene:>5s} | {len(sub):4d} | {bacc:6.3f} | {mprob:5.3f} | {rmse:6.3f}")

        results["cross_protein_drugs"] = {
            "note": "Drugs tested on both EGFR (NSCLC) and ERBB2 (breast) contexts. "
                    "Afatinib is a pan-ERBB inhibitor targeting both receptors. "
                    "Cross-receptor consistency validates the model's biological learning."
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: MUTATION-STRATIFIED BIOLOGICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    # Two parallel stratifications (gene-aware, added 2026-06-28 to address
    # Failure-3 / ERBB2 underrepresentation in eval reporting):
    #   3a) EGFR mutation groups — driven by activating point/del mutations
    #       (L858R, exon19del, T790M, C797S, etc.).  Existing behaviour.
    #   3b) ERBB2 HER2-amplification tier groups — HER2 oncogenicity in
    #       breast cancer is amplification-driven, not point-mutation-driven
    #       (Hudis 2007 PMID 17626692; Citri & Yarden 2006 PMID 16829981).
    #       We stratify ERBB2 samples by mutation_classes
    #       ('HER2_amplified', 'ERBB2_wild_type', etc.) so the model's
    #       behaviour on HER2-amp cell lines is reported symmetrically.
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*55}")
    print(f"  PART 3: Mutation-Stratified Analysis (ALL samples)")
    print(f"  {'='*55}")
    print(f"  NOTE: This runs on ALL EGFR-mutant and ERBB2-amplified samples,")
    print(f"  not just the held-out test set — biological analysis.")

    # Run model on ALL samples for biological analysis
    all_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["model"]["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )
    all_ic50_true, all_ic50_pred, all_cls_true, all_cls_prob = collect_predictions(
        model, all_loader
    )

    # Identify EGFR-mutant samples (existing logic, unchanged)
    mc = df["mutation_classes"].fillna("wild_type").str.lower()
    is_mutant = mc.str.contains("pathogenic|cmp_driver", regex=True)
    # And gene-aware masks
    has_target = "target_protein" in df.columns
    is_egfr = (df["target_protein"] == "EGFR") if has_target else pd.Series([True]*len(df))
    is_erbb2 = (df["target_protein"] == "ERBB2") if has_target else pd.Series([False]*len(df))

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
    print(f"  {'-'*80}")

    for mut_name, indices in sorted(mutation_groups.items(),
                                     key=lambda x: -len(x[1])):
        idx_arr = np.array(indices)
        probs = all_cls_prob[idx_arr]
        labels = all_cls_true[idx_arr]
        ic50_t = all_ic50_true[idx_arr]
        ic50_p = all_ic50_pred[idx_arr]

        n_sens_m = int((labels == 0).sum())
        n_res_m = int((labels == 1).sum())
        mean_prob = float(probs.mean())
        mean_ic50_t = float(ic50_t.mean())
        mean_ic50_p = float(ic50_p.mean())

        # Biological interpretation
        if n_sens_m > n_res_m:
            note = "mostly sensitive"
        elif n_res_m > 3 * n_sens_m:
            note = "mostly resistant"
        else:
            note = "mixed"

        mutation_results[mut_name] = {
            "n_samples": len(indices),
            "n_sensitive": n_sens_m,
            "n_resistant": n_res_m,
            "mean_resist_prob": mean_prob,
            "mean_ic50_true": mean_ic50_t,
            "mean_ic50_pred": mean_ic50_p,
            "sample_indices": indices,
        }

        print(f"  {mut_name:<25s} | {len(indices):3d} | {n_sens_m:4d} | {n_res_m:4d} | "
              f"{mean_prob:5.3f} | {mean_ic50_t:7.2f} | {mean_ic50_p:7.2f} | {note}")

    results["mutation_stratified"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "sample_indices"}
        for k, v in mutation_results.items()
    }

    # ── 3b) ERBB2 amplification-tier groups (added 2026-06-28) ────────────
    # HER2 oncogenicity in breast cancer is amplification-driven.  Group
    # by mutation_classes (e.g. 'HER2_amplified', 'ERBB2_wild_type') —
    # the same column step06 populates for ERBB2 rows.
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
              f"{'mProb':>5s} | {'mIC50_T':>7s} | {'mIC50_P':>7s} | {'Note'}")
        print(f"  {'-'*80}")
        for tier_name, indices in sorted(erbb2_groups.items(),
                                          key=lambda x: -len(x[1])):
            idx_arr = np.array(indices)
            probs = all_cls_prob[idx_arr]
            labels = all_cls_true[idx_arr]
            ic50_t = all_ic50_true[idx_arr]
            ic50_p = all_ic50_pred[idx_arr]

            n_sens_m = int((labels == 0).sum())
            n_res_m = int((labels == 1).sum())
            mean_prob = float(probs.mean())
            mean_ic50_t = float(ic50_t.mean())
            mean_ic50_p = float(ic50_p.mean())

            if n_sens_m > n_res_m:
                note = "mostly sensitive"
            elif n_res_m > 3 * n_sens_m:
                note = "mostly resistant"
            else:
                note = "mixed"

            erbb2_results[tier_name] = {
                "n_samples": len(indices),
                "n_sensitive": n_sens_m,
                "n_resistant": n_res_m,
                "mean_resist_prob": mean_prob,
                "mean_ic50_true": mean_ic50_t,
                "mean_ic50_pred": mean_ic50_p,
                "sample_indices": indices,
            }
            print(f"  {tier_name:<25s} | {len(indices):3d} | {n_sens_m:4d} | "
                  f"{n_res_m:4d} | {mean_prob:5.3f} | {mean_ic50_t:7.2f} | "
                  f"{mean_ic50_p:7.2f} | {note}")

        results["erbb2_amp_stratified"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "sample_indices"}
            for k, v in erbb2_results.items()
        }

    # Key biological question (EGFR): Do activating mutations show LOWER resist?
    if mutation_results:
        activating_probs = []
        for mut_name, data in mutation_results.items():
            if data["n_sensitive"] > 0:
                activating_probs.extend(
                    all_cls_prob[np.array(data["sample_indices"])].tolist()
                )

        wt_indices = [i for i in range(len(df))
                      if is_egfr.iloc[i] and not is_mutant.iloc[i]]
        if wt_indices and activating_probs:
            wt_mean_prob = float(all_cls_prob[np.array(wt_indices)].mean())
            mut_mean_prob = float(np.mean(activating_probs))
            print(f"\n  Biological Insight (EGFR):")
            print(f"    EGFR WT/VUS mean resist prob:   {wt_mean_prob:.3f}")
            print(f"    EGFR-mutant mean prob:          {mut_mean_prob:.3f}")
            diff = wt_mean_prob - mut_mean_prob
            print(f"    Difference: {diff:+.3f} "
                  f"({'✓ mutants more sensitive' if diff > 0 else '⚠ unexpected'})")

    # Key biological question (ERBB2): HER2-amplified more sensitive than WT?
    if erbb2_results:
        amp_indices = []
        wt_indices_erbb2 = []
        for tier_name, data in erbb2_results.items():
            tnorm = tier_name.lower()
            if "amplif" in tnorm:
                amp_indices.extend(data["sample_indices"])
            elif "wild" in tnorm or "wt" in tnorm:
                wt_indices_erbb2.extend(data["sample_indices"])

        if amp_indices and wt_indices_erbb2:
            amp_mean = float(all_cls_prob[np.array(amp_indices)].mean())
            wt_mean_erbb2 = float(all_cls_prob[np.array(wt_indices_erbb2)].mean())
            print(f"\n  Biological Insight (HER2):")
            print(f"    ERBB2 wild_type mean resist prob: {wt_mean_erbb2:.3f}")
            print(f"    HER2-amplified mean prob:         {amp_mean:.3f}")
            diff_h = wt_mean_erbb2 - amp_mean
            print(f"    Difference: {diff_h:+.3f} "
                  f"({'✓ HER2-amp more sensitive' if diff_h > 0 else '⚠ unexpected'})")


    # ══════════════════════════════════════════════════════════════════════════
    # PART 3b: GLYCO-STATE STRATIFIED EVALUATION (PTM-BDL §2f, added 2026-06-28)
    # ══════════════════════════════════════════════════════════════════════════
    # Stratify samples by their mean glyco-slot occupancy (low / mid / high)
    # crossed with EGFR-mutation class.  The proposal §2f pass-criterion
    # requires this block to exist with 3 bins and ≥ 1 sample per bin.
    print(f"\n  {'='*55}")
    print(f"  PART 3b: Glyco-State Stratified Evaluation (PTM-BDL §2f)")
    print(f"  {'='*55}")

    glyco_cols = [f"glyco_slot{i:02d}" for i in range(12)]
    have_glyco = all(c in df.columns for c in glyco_cols)
    glyco_state_stratified = {}
    if have_glyco:
        glyco_mean = df[glyco_cols].mean(axis=1).values  # per-sample mean occupancy
        # Bin by per-sample tertiles
        q1, q2 = float(np.nanpercentile(glyco_mean, 33)), \
                  float(np.nanpercentile(glyco_mean, 66))
        bins = ["low", "mid", "high"]
        def _bin_of(v):
            if not np.isfinite(v):
                return "mid"
            if v <= q1:
                return "low"
            if v >= q2:
                return "high"
            return "mid"
        glyco_bins = np.array([_bin_of(v) for v in glyco_mean])

        # Mutation class buckets (re-using EGFR/ERBB2 strata from earlier).
        # We aggregate (mutation_class × glyco_bin) → samples.
        mc_str = df["mutation_classes"].fillna("wild_type").astype(str).values
        print(f"\n  Glyco-mean tertile cutoffs: q33={q1:.3f}, q66={q2:.3f}")
        print(f"  {'Glyco bin':<10s} | {'Mutation class':<25s} | "
              f"{'N':>4s} | {'BAcc':>6s} | {'AUROC':>6s} | {'RMSE':>6s} | {'mProb':>5s}")
        print(f"  {'-'*78}")
        for bin_name in bins:
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
                    "mean_glyco_mean": float(glyco_mean[idx_arr].mean()),
                }
                if len(set(labels)) > 1:
                    entry["bacc"] = float(balanced_accuracy_score(labels, preds))
                    try:
                        entry["auroc"] = float(roc_auc_score(labels, probs))
                    except Exception:
                        entry["auroc"] = 0.0
                if len(ic50_t) > 0:
                    entry["rmse"] = float(np.sqrt(((ic50_t - ic50_p) ** 2).mean()))
                glyco_state_stratified[f"{bin_name}__{mclass}"] = entry
                print(f"  {bin_name:<10s} | {mclass:<25s} | "
                      f"{entry['n_samples']:4d} | "
                      f"{entry.get('bacc',0):6.3f} | "
                      f"{entry.get('auroc',0):6.3f} | "
                      f"{entry.get('rmse',0):6.3f} | "
                      f"{entry['mean_resist_prob']:5.3f}")

        # Marginal pass criterion: ≥ 3 distinct bins with ≥ 1 sample each
        bins_present = {k.split("__")[0] for k in glyco_state_stratified.keys()}
        n_bins_present = len(bins_present)
        print(f"\n  Bins observed: {sorted(bins_present)}  "
              f"(≥3 required: {'✓' if n_bins_present >= 3 else '✗'})")
        results["glyco_state_stratified"] = {
            "tertiles": {"q33": q1, "q66": q2},
            "entries": glyco_state_stratified,
            "n_bins_present": int(n_bins_present),
            "pass": bool(n_bins_present >= 3 and len(glyco_state_stratified) >= 3),
        }
    else:
        print(f"  ⚠ Glyco columns not found in dataset; skipping block.")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4: CONFIDENCE-AWARE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*55}")
    print(f"  PART 4: Confidence-Aware Analysis (Propagation Validation)")
    print(f"  {'='*55}")


    prop_conf = df["propagation_confidence"].values
    high_conf_idx = np.where(prop_conf >= 0.80)[0]
    med_conf_idx = np.where((prop_conf >= 0.40) & (prop_conf < 0.80))[0]
    low_conf_idx = np.where(prop_conf < 0.40)[0]

    confidence_results = {}
    for label, idx_group in [("high_confidence (≥0.80)", high_conf_idx),
                              ("medium_confidence (0.40-0.80)", med_conf_idx),
                              ("low_confidence (<0.40)", low_conf_idx)]:
        if len(idx_group) == 0:
            continue

        probs = all_cls_prob[idx_group]
        labels_g = all_cls_true[idx_group]
        ic50_t_g = all_ic50_true[idx_group]
        ic50_p_g = all_ic50_pred[idx_group]

        preds_g = (probs > RESIST_THRESHOLD).astype(float)
        n_correct = int((preds_g == labels_g).sum())

        conf_metrics = {
            "n_samples": len(idx_group),
            "n_sensitive": int((labels_g == 0).sum()),
            "n_resistant": int((labels_g == 1).sum()),
            "accuracy": n_correct / len(idx_group),
            "mean_prob": float(probs.mean()),
            "mean_ic50_pred": float(ic50_p_g.mean()),
            "mean_ic50_true": float(ic50_t_g.mean()),
        }

        if len(set(labels_g)) > 1:
            conf_metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(labels_g, preds_g))
        if len(ic50_t_g) > 2 and np.std(ic50_p_g) > 1e-8:
            conf_metrics["pearson_r"] = float(
                np.corrcoef(ic50_t_g, ic50_p_g)[0, 1])
            conf_metrics["rmse"] = float(
                np.sqrt(mean_squared_error(ic50_t_g, ic50_p_g)))

        confidence_results[label] = conf_metrics

        print(f"\n  {label}:")
        print(f"    Samples: {conf_metrics['n_samples']} "
              f"({conf_metrics['n_sensitive']} sensitive, "
              f"{conf_metrics['n_resistant']} resistant)")
        print(f"    Accuracy: {conf_metrics['accuracy']:.3f}")
        if "balanced_accuracy" in conf_metrics:
            print(f"    BAcc: {conf_metrics['balanced_accuracy']:.3f}")
        print(f"    Mean resist prob: {conf_metrics['mean_prob']:.3f}")
        if "rmse" in conf_metrics:
            print(f"    RMSE: {conf_metrics['rmse']:.3f}, "
                  f"R: {conf_metrics.get('pearson_r', 0):.3f}")

    results["confidence_analysis"] = confidence_results

    # Propagation validation insight
    if "high_confidence (≥0.80)" in confidence_results and \
       "medium_confidence (0.40-0.80)" in confidence_results:
        hc = confidence_results["high_confidence (≥0.80)"]
        mc_res = confidence_results["medium_confidence (0.40-0.80)"]
        print(f"\n  Propagation Validation:")
        print(f"    Measured samples (n={hc['n_samples']}):   "
              f"accuracy={hc['accuracy']:.3f}")
        print(f"    Propagated samples (n={mc_res['n_samples']}): "
              f"accuracy={mc_res['accuracy']:.3f}")
        if hc['accuracy'] >= mc_res['accuracy']:
            print(f"    ✓ Model performs as well or better on measured samples")
        else:
            print(f"    ⚠ Model performs better on propagated — possible artifact")

    # ══════════════════════════════════════════════════════════════════════════
    # PRINT FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  {'='*55}")
    print(f"  FINAL SUMMARY")
    print(f"  {'='*55}")
    print(f"    Test BAcc:     {model_bacc:.4f}")
    print(f"    Test AUROC:    {classification.get('auroc', 0):.4f}")
    print(f"    Test RMSE:     {model_rmse:.4f}")
    print(f"    Test R:        {regression.get('pearson_r', 0):.4f}")
    print(f"    Drugs analyzed: {len(drug_results)}")
    print(f"    Mutation groups: {len(mutation_results)}")

    # ── Randomized PTM control reference (from step11b) ───────────────────
    # The randomized PTM control (step11b Part 3) is the PRIMARY falsification
    # test: if shuffling PTM features across samples doesn't hurt the model,
    # then PTM is not contributing biological signal.  We load the result
    # here and include it in the evaluation report for completeness.
    rand_path = RESULTS_DIR / "randomized_ptm_control.json"
    if rand_path.exists():
        with open(rand_path) as f:
            rand_ctrl = json.load(f)
        primary_arm = rand_ctrl.get("primary_arm", "both_shuffled")
        primary_pass = rand_ctrl.get("primary_pass", False)
        drops = rand_ctrl.get("arms", {}).get(primary_arm, {}).get("drops", {})
        results["randomized_ptm_control"] = {
            "primary_arm": primary_arm,
            "primary_pass": primary_pass,
            "drop_auroc": drops.get("drop_auroc", 0),
            "drop_bacc": drops.get("drop_bacc", 0),
            "drop_auprc_sensitive": drops.get("drop_auprc_sensitive", 0),
            "note": "Positive drop = real PTM beats shuffled (model uses PTM biology)"
        }
        print(f"\n  RANDOMIZED PTM CONTROL (from step11b):")
        print(f"    Primary arm: {primary_arm}")
        print(f"    ΔAUROC: {drops.get('drop_auroc', 0):+.4f}  "
              f"(positive = PTM helps)")
        print(f"    ΔBAcc:  {drops.get('drop_bacc', 0):+.4f}")
        print(f"    Pass:   {'✓ PASS' if primary_pass else '✗ FAIL'}")
    else:
        print(f"\n  ⚠ randomized_ptm_control.json not found — run step11b first")

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
        r_val = regression.get("pearson_r", 0)
        ax.set_title(f"IC50 Prediction (R={r_val:.3f}, RMSE={model_rmse:.3f})")

        # Plot 2: Resistance probability histogram
        ax = axes[0, 1]
        sens_probs = y_prob_cls[y_true_cls == 0]
        res_probs = y_prob_cls[y_true_cls == 1]
        if len(sens_probs) > 0:
            ax.hist(sens_probs, bins=15, alpha=0.7,
                    label=f"Sensitive (n={len(sens_probs)})", color="green", density=True)
        if len(res_probs) > 0:
            ax.hist(res_probs, bins=15, alpha=0.7,
                    label=f"Resistant (n={len(res_probs)})", color="red", density=True)
        ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold")
        ax.set_xlabel("Predicted Resistance Probability")
        ax.set_ylabel("Density")
        ax.set_title(f"Classification (BAcc={model_bacc:.3f})")
        ax.legend()

        # Plot 3: Drug-specific BAcc comparison
        ax = axes[1, 0]
        drug_names = sorted(drug_results.keys())
        drug_baccs = [drug_results[d].get("balanced_accuracy", 0) for d in drug_names]
        drug_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(drug_names)]
        bars = ax.bar(drug_names, drug_baccs, color=drug_colors, alpha=0.85,
                      edgecolor="black")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title("Drug-Specific Performance (Test Set)")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        for bar, v in zip(bars, drug_baccs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Plot 4: Mutation-group mean predictions
        ax = axes[1, 1]
        if mutation_results:
            mut_names = sorted(mutation_results.keys(),
                               key=lambda x: mutation_results[x]["mean_resist_prob"])
            mut_probs = [mutation_results[m]["mean_resist_prob"] for m in mut_names]
            mut_n = [mutation_results[m]["n_samples"] for m in mut_names]
            # Shorten names for display
            short_names = [m[:20] + "..." if len(m) > 20 else m for m in mut_names]
            y_pos = range(len(mut_names))
            bars = ax.barh(y_pos, mut_probs, color="steelblue", alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(short_names, fontsize=8)
            ax.set_xlabel("Mean Resistance Probability")
            ax.set_title("EGFR Mutation Groups (All Samples)")
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
            for i, (bar, n) in enumerate(zip(bars, mut_n)):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"n={n}", va="center", fontsize=8)

        plt.suptitle("Model Evaluation — Comprehensive Analysis",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig_path = FIGURES_DIR / "evaluation_plots.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure saved: {fig_path}")
    except Exception as e:
        print(f"  ⚠ Could not generate figure: {e}")

    print("\n✓ Step 12 complete!")


if __name__ == "__main__":
    evaluate()
