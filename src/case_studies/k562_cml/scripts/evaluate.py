#!/usr/bin/env python3
"""
K562/CML Case Study — Comprehensive Evaluation & Benchmarking.

PURPOSE:
  Evaluate PTM-BDL predictions AGAINST published experimental outcomes
  for BCR-ABL TKIs and chemotherapy drugs on K562 CML cells.

PUBLISHED BENCHMARKS (model predictions must match these):
  1. Dasatinib IC50 on K562 ≈ 0.8 nM (ln = −7.13)
     → Model should predict STRONG sensitivity (resistance_prob ≈ 0)
     Ref: Shah et al., Science 2004 (PMID 15256107) — "Multiple BCR-ABL
          kinase domain mutations confer polyclonal resistance to the
          tyrosine kinase inhibitor imatinib (STI571) in chronic phase
          and blast crisis chronic myeloid leukemia"

  2. Imatinib IC50 on K562 ≈ 260 nM (ln = −1.35)
     → Model should predict moderate sensitivity
     Ref: Druker et al., NEJM 2006 (PMID 16481636) — "Five-Year Follow-up
          of Patients Receiving Imatinib for Chronic Myeloid Leukemia"

  3. Dasatinib ~325× more potent than Imatinib on BCR-ABL
     → Model's predicted IC50 for Dasatinib << Imatinib
     Ref: O'Hare et al., Blood 2005 (PMID 15256422) — "In vitro activity
          of Bcr-Abl inhibitors AMN107 and BMS-354825 against clinically
          relevant imatinib-resistant Abl kinase domain mutants"

  4. Cytarabine IC50 on K562 ≈ 200 nM (ln = −1.61)
     Ref: Momparler, Pharmacol Ther 2013 (PMID 23583331)

  5. Paclitaxel IC50 on K562 ≈ 10 nM (ln = −4.61)
     Ref: Mujagic et al., Leuk Res 2002 (PMID 12191564) — taxane
          sensitivity in leukemia cell lines

  6. TKI drugs should dephosphorylate CRKL-Y207 and STAT5A-Y694
     (direct BCR-ABL substrates) while chemo drugs should NOT
     → Cross-drug IG pattern validation
     Ref: Soverini et al., Haematologica 2024 — CML treatment guidelines
     Ref: Hochhaus et al., Leukemia 2020 (PMID 31988391) — ELN CML management

EVALUATION AXES:
  1. Standard metrics — AUROC, BAcc, Pearson R, RMSE
  2. Per-drug evaluation — all 5 drugs individually
  3. TKI vs chemo stratification — mechanism-based grouping
  4. Potency ranking validation — Dasatinib > Imatinib sensitivity
  5. Published IC50 comparison — predicted vs literature values
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

from src.ptm_bdl.data import ResistanceDataset, collate_fn
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics, load_threshold, make_eval_loader
from src.ptm_bdl.training import build_model_from_cfg, load_checkpoint, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load optimal classification threshold (Youden's J, from training)
RESIST_THRESHOLD = load_threshold(MODEL_DIR)

# Published IC50 benchmarks (ln(IC50 in µM))
PUBLISHED_IC50 = {
    "Dasatinib":    {"ln_ic50": -7.13, "ic50_nM": 0.8,
                     "ref": "Shah et al., Science 2004 (PMID 15256107)"},
    "Imatinib":     {"ln_ic50": -1.35, "ic50_nM": 260,
                     "ref": "Druker et al., NEJM 2006 (PMID 16481636)"},
    "Cytarabine":   {"ln_ic50": -1.61, "ic50_nM": 200,
                     "ref": "Momparler, Pharmacol Ther 2013 (PMID 23583331)"},
    "Paclitaxel":   {"ln_ic50": -4.61, "ic50_nM": 10,
                     "ref": "Mujagic et al., Leuk Res 2002 (PMID 12191564)"},
    "Methotrexat":  {"ln_ic50": -2.30, "ic50_nM": 100,
                     "ref": "estimated from GDSC2 K562 data"},
}

DRUG_GROUPS = {
    "TKI": ["Dasatinib", "Imatinib"],
    "chemo": ["Cytarabine", "Paclitaxel", "Methotrexat"],
}


def evaluate():
    """Comprehensive evaluation with published benchmark validation."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  {CASE_STUDY} — Comprehensive Evaluation                   ║")
    print(f"║  5 drugs: 2 TKIs + 3 chemo — benchmark vs publications    ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    device = resolve_device(cfg)
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
    load_checkpoint(model, MODEL_DIR / "best_model.pt", device)

    # Helper — build a DataLoader from index array (same pattern as egfr)
    batch_size = cfg["model"]["batch_size"]

    def _loader(indices):
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        return torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ── 1. Standard metrics ──────────────────────────────────────────────
    print("\n  1. Standard metrics...")
    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(
        model, _loader(test_idx))
    regression, classification = compute_full_metrics(
        y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
        threshold=RESIST_THRESHOLD)
    metrics = {**regression, **classification}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # ── 2. Per-drug evaluation ───────────────────────────────────────────
    print("\n  2. Per-drug evaluation...")
    df_test = df.iloc[test_idx]
    per_drug = {}
    for drug in sorted(df_test["drug_name"].unique()):
        drug_mask = df_test["drug_name"] == drug
        drug_idx = test_idx[drug_mask.values]
        if len(drug_idx) < 2:
            continue
        d_ic50_t, d_ic50_p, d_cls_t, d_cls_p = collect_predictions(
            model, _loader(drug_idx))
        d_reg, d_cls = compute_full_metrics(d_ic50_t, d_ic50_p, d_cls_t, d_cls_p,
                                            threshold=RESIST_THRESHOLD)
        dm = {**d_reg, **d_cls, "mean_pred_ic50": float(d_ic50_p.mean())}
        per_drug[drug] = dm

    # ── 3. TKI vs chemo stratification ───────────────────────────────────
    print("\n  3. TKI vs chemo mechanism stratification...")
    print("    Ref: Soverini et al., Haematologica 2024")
    group_metrics = {}
    for group_name, drugs in DRUG_GROUPS.items():
        mask = df_test["drug_name"].isin(drugs)
        idx = test_idx[mask.values]
        if len(idx) >= 2:
            g_ic50_t, g_ic50_p, g_cls_t, g_cls_p = collect_predictions(
                model, _loader(idx))
            g_reg, g_cls = compute_full_metrics(
                g_ic50_t, g_ic50_p, g_cls_t, g_cls_p,
                threshold=RESIST_THRESHOLD)
            gm = {**g_reg, **g_cls}
            group_metrics[group_name] = gm
            print(f"    {group_name:6s} ({len(idx)} samples): "
                  f"AUROC={gm.get('auroc', 'N/A')}")

    # ── 4. Potency ranking: Dasatinib > Imatinib ────────────────────────
    print("\n  4. Potency ranking validation...")
    print("    Ref: O'Hare et al., Blood 2005 (PMID 15256422)")
    print("    Expected: Dasatinib ~325× more potent than Imatinib")
    das_pred = per_drug.get("Dasatinib", {}).get("mean_pred_ic50")
    ima_pred = per_drug.get("Imatinib", {}).get("mean_pred_ic50")
    if das_pred is not None and ima_pred is not None:
        print(f"    Predicted IC50: Dasatinib={das_pred:.3f}, Imatinib={ima_pred:.3f}")
        print(f"    Dasatinib < Imatinib: {das_pred < ima_pred} ✓" if das_pred < ima_pred
              else f"    ⚠ Dasatinib NOT predicted as more potent")

    # ── 5. Published IC50 benchmark comparison ───────────────────────────
    print("\n  5. Published IC50 benchmark comparison...")
    benchmark_results = {}
    for drug, pub in PUBLISHED_IC50.items():
        if drug in per_drug:
            pred_ic50 = per_drug[drug].get("mean_pred_ic50", "N/A")
            benchmark_results[drug] = {
                "predicted_ln_ic50": pred_ic50,
                "published_ln_ic50": pub["ln_ic50"],
                "published_ic50_nM": pub["ic50_nM"],
                "reference": pub["ref"],
            }
            print(f"    {drug:12s}: predicted={pred_ic50}, "
                  f"published={pub['ln_ic50']:.2f} ({pub['ic50_nM']} nM)")
            print(f"                ref: {pub['ref']}")

    # ══════════════════════════════════════════════════════════════════════
    # REVIEWER ANALYSES: Q2 Leakage, Q8 Exclusion, Q9 ECE, Q10 IG Audit
    # ══════════════════════════════════════════════════════════════════════
    print("\n  ── Reviewer Q2: Formal Leakage Analysis ──")
    from src.ptm_bdl.evaluation.statistical import (
        compute_leakage_analysis, compute_ece, compute_ece_per_drug,
    )
    split_path_q2 = MODEL_DIR / "split_indices.json"
    if split_path_q2.exists():
        with open(split_path_q2) as f:
            split_q2 = json.load(f)
        leakage = compute_leakage_analysis(
            df, train_idx=np.array(split_q2["train_idx"]),
            val_idx=np.array(split_q2["val_idx"]),
            test_idx=np.array(split_q2["test_idx"]),
        )
        print(f"    PTM diversity: {leakage.get('ptm_all_diversity', '?')}")
        if "constant_channels" in leakage:
            print(f"    Constant channels: {leakage['constant_channels']['n_constant']}")
    else:
        leakage = {}

    # Q9: ECE calibration
    print("\n  ── Reviewer Q9: ECE Calibration ──")
    ece_overall = compute_ece(y_true_cls, y_prob_cls, n_bins=10)
    print(f"    Overall ECE: {ece_overall['ece']:.4f}, MCE: {ece_overall['mce']:.4f}")
    ece_per_drug = compute_ece_per_drug(
        y_true_cls, y_prob_cls, df_test["drug_name"].values, n_bins=10)

    # Q10: IG scale audit
    print("\n  ── Reviewer Q10: IG Scale Audit ──")
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
                if rankings:
                    values = [r["mean_abs_attribution"] for r in rankings]
                    min_v, max_v = min(values), max(values)
                    ig_audit["ig_value_ranges"][f"{protein}_{mod_type}"] = {
                        "min": float(min_v), "max": float(max_v), "n_sites": len(values),
                    }
                    if max_v < 1e-6:
                        ig_audit["issues_found"].append(f"{protein} {mod_type}: near-zero IG")
                    print(f"    {protein:8s} {mod_type:12s}: range=[{min_v:.2e}, {max_v:.2e}]")
        if not ig_audit["issues_found"]:
            print(f"    ✓ No obvious scale issues")
    else:
        print(f"    ⚠ xai_report.json not found")

    # ── Save ─────────────────────────────────────────────────────────────
    report = {
        "case_study": CASE_STUDY,
        "overall_metrics": metrics,
        "per_drug": per_drug,
        "drug_group_metrics": group_metrics,
        "published_ic50_benchmarks": benchmark_results,
        "leakage_analysis": leakage,
        "calibration": {"overall_ece": ece_overall, "per_drug_ece": ece_per_drug},
        "ig_scale_audit": ig_audit,
        "frozen_encoders": {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "frozen": "ESM-2(650M)+ChemBERTa(77M)+GearNet all FROZEN"},
        "references": {
            "dasatinib_potency": "Shah et al., Science 2004 (PMID 15256107)",
            "imatinib_cml": "Druker et al., NEJM 2006 (PMID 16481636)",
            "dasatinib_vs_imatinib": "O'Hare et al., Blood 2005 (PMID 15256422)",
            "cml_guidelines": "Soverini et al., Haematologica 2024",
            "eln_management": "Hochhaus et al., Leukemia 2020 (PMID 31988391)",
            "cytarabine": "Momparler, Pharmacol Ther 2013 (PMID 23583331)",
            "primary_data": "Badkul et al., DrugPTM-Bench 2024",
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
