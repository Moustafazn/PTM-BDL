"""
Cross-dataset evaluation: Train on GDSC2, evaluate against CTRPv2.

Tests whether the GDSC-trained model's predictions generalize to an
independent drug sensitivity dataset (CTRPv2) that uses different
viability assays, dose ranges, and fitting algorithms.

Ref: Sada Del Real et al., Brief Bioinf 2026 — recommends cross-dataset
     evaluation as a key generalization axis for DRP benchmarking.

The module is case-study-agnostic — drug mappings come from the caller.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Subset


def _normalize_cell_name(name) -> str:
    """Normalize cell line names for cross-database matching."""
    if not name or not isinstance(name, str):
        return ""
    name = name.upper().strip()
    for prefix in ["NCI-", "NCI_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return re.sub(r"[-_ ]", "", name)


def run_cross_dataset_ctrp(
        model,
        dataset,
        ctrp_csv_path: str,
        collate_fn,
        batch_size: int = 16,
        device: str = "cpu",
        drug_name_map: dict | None = None,
) -> dict:
    """
    Cross-dataset evaluation: GDSC-trained model vs CTRPv2 ground truth.

    For each (cell_line, drug) pair present in BOTH GDSC and CTRPv2:
      1. Use the GDSC-trained model to predict ln(IC50)
      2. Compare predictions against CTRPv2 IC50 values
      3. Compute Pearson R, Spearman rho, RMSE across the overlap

    This tests cross-assay generalization. GDSC2 and CTRPv2 use different
    viability protocols, dose ranges, and fitting algorithms, so perfect
    correlation is NOT expected. Pearson R > 0.3 is meaningful.

    Args:
        model: Trained PTM-BDL model (on GDSC data).
        dataset: ResistanceDataset (GDSC-based).
        ctrp_csv_path: Path to processed CTRP CSV.
        collate_fn: Collation function.
        batch_size: Batch size for inference.
        device: Device for inference.
        drug_name_map: Optional dict mapping GDSC drug names to CTRP names.

    Returns:
        Dict with overall and per-drug cross-dataset metrics.
    """
    print(f"\n  Cross-Dataset Evaluation: GDSC -> CTRPv2")
    print(f"  {'=' * 50}")

    ctrp_path = Path(ctrp_csv_path)
    if not ctrp_path.exists():
        print(f"    CTRP data not found: {ctrp_path}")
        print(f"    Run: python -m src.case_studies.common.download_ctrp")
        return {"status": "skipped", "reason": "CTRP data not found"}

    df_ctrp = pd.read_csv(ctrp_path)
    print(f"    CTRP records: {len(df_ctrp)}")

    # Normalize CTRP cell line names
    cell_col_ctrp = ("cell_line_norm" if "cell_line_norm" in df_ctrp.columns
                     else "cell_line_name")
    df_ctrp["cell_norm"] = df_ctrp[cell_col_ctrp].apply(_normalize_cell_name)

    # Get CTRP drug name column
    drug_col_ctrp = ("drug_name_canonical"
                     if "drug_name_canonical" in df_ctrp.columns
                     else "canonical_drug_name"
                     if "canonical_drug_name" in df_ctrp.columns
                     else "drug_name")

    # Get CTRP IC50 column
    ic50_col_ctrp = None
    for col in ["ln_ic50_ctrp", "ic50_recomputed", "ln_ic50"]:
        if col in df_ctrp.columns:
            ic50_col_ctrp = col
            break
    if ic50_col_ctrp is None:
        print(f"    No IC50 column found in CTRP data")
        return {"status": "skipped", "reason": "No IC50 column in CTRP"}

    # Normalize GDSC dataset
    df_gdsc = dataset.df.copy()
    cell_col_gdsc = ("cell_line_name" if "cell_line_name" in df_gdsc.columns
                     else "cell_line")
    df_gdsc["cell_norm"] = df_gdsc[cell_col_gdsc].apply(_normalize_cell_name)

    if drug_name_map:
        df_gdsc["drug_mapped"] = df_gdsc["drug_name"].map(
            lambda d: drug_name_map.get(d, d))
    else:
        df_gdsc["drug_mapped"] = df_gdsc["drug_name"]

    # Find overlapping (cell_line, drug) pairs
    gdsc_pairs = set(zip(df_gdsc["cell_norm"], df_gdsc["drug_mapped"]))
    ctrp_pairs = set(zip(df_ctrp["cell_norm"], df_ctrp[drug_col_ctrp]))
    overlap = gdsc_pairs & ctrp_pairs

    print(f"    GDSC (cell, drug) pairs: {len(gdsc_pairs)}")
    print(f"    CTRP (cell, drug) pairs: {len(ctrp_pairs)}")
    print(f"    Overlap: {len(overlap)} pairs")

    if len(overlap) < 5:
        print(f"    Too few overlapping pairs for evaluation")
        return {"status": "skipped",
                "reason": f"Only {len(overlap)} overlap",
                "n_overlap": len(overlap)}

    # Build CTRP lookup: (cell_norm, drug) -> ln_IC50
    ctrp_lookup = {}
    for _, row in df_ctrp.iterrows():
        key = (row["cell_norm"], row[drug_col_ctrp])
        val = row.get(ic50_col_ctrp)
        if pd.notna(val):
            ctrp_lookup[key] = float(val)

    # Match GDSC samples to CTRP
    overlap_gdsc_idx = []
    overlap_ctrp_ic50 = []
    overlap_drugs = []

    for idx in range(len(df_gdsc)):
        cell = df_gdsc.iloc[idx]["cell_norm"]
        drug = df_gdsc.iloc[idx]["drug_mapped"]
        key = (cell, drug)
        if key in ctrp_lookup:
            overlap_gdsc_idx.append(idx)
            overlap_ctrp_ic50.append(ctrp_lookup[key])
            overlap_drugs.append(drug)

    if len(overlap_gdsc_idx) < 5:
        return {"status": "skipped", "reason": "Too few matched samples"}

    print(f"    Matched GDSC samples: {len(overlap_gdsc_idx)}")
    drugs_in_overlap = sorted(set(overlap_drugs))
    for d in drugs_in_overlap:
        n = sum(1 for dd in overlap_drugs if dd == d)
        print(f"      {d}: {n} samples")

    # Run model inference on overlapping GDSC samples
    model.eval()
    overlap_subset = Subset(dataset, overlap_gdsc_idx)
    overlap_loader = DataLoader(
        overlap_subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    gdsc_ic50_pred = []
    gdsc_ic50_true = []

    with torch.no_grad():
        for batch in overlap_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            ic50_pred, resist_pred = model(
                seq_embeddings=batch_dev["seq_emb"],
                struct_embeddings=batch_dev["struct_emb"],
                drug_pooled=batch_dev["drug_pooled"],
                drug_embeddings=batch_dev.get("drug_emb"),
                ptm_vector=batch_dev["ptm_vector"],
                delta_ptm_vector=batch_dev["delta_ptm_vector"],
                target_protein=batch_dev["target_protein"],
            )
            gdsc_ic50_pred.append(ic50_pred.cpu().numpy())
            gdsc_ic50_true.append(batch["ln_ic50"].numpy())

    y_pred = np.concatenate(gdsc_ic50_pred).flatten()
    y_true_gdsc = np.concatenate(gdsc_ic50_true).flatten()
    y_ctrp = np.array(overlap_ctrp_ic50)
    drug_arr = np.array(overlap_drugs)

    # ── Overall cross-dataset metrics ─────────────────────────────────
    overall = {"n_samples": len(y_pred)}

    # Model prediction vs CTRP ground truth
    if (len(y_pred) > 2
            and np.std(y_pred) > 1e-8
            and np.std(y_ctrp) > 1e-8):
        overall["pred_vs_ctrp_pearson_r"] = float(
            np.corrcoef(y_pred, y_ctrp)[0, 1])
        sr = scipy_stats.spearmanr(y_pred, y_ctrp)
        overall["pred_vs_ctrp_spearman_rho"] = float(
            sr.statistic if hasattr(sr, 'statistic') else sr[0])
        overall["pred_vs_ctrp_rmse"] = float(
            np.sqrt(((y_pred - y_ctrp) ** 2).mean()))

    # GDSC truth vs CTRP truth (assay concordance baseline)
    if (len(y_true_gdsc) > 2
            and np.std(y_true_gdsc) > 1e-8
            and np.std(y_ctrp) > 1e-8):
        overall["gdsc_vs_ctrp_pearson_r"] = float(
            np.corrcoef(y_true_gdsc, y_ctrp)[0, 1])
        sr2 = scipy_stats.spearmanr(y_true_gdsc, y_ctrp)
        overall["gdsc_vs_ctrp_spearman_rho"] = float(
            sr2.statistic if hasattr(sr2, 'statistic') else sr2[0])

    print(f"\n    Overall cross-dataset results:")
    print(f"      Model pred vs CTRP:  PCC="
          f"{overall.get('pred_vs_ctrp_pearson_r', 0):.3f}")
    print(f"      GDSC truth vs CTRP:  PCC="
          f"{overall.get('gdsc_vs_ctrp_pearson_r', 0):.3f} "
          f"(assay concordance baseline)")

    # ── Per-drug cross-dataset metrics ────────────────────────────────
    per_drug = {}
    for drug in drugs_in_overlap:
        mask = drug_arr == drug
        if mask.sum() < 3:
            per_drug[drug] = {"n_samples": int(mask.sum()),
                              "note": "too few samples"}
            continue
        pred_d = y_pred[mask]
        ctrp_d = y_ctrp[mask]
        gdsc_d = y_true_gdsc[mask]
        entry = {"n_samples": int(mask.sum())}
        if np.std(pred_d) > 1e-8 and np.std(ctrp_d) > 1e-8:
            entry["pred_vs_ctrp_pearson_r"] = float(
                np.corrcoef(pred_d, ctrp_d)[0, 1])
        if np.std(gdsc_d) > 1e-8 and np.std(ctrp_d) > 1e-8:
            entry["gdsc_vs_ctrp_pearson_r"] = float(
                np.corrcoef(gdsc_d, ctrp_d)[0, 1])
        per_drug[drug] = entry
        pcc = entry.get('pred_vs_ctrp_pearson_r', 0)
        print(f"      {drug:15s}: n={entry['n_samples']:4d}, "
              f"pred->CTRP PCC={pcc:.3f}")

    return {
        "method": "Cross-dataset (GDSC2 -> CTRPv2)",
        "description": (
            "GDSC-trained model predictions evaluated against CTRPv2 "
            "IC50 ground truth for overlapping (cell_line, drug) pairs. "
            "Ref: Basu et al. Cell 2013; Rees et al. Nat Chem Biol 2016."
        ),
        "ctrp_source": str(ctrp_csv_path),
        "overall": overall,
        "per_drug": per_drug,
        "drugs_in_overlap": drugs_in_overlap,
        "n_overlap_pairs": len(overlap_gdsc_idx),
    }
