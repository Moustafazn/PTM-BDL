#!/usr/bin/env python3
"""
Extended evaluation metrics — runs ONCE for ALL three case studies.

Computes supplementary metrics not covered by the main evaluation:
  A. Brier score (calibration complement to ECE)
  B. PTM-vector-grouped split overlap analysis
  C. IG pad-token audit
  D. Data accounting table (precise sample/cell/drug/protein counts)

Outputs:
  results/<case_study>/extended_evaluation.json   (per case study)
  results/extended_evaluation_summary.json        (cross-study)

Usage:
    python -m src.case_studies.common.extended_evaluation
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CASE_STUDIES = ["egfr_erbb2_tki", "hela_hdac", "k562_cml"]


# ══════════════════════════════════════════════════════════════════════════════
# A. BRIER SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_brier(results_dir: Path) -> dict:
    """Compute Brier score from saved test predictions."""
    npz = results_dir / "test_predictions.npz"
    if not npz.exists():
        return {"status": "skipped", "reason": "test_predictions.npz not found"}
    data = np.load(npz)
    y_true = data.get("y_true_cls", data.get("resist_true"))
    y_prob = data.get("y_prob_cls", data.get("resist_prob"))
    if y_true is None or y_prob is None:
        return {"status": "skipped", "reason": f"keys: {list(data.keys())}"}
    from sklearn.metrics import brier_score_loss
    brier = float(brier_score_loss(y_true, y_prob))
    return {"brier_score": brier, "n_samples": int(len(y_true)),
            "quality": ("good" if brier < 0.15 else
                        "moderate" if brier < 0.25 else "poor")}


# ══════════════════════════════════════════════════════════════════════════════
# B. PTM-VECTOR-GROUPED SPLIT OVERLAP
# ══════════════════════════════════════════════════════════════════════════════

def analyse_ptm_grouped_splits(case_study: str) -> dict:
    """Quantify PTM-vector overlap between train and test splits."""
    from src.ptm_bdl.config import load_config
    cfg = load_config(case_study=case_study)
    split_path = PROJECT_ROOT / cfg["paths"]["models"] / case_study / "split_indices.json"
    if not split_path.exists():
        return {"status": "skipped", "reason": "split_indices.json not found"}
    with open(split_path) as f:
        split = json.load(f)
    train_idx, test_idx = np.array(split["train_idx"]), np.array(split["test_idx"])

    # Load CSV  (case-study-specific path first to avoid cross-study collisions)
    for cand in [PROJECT_ROOT / cfg["paths"]["processed_data"] / case_study / "multimodal_dataset.csv",
                 PROJECT_ROOT / cfg["paths"]["processed_data"] / "multimodal_dataset.csv"]:
        if cand.exists():
            df = pd.read_csv(cand); break
    else:
        return {"status": "skipped", "reason": "CSV not found"}

    ptm_cols = sorted([c for c in df.columns
                       if (c.startswith("ptm_") or "_slot" in c)
                       and not c.startswith("delta_")
                       and df[c].dtype in ("float64", "float32")])
    delta_cols = sorted([c for c in df.columns
                         if c.startswith("delta_ptm_") or
                         (c.startswith("delta_") and "_slot" in c)])
    all_cols = ptm_cols + delta_cols
    if not all_cols:
        return {"status": "skipped", "reason": "no PTM columns"}

    def fps(idx):
        return set(df.iloc[idx][all_cols].round(6).apply(tuple, axis=1))
    train_fp, test_fp = fps(train_idx), fps(test_idx)
    overlap = train_fp & test_fp

    test_per = df.iloc[test_idx][all_cols].round(6).apply(tuple, axis=1)
    n_overlap = int(test_per.isin(overlap).sum())
    return {
        "n_unique_train": len(train_fp), "n_unique_test": len(test_fp),
        "n_overlapping_vectors": len(overlap),
        "test_samples_with_overlap": n_overlap,
        "test_samples_unique": len(test_idx) - n_overlap,
        "overlap_fraction": round(n_overlap / max(len(test_idx), 1), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# C. IG PAD-TOKEN AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def audit_ig_pad_tokens(results_dir: Path) -> dict:
    """Check for pad tokens in top-ranked IG sites."""
    xai_path = results_dir / "xai_report.json"
    if not xai_path.exists():
        return {"status": "skipped", "reason": "xai_report.json not found"}
    with open(xai_path) as f:
        xai = json.load(f)

    pad_issues, total, pad_top10 = [], 0, 0
    for key in xai:
        if not key.startswith("integrated_gradients_"):
            continue
        section = xai[key]
        if not isinstance(section, dict):
            continue
        for protein, pdata in section.items():
            if not isinstance(pdata, dict):
                continue
            for rk in ["resist_site_ranking", "site_ranking"]:
                for entry in pdata.get(rk, []):
                    total += 1
                    site = str(entry.get("site", entry.get("label", "")))
                    rank = entry.get("rank", 999)
                    if "pad" in site.lower() or site == "":
                        if rank <= 10:
                            pad_top10 += 1
                            pad_issues.append({"section": key, "protein": protein,
                                               "rank": rank, "site": site})
    return {"total_ranked": total, "pad_in_top10": pad_top10,
            "issues": pad_issues[:20]}


# ══════════════════════════════════════════════════════════════════════════════
# D. DATA ACCOUNTING
# ══════════════════════════════════════════════════════════════════════════════

def build_data_accounting(case_study: str) -> dict:
    """Precise sample/cell/drug/protein counts for one case study."""
    from src.ptm_bdl.config import load_config
    cfg = load_config(case_study=case_study)
    for cand in [PROJECT_ROOT / cfg["paths"]["processed_data"] / case_study / "multimodal_dataset.csv",
                 PROJECT_ROOT / cfg["paths"]["processed_data"] / "multimodal_dataset.csv"]:
        if cand.exists():
            df = pd.read_csv(cand); break
    else:
        return {"status": "skipped"}

    cell_col = "cell_line_name" if "cell_line_name" in df.columns else "cell_line"
    drugs = sorted(df["drug_name"].unique().tolist())
    proteins = sorted(df["target_protein"].unique().tolist()) if "target_protein" in df.columns else []
    n_s = int((df["resistance_label"] == 0).sum())
    n_r = int((df["resistance_label"] == 1).sum())

    pair_cols = [c for c in [cell_col, "drug_name", "target_protein"] if c in df.columns]
    n_pairs = int(df[pair_cols].drop_duplicates().shape[0])

    acc = {"total_samples": len(df), "unique_cell_lines": int(df[cell_col].nunique()),
           "unique_drugs": len(drugs), "drug_names": drugs,
           "unique_proteins": len(proteins), "protein_names": proteins,
           "unique_pairs": n_pairs, "pair_definition": " × ".join(pair_cols),
           "n_sensitive": n_s, "n_resistant": n_r,
           "class_ratio": f"{n_r}:{n_s}"}
    if "target_protein" in df.columns:
        acc["per_protein"] = {
            p: {"samples": int((df["target_protein"] == p).sum()),
                "cells": int(df.loc[df["target_protein"] == p, cell_col].nunique()),
                "drugs": int(df.loc[df["target_protein"] == p, "drug_name"].nunique())}
            for p in proteins}
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Extended Evaluation — All Case Studies                    ║")
    print("║  Brier · PTM-grouped splits · IG pad audit · Accounting   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    from src.ptm_bdl.config import load_config
    summary = {}

    for cs in CASE_STUDIES:
        print(f"\n{'='*60}\n  {cs.upper()}\n{'='*60}")
        cfg = load_config(case_study=cs)
        rdir = PROJECT_ROOT / cfg["paths"]["results"] / cs
        rdir.mkdir(parents=True, exist_ok=True)
        report = {"case_study": cs, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        # A
        print("  A. Brier score...")
        b = compute_brier(rdir); report["brier_score"] = b
        if "brier_score" in b:
            print(f"     Brier = {b['brier_score']:.4f} ({b['quality']})")
        else:
            print(f"     {b.get('reason','skipped')}")

        # B
        print("  B. PTM-vector overlap...")
        g = analyse_ptm_grouped_splits(cs); report["ptm_grouped_splits"] = g
        if "overlap_fraction" in g:
            print(f"     Overlap: {g['overlap_fraction']:.1%} of test samples")
        else:
            print(f"     {g.get('reason','skipped')}")

        # C
        print("  C. IG pad-token audit...")
        ig = audit_ig_pad_tokens(rdir); report["ig_pad_audit"] = ig
        if "total_ranked" in ig:
            print(f"     Ranked: {ig['total_ranked']}, pad in top-10: {ig['pad_in_top10']}")
        else:
            print(f"     {ig.get('reason','skipped')}")

        # D
        print("  D. Data accounting...")
        a = build_data_accounting(cs); report["data_accounting"] = a
        if "total_samples" in a:
            print(f"     {a['total_samples']} samples, {a['unique_cell_lines']} cells, "
                  f"{a['unique_drugs']} drugs, {a['unique_proteins']} proteins")
        else:
            print(f"     skipped")

        out = rdir / "extended_evaluation.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  ✓ Saved: {out}")
        summary[cs] = report

    # Cross-study table
    print(f"\n  {'Case Study':<18s} | {'Samples':>7s} | {'Cells':>5s} | "
          f"{'Drugs':>5s} | {'Proteins':>8s} | {'R:S':<10s}")
    print(f"  {'-'*65}")
    for cs in CASE_STUDIES:
        a = summary[cs].get("data_accounting", {})
        if "total_samples" in a:
            print(f"  {cs:<18s} | {a['total_samples']:>7d} | "
                  f"{a['unique_cell_lines']:>5d} | {a['unique_drugs']:>5d} | "
                  f"{a['unique_proteins']:>8d} | {a['class_ratio']:<10s}")

    out = PROJECT_ROOT / "results" / "extended_evaluation_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  ✓ Summary: {out}\n✓ Done!")


if __name__ == "__main__":
    main()

