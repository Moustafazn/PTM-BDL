#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 05 — Process Drug-PTM Data: HeLa / HDAC Inhibitor Case Study         ║
║  (Phosphorylation + Acetylation — NEW PTM type for the tool)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    1. Extract phosphorylation AND acetylation dose-response data from        ║
║       DrugPTM-Bench PTM_CellLine_HeLa.csv (508 MB, 980,608 rows).          ║
║    2. Scan ALL 7 DrugPTM-Bench cell line files for BASELINE PTM levels      ║
║       (dose=0) of HDAC target genes: HDAC1, EP300, HIST1H4A, CREBBP.       ║
║       This cross-cell-line baseline is used by step06 to build a multi-     ║
║       cell-line dataset with GDSC IC50 labels (~900 cell lines).            ║
║                                                                              ║
║    This is the FIRST case study to use acetylation (acetyl_K) — a NEW       ║
║    PTM type that the tool has never seen. If PTM-BDL processes         ║
║    acetylation tokens alongside phospho tokens without code changes,        ║
║    it proves the unified tool claim.                                   ║
║                                                                              ║
║  DATA SOURCE:                                                                ║
║    DrugPTM-Bench — Badkul et al., 2026 (PMID 30394195)                     ║
║    Primary: data/raw/drugptm/30394195/PTM_CellLine_HeLa.csv                ║
║    Baseline scan: data/raw/drugptm/30394195/PTM_CellLine_*.csv (all 7)     ║
║                                                                              ║
║  PTM TYPES IN DATA (verified):                                               ║
║    phosphorylation  921,623 rows (94%) — residues S, T, Y                   ║
║    acetylation       58,985 rows  (6%) — residue K (lysine)                 ║
║                                                                              ║
║  DRUGS (6):                                                                  ║
║    Vorinostat (SAHA)  — pan-HDAC inhibitor (GDSC ID 1012)                   ║
║    Romidepsin         — HDAC class I selective (GDSC ID 1659)               ║
║    CUDC-101           — triple HDAC/EGFR/HER2 (GDSC ID 1578)               ║
║    A485               — p300/CBP HAT inhibitor (research compound)          ║
║    A486               — inactive control for A485                           ║
║    Curcumin           — natural polyphenol HDAC modulator                   ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/processed/drugptm/hela_hdac_ptm_responses.csv                       ║
║    data/processed/drugptm/hela_hdac_drug_catalog.csv                        ║
║    data/processed/drugptm/hela_hdac_baseline_ptm.csv   ← NEW               ║
║                                                                              ║
║  UNIFIED SCHEMA (each row = one PTM site measurement):                       ║
║    cell_line, drug_name, protein (gene name), ptm_site, ptm_residue,        ║
║    ptm_type (phosphorylation | acetylation), ptm_modification_type          ║
║    (phospho_S | phospho_T | phospho_Y | acetyl_K),                          ║
║    baseline_intensity, max_dose_intensity, log2_fold_change,                 ║
║    EC50, pEC50, curve_effect_size, R2, n_doses, drug_smiles,                ║
║    data_source                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="hela_hdac")

RAW_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "drugptm"
PMID_DIR = RAW_DIR / "30394195"
OUT_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"] / "drugptm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

KEEP_COLUMNS = [
    "Sequence", "PTM Index", "PTM Residue", "PTM type",
    "Proteins", "Leading proteins", "Protein names", "Gene names",
    "Cell Line", "Timepoint (min)", "Chemical Name",
    "Dosage (µg)", "N duplicates",
    "Signal Intensity", "Signal Ratio",
    "Log EC50", "Curve slope", "Curve top", "Curve bottom",
    "R2", "EC50", "pEC50", "Curve effect size",
    "Canonical SMILES",
]

# PTM type → modification subtype mapping
PTM_SUBTYPE_MAP = {
    ("phosphorylation", "S"): "phospho_S",
    ("phosphorylation", "T"): "phospho_T",
    ("phosphorylation", "Y"): "phospho_Y",
    ("acetylation", "K"): "acetyl_K",
}

# Target genes for cross-cell-line baseline scan
TARGET_GENES = cfg["project"].get("target_genes_for_baseline", [
    "HDAC1", "HDAC2", "HDAC3", "HDAC6",
    "EP300", "CREBBP", "HIST1H4A", "H2AFZ",
])


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Extract HeLa drug-response PTM data (phospho + acetyl)
# ══════════════════════════════════════════════════════════════════════════════

def extract_hela_ptm_data() -> pd.DataFrame:
    """
    Stream through PTM_CellLine_HeLa.csv in chunks and extract ALL rows
    (both phosphorylation and acetylation).

    Unlike the EGFR case study which filters by gene, here we keep ALL
    genes because HDAC inhibitors affect the global proteome — histones,
    EP300, CREBBP, and thousands of other proteins.
    """
    hela_path = PMID_DIR / "PTM_CellLine_HeLa.csv"

    if not hela_path.exists():
        print(f"  ✗ File not found: {hela_path}")
        print("    Place the DrugPTM-Bench HeLa CSV file at:")
        print(f"      {hela_path}")
        print("    Download from: https://github.com/Xie-lab/DrugPTM-Bench")
        return pd.DataFrame()

    print(f"\n  Reading: {hela_path.name} ({hela_path.stat().st_size / 1e6:.0f} MB)")

    frames: list[pd.DataFrame] = []
    n_total = 0
    n_phospho = 0
    n_acetyl = 0

    for chunk_i, chunk in enumerate(pd.read_csv(hela_path, chunksize=CHUNK_SIZE,
                                                 low_memory=False)):
        n_total += len(chunk)

        # Keep only columns we need
        avail = [c for c in KEEP_COLUMNS if c in chunk.columns]
        frames.append(chunk[avail])

        # Count PTM types
        if "PTM type" in chunk.columns:
            ptm_counts = chunk["PTM type"].value_counts()
            n_phospho += ptm_counts.get("phosphorylation", 0)
            n_acetyl += ptm_counts.get("acetylation", 0)

        if (chunk_i + 1) % 5 == 0:
            print(f"    ... {n_total:,} rows processed "
                  f"(phospho: {n_phospho:,}, acetyl: {n_acetyl:,})")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  ✓ Total rows extracted: {len(df):,}")
    print(f"    Phosphorylation: {n_phospho:,} ({100*n_phospho/n_total:.1f}%)")
    print(f"    Acetylation:     {n_acetyl:,} ({100*n_acetyl/n_total:.1f}%)")
    print(f"    Unique drugs:    {df['Chemical Name'].nunique()}")
    print(f"    Unique genes:    {df['Gene names'].nunique()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Cross-cell-line baseline PTM scan
# ══════════════════════════════════════════════════════════════════════════════

def extract_baseline_ptm_all_cell_lines() -> pd.DataFrame:
    """
    Scan ALL 7 DrugPTM-Bench cell line files for baseline (dose=0) PTM
    levels of HDAC target genes.

    This is biologically correct because:
      - Baseline PTM state of HDAC1/EP300/histones reflects the cell's
        epigenetic dependency
      - Cells with different baseline histone acetylation respond differently
        to HDAC inhibitors (Marks & Xu 2009; Seto & Yoshida 2014;
        Hartl et al., Cell Reports 2024)

    For each target gene × cell line, we extract the dose=0 (DMSO control)
    Signal Intensity, which represents the untreated baseline PTM level.

    Returns DataFrame with columns:
        cell_line, gene, ptm_site, ptm_residue, ptm_type,
        ptm_modification_type, baseline_intensity, n_replicates
    """
    print("\n" + "=" * 70)
    print("PART 2: Cross-Cell-Line Baseline PTM Scan")
    print("=" * 70)
    print(f"  Target genes: {TARGET_GENES}")

    all_files = cfg["drugptm_bench"].get("all_cell_line_files", [])
    if not all_files:
        print("  ⚠ No cell line files configured in drugptm_bench.all_cell_line_files")
        return pd.DataFrame()

    # Build a set for fast gene matching
    target_gene_set = set(TARGET_GENES)

    all_baseline_frames: list[pd.DataFrame] = []

    for fname in all_files:
        fpath = PMID_DIR / fname
        if not fpath.exists():
            print(f"  ✗ {fname} not found — skipping")
            continue

        cell_line_name = fname.replace("PTM_CellLine_", "").replace(".csv", "")
        n_total = 0
        n_target = 0
        target_frames: list[pd.DataFrame] = []

        print(f"\n  Scanning {fname} for target genes...")
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            n_total += len(chunk)

            # Filter for target genes
            if "Gene names" not in chunk.columns:
                continue
            # Build regex pattern for all target genes at once (fast)
            pattern = "|".join(target_gene_set)
            mask = chunk["Gene names"].astype(str).str.contains(
                pattern, na=False, regex=True)
            target_chunk = chunk.loc[mask].copy()

            if not target_chunk.empty:
                # Keep only dose=0 (baseline / DMSO control) rows
                if "Dosage (µg)" in target_chunk.columns:
                    target_chunk["Dosage (µg)"] = pd.to_numeric(
                        target_chunk["Dosage (µg)"], errors="coerce")
                    baseline_chunk = target_chunk[
                        target_chunk["Dosage (µg)"] == 0.0
                    ].copy()
                else:
                    baseline_chunk = target_chunk.copy()

                if not baseline_chunk.empty:
                    avail = [c for c in KEEP_COLUMNS if c in baseline_chunk.columns]
                    target_frames.append(baseline_chunk[avail])
                    n_target += len(baseline_chunk)

        print(f"    {cell_line_name:>15s}: {n_total:>10,} rows total → "
              f"{n_target:>6,} baseline target gene rows")

        if target_frames:
            df_cell = pd.concat(target_frames, ignore_index=True)
            df_cell["source_cell_line"] = cell_line_name
            all_baseline_frames.append(df_cell)

    if not all_baseline_frames:
        print("  ⚠ No baseline target gene rows found in any file.")
        return pd.DataFrame()

    df_all = pd.concat(all_baseline_frames, ignore_index=True)

    # Build per-site baseline summaries
    print(f"\n  Building baseline summaries from {len(df_all):,} rows...")

    if "Signal Intensity" in df_all.columns:
        df_all["Signal Intensity"] = pd.to_numeric(
            df_all["Signal Intensity"], errors="coerce")

    df_all["ptm_site"] = (df_all["PTM Residue"].astype(str)
                          + df_all["PTM Index"].astype(str))

    group_cols = ["source_cell_line", "Gene names", "ptm_site",
                  "PTM Residue", "PTM type"]
    avail_group = [c for c in group_cols if c in df_all.columns]

    records = []
    for keys, grp in df_all.groupby(avail_group, dropna=False):
        cell_line = keys[0] if len(keys) > 0 else ""
        gene = keys[1] if len(keys) > 1 else ""
        site = keys[2] if len(keys) > 2 else ""
        residue = keys[3] if len(keys) > 3 else ""
        ptm_type = keys[4] if len(keys) > 4 else "phosphorylation"

        baseline_int = grp["Signal Intensity"].mean() if (
            "Signal Intensity" in grp.columns and
            grp["Signal Intensity"].notna().any()
        ) else np.nan

        ptm_type_str = str(ptm_type).strip()
        residue_str = str(residue).strip()
        mod_type = PTM_SUBTYPE_MAP.get(
            (ptm_type_str, residue_str), f"{ptm_type_str}_{residue_str}")

        records.append({
            "cell_line": str(cell_line).strip(),
            "gene": str(gene).strip(),
            "ptm_site": str(site).strip(),
            "ptm_residue": residue_str,
            "ptm_type": ptm_type_str,
            "ptm_modification_type": mod_type,
            "baseline_intensity": (round(baseline_int, 4)
                                   if pd.notna(baseline_int) else np.nan),
            "n_replicates": len(grp),
        })

    df_baseline = pd.DataFrame(records)

    print(f"\n  ✓ Baseline PTM summaries: {len(df_baseline):,}")
    if not df_baseline.empty:
        print(f"    Cell lines: {sorted(df_baseline['cell_line'].unique())}")
        print(f"    Genes:      {df_baseline['gene'].nunique()}")
        print(f"    PTM types:  {sorted(df_baseline['ptm_type'].unique())}")
        for cl in sorted(df_baseline["cell_line"].unique()):
            sub = df_baseline[df_baseline["cell_line"] == cl]
            print(f"      {cl:>15s}: {len(sub):>6,} sites across "
                  f"{sub['gene'].nunique()} genes")

    return df_baseline


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SUMMARIES: Aggregate dose-response into per-site summaries
# ══════════════════════════════════════════════════════════════════════════════

def build_hela_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the raw HeLa dose-response rows into one summary row per
    (drug, gene, ptm_site, ptm_residue, ptm_type) combination.

    For each group we capture:
      • baseline_intensity — Signal Intensity at Dosage == 0 (DMSO control)
      • max_dose_intensity — Signal Intensity at the highest dosage
      • log2_fold_change   — log2(max_dose / baseline)
      • EC50, pEC50, curve effect size, R² from curve fitting
      • n_doses — number of dose points measured
      • ptm_modification_type — phospho_S/T/Y or acetyl_K
    """
    print("\n  Building per-site dose-response summaries...")

    if df.empty:
        print("    ⚠ No data to summarise.")
        return pd.DataFrame()

    # Ensure numeric types
    for col in ["Signal Intensity", "Signal Ratio", "Dosage (µg)",
                "Log EC50", "EC50", "pEC50", "Curve effect size", "R2",
                "Curve slope", "Curve top", "Curve bottom"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a unique site label (e.g. "K1499", "S421", "Y204")
    df["ptm_site"] = df["PTM Residue"].astype(str) + df["PTM Index"].astype(str)

    group_cols = ["Chemical Name", "Gene names",
                  "Sequence", "ptm_site", "PTM Residue", "PTM type"]

    records = []
    n_groups = 0

    for keys, grp in df.groupby(group_cols, dropna=False):
        drug, gene, seq, site, residue, ptm_type = keys
        n_groups += 1

        # Baseline (DMSO control, Dosage == 0)
        baseline_rows = grp[grp["Dosage (µg)"] == 0.0]
        baseline_int = (baseline_rows["Signal Intensity"].mean()
                        if not baseline_rows.empty else np.nan)

        # Max dose
        max_dose = grp["Dosage (µg)"].max()
        max_dose_rows = grp[grp["Dosage (µg)"] == max_dose]
        max_dose_int = (max_dose_rows["Signal Intensity"].mean()
                        if not max_dose_rows.empty else np.nan)

        # Log2 fold-change
        if (pd.notna(baseline_int) and baseline_int > 0
                and pd.notna(max_dose_int)):
            log2fc = np.log2((max_dose_int + 0.01) / (baseline_int + 0.01))
        else:
            log2fc = np.nan

        # Curve fit parameters
        ec50 = (grp["EC50"].dropna().iloc[0]
                if "EC50" in grp.columns and grp["EC50"].notna().any()
                else np.nan)
        pec50 = (grp["pEC50"].dropna().iloc[0]
                 if "pEC50" in grp.columns and grp["pEC50"].notna().any()
                 else np.nan)
        eff_size = (grp["Curve effect size"].dropna().iloc[0]
                    if "Curve effect size" in grp.columns
                    and grp["Curve effect size"].notna().any()
                    else np.nan)
        r2 = (grp["R2"].dropna().iloc[0]
              if "R2" in grp.columns and grp["R2"].notna().any()
              else np.nan)
        smiles = (grp["Canonical SMILES"].dropna().iloc[0]
                  if "Canonical SMILES" in grp.columns
                  and grp["Canonical SMILES"].notna().any()
                  else "")

        # Map to modification subtype
        ptm_type_str = str(ptm_type).strip()
        residue_str = str(residue).strip()
        mod_type = PTM_SUBTYPE_MAP.get(
            (ptm_type_str, residue_str), f"{ptm_type_str}_{residue_str}")

        records.append({
            "cell_line": "HeLa",
            "drug_name": str(drug).strip(),
            "protein": str(gene).strip() if pd.notna(gene) else "",
            "ptm_site": str(site).strip(),
            "ptm_residue": residue_str,
            "ptm_type": ptm_type_str,
            "ptm_modification_type": mod_type,
            "peptide_sequence": str(seq).strip() if pd.notna(seq) else "",
            "baseline_intensity": (round(baseline_int, 4)
                                   if pd.notna(baseline_int) else np.nan),
            "max_dose_intensity": (round(max_dose_int, 4)
                                   if pd.notna(max_dose_int) else np.nan),
            "max_dose_ug": max_dose,
            "log2_fold_change": (round(log2fc, 4)
                                 if pd.notna(log2fc) else np.nan),
            "EC50": ec50,
            "pEC50": pec50,
            "curve_effect_size": eff_size,
            "R2": r2,
            "n_doses": int(grp["Dosage (µg)"].nunique()),
            "drug_smiles": str(smiles).strip(),
            "data_source": "drugptm_bench",
        })

        if n_groups % 10_000 == 0:
            print(f"    ... processed {n_groups:,} groups")

    df_summary = pd.DataFrame(records)

    print(f"\n  ✓ {len(df_summary):,} dose-response summaries built")
    if not df_summary.empty:
        print(f"    Drugs:     {sorted(df_summary['drug_name'].unique())}")
        print(f"    PTM types: {sorted(df_summary['ptm_type'].unique())}")
        print(f"    Mod types: {sorted(df_summary['ptm_modification_type'].unique())}")
        print(f"    Genes:     {df_summary['protein'].nunique():,}")
        print(f"    Sites:     {df_summary['ptm_site'].nunique():,}")

        # Per-PTM-type breakdown
        for pt in sorted(df_summary["ptm_type"].unique()):
            sub = df_summary[df_summary["ptm_type"] == pt]
            print(f"\n    {pt}:")
            print(f"      Summaries: {len(sub):,}")
            print(f"      Genes:     {sub['protein'].nunique():,}")
            print(f"      Sites:     {sub['ptm_site'].nunique():,}")
            if sub["log2_fold_change"].notna().any():
                print(f"      log2FC:    {sub['log2_fold_change'].min():+.3f} "
                      f"to {sub['log2_fold_change'].max():+.3f} "
                      f"(mean {sub['log2_fold_change'].mean():+.3f})")

    return df_summary


# ══════════════════════════════════════════════════════════════════════════════
# CATALOG: Build drug catalog from HeLa data
# ══════════════════════════════════════════════════════════════════════════════

def build_drug_catalog(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Build a catalog of drugs with their PTM coverage."""
    if df_summary.empty:
        return pd.DataFrame()

    records = []
    for drug in sorted(df_summary["drug_name"].unique()):
        sub = df_summary[df_summary["drug_name"] == drug]
        records.append({
            "drug_name": drug,
            "cell_line": "HeLa",
            "n_summaries": len(sub),
            "n_genes": sub["protein"].nunique(),
            "n_phospho": len(sub[sub["ptm_type"] == "phosphorylation"]),
            "n_acetyl": len(sub[sub["ptm_type"] == "acetylation"]),
            "ptm_types": ", ".join(sorted(sub["ptm_type"].unique())),
            "drug_smiles": (sub["drug_smiles"].dropna().iloc[0]
                            if sub["drug_smiles"].notna().any() else ""),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(df_summary: pd.DataFrame, df_catalog: pd.DataFrame,
                 df_baseline: pd.DataFrame):
    """Save processed data to CSV."""
    summary_path = OUT_DIR / "hela_hdac_ptm_responses.csv"
    catalog_path = OUT_DIR / "hela_hdac_drug_catalog.csv"
    baseline_path = OUT_DIR / "hela_hdac_baseline_ptm.csv"

    df_summary.to_csv(summary_path, index=False)
    print(f"\n  ✓ Saved: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"    {len(df_summary):,} rows")

    df_catalog.to_csv(catalog_path, index=False)
    print(f"  ✓ Saved: {catalog_path.relative_to(PROJECT_ROOT)}")
    print(f"    {len(df_catalog)} drugs")

    if not df_baseline.empty:
        df_baseline.to_csv(baseline_path, index=False)
        print(f"  ✓ Saved: {baseline_path.relative_to(PROJECT_ROOT)}")
        print(f"    {len(df_baseline):,} baseline PTM sites across "
              f"{df_baseline['cell_line'].nunique()} cell lines")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 05 — HeLa / HDAC Inhibitor Drug-PTM Data Processing     ║")
    print("║  PTM types: phosphorylation (S/T/Y) + acetylation (K) — NEW   ║")
    print("║  Source: DrugPTM-Bench PTM_CellLine_*.csv (all 7 cell lines)  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Part 1: Extract HeLa drug-response data (phospho + acetyl)
    print("\n" + "=" * 70)
    print("PART 1: HeLa Drug-Response PTM Extraction")
    print("=" * 70)

    df_raw = extract_hela_ptm_data()

    if df_raw.empty:
        print("\n  ✗ No HeLa data extracted. Check that the CSV exists.")
        exit(1)

    # Build per-site dose-response summaries
    df_summary = build_hela_summaries(df_raw)

    # Build drug catalog
    df_catalog = build_drug_catalog(df_summary)

    # Part 2: Cross-cell-line baseline PTM scan
    df_baseline = extract_baseline_ptm_all_cell_lines()

    # Save all outputs
    save_outputs(df_summary, df_catalog, df_baseline)

    print("\n✓ Step 05 complete! HeLa phospho+acetyl drug-PTM data + "
          "cross-cell-line baseline ready for harmonization (Step 06).")
