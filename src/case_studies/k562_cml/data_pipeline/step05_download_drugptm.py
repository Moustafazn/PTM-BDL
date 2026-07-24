#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 05 — Process Drug-PTM Data: K562 / CML (BCR-ABL) Case Study         ║
║  (Phosphorylation under TKI + chemotherapy drugs)                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    1. Extract phosphorylation dose-response data from DrugPTM-Bench          ║
║       PTM_CellLine_K562.csv (991 MB, 1,608,421 rows).                       ║
║    2. Scan ALL 7 DrugPTM-Bench cell line files for BASELINE PTM levels      ║
║       (dose=0) of BCR-ABL pathway genes: ABL1, CRKL, STAT5A.               ║
║       This cross-cell-line baseline is used by step06 to build a multi-     ║
║       cell-line dataset with GDSC IC50 labels (~900 cell lines).            ║
║    3. Extract Dasatinib dose-response from A431 (shared drug across K562    ║
║       and A431), providing cross-cell-line drug-response data.              ║
║                                                                              ║
║    K562 is a CML cell line driven by BCR-ABL fusion — a completely          ║
║    different kinase system from EGFR/HER2. It also has non-TKI drugs        ║
║    (Cytarabine, Paclitaxel, Methotrexate) that work through DNA            ║
║    synthesis and microtubule pathways — proving the framework handles       ║
║    diverse drug mechanisms.                                                  ║
║                                                                              ║
║  DATA SOURCE:                                                                ║
║    DrugPTM-Bench — Badkul et al., 2026 (PMID 30394195)                     ║
║    Primary: data/raw/drugptm/30394195/PTM_CellLine_K562.csv                ║
║    Baseline scan: data/raw/drugptm/30394195/PTM_CellLine_*.csv (all 7)     ║
║                                                                              ║
║  DRUGS (5 in K562):                                                          ║
║    Dasatinib     — BCR-ABL/SRC TKI (GDSC ID 1066) — 1,081,759 rows        ║
║    Imatinib      — BCR-ABL TKI (GDSC ID 1003) — 241,582 rows              ║
║    Cytarabine    — nucleoside analog (GDSC ID 1006) — 98,860 rows          ║
║    Paclitaxel    — taxane (GDSC ID 1080) — 98,590 rows                     ║
║    Methotrexat   — antifolate (GDSC ID 1007) — 87,630 rows                 ║
║                                                                              ║
║  CROSS-CELL-LINE DRUG:                                                       ║
║    Dasatinib also tested in A431 — provides cross-cell-line drug response   ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/processed/drugptm/k562_cml_ptm_responses.csv                        ║
║    data/processed/drugptm/k562_cml_drug_catalog.csv                         ║
║    data/processed/drugptm/k562_cml_baseline_ptm.csv                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="k562_cml")

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

PTM_SUBTYPE_MAP = {
    ("phosphorylation", "S"): "phospho_S",
    ("phosphorylation", "T"): "phospho_T",
    ("phosphorylation", "Y"): "phospho_Y",
}

# Target genes for cross-cell-line baseline scan
TARGET_GENES = cfg["project"].get("target_genes_for_baseline", [
    "ABL1", "BCR", "CRKL", "STAT5A", "STAT5B", "SRC", "LYN",
])

# Drugs shared across cell lines (for cross-cell-line drug-response extraction)
SHARED_DRUGS = {"Dasatinib"}  # Dasatinib in A431 + K562
SHARED_DRUG_CELL_LINES = {"A431"}  # Additional cell lines to extract drug-response from


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Extract K562 drug-response PTM data
# ══════════════════════════════════════════════════════════════════════════════

def extract_k562_ptm_data() -> pd.DataFrame:
    """Stream through PTM_CellLine_K562.csv and extract all rows."""
    k562_path = PMID_DIR / "PTM_CellLine_K562.csv"

    if not k562_path.exists():
        print(f"  ✗ File not found: {k562_path}")
        print("    Download from: https://github.com/Xie-lab/DrugPTM-Bench")
        return pd.DataFrame()

    print(f"\n  Reading: {k562_path.name} ({k562_path.stat().st_size / 1e6:.0f} MB)")

    frames: list[pd.DataFrame] = []
    n_total = 0

    for chunk_i, chunk in enumerate(pd.read_csv(k562_path, chunksize=CHUNK_SIZE,
                                                 low_memory=False)):
        n_total += len(chunk)
        avail = [c for c in KEEP_COLUMNS if c in chunk.columns]
        frames.append(chunk[avail])

        if (chunk_i + 1) % 5 == 0:
            print(f"    ... {n_total:,} rows processed")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  ✓ Total rows: {len(df):,}")
    print(f"    Unique drugs: {df['Chemical Name'].nunique()}")
    print(f"    Unique genes: {df['Gene names'].nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 1b: Extract shared-drug data from other cell lines (A431 Dasatinib)
# ══════════════════════════════════════════════════════════════════════════════

def extract_shared_drug_data() -> pd.DataFrame:
    """
    Extract dose-response data for shared drugs (Dasatinib) from additional
    cell lines (A431). This provides cross-cell-line drug response data.

    Dasatinib is tested in both K562 (CML, BCR-ABL+) and A431 (epidermoid,
    EGFR WT), allowing comparison of TKI response across kinase systems.
    """
    print("\n  Extracting shared-drug data from additional cell lines...")

    frames: list[pd.DataFrame] = []

    for cl_name in SHARED_DRUG_CELL_LINES:
        fpath = PMID_DIR / f"PTM_CellLine_{cl_name}.csv"
        if not fpath.exists():
            print(f"    ✗ {fpath.name} not found — skipping")
            continue

        n_total = 0
        n_shared = 0

        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            n_total += len(chunk)

            if "Chemical Name" not in chunk.columns:
                continue

            # Filter for shared drugs only
            mask = chunk["Chemical Name"].astype(str).isin(SHARED_DRUGS)
            shared_chunk = chunk.loc[mask].copy()

            if not shared_chunk.empty:
                avail = [c for c in KEEP_COLUMNS if c in shared_chunk.columns]
                frames.append(shared_chunk[avail])
                n_shared += len(shared_chunk)

        print(f"    {cl_name:>15s}: {n_total:>10,} rows → "
              f"{n_shared:>6,} shared-drug rows")

    if not frames:
        print("    ⚠ No shared-drug data found.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"    Total shared-drug rows: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Cross-cell-line baseline PTM scan
# ══════════════════════════════════════════════════════════════════════════════

def extract_baseline_ptm_all_cell_lines() -> pd.DataFrame:
    """
    Scan ALL 7 DrugPTM-Bench cell line files for baseline (dose=0) PTM
    levels of BCR-ABL pathway genes.

    This is biologically correct because:
      - Baseline ABL1/CRKL/STAT5 phospho levels reflect BCR-ABL signaling
        pathway activity (Shah et al., Science 2004; O'Hare et al., Blood 2005)
      - Cells without BCR-ABL fusion have low ABL1 substrate phosphorylation
      - This difference in baseline PTM state determines TKI sensitivity

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

def build_k562_summaries(df: pd.DataFrame,
                         cell_line_label: str = "K562") -> pd.DataFrame:
    """Aggregate dose-response rows into per-site summaries."""
    print(f"\n  Building per-site dose-response summaries ({cell_line_label})...")

    if df.empty:
        return pd.DataFrame()

    for col in ["Signal Intensity", "Signal Ratio", "Dosage (µg)",
                "Log EC50", "EC50", "pEC50", "Curve effect size", "R2",
                "Curve slope", "Curve top", "Curve bottom"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ptm_site"] = df["PTM Residue"].astype(str) + df["PTM Index"].astype(str)

    # Determine cell line from data if available
    if "Cell Line" in df.columns and df["Cell Line"].notna().any():
        pass  # use per-row cell line
    else:
        df["Cell Line"] = cell_line_label

    group_cols = ["Cell Line", "Chemical Name", "Gene names",
                  "Sequence", "ptm_site", "PTM Residue", "PTM type"]

    records = []
    n_groups = 0

    for keys, grp in df.groupby(group_cols, dropna=False):
        cl, drug, gene, seq, site, residue, ptm_type = keys
        n_groups += 1

        bl = grp[grp["Dosage (µg)"] == 0.0]
        bl_int = bl["Signal Intensity"].mean() if not bl.empty else np.nan
        mx = grp["Dosage (µg)"].max()
        mx_rows = grp[grp["Dosage (µg)"] == mx]
        mx_int = mx_rows["Signal Intensity"].mean() if not mx_rows.empty else np.nan

        if pd.notna(bl_int) and bl_int > 0 and pd.notna(mx_int):
            log2fc = np.log2((mx_int + 0.01) / (bl_int + 0.01))
        else:
            log2fc = np.nan

        ec50 = grp["EC50"].dropna().iloc[0] if grp["EC50"].notna().any() else np.nan
        pec50 = grp["pEC50"].dropna().iloc[0] if grp["pEC50"].notna().any() else np.nan
        eff = (grp["Curve effect size"].dropna().iloc[0]
               if "Curve effect size" in grp.columns and grp["Curve effect size"].notna().any()
               else np.nan)
        r2 = grp["R2"].dropna().iloc[0] if grp["R2"].notna().any() else np.nan
        smiles = (grp["Canonical SMILES"].dropna().iloc[0]
                  if "Canonical SMILES" in grp.columns and grp["Canonical SMILES"].notna().any()
                  else "")

        ptm_type_str = str(ptm_type).strip()
        residue_str = str(residue).strip()
        mod_type = PTM_SUBTYPE_MAP.get((ptm_type_str, residue_str),
                                        f"{ptm_type_str}_{residue_str}")

        records.append({
            "cell_line": str(cl).strip() if pd.notna(cl) else cell_line_label,
            "drug_name": str(drug).strip(),
            "protein": str(gene).strip() if pd.notna(gene) else "",
            "ptm_site": str(site).strip(),
            "ptm_residue": residue_str,
            "ptm_type": ptm_type_str,
            "ptm_modification_type": mod_type,
            "peptide_sequence": str(seq).strip() if pd.notna(seq) else "",
            "baseline_intensity": round(bl_int, 4) if pd.notna(bl_int) else np.nan,
            "max_dose_intensity": round(mx_int, 4) if pd.notna(mx_int) else np.nan,
            "max_dose_ug": mx,
            "log2_fold_change": round(log2fc, 4) if pd.notna(log2fc) else np.nan,
            "EC50": ec50, "pEC50": pec50,
            "curve_effect_size": eff, "R2": r2,
            "n_doses": int(grp["Dosage (µg)"].nunique()),
            "drug_smiles": str(smiles).strip(),
            "data_source": "drugptm_bench",
        })

        if n_groups % 10_000 == 0:
            print(f"    ... processed {n_groups:,} groups")

    df_summary = pd.DataFrame(records)
    print(f"\n  ✓ {len(df_summary):,} dose-response summaries built")
    if not df_summary.empty:
        print(f"    Cell lines: {sorted(df_summary['cell_line'].unique())}")
        print(f"    Drugs:      {sorted(df_summary['drug_name'].unique())}")
        print(f"    Mod types:  {sorted(df_summary['ptm_modification_type'].unique())}")
        print(f"    Genes:      {df_summary['protein'].nunique():,}")
        print(f"    Sites:      {df_summary['ptm_site'].nunique():,}")

        for drug in sorted(df_summary["drug_name"].unique()):
            sub = df_summary[df_summary["drug_name"] == drug]
            fc = sub["log2_fold_change"]
            cl_str = ", ".join(sorted(sub["cell_line"].unique()))
            print(f"\n    {drug} ({cl_str}): {len(sub):,} summaries, "
                  f"log2FC {fc.min():+.3f} to {fc.max():+.3f} "
                  f"(mean {fc.mean():+.3f})")

    return df_summary


def build_drug_catalog(df_summary: pd.DataFrame) -> pd.DataFrame:
    if df_summary.empty:
        return pd.DataFrame()
    records = []
    for drug in sorted(df_summary["drug_name"].unique()):
        sub = df_summary[df_summary["drug_name"] == drug]
        records.append({
            "drug_name": drug,
            "cell_lines": ", ".join(sorted(sub["cell_line"].unique())),
            "n_summaries": len(sub),
            "n_genes": sub["protein"].nunique(),
            "ptm_types": ", ".join(sorted(sub["ptm_type"].unique())),
            "drug_smiles": (sub["drug_smiles"].dropna().iloc[0]
                            if sub["drug_smiles"].notna().any() else ""),
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 05 — K562 / CML Drug-PTM Data Processing                ║")
    print("║  Drugs: Dasatinib, Imatinib + Cytarabine, Paclitaxel, MTX     ║")
    print("║  Source: DrugPTM-Bench PTM_CellLine_*.csv (all 7 cell lines)  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Part 1a: Extract K562 drug-response data
    print("\n" + "=" * 70)
    print("PART 1a: K562 Drug-Response PTM Extraction")
    print("=" * 70)

    df_raw = extract_k562_ptm_data()
    if df_raw.empty:
        print("\n  ✗ No K562 data. Check CSV exists.")
        exit(1)

    df_k562_summary = build_k562_summaries(df_raw, cell_line_label="K562")

    # Part 1b: Extract shared-drug data from A431 (Dasatinib)
    print("\n" + "=" * 70)
    print("PART 1b: Shared-Drug Cross-Cell-Line Extraction")
    print("=" * 70)

    df_shared = extract_shared_drug_data()
    if not df_shared.empty:
        df_shared_summary = build_k562_summaries(df_shared)
        # Merge K562 + A431 summaries
        df_summary = pd.concat([df_k562_summary, df_shared_summary],
                               ignore_index=True)
    else:
        df_summary = df_k562_summary

    df_catalog = build_drug_catalog(df_summary)

    # Part 2: Cross-cell-line baseline PTM scan
    df_baseline = extract_baseline_ptm_all_cell_lines()

    # Save all outputs
    summary_path = OUT_DIR / "k562_cml_ptm_responses.csv"
    catalog_path = OUT_DIR / "k562_cml_drug_catalog.csv"
    baseline_path = OUT_DIR / "k562_cml_baseline_ptm.csv"

    df_summary.to_csv(summary_path, index=False)
    df_catalog.to_csv(catalog_path, index=False)
    print(f"\n  ✓ Saved: {summary_path.relative_to(PROJECT_ROOT)} ({len(df_summary):,} rows)")
    print(f"  ✓ Saved: {catalog_path.relative_to(PROJECT_ROOT)} ({len(df_catalog)} drugs)")

    if not df_baseline.empty:
        df_baseline.to_csv(baseline_path, index=False)
        print(f"  ✓ Saved: {baseline_path.relative_to(PROJECT_ROOT)} "
              f"({len(df_baseline):,} baseline sites across "
              f"{df_baseline['cell_line'].nunique()} cell lines)")

    print("\n✓ Step 05 complete! K562 drug-PTM data + cross-cell-line "
          "baseline ready for harmonization (Step 06).")
