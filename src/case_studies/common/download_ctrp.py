#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Cross-Dataset Testing: Process CTRPv2 Drug Response Data (Reviewer Q4)     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Process Cancer Therapeutics Response Portal v2 (CTRPv2) drug response    ║
║    data for cross-dataset generalization testing.                            ║
║    Train on GDSC2 → Test on CTRPv2 (or vice versa).                        ║
║                                                                              ║
║  DATA SOURCE:                                                                ║
║    ORCESTRA PharmacoGx: https://orcestra.ca/pset/canonical                  ║
║    → CTRPv2_2015 PharmacoSet (PSet_CTRPv2.rds, ~40 MB)                     ║
║    Original: Basu et al., Cell 2013 (PMID 23993102)                        ║
║    Updated: Rees et al., Nat Chem Biol 2016 (PMID 26656090) — CTRPv2      ║
║                                                                              ║
║  PREREQUISITE (one-time):                                                    ║
║    The PSet_CTRPv2.rds file is a PharmacoSet R object. It must first be    ║
║    extracted to flat files using the R script:                               ║
║                                                                              ║
║      Rscript src/case_studies/common/extract_ctrp_from_rds.R               ║
║                                                                              ║
║    This produces:                                                            ║
║      data/raw/ctrp/ctrp_sensitivity_summary.csv  (pre-merged, Python-ready)║
║      data/raw/ctrp/v20.data.curves_post_qc.txt   (raw curves)             ║
║      data/raw/ctrp/v20.meta.per_cell_line.txt     (cell line metadata)     ║
║      data/raw/ctrp/v20.meta.per_compound.txt      (compound metadata)      ║
║                                                                              ║
║  OVERLAPPING DRUGS (CTRPv2 ↔ our case studies):                             ║
║    CS1: Erlotinib, Gefitinib, Lapatinib, Afatinib                          ║
║    CS2: Vorinostat (SAHA)                                                    ║
║    CS3: Imatinib, Dasatinib, Paclitaxel, Cytarabine, Methotrexate         ║
║                                                                              ║
║  USAGE:                                                                      ║
║    python -m src.case_studies.common.download_ctrp                           ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/processed/ctrp/ctrp_drug_responses.csv                              ║
║    data/processed/ctrp/ctrp_cell_line_map.json                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ctrp"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ctrp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Drug name mapping: CTRPv2 name (lower) → our canonical name
CTRP_DRUG_MAP = {
    # CS1: EGFR/ERBB2 TKIs
    "erlotinib": "Erlotinib",
    "gefitinib": "Gefitinib",
    "lapatinib": "Lapatinib",
    "afatinib": "Afatinib",
    # CS2: HDAC inhibitors
    "vorinostat": "Vorinostat",
    "saha": "Vorinostat",
    "romidepsin": "Romidepsin",
    # CS3: BCR-ABL TKIs + chemo
    "imatinib": "Imatinib",
    "dasatinib": "Dasatinib",
    "paclitaxel": "Paclitaxel",
    "cytarabine": "Cytarabine",
    "methotrexate": "Methotrexat",
    "methotrexat": "Methotrexat",
}


def normalize_cell_line_name(name):
    """Normalize cell line names for cross-database matching."""
    if pd.isna(name):
        return name
    name = str(name).upper().strip()
    for prefix in ["NCI-", "NCI_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("-", "").replace(" ", "")


def process_from_summary_csv():
    """
    Process CTRPv2 from the pre-merged summary CSV (produced by R extraction).

    This is the PRIMARY path — reads the ctrp_sensitivity_summary.csv that
    the R script produces from the PharmacoSet RDS object.
    """
    summary_path = RAW_DIR / "ctrp_sensitivity_summary.csv"
    if not summary_path.exists():
        return None

    print(f"  Loading pre-merged summary: {summary_path}")
    df = pd.read_csv(summary_path)
    print(f"    Total records: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # Normalize drug names for matching
    if "drug_name" not in df.columns:
        for col in ["drugid", "cpd_name", "compound"]:
            if col in df.columns:
                df = df.rename(columns={col: "drug_name"})
                break

    if "cell_line_name" not in df.columns:
        for col in ["cellid", "ccl_name", "cell_line"]:
            if col in df.columns:
                df = df.rename(columns={col: "cell_line_name"})
                break

    if "drug_name" not in df.columns or "cell_line_name" not in df.columns:
        print(f"    Could not find drug/cell columns. Available: {list(df.columns)}")
        return None

    df["drug_name_lower"] = df["drug_name"].astype(str).str.lower().str.strip()
    df["cell_line_norm"] = df["cell_line_name"].apply(normalize_cell_line_name)

    # Filter to our drugs of interest
    mask = df["drug_name_lower"].isin(set(CTRP_DRUG_MAP.keys()))
    df_filtered = df[mask].copy()
    df_filtered["canonical_drug_name"] = df_filtered["drug_name_lower"].map(
        CTRP_DRUG_MAP)

    print(f"    Filtered to our drugs: {len(df_filtered)} records")
    for drug in sorted(df_filtered["canonical_drug_name"].dropna().unique()):
        n = (df_filtered["canonical_drug_name"] == drug).sum()
        print(f"      {drug}: {n} records")

    return df_filtered


def process_from_flat_files():
    """
    Process CTRPv2 from flat tab-separated files (legacy or R-extracted).
    """
    curves_path = RAW_DIR / "v20.data.curves_post_qc.txt"
    cells_path = RAW_DIR / "v20.meta.per_cell_line.txt"
    compounds_path = RAW_DIR / "v20.meta.per_compound.txt"

    missing = [p.name for p in [curves_path, cells_path, compounds_path]
               if not p.exists()]
    if missing:
        return None

    print(f"  Loading flat files...")
    df_curves = pd.read_csv(curves_path, sep="\t")
    df_cells = pd.read_csv(cells_path, sep="\t")
    df_compounds = pd.read_csv(compounds_path, sep="\t")
    print(f"    Curves: {len(df_curves)}, Cells: {len(df_cells)}, "
          f"Compounds: {len(df_compounds)}")

    if "cellid" in df_curves.columns and "drugid" in df_curves.columns:
        df = df_curves.copy()
        df = df.rename(columns={"cellid": "cell_line_name",
                                 "drugid": "drug_name"})
    else:
        experiments_path = RAW_DIR / "v20.meta.per_experiment.txt"
        if experiments_path.exists():
            df_exp = pd.read_csv(experiments_path, sep="\t")
            df = df_curves.merge(
                df_exp[["experiment_id", "master_ccl_id", "master_cpd_id"]],
                on="experiment_id", how="left"
            )
            df = df.merge(
                df_cells[["master_ccl_id", "ccl_name"]].drop_duplicates(),
                on="master_ccl_id", how="left"
            )
            df = df.merge(
                df_compounds[["master_cpd_id", "cpd_name"]].drop_duplicates(),
                on="master_cpd_id", how="left"
            )
            df = df.rename(columns={"ccl_name": "cell_line_name",
                                     "cpd_name": "drug_name"})
        else:
            print("    Cannot merge: no experiment metadata or cellid/drugid")
            return None

    df["drug_name_lower"] = df["drug_name"].astype(str).str.lower().str.strip()
    df["cell_line_norm"] = df["cell_line_name"].apply(normalize_cell_line_name)

    mask = df["drug_name_lower"].isin(set(CTRP_DRUG_MAP.keys()))
    df_filtered = df[mask].copy()
    df_filtered["canonical_drug_name"] = df_filtered["drug_name_lower"].map(
        CTRP_DRUG_MAP)

    print(f"    Filtered to our drugs: {len(df_filtered)} records")
    return df_filtered


def compute_response_metrics(df):
    """Compute IC50-equivalent and resistance labels from CTRPv2 data."""
    auc_col = None
    for col in ["auc_recomputed", "auc_published", "area_under_curve", "AUC"]:
        if col in df.columns:
            auc_col = col
            break

    ic50_col = None
    for col in ["ic50_recomputed", "ic50_published",
                "apparent_ec50_umol", "ec50"]:
        if col in df.columns:
            ic50_col = col
            break

    if ic50_col:
        df["ln_ic50_ctrp"] = np.log(
            pd.to_numeric(df[ic50_col], errors="coerce").clip(lower=1e-6))

    if auc_col:
        df["auc_ctrp"] = pd.to_numeric(df[auc_col], errors="coerce")
        median_auc = df["auc_ctrp"].median()
        df["resistance_label_ctrp"] = (df["auc_ctrp"] > median_auc).astype(int)
        print(f"    AUC column: {auc_col}, median={median_auc:.3f}")

    return df


def process_ctrp_data():
    """Main entry: process CTRPv2 data."""
    print("=" * 62)
    print("  Processing CTRPv2 Drug Response Data (Reviewer Q4)")
    print("=" * 62)

    rds_path = RAW_DIR / "PSet_CTRPv2.rds"
    summary_path = RAW_DIR / "ctrp_sensitivity_summary.csv"

    if rds_path.exists() and not summary_path.exists():
        print(f"\n  Found PSet_CTRPv2.rds but no extracted CSVs.")
        print(f"  Run the R extraction script first:")
        print(f"    Rscript src/case_studies/common/extract_ctrp_from_rds.R")
        return None

    if not rds_path.exists() and not summary_path.exists():
        print(f"\n  No CTRPv2 data found in {RAW_DIR}/")
        print(f"\n  Download instructions:")
        print(f"    1. Go to: https://orcestra.ca/pset/canonical")
        print(f"    2. Find 'CTRPv2' -> click 'CTRPv2_2015' -> Download")
        print(f"    3. Save as: data/raw/ctrp/PSet_CTRPv2.rds")
        print(f"    4. Rscript src/case_studies/common/extract_ctrp_from_rds.R")
        return None

    df = process_from_summary_csv()
    if df is None:
        df = process_from_flat_files()
    if df is None:
        print("  Could not process CTRPv2 data.")
        return None

    print("\n  Computing response metrics...")
    df = compute_response_metrics(df)

    # Save
    out_cols = ["cell_line_name", "cell_line_norm", "canonical_drug_name",
                "drug_name"]
    for c in ["ln_ic50_ctrp", "auc_ctrp", "resistance_label_ctrp"]:
        if c in df.columns:
            out_cols.append(c)

    existing_cols = [c for c in out_cols if c in df.columns]
    df_out = df[existing_cols].copy()
    df_out = df_out.rename(columns={"canonical_drug_name": "drug_name_canonical"})

    out_path = OUT_DIR / "ctrp_drug_responses.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(df_out)} records)")

    cell_map = {}
    for _, row in df_out.drop_duplicates("cell_line_norm").iterrows():
        cell_map[row["cell_line_norm"]] = {"ctrp_name": row["cell_line_name"]}
    map_path = OUT_DIR / "ctrp_cell_line_map.json"
    with open(map_path, "w") as f:
        json.dump(cell_map, f, indent=2)
    print(f"  Saved: {map_path} ({len(cell_map)} cell lines)")

    print(f"\n  Summary:")
    print(f"    Total CTRPv2 records (our drugs): {len(df_out)}")
    print(f"    Cell lines: {df_out['cell_line_norm'].nunique()}")
    drugs_col = ("drug_name_canonical"
                 if "drug_name_canonical" in df_out.columns else "drug_name")
    print(f"    Drugs: {sorted(df_out[drugs_col].dropna().unique())}")

    return df_out


if __name__ == "__main__":
    process_ctrp_data()
