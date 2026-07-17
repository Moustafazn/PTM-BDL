#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 06 — Data Harmonization: Build Unified Multimodal Dataset             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THIS IS THE HARDEST AND MOST IMPORTANT STEP IN THE ENTIRE PIPELINE.        ║
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Merge all heterogeneous data sources (Steps 01-05) into a single,         ║
║    unified multimodal dataset where each sample contains:                    ║
║      • Cell line identity + drug identity                                    ║
║      • EGFR mutation profile → determines which protein sequence            ║
║      • Matched PDB structure → 3D conformation for that mutation state      ║
║      • PTM state vector → phosphorylation levels at key sites               ║
║      • Drug-PTM phospho features → real experimental phospho changes        ║
║      • Drug SMILES → chemical structure                                     ║
║      • IC50 response → ground-truth resistance label                        ║
║                                                                              ║
║  WHY THIS IS HARD:                                                           ║
║    All data sources use DIFFERENT identifiers, DIFFERENT formats, and        ║
║    come from DIFFERENT experiments:                                          ║
║    • GDSC uses COSMIC IDs + cell line names for drug response                ║
║    • CCLE/DepMap uses ModelIDs for mutations                                 ║
║    • PDB structures map to specific mutation combinations                    ║
║    • PTM data maps to sequence positions, not cell lines                     ║
║    • Drug-PTM has 5 sources with different schemas and contexts              ║
║    • Drug names vary: "Osimertinib" vs "AZD9291" vs drug ID 2156           ║
║                                                                              ║
║    The biggest scientific contribution may be creating a biologically        ║
║    coherent multimodal dataset for resistance modeling.                      ║
║                                                                              ║
║  DATA GAP FIX:                                                               ║
║    PC-9 and HCC827 — the most important EGFR-mutant NSCLC cell lines for   ║
║    Osimertinib resistance research — are present in our Drug-PTM data        ║
║    (Tozuka 2024, Hsu 2025) but are MISSING from GDSC. This script adds      ║
║    well-characterized literature IC50 values (Cross et al., Cancer Discov    ║
║    2014, PMID 25351743) so they can be included in the model.               ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/processed/multimodal_dataset.csv      (final unified table)          ║
║    data/processed/dataset_summary.json        (statistics)                   ║
║                                                                              ║
║  EACH ROW IN THE OUTPUT REPRESENTS:                                          ║
║    One (cell_line, drug) pair with ALL modalities linked:                    ║
║    ┌──────────────────────────────────────────────────────────────────────┐ ║
║    │ cell_line | drug | mutations | sequence_id | pdb_id | ptm_vector |  │ ║
║    │ phospho_features | smiles | ln_IC50 | resistance_label              │ ║
║    └──────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"]
PROCESSED_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"]
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Cell-Line and Drug Name Normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_cell_line_name(name):
    """
    Standardize cell line names across databases.

    Different databases use different naming conventions for the same cell line:
      GDSC:  "NCI-H1975"    CCLE:  "H1975"    Literature:  "H-1975"

    This function normalizes to uppercase, removes "NCI-" prefix and hyphens.
    """
    if pd.isna(name):
        return name
    name = str(name).upper().strip()
    for prefix in ["NCI-", "NCI_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("-", "")
    return name


def normalize_drug_name(name):
    """Normalize drug names for matching across datasets."""
    if pd.isna(name):
        return name
    return str(name).strip().lower()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Load All Data Sources
# ══════════════════════════════════════════════════════════════════════════════

def load_gdsc_responses():
    """
    Load GDSC drug response data (from Step 01) — ERBB family (EGFR + HER2).

    Step 01 now outputs the ERBB family file with target_protein column,
    containing BOTH NSCLC (EGFR) and breast cancer (ERBB2) cell lines.
    
    KEY FINDINGS (Section 7a HER2_EXPANSION_PLAN.md):
    - 943 records: 638 EGFR (NSCLC) + 305 ERBB2 (breast)
    - ALL EGFR drugs also tested on 52 breast cancer cell lines
    - Sapitinib replaced Neratinib (not in GDSC2)
    - target_protein column determines EGFR vs ERBB2 context

    Returns DataFrame with columns:
    - CELL_LINE_NAME: cell line identifier (used for cross-referencing)
    - DRUG_NAME: drug name
    - DRUG_ID: GDSC drug ID
    - target_protein: "EGFR" or "ERBB2"
    - LN_IC50: natural log of IC50 (our regression target)
    - AUC: area under dose-response curve (alternative target)
    - resistance_label: binary classification (0=sensitive, 1=resistant)
    """
    print("  Loading GDSC drug response data (ERBB family: EGFR + HER2)...")

    # Try ERBB family file first (from HER2 expansion)
    erbb_path = PROCESSED_DIR / "gdsc" / "gdsc_erbb_tki_responses.csv"
    nsclc_path = PROCESSED_DIR / "gdsc" / "gdsc_nsclc_egfr_tki_responses.csv"

    if erbb_path.exists():
        df = pd.read_csv(erbb_path)
        # Ensure target_protein column exists
        if "target_protein" not in df.columns:
            df["target_protein"] = "EGFR"
        n_egfr = len(df[df["target_protein"] == "EGFR"])
        n_erbb2 = len(df[df["target_protein"] == "ERBB2"])
        print(f"    ✓ Loaded ERBB family responses: {len(df)} records "
              f"(EGFR={n_egfr}, ERBB2={n_erbb2})")
    elif nsclc_path.exists():
        df = pd.read_csv(nsclc_path)
        df["target_protein"] = "EGFR"  # backward compat
        print(f"    ✓ Loaded NSCLC-only responses: {len(df)} records (EGFR only)")
    else:
        print("    ✗ GDSC data not found. Run step01 first.")
        print("    → Creating minimal placeholder for development...")
        df = create_placeholder_gdsc()
        df["target_protein"] = "EGFR"

    return df


def load_mutation_profiles():
    """
    Load EGFR mutation profiles per cell line (from Step 02).

    Returns DataFrame mapping cell_line → EGFR mutation profile.
    This is critical because the mutation profile determines:
    1. Which mutant sequence to use for ESM-2
    2. Which PDB structure best represents the protein
    3. What PTM state to expect
    """
    print("  Loading mutation profiles...")

    path = PROCESSED_DIR / "ccle" / "egfr_mutations_by_cell_line.csv"

    if path.exists():
        df = pd.read_csv(path)
        # Normalize column names from Step 02 output
        if "CellLineName" in df.columns and "cell_line" not in df.columns:
            df = df.rename(columns={"CellLineName": "cell_line"})
        if "mutation_class" in df.columns and "mutation_classes" not in df.columns:
            df = df.rename(columns={"mutation_class": "mutation_classes"})
        print(f"    ✓ Loaded mutation profiles: {len(df)} cell lines")
    else:
        print("    ✗ Mutation data not found. Run step02 first.")
        df = create_placeholder_mutations()

    return df


def load_ptm_data():
    """
    Load PTM state vectors (from Step 04).

    Returns dict mapping mutation_background → {position: phospho_level}
    """
    print("  Loading PTM state vectors...")

    vectors_path = PROCESSED_DIR / "ptm" / "egfr_ptm_state_vectors.json"
    sites_path = PROCESSED_DIR / "ptm" / "egfr_phosphorylation_sites.csv"

    ptm_vectors = {}
    ptm_sites = None

    if vectors_path.exists():
        with open(vectors_path) as f:
            ptm_vectors = json.load(f)
        print(f"    ✓ Loaded PTM vectors for {len(ptm_vectors)} backgrounds")

    if sites_path.exists():
        ptm_sites = pd.read_csv(sites_path)
        print(f"    ✓ Loaded {len(ptm_sites)} phosphorylation sites")

    if not ptm_vectors:
        print("    ✗ PTM data not found. Run step04 first.")
        ptm_vectors = create_placeholder_ptm_vectors()

    return ptm_vectors, ptm_sites


def load_drugptm_data():
    """
    Load drug-induced PTM measurements (from Step 05).

    Returns DataFrame with per-site PTM measurements covering BOTH
    phosphorylation and N-glycosylation for BOTH EGFR and ERBB2.

    PTM-BDL multi-PTM expansion (2026-06-28):
      Step 05 now writes a single unified file
      `data/processed/drugptm/drugptm_multiptm_responses.csv`
      that contains the PTM-BDL schema columns:
          target_protein        ∈ {EGFR, ERBB2}
          ptm_modification_type ∈ {phospho_Y, phospho_S, phospho_T,
                                    phospho_other, glyco_N}
      We prefer this file when present so the glyco rows are available
      to the rest of step06.  The legacy phospho-only CSV
      `drugptm_egfr_phospho_responses.csv` is kept as a fallback for
      back-compat with older step05 runs.

    Sources merged here:
      Phospho (A–I) — DrugPTM-Bench, Tozuka 2024, Hsu 2025, PNAS 2025,
                       FEBS 2025, Cancer Res 2021, MCP 2025, Remsing Rix
                       2022, Ruprecht 2017.
      Glyco   (J–N) — MCP 2025 / MCP 2025b GP rows, ErbB2 Glycoform Atlas
                       (Taniguchi 2024), ST6Gal1→ErbB2 (Garnham 2021),
                       EGFR Fucosylation (Sethi 2020).
    """
    print("  Loading Drug-PTM response data...")

    new_path = PROCESSED_DIR / "drugptm" / "drugptm_multiptm_responses.csv"
    legacy_path = PROCESSED_DIR / "drugptm" / "drugptm_egfr_phospho_responses.csv"

    if new_path.exists():
        df = pd.read_csv(new_path)
        sources = df["data_source"].value_counts()
        n_phospho = int((df.get("ptm_modification_type",
                                pd.Series(dtype=str))
                           .astype(str)
                           .str.startswith("phospho")).sum())
        n_glyco = int((df.get("ptm_modification_type",
                              pd.Series(dtype=str))
                         .astype(str) == "glyco_N").sum())
        print(f"    ✓ Loaded unified multi-PTM data: {len(df)} records "
              f"({n_phospho} phospho + {n_glyco} glyco)")
        for src, count in sources.items():
            print(f"      {src}: {count} rows")
        return df

    if legacy_path.exists():
        df = pd.read_csv(legacy_path)
        sources = df["data_source"].value_counts()
        print(f"    ⚠ Loaded LEGACY phospho-only Drug-PTM data: "
              f"{len(df)} records")
        print(f"      (re-run step05 to enable the PTM-BDL glyco branch)")
        for src, count in sources.items():
            print(f"      {src}: {count} rows")
        return df

    print("    ✗ DrugPTM data not found. Run step05 first.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1b: Per-Cell-Line PTM Modulators (added 2026-06-28)
# ══════════════════════════════════════════════════════════════════════════════
# These three helpers fix the failure mode documented in
# results/COMPREHENSIVE_EVALUATION_28_june.md §1.1:
#
#   • Failure 1: Randomized PTM control: shuffled > real
#       (drop_BAcc=-0.042, drop_AUROC=-0.010)
#   • Failure 2: Mutation-group prediction collapse — all 8 EGFR mutation
#       groups predict ~0.186 because only ~5 unique PTM vectors exist
#       across 951 samples.
#
# ROOT CAUSE (verified empirically on 2026-06-28 dataset):
#   df[ptm_*].drop_duplicates().shape == (5, 12)
#   df[delta_ptm_*].drop_duplicates().shape == (24, 12)
#   → PTM input is deterministic f(mutation, drug), redundant with sequence
#     and drug embeddings; the model can / does ignore it.
#
# FIX STRATEGY (Solutions A + B + C of the proposal):
#   (A) For 38 high-confidence samples with measured per-site log2FC,
#       use the actual measurements (overrides mutation-class baseline).
#   (B) For ~900 samples without measurements, apply per-cell-line
#       biological modulators (KRAS / MET / PIK3CA / TP53 / PTEN / tissue)
#       loaded from CCLE somatic mutations + model metadata.
#   (C) Differentiate ERBB2 samples by HER2-amp tier + co-mutation status.
#
# Each modulator magnitude is tied to a published PMID — see
# config['ptm_modulators'].
# ══════════════════════════════════════════════════════════════════════════════

# Module-level caches (loaded once on first call)
_CELL_LINE_COMUTATIONS = None
_MEASURED_PTM_LOOKUP = None
_MEASURED_DELTA_LOOKUP = None


def load_cell_line_comutations():
    """
    Build per-cell-line co-mutation + tissue + HER2-amp context dict.

    Reads:
      • data/raw/ccle/ccle_somatic_mutations.csv  (huge — streamed)
      • data/raw/ccle/ccle_model_info.csv         (cell-line metadata)

    Returns dict keyed by NORMALIZED cell-line name. Value:
      {
        'ModelID':           str,
        'tissue':            str  (OncotreeLineage; lower-cased),
        'subtype':           str  (OncotreePrimaryDisease; lower-cased),
        'engineered_details':str  (EngineeredModelDetails; lower-cased),
        'kras_activating':   bool (G12/G13/Q61 in KRAS),
        'pik3ca_activating': bool (E542K/E545K/H1047R in PIK3CA),
        'tp53_lof':          bool (any nonsense/frameshift/missense in TP53),
        'pten_loss':         bool (any nonsense/frameshift/missense in PTEN),
        'met_amplified':     bool (in curated met_amplified_lines OR
                                   EngineeredModelDetails contains "MET amplified"),
        'her2_amp_tier':     str  ('high'/'intermediate'/'baseline'),
        'er_status':         str  ('positive'/'negative'/'unknown')
                                   (proxy from OncotreeSubtype luminal vs basal),
      }

    Only mutation rows with HugoSymbol in {KRAS, PIK3CA, TP53, PTEN} are
    examined — keeps memory + time tractable on the 595 MB mutation table.
    """
    global _CELL_LINE_COMUTATIONS
    if _CELL_LINE_COMUTATIONS is not None:
        return _CELL_LINE_COMUTATIONS

    print("  Loading per-cell-line co-mutation context from CCLE...")

    model_path = RAW_DIR / "ccle" / "ccle_model_info.csv"
    mut_path = RAW_DIR / "ccle" / "ccle_somatic_mutations.csv"
    if not model_path.exists() or not mut_path.exists():
        print("    ⚠ CCLE raw files not found — falling back to empty context")
        _CELL_LINE_COMUTATIONS = {}
        return _CELL_LINE_COMUTATIONS

    # ── 1. Load cell-line metadata ────────────────────────────────────────
    cols_needed = [
        "ModelID", "CellLineName", "StrippedCellLineName",
        "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
        "EngineeredModelDetails", "CCLEName",
    ]
    df_models = pd.read_csv(model_path, usecols=lambda c: c in cols_needed)
    df_models = df_models.fillna("")
    print(f"    ✓ Loaded {len(df_models)} CCLE models")

    # Build modelID → metadata map and also build the cell-line-name index
    modelid_to_meta = {}
    name_to_modelid = {}
    for _, row in df_models.iterrows():
        mid = row["ModelID"]
        tissue = str(row.get("OncotreeLineage", "")).lower()
        subtype = str(row.get("OncotreePrimaryDisease", "")).lower()
        ot_subtype = str(row.get("OncotreeSubtype", "")).lower()
        eng = str(row.get("EngineeredModelDetails", "")).lower()

        # ER status proxy from OncotreeSubtype (luminal/lumA/lumB → +,
        # "triple negative" or "basal" → −).  Best-effort; many gaps.
        er = "unknown"
        if any(t in ot_subtype for t in
               ["luminal", "lumb", "luma", "lum a", "lum b", "er+", "er positive"]):
            er = "positive"
        elif any(t in ot_subtype for t in
                 ["triple negative", "basal", "tnbc"]):
            er = "negative"

        meta = {
            "ModelID": mid,
            "tissue": tissue,
            "subtype": subtype,
            "engineered_details": eng,
            "kras_activating": False,
            "pik3ca_activating": False,
            "tp53_lof": False,
            "pten_loss": False,
            "met_amplified": "met amplif" in eng or "met-amplif" in eng,
            "her2_amp_tier": "baseline",
            "er_status": er,
        }
        modelid_to_meta[mid] = meta

        # Multiple names per cell line → all index to same ModelID
        for name_field in ["CellLineName", "StrippedCellLineName", "CCLEName"]:
            n = str(row.get(name_field, ""))
            if n:
                norm = normalize_cell_line_name(n)
                if norm:
                    name_to_modelid[norm] = mid

    # ── 2. Stream the somatic mutations CSV, keep only rows of interest ──
    interesting_genes = {"KRAS", "PIK3CA", "TP53", "PTEN"}
    activating_kras = {"G12", "G13", "Q61"}     # prefix match on ProteinChange
    activating_pik3ca = {"E542K", "E545K", "H1047R"}
    damaging_consequences = (
        "missense_variant", "stop_gained", "frameshift_variant",
        "splice_acceptor_variant", "splice_donor_variant",
        "start_lost", "stop_lost",
    )

    n_processed = 0
    chunk_iter = pd.read_csv(
        mut_path, chunksize=200_000,
        usecols=lambda c: c in {
            "ModelID", "HugoSymbol", "ProteinChange",
            "VariantInfo", "VepImpact", "LikelyLoF",
        },
        dtype=str, low_memory=True,
    )
    for chunk in chunk_iter:
        sub = chunk[chunk["HugoSymbol"].isin(interesting_genes)]
        for _, row in sub.iterrows():
            mid = row.get("ModelID", "")
            if mid not in modelid_to_meta:
                continue
            gene = row["HugoSymbol"]
            pc = str(row.get("ProteinChange", "")).replace("p.", "")
            vi = str(row.get("VariantInfo", "")).lower()
            impact = str(row.get("VepImpact", "")).upper()
            llof = str(row.get("LikelyLoF", "")).lower()

            if gene == "KRAS":
                # G12X, G13X, Q61X are well-established activating mutations
                if any(pc.startswith(p) for p in activating_kras):
                    modelid_to_meta[mid]["kras_activating"] = True

            elif gene == "PIK3CA":
                if pc in activating_pik3ca:
                    modelid_to_meta[mid]["pik3ca_activating"] = True
                # Also flag if E545X / H1047X (hotspots)
                elif pc.startswith("E545") or pc.startswith("H1047") \
                        or pc.startswith("E542"):
                    modelid_to_meta[mid]["pik3ca_activating"] = True

            elif gene == "TP53":
                # Loss-of-function: nonsense, frameshift, splice, damaging missense.
                # Even missense in TP53 is dominant-negative.
                if any(c in vi for c in damaging_consequences) or impact == "HIGH":
                    modelid_to_meta[mid]["tp53_lof"] = True
                elif "missense" in vi:
                    modelid_to_meta[mid]["tp53_lof"] = True
                elif llof in ("yes", "true", "1"):
                    modelid_to_meta[mid]["tp53_lof"] = True

            elif gene == "PTEN":
                if any(c in vi for c in damaging_consequences) or impact == "HIGH":
                    modelid_to_meta[mid]["pten_loss"] = True
                elif llof in ("yes", "true", "1"):
                    modelid_to_meta[mid]["pten_loss"] = True

        n_processed += len(chunk)

    print(f"    ✓ Streamed {n_processed:,} mutation rows (kept "
          f"{sum(1 for m in modelid_to_meta.values() if any([m['kras_activating'], m['pik3ca_activating'], m['tp53_lof'], m['pten_loss']]))} "
          f"cell lines with ≥1 driver co-mutation)")

    # ── 3. Apply curated HER2-amp + MET-amp lists from config ────────────
    modulators_cfg = cfg.get("ptm_modulators", {})
    her2_tiers = modulators_cfg.get("her2_amp_tiers", {})
    met_lines = modulators_cfg.get("met_amplified_lines", [])

    high_set = {normalize_cell_line_name(n) for n in her2_tiers.get("high", [])}
    intermediate_set = {normalize_cell_line_name(n)
                        for n in her2_tiers.get("intermediate", [])}
    met_set = {normalize_cell_line_name(n) for n in met_lines}

    n_high = n_inter = n_met = 0
    for norm_name, mid in name_to_modelid.items():
        meta = modelid_to_meta.get(mid)
        if not meta:
            continue
        if norm_name in high_set:
            meta["her2_amp_tier"] = "high"
            n_high += 1
        elif norm_name in intermediate_set:
            meta["her2_amp_tier"] = "intermediate"
            n_inter += 1
        if norm_name in met_set:
            meta["met_amplified"] = True
            n_met += 1

    # ── 4. Build the public dict keyed by normalized cell-line name ──────
    out = {}
    for norm_name, mid in name_to_modelid.items():
        meta = modelid_to_meta.get(mid)
        if meta is not None:
            out[norm_name] = meta

    print(f"    ✓ Built co-mutation context for {len(out)} normalized cell lines")
    print(f"      HER2-amp high:        {n_high}")
    print(f"      HER2-amp intermediate:{n_inter}")
    print(f"      MET amplified:        {n_met}")

    _CELL_LINE_COMUTATIONS = out
    return out


def build_measured_ptm_lookup(df_drugptm):
    """
    Build two lookup dicts from high-confidence per-site phospho measurements.

    Returns (baseline_lookup, delta_lookup).

    baseline_lookup[(cell_line_norm, gene)][position] = relative_log2_intensity
        — For PARENTAL/UNTREATED ratios (Tozuka 2024, ruprecht_2017 baselines).
          When present, these override the mutation-class baseline ptm_vector
          for that cell line.

    delta_lookup[(cell_line_norm, drug_name_norm, gene)][position] = log2FC
        — Drug-induced fold change per site.  Overrides delta_ptm scaling
          for the specific (cell_line, drug) combinations measured.

    SOURCES feeding the delta_lookup:
        • pnas_2025      H1975, HCC4006 × Osimertinib
        • tozuka_2024    HCC827, PC-9   × Osimertinib
        • cancerres_2021 H1975          × Osimertinib, Rociletinib
        • mcp_2025       H1975, PC-9    × Osimertinib
        • hsu_2025       PC-9           × Osimertinib
        • drugptm_bench  A431           × Afatinib, Gefitinib
        • ruprecht_2017  BT-474         × Lapatinib  (ERBB2)

    POSITION PARSING:
      ptm_site strings come in two forms:
        'Y1092' / 'S991'  — UniProt precursor positions (parsed → 1092 / 991)
        'Y2' / 'S26'      — DrugPTM-Bench's local peptide indices; we SKIP
                            these because they are not UniProt residue
                            positions and cannot be aligned to our 12-site
                            schema without a peptide→protein remap.

    Only sites whose parsed position falls within the 12-site EGFR
    {869, 991, 998, 1016, 1039, 1041, 1069, 1092, 1110, 1125, 1172, 1197}
    or 10-site ERBB2 {686, 1005, 1054, 1099, 1139, 1151, 1196, 1221, 1222, 1248}
    schemas are kept.  All other sites belong to ESM-derived global pY data
    (Y190, Y362, …) and are not part of our ptm_vector.
    """
    global _MEASURED_PTM_LOOKUP, _MEASURED_DELTA_LOOKUP
    if _MEASURED_DELTA_LOOKUP is not None:
        return _MEASURED_PTM_LOOKUP, _MEASURED_DELTA_LOOKUP

    print("  Building measured-PTM lookups from high-confidence sources...")

    egfr_positions = {869, 991, 998, 1016, 1039, 1041,
                      1069, 1092, 1110, 1125, 1172, 1197}
    erbb2_positions = {686, 1005, 1054, 1099, 1139, 1151,
                       1196, 1221, 1222, 1248}

    baseline_lookup = {}    # (cell_line_norm, gene) -> {pos: log2_baseline}
    delta_lookup = {}       # (cell_line_norm, drug_name_norm, gene) -> {pos: log2FC}

    if df_drugptm is None or df_drugptm.empty:
        _MEASURED_PTM_LOOKUP = baseline_lookup
        _MEASURED_DELTA_LOOKUP = delta_lookup
        return baseline_lookup, delta_lookup

    # Only keep rows where ptm_site is a true protein position (e.g. "Y1092"
    # or "S991") — i.e. the numeric part is large enough to be a UniProt
    # residue number, not a peptide-local index.
    import re
    pos_re = re.compile(r"^([STY])(\d+)$")

    n_kept_baseline = 0
    n_kept_delta = 0

    for _, row in df_drugptm.iterrows():
        site = str(row.get("ptm_site", "")).strip()
        gene = str(row.get("gene", row.get("protein", ""))).upper().strip()
        log2fc = row.get("log2_fold_change")
        if pd.isna(log2fc) or not site or not gene:
            continue
        m = pos_re.match(site)
        if not m:
            continue
        pos = int(m.group(2))

        # Validate position against our 12-site schema
        if gene == "EGFR" and pos not in egfr_positions:
            continue
        if gene == "ERBB2" and pos not in erbb2_positions:
            continue
        if gene not in ("EGFR", "ERBB2"):
            continue

        cell = normalize_cell_line_name(row.get("cell_line", ""))
        drug = normalize_drug_name(row.get("drug_name", ""))
        if not cell:
            continue

        # ── delta_lookup is the main payload (drug-induced changes) ─────
        if drug and drug != "none":
            key = (cell, drug, gene)
            delta_lookup.setdefault(key, {})
            # If multiple measurements for same site (different sources),
            # take the mean.
            if pos in delta_lookup[key]:
                prev = delta_lookup[key][pos]
                delta_lookup[key][pos] = (prev + float(log2fc)) / 2.0
            else:
                delta_lookup[key][pos] = float(log2fc)
                n_kept_delta += 1

        # ── baseline_lookup uses parental-vs-resistant / no-drug rows ──
        # When data_source is febs_2025 the drug is 'none' and the row
        # represents a tumor-vs-WT contrast — these inform the baseline
        # log2 intensity for activating-mutation lines.
        ctx = str(row.get("resistance_context", "")).lower()
        if drug == "none" or "parental" in ctx or "baseline" in ctx:
            key = (cell, gene)
            baseline_lookup.setdefault(key, {})
            if pos not in baseline_lookup[key]:
                baseline_lookup[key][pos] = float(log2fc)
                n_kept_baseline += 1

    print(f"    ✓ Measured-delta lookup: {len(delta_lookup)} (cell, drug, gene) "
          f"combos covering {n_kept_delta} site-measurements")
    print(f"    ✓ Measured-baseline lookup: {len(baseline_lookup)} (cell, gene) "
          f"combos covering {n_kept_baseline} site-measurements")
    # Diagnostic: which (cell, drug, gene) combos got delta data
    for key in sorted(delta_lookup.keys())[:15]:
        n = len(delta_lookup[key])
        print(f"      delta: {key[0]:10s} × {key[1]:14s} ({key[2]}) → {n} sites")
    if len(delta_lookup) > 15:
        print(f"      ... ({len(delta_lookup) - 15} more)")

    _MEASURED_PTM_LOOKUP = baseline_lookup
    _MEASURED_DELTA_LOOKUP = delta_lookup
    return baseline_lookup, delta_lookup


def compute_per_sample_ptm_vector(
    cell_line_norm: str,
    target_protein: str,
    ptm_background: str,
    ptm_vectors: dict,
    erbb2_ptm_vectors: dict,
    sites: list,
    comutations_dict: dict,
    measured_baseline_lookup: dict,
) -> list:
    """
    Compute the 12-element ptm_vector for a single (cell_line, gene) pair.

    PRECEDENCE (per Solutions A + B of the proposal):
       1) Measured baseline (only when measured_baseline_lookup has data)
       2) Mutation-class baseline × per-cell-line modulators (KRAS, MET,
          PIK3CA, TP53, PTEN, tissue, HER2-amp tier)
       3) Mutation-class baseline only (fallback for cell lines not in CCLE)

    Returns the 12-element list of PTM values in the canonical site order:
       EGFR:  [Y869, S991, Y998, Y1016, S1039, T1041,
               Y1069, Y1092, Y1110, Y1125, Y1172, Y1197]
       ERBB2: [T686, Y1005, S1054, T1099, Y1139, S1151,
               Y1196, Y1221, Y1222, Y1248, 0, 0]
    """
    target_protein = target_protein or "EGFR"
    is_erbb2 = (target_protein == "ERBB2")
    modulators_cfg = cfg.get("ptm_modulators", {})
    gene_mods = modulators_cfg.get(target_protein, {})

    # Get the mutation-class baseline vector (this is what the OLD code did)
    if is_erbb2:
        erbb2_sites_cfg = cfg["ptm"]["ERBB2"]["phospho_sites"]
        baseline_vec = []
        for i, (_aa, _pos) in enumerate(sites):  # sites is EGFR-shaped (12)
            if i < len(erbb2_sites_cfg):
                erbb2_pos = str(erbb2_sites_cfg[i]["position"])
                v = erbb2_ptm_vectors.get(ptm_background, {}).get(erbb2_pos, 1.0)
            else:
                v = 0.0  # zero-padded slot
            baseline_vec.append(float(v))
    else:
        baseline_vec = []
        for (_aa, pos) in sites:
            bg = ptm_vectors.get(ptm_background, {})
            v = bg.get(str(pos), bg.get(pos, 1.0))
            baseline_vec.append(float(v))

    # ── Solution A: measured baseline override ──────────────────────────
    measured = measured_baseline_lookup.get((cell_line_norm, target_protein), {})
    if measured:
        if is_erbb2:
            erbb2_sites_cfg = cfg["ptm"]["ERBB2"]["phospho_sites"]
            for i, _ in enumerate(sites):
                if i < len(erbb2_sites_cfg):
                    erbb2_pos = erbb2_sites_cfg[i]["position"]
                    if erbb2_pos in measured:
                        # Convert log2FC to multiplicative factor:
                        # baseline 1.0 × 2^log2FC
                        baseline_vec[i] = 1.0 * (2.0 ** measured[erbb2_pos])
        else:
            for i, (_aa, pos) in enumerate(sites):
                if pos in measured:
                    baseline_vec[i] = 1.0 * (2.0 ** measured[pos])

    # ── Solution B: per-cell-line modulators (only when CCLE context exists) ──
    ctx = comutations_dict.get(cell_line_norm, {})
    if not ctx:
        return baseline_vec

    if not is_erbb2:
        # EGFR modulator schema — uses UniProt precursor positions.
        # Build a position → multiplier map from the additive deltas.
        deltas = {pos: 0.0 for (_aa, pos) in sites}

        if ctx.get("kras_activating", False):
            kras_d = float(gene_mods.get("kras_activating", 0.0))
            deltas[1092] += kras_d
            deltas[1110] += kras_d

        if ctx.get("met_amplified", False):
            deltas[869] += float(gene_mods.get("met_amp_Y869", 0.0))
            deltas[1110] += float(gene_mods.get("met_amp_Y1110", 0.0))

        if ctx.get("pik3ca_activating", False):
            deltas[1197] += float(gene_mods.get("pik3ca_Y1197", 0.0))

        if ctx.get("tp53_lof", False):
            deltas[998] += float(gene_mods.get("tp53_lof_Y998", 0.0))
            deltas[1069] += float(gene_mods.get("tp53_lof_Y1069", 0.0))

        if ctx.get("pten_loss", False):
            deltas[1197] += float(gene_mods.get("pten_loss_Y1197", 0.0))

        # Tissue-of-origin modulators
        tissue = ctx.get("tissue", "")
        if "breast" in tissue:
            d = float(gene_mods.get("tissue_breast_Y1092_Y1197", 0.0))
            deltas[1092] += d
            deltas[1197] += d
        if any(s in tissue for s in ("squamous", "head", "skin")):
            deltas[869] += float(gene_mods.get("tissue_squamous_Y869", 0.0))

        # Apply: new = baseline × (1 + Δ) clipped to a sane range
        for i, (_aa, pos) in enumerate(sites):
            mult = 1.0 + deltas.get(pos, 0.0)
            mult = max(0.1, min(mult, 5.0))   # safety clip
            baseline_vec[i] = baseline_vec[i] * mult

    else:
        # ERBB2 modulator schema
        erbb2_sites_cfg = cfg["ptm"]["ERBB2"]["phospho_sites"]
        her2_amp_table = gene_mods.get("her2_amp_multiplier",
                                       {"baseline": 1.0, "intermediate": 1.2, "high": 1.5})
        tier = ctx.get("her2_amp_tier", "baseline")
        amp_mult = float(her2_amp_table.get(tier, 1.0))

        # All 7 tyrosine sites get the HER2-amp multiplier
        # (Y1005, Y1139, Y1196, Y1221, Y1222, Y1248)
        her2_tyrosines = {1005, 1139, 1196, 1221, 1222, 1248}

        # Additive deltas per position (applied multiplicatively as (1 + Δ))
        deltas = {}
        if ctx.get("pik3ca_activating", False):
            deltas[1248] = deltas.get(1248, 0.0) + float(gene_mods.get("pik3ca_Y1248", 0.0))
        if ctx.get("pten_loss", False):
            deltas[1248] = deltas.get(1248, 0.0) + float(gene_mods.get("pten_Y1248", 0.0))
        if ctx.get("er_status", "") == "positive":
            deltas[1005] = deltas.get(1005, 0.0) + float(gene_mods.get("er_plus_Y1005", 0.0))

        for i, _ in enumerate(sites):
            if i >= len(erbb2_sites_cfg):
                continue   # padded slot stays at 0
            erbb2_pos = erbb2_sites_cfg[i]["position"]
            # Apply HER2-amp multiplier to tyrosines (auto-phospho sites)
            mult = 1.0
            if erbb2_pos in her2_tyrosines:
                mult *= amp_mult
            mult *= (1.0 + deltas.get(erbb2_pos, 0.0))
            mult = max(0.1, min(mult, 5.0))
            baseline_vec[i] = baseline_vec[i] * mult

    return baseline_vec


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1c: N-Glycosylation State Vector Pipeline (NEW — PTM-BDL §3, §7.4)
# ══════════════════════════════════════════════════════════════════════════════
#
# These four helpers are the GLYCO analogue of the phospho pipeline above
# (load_ptm_data, build_measured_ptm_lookup, compute_per_sample_ptm_vector).
# They produce the second PTM channel that the PTM-BDL self-attention block
# in step10 consumes (modification type `glyco_N`, type_id = 3).
#
# DESIGN: glyco is intentionally SIMPLER than phospho because we have no
# mutation-conditional glyco baseline:
#   1) baseline ← step04 `<gene>_glyco_state_vectors.json` (1.0 across
#      all 12 sites for both genes; HER2 has 1.5 for the amplified bg).
#   2) measured baseline override ← per-(cell, gene) site averages
#      from step05's drugptm_multiptm_responses.csv (filter ptm_modification_type
#      == "glyco_N" and convert log2FC → 2^log2FC × baseline).
#   3) measured drug delta override ← per-(cell, drug, gene) site averages,
#      drug-conditioned glyco changes (only the MCP 2025 / 2025b rows have
#      these; everything else stays at delta = 0).
# Per-cell-line modulators are NOT applied to glyco — there is no published
# evidence base equivalent to the phospho HER2-amp / MET-amp magnitudes
# from CPTAC etc., so we deliberately keep the glyco channel data-driven.
# ══════════════════════════════════════════════════════════════════════════════

_MEASURED_GLYCO_BASELINE_LOOKUP = None
_MEASURED_GLYCO_DELTA_LOOKUP = None


def load_glyco_state_vectors():
    """
    Load step04's per-gene N-glyco state vectors and the per-gene canonical
    glyco-site lists from `cfg["ptm"][<gene>]["glyco_sites"]`.

    Returns:
        (egfr_glyco_vectors, erbb2_glyco_vectors,
         egfr_glyco_sites,   erbb2_glyco_sites)

    Where:
        egfr_glyco_vectors[bg][position] = baseline_value (float)
        egfr_glyco_sites = list of {position, residue, amino_acid, ...} dicts
                           (length 12 for EGFR, length 7 for ERBB2 — step06
                           pads ERBB2 to 12 below)
    """
    print("  Loading N-glyco state vectors (step04 PTM-BDL multi-PTM)...")

    egfr_path = PROCESSED_DIR / "ptm" / "egfr_glyco_state_vectors.json"
    erbb2_path = PROCESSED_DIR / "ptm" / "erbb2_glyco_state_vectors.json"

    egfr_vectors = {}
    erbb2_vectors = {}
    if egfr_path.exists():
        with open(egfr_path) as f:
            egfr_vectors = json.load(f)
        print(f"    ✓ EGFR glyco vectors: {len(egfr_vectors)} backgrounds")
    else:
        print(f"    ⚠ EGFR glyco vectors not found — using baseline=1.0 fallback")
    if erbb2_path.exists():
        with open(erbb2_path) as f:
            erbb2_vectors = json.load(f)
        print(f"    ✓ ERBB2 glyco vectors: {len(erbb2_vectors)} backgrounds")
    else:
        print(f"    ⚠ ERBB2 glyco vectors not found — using baseline=1.0 fallback")

    egfr_sites = cfg["ptm"]["EGFR"].get("glyco_sites", [])
    erbb2_sites = cfg["ptm"]["ERBB2"].get("glyco_sites", [])

    return egfr_vectors, erbb2_vectors, egfr_sites, erbb2_sites


def build_measured_glyco_lookup(df_drugptm):
    """
    Build (baseline_lookup, delta_lookup) for N-glycosylation, parallel to
    `build_measured_ptm_lookup` for phospho.

    baseline_lookup[(cell_line_norm, gene)][position] = log2_baseline
        From rows where drug_name == "none" or resistance_context contains
        "baseline" / "reference" — i.e. parental / untreated glyco
        measurements (Taniguchi 2024 atlas, Sethi 2020 DMSO occupancy).
        These OVERRIDE the flat-1.0 baseline from step04.

    delta_lookup[(cell_line_norm, drug_name_norm, gene)][position] = log2FC
        From rows where drug_name is a real TKI and the row is a glyco
        measurement — the drug-conditioned glyco change (only MCP 2025
        / MCP 2025b provide these so far; ST6Gal1 Garnham 2021 is included
        but its log2FC is NaN so it does not contribute).

    Position parsing: ptm_site = "N<pos>" → pos = int(pos).
    We accept positions only when they are present in the canonical
    glyco-site list for the gene (so non-canonical sites are silently
    dropped).
    """
    global _MEASURED_GLYCO_BASELINE_LOOKUP, _MEASURED_GLYCO_DELTA_LOOKUP
    if _MEASURED_GLYCO_DELTA_LOOKUP is not None:
        return _MEASURED_GLYCO_BASELINE_LOOKUP, _MEASURED_GLYCO_DELTA_LOOKUP

    print("  Building measured-glyco lookups from step05 unified data...")

    egfr_positions = {s["position"]
                      for s in cfg["ptm"]["EGFR"].get("glyco_sites", [])}
    erbb2_positions = {s["position"]
                       for s in cfg["ptm"]["ERBB2"].get("glyco_sites", [])}

    baseline_lookup = {}
    delta_lookup = {}

    if df_drugptm is None or df_drugptm.empty:
        _MEASURED_GLYCO_BASELINE_LOOKUP = baseline_lookup
        _MEASURED_GLYCO_DELTA_LOOKUP = delta_lookup
        return baseline_lookup, delta_lookup

    # Filter to glyco_N rows only.  We use ptm_modification_type when
    # available (new step05 schema); otherwise fall back to ptm_type startswith
    # "N-glyco".
    if "ptm_modification_type" in df_drugptm.columns:
        df_glyco = df_drugptm[
            df_drugptm["ptm_modification_type"].astype(str) == "glyco_N"
        ]
    else:
        df_glyco = df_drugptm[
            df_drugptm.get("ptm_type", pd.Series(dtype=str))
                      .astype(str)
                      .str.lower()
                      .str.startswith("n-glyco")
        ]
    if df_glyco.empty:
        print("    ⚠ No glyco rows found in drugptm input — glyco lookups empty.")
        _MEASURED_GLYCO_BASELINE_LOOKUP = baseline_lookup
        _MEASURED_GLYCO_DELTA_LOOKUP = delta_lookup
        return baseline_lookup, delta_lookup

    import re
    pos_re = re.compile(r"^N(\d+)$")

    n_kept_baseline = 0
    n_kept_delta = 0

    # Aggregate to per-(cell, gene[, drug], position) means before storage —
    # there are typically many glycoforms per site, so we collapse to the
    # site-level log2FC mean (the per-glycoform attributions still live in
    # the raw CSV and are consumed by step13 IG).
    agg_keys_baseline: dict = {}   # (cell, gene, pos) → list[float]
    agg_keys_delta: dict = {}      # (cell, gene, drug, pos) → list[float]

    for _, row in df_glyco.iterrows():
        site = str(row.get("ptm_site", "")).strip()
        gene = str(row.get("target_protein",
                           row.get("protein", ""))).upper().strip()
        if gene not in ("EGFR", "ERBB2"):
            continue
        m = pos_re.match(site)
        if not m:
            continue
        pos = int(m.group(1))
        if gene == "EGFR" and pos not in egfr_positions:
            continue
        if gene == "ERBB2" and pos not in erbb2_positions:
            continue

        cell = normalize_cell_line_name(row.get("cell_line", ""))
        drug = normalize_drug_name(row.get("drug_name", ""))
        if not cell:
            continue

        log2fc = row.get("log2_fold_change")
        baseline_intensity = row.get("baseline_intensity")
        ctx = str(row.get("resistance_context", "")).lower()

        # ── Delta lookup ────────────────────────────────────────────────
        # Drug-conditioned glyco fold-change (cell × drug × site).  Only
        # the rows where (a) drug != "none" AND (b) log2FC is finite
        # contribute.  This is currently MCP 2025 / 2025b (Osimertinib).
        if (drug and drug != "none"
                and log2fc is not None and pd.notna(log2fc)):
            try:
                agg_keys_delta.setdefault((cell, gene, drug, pos),
                                          []).append(float(log2fc))
            except (TypeError, ValueError):
                pass

        # ── Baseline lookup ────────────────────────────────────────────
        # Parental / untreated / reference rows.  Use `baseline_intensity`
        # for percent-style data (atlas) and log2FC for the few rows with
        # a drug=DMSO comparison.  We store a SINGLE log2-scale number
        # per (cell, gene, pos); for percent data we convert to log2 of
        # the fraction so baseline_lookup is always comparable.
        is_reference = (drug == "none" or "baseline" in ctx
                        or "reference" in ctx or "glycoform_reference" in ctx
                        or "occupancy_reference" in ctx)
        if is_reference:
            val = None
            if log2fc is not None and pd.notna(log2fc):
                try:
                    val = float(log2fc)
                except (TypeError, ValueError):
                    pass
            elif baseline_intensity is not None and pd.notna(baseline_intensity):
                # Atlas %/PSM-count style — convert to log2 of a normalised
                # fraction.  For PSM counts (st6gal1) the absolute scale is
                # arbitrary, so we just store log2(value + 1) which keeps
                # the relative ordering between sites.
                try:
                    val = float(np.log2(float(baseline_intensity) + 1.0))
                except (TypeError, ValueError):
                    pass
            if val is not None:
                agg_keys_baseline.setdefault((cell, gene, pos),
                                             []).append(val)

    # Collapse to means
    for (cell, gene, drug, pos), vals in agg_keys_delta.items():
        if not vals:
            continue
        delta_lookup.setdefault((cell, drug, gene), {})[pos] = float(np.mean(vals))
        n_kept_delta += 1
    for (cell, gene, pos), vals in agg_keys_baseline.items():
        if not vals:
            continue
        baseline_lookup.setdefault((cell, gene), {})[pos] = float(np.mean(vals))
        n_kept_baseline += 1

    print(f"    ✓ Glyco delta lookup:    {len(delta_lookup)} (cell, drug, gene) "
          f"combos covering {n_kept_delta} site-measurements")
    print(f"    ✓ Glyco baseline lookup: {len(baseline_lookup)} (cell, gene) "
          f"combos covering {n_kept_baseline} site-measurements")
    for key in sorted(delta_lookup.keys())[:8]:
        n = len(delta_lookup[key])
        print(f"      glyco-delta: {key[0]:20s} × {key[1]:12s} ({key[2]}) → {n} sites")
    if len(delta_lookup) > 8:
        print(f"      ... ({len(delta_lookup) - 8} more)")

    _MEASURED_GLYCO_BASELINE_LOOKUP = baseline_lookup
    _MEASURED_GLYCO_DELTA_LOOKUP = delta_lookup
    return baseline_lookup, delta_lookup


def compute_per_sample_glyco_vector(
    cell_line_norm: str,
    target_protein: str,
    egfr_glyco_vectors: dict,
    erbb2_glyco_vectors: dict,
    egfr_sites: list,
    erbb2_sites: list,
    glyco_baseline_lookup: dict,
    comutations_dict: dict,
    glyco_dim: int = 12,
) -> list:
    """
    Compute the 12-element glyco vector for one (cell_line, gene) pair.

    Precedence (mirrors compute_per_sample_ptm_vector):
       1) Measured baseline (from `glyco_baseline_lookup`) — override
          per matched site.  Stored value is in log2 space, so we
          convert back to multiplicative: baseline_vec[i] = 2^log2.
       2) Step04 background vector (1.0 across all sites for EGFR;
          1.0 or 1.5 for ERBB2 depending on HER2-amp tier).
       3) Padding: positions beyond the canonical site list are 0.0.

    Returns a list of length `glyco_dim` (default 12).  ERBB2 has 7 real
    sites, the remaining 5 slots are zero-padded.
    """
    target_protein = target_protein or "EGFR"
    is_erbb2 = (target_protein == "ERBB2")

    if is_erbb2:
        # Pick baseline background based on HER2-amp tier (if known).
        ctx = comutations_dict.get(cell_line_norm, {}) if comutations_dict else {}
        tier = ctx.get("her2_amp_tier", "baseline")
        if tier == "high":
            bg = "HER2_amplified_glyco_level"
        else:
            bg = "wt_glyco_level"
        bg_dict = erbb2_glyco_vectors.get(bg) \
                  or erbb2_glyco_vectors.get("wt_glyco_level", {})
        sites = erbb2_sites
    else:
        bg_dict = (egfr_glyco_vectors.get("wt_glyco_level")
                   or {})
        sites = egfr_sites

    vec = []
    for i in range(glyco_dim):
        if i < len(sites):
            pos = sites[i]["position"]
            # JSON dict keys are stringified
            v = bg_dict.get(str(pos), bg_dict.get(pos, 1.0))
            try:
                vec.append(float(v))
            except (TypeError, ValueError):
                vec.append(1.0)
        else:
            vec.append(0.0)   # padded slot

    # Solution A: measured baseline override
    measured = (glyco_baseline_lookup or {}).get(
        (cell_line_norm, target_protein), {}
    )
    if measured:
        for i in range(min(glyco_dim, len(sites))):
            pos = sites[i]["position"]
            if pos in measured:
                try:
                    vec[i] = float(2.0 ** measured[pos])
                except (TypeError, ValueError, OverflowError):
                    pass

    # Safety clip — matches the phospho `safety clip` in
    # compute_per_sample_ptm_vector (max=5.0).  Without this, the PSM-count-
    # derived baseline from ST6Gal1 Garnham 2021 (BT-474) produces values
    # like 51.0 at N56, which would dominate the PTM-BDL attention pool.
    # Padded slots (value == 0.0) are left alone so the model can still
    # distinguish "no real site here" from "very low signal".
    for i in range(glyco_dim):
        v = vec[i]
        if v > 0.0:
            vec[i] = max(0.1, min(v, 5.0))

    return vec


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Placeholder Data Generators (for development without downloads)
# ══════════════════════════════════════════════════════════════════════════════

def create_placeholder_gdsc():
    """Create minimal GDSC-like data for development."""
    np.random.seed(42)

    records = []
    cell_lines = {
        "HCC827": ("exon19del", True),
        "PC-9": ("exon19del", True),
        "H3255": ("L858R", True),
        "H1975": ("L858R_T790M", True),
        "H1650": ("exon19del", True),
        "A549": ("wild_type", False),
        "H460": ("wild_type", False),
        "H358": ("wild_type", False),
    }

    drugs = {
        "Osimertinib": 1919, "Gefitinib": 1010,
        "Afatinib": 1032, "Erlotinib": 1168,
    }

    for cl, (mut, is_egfr_mut) in cell_lines.items():
        for drug_name, drug_id in drugs.items():
            if is_egfr_mut:
                if "T790M" in mut and drug_name in ["Gefitinib", "Erlotinib", "Afatinib"]:
                    ln_ic50 = np.random.normal(3.0, 0.5)
                elif drug_name == "Osimertinib" or "T790M" not in mut:
                    ln_ic50 = np.random.normal(-2.0, 0.5)
                else:
                    ln_ic50 = np.random.normal(0.0, 1.0)
            else:
                ln_ic50 = np.random.normal(2.5, 0.5)

            records.append({
                "CELL_LINE_NAME": cl,
                "DRUG_NAME": drug_name,
                "DRUG_ID": drug_id,
                "LN_IC50": round(ln_ic50, 4),
                "AUC": round(1 / (1 + np.exp(-ln_ic50)), 4),
                "resistance_label": 1 if ln_ic50 > 0 else 0,
                "TCGA_DESC": "LUAD",
            })

    return pd.DataFrame(records)


def create_placeholder_mutations():
    """Create minimal mutation profile data."""
    return pd.DataFrame([
        {"cell_line": "HCC827",  "egfr_mutations": "exon19del (E746-A750)", "mutation_classes": "activating"},
        {"cell_line": "PC-9",    "egfr_mutations": "exon19del (E746-A750)", "mutation_classes": "activating"},
        {"cell_line": "H3255",   "egfr_mutations": "L858R",                 "mutation_classes": "activating"},
        {"cell_line": "H1975",   "egfr_mutations": "L858R; T790M",          "mutation_classes": "activating; resistance"},
        {"cell_line": "H1650",   "egfr_mutations": "exon19del (E746-A750)", "mutation_classes": "activating"},
        {"cell_line": "HCC4006", "egfr_mutations": "exon19del (E746-A750)", "mutation_classes": "activating"},
        {"cell_line": "A549",    "egfr_mutations": "wild_type",             "mutation_classes": "wild_type"},
        {"cell_line": "H460",    "egfr_mutations": "wild_type",             "mutation_classes": "wild_type"},
        {"cell_line": "H358",    "egfr_mutations": "wild_type",             "mutation_classes": "wild_type"},
    ])


def create_placeholder_ptm_vectors():
    """Create minimal PTM vectors (12 phosphorylation sites, UniProt precursor numbering)."""
    return {
        "wt_phospho_level": {
            869: 1.0, 991: 1.0, 998: 1.0, 1016: 1.0, 1039: 1.0, 1041: 1.0,
            1069: 1.0, 1092: 1.0, 1110: 1.0, 1125: 1.0, 1172: 1.0, 1197: 1.0,
        },
        "L858R_phospho_level": {
            869: 2.5, 991: 1.3, 998: 1.8, 1016: 2.0, 1039: 1.2, 1041: 1.5,
            1069: 0.6, 1092: 4.0, 1110: 2.5, 1125: 2.0, 1172: 2.0, 1197: 3.5,
        },
        "T790M_phospho_level": {
            869: 1.2, 991: 1.0, 998: 1.1, 1016: 1.1, 1039: 1.0, 1041: 1.0,
            1069: 0.8, 1092: 1.5, 1110: 1.2, 1125: 1.1, 1172: 1.0, 1197: 1.3,
        },
        "L858R_T790M_phospho_level": {
            869: 3.0, 991: 1.4, 998: 2.0, 1016: 2.2, 1039: 1.3, 1041: 1.6,
            1069: 0.4, 1092: 5.0, 1110: 3.0, 1125: 2.3, 1172: 2.2, 1197: 4.0,
        },
        "C797S_phospho_level": {
            869: 1.1, 991: 1.0, 998: 1.0, 1016: 1.0, 1039: 1.0, 1041: 1.0,
            1069: 0.9, 1092: 1.3, 1110: 1.1, 1125: 1.0, 1172: 1.0, 1197: 1.2,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Literature IC50 for Cell Lines Missing from GDSC
# ══════════════════════════════════════════════════════════════════════════════

def add_literature_ic50_records(df_response):
    """
    Add literature-curated IC50 values for PC-9, HCC827, and HCC4006.

    These are the gold-standard exon19del NSCLC cell lines used in virtually
    every Osimertinib resistance study. They are present in our Drug-PTM
    phosphoproteomic data (Tozuka 2024, Hsu 2025) but are NOT screened in the
    GDSC panel. Without adding them, we lose the ability to connect real
    phosphoproteomic measurements to drug response labels.

    IC50 values (converted to LN_IC50 = ln(IC50 in µM)):
    ─────────────────────────────────────────────────────

    PC-9 (exon19del E746-A750):
      Osimertinib  IC50 ≈ 15 nM  → ln(0.015) = −4.200   Cross et al. 2014
      Gefitinib    IC50 ≈ 30 nM  → ln(0.030) = −3.507   Koizumi et al. 2005
      Afatinib     IC50 ≈ 1 nM   → ln(0.001) = −6.908   Li et al. 2008
      Erlotinib    IC50 ≈ 50 nM  → ln(0.050) = −2.996   Engelman et al. 2007

    HCC827 (exon19del E746-A750):
      Osimertinib  IC50 ≈ 6 nM   → ln(0.006) = −5.116   Cross et al. 2014
      Gefitinib    IC50 ≈ 5 nM   → ln(0.005) = −5.298   Engelman et al. 2007
      Afatinib     IC50 ≈ 0.3 nM → ln(0.0003) = −8.112  Li et al. 2008
      Erlotinib    IC50 ≈ 15 nM  → ln(0.015) = −4.200   Engelman et al. 2007

    AUC estimates are calibrated against GDSC values for comparable sensitive
    cell lines (H3255 + Afatinib AUC=0.419, H1975 + Osimertinib AUC=0.521).

    References:
    • Cross et al., Cancer Discov 2014; 4(9):1046-61 (PMID: 25351743)
    • Engelman et al., Science 2007; 316(5827):1039-43 (PMID: 17463250)
    • Koizumi et al., Mol Cancer Ther 2005; 4(7):1014-21
    • Li et al., Oncogene 2008; 27:4702-11 (PMID: 18408761)
    """
    print("\n  Adding literature IC50 for cell lines missing from GDSC...")

    cl_col = "CELL_LINE_NAME" if "CELL_LINE_NAME" in df_response.columns else "cell_line"
    drug_col = "DRUG_NAME" if "DRUG_NAME" in df_response.columns else "drug_name"

    # Build set of existing (normalized_cell_line, drug) pairs
    existing = set()
    for _, row in df_response.iterrows():
        key = (normalize_cell_line_name(str(row[cl_col])), str(row[drug_col]))
        existing.add(key)

    # Literature IC50 records
    literature_records = [
        # ── PC-9 (exon19del E746-A750) ────────────────────────────────────
        {cl_col: "PC-9", drug_col: "Osimertinib", "DRUG_ID": 1919,
         "LN_IC50": -4.200, "AUC": 0.520, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "25351743"},
        {cl_col: "PC-9", drug_col: "Gefitinib", "DRUG_ID": 1010,
         "LN_IC50": -3.507, "AUC": 0.580, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature"},
        {cl_col: "PC-9", drug_col: "Afatinib", "DRUG_ID": 1032,
         "LN_IC50": -6.908, "AUC": 0.400, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "18408761"},
        {cl_col: "PC-9", drug_col: "Erlotinib", "DRUG_ID": 1168,
         "LN_IC50": -2.996, "AUC": 0.620, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature"},

        # ── HCC827 (exon19del E746-A750) ──────────────────────────────────
        {cl_col: "HCC827", drug_col: "Osimertinib", "DRUG_ID": 1919,
         "LN_IC50": -5.116, "AUC": 0.450, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "25351743"},
        {cl_col: "HCC827", drug_col: "Gefitinib", "DRUG_ID": 1010,
         "LN_IC50": -5.298, "AUC": 0.430, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "17463250"},
        {cl_col: "HCC827", drug_col: "Afatinib", "DRUG_ID": 1032,
         "LN_IC50": -8.112, "AUC": 0.350, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "18408761"},
        {cl_col: "HCC827", drug_col: "Erlotinib", "DRUG_ID": 1168,
         "LN_IC50": -4.200, "AUC": 0.500, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "17463250"},

        # ── HCC4006 (exon19del E746-A750) ─────────────────────────────────
        # Source: Chmielecki et al., Sci Transl Med 2011; Cross et al., 2014
        {cl_col: "HCC4006", drug_col: "Osimertinib", "DRUG_ID": 1919,
         "LN_IC50": -4.500, "AUC": 0.500, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature", "ic50_pmid": "25351743"},
        {cl_col: "HCC4006", drug_col: "Gefitinib", "DRUG_ID": 1010,
         "LN_IC50": -3.800, "AUC": 0.560, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature"},
        {cl_col: "HCC4006", drug_col: "Afatinib", "DRUG_ID": 1032,
         "LN_IC50": -5.500, "AUC": 0.420, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature"},
        {cl_col: "HCC4006", drug_col: "Erlotinib", "DRUG_ID": 1168,
         "LN_IC50": -3.200, "AUC": 0.600, "resistance_label": 0,
         "TCGA_DESC": "Non-Small Cell Lung Carcinoma",
         "ic50_source": "literature"},
    ]

    # Only add records not already in GDSC
    new_rows = []
    skipped = []
    for rec in literature_records:
        key = (normalize_cell_line_name(rec[cl_col]), rec[drug_col])
        if key not in existing:
            new_rows.append(rec)
        else:
            skipped.append(f"{rec[cl_col]}+{rec[drug_col]}")

    if skipped:
        print(f"    Skipped (already in GDSC): {', '.join(skipped)}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)

        # Add ic50_source column to original GDSC records
        if "ic50_source" not in df_response.columns:
            df_response["ic50_source"] = "gdsc"
        if "ic50_pmid" not in df_response.columns:
            df_response["ic50_pmid"] = np.nan

        # Ensure column alignment
        for col in df_response.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        for col in df_new.columns:
            if col not in df_response.columns:
                df_response[col] = np.nan

        df_combined = pd.concat(
            [df_response, df_new[df_response.columns]],
            ignore_index=True,
        )

        for rec in new_rows:
            src = f" (PMID {rec.get('ic50_pmid', '—')})" if rec.get("ic50_pmid") else ""
            print(f"    + {rec[cl_col]:8s} + {rec[drug_col]:14s} "
                  f"LN_IC50={rec['LN_IC50']:+.3f}  label={rec['resistance_label']}{src}")

        print(f"    ✓ Added {len(new_rows)} literature IC50 records")
        return df_combined
    else:
        print("    All literature cell lines already present in GDSC data")
        return df_response


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Drug-PTM Phosphoproteomic Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def build_phospho_features(df_drugptm):
    """
    Transform per-site Drug-PTM data into per-(cell_line, drug) aggregate features.

    The Drug-PTM data from Step 05 contains one row per EGFR phosphosite per
    (cell_line, drug) experiment. This function pivots that data into a single
    row per (cell_line, drug) pair with summary statistics:

    A) DrugPTM-Bench (dose-response):
       log2_fold_change = log2(max_dose_intensity / baseline_intensity)

    B) Tozuka 2024 (parental vs resistant):
       log2_fold_change = resistant_log2 − parental_log2

    C) Hsu 2025 (temporal dynamics):
       fc_acute_5min, fc_sustained_6h, fc_dtp_persister, fc_dtp_rebound

    D) PNAS 2025 (pY phosphoproteome):
       log2FC = log2(Osi_TMT / DMSO_TMT) per cell line (H1975, HCC4006)

    E) FEBS 2025 (tumor phospho signatures):
       log2FC = FC(EGFR-mutant tumors vs WT tumors) from patient LUAD tissue

    Output columns:
    ───────────────
    phospho_mean_log2fc       Mean drug-induced log2FC across EGFR sites
    phospho_min_log2fc        Most dephosphorylated site
    phospho_max_log2fc        Most hyperphosphorylated site
    phospho_std_log2fc        Variability of drug effect across sites
    phospho_n_sites           Number of EGFR phosphosites measured
    phospho_n_down            Sites with log2FC < −1.0
    phospho_n_up              Sites with log2FC > +1.0
    phospho_data_sources      Which datasets contributed
    phospho_contexts          Resistance context(s)
    phospho_fc_acute          Mean acute response (5 min, Hsu only)
    phospho_fc_sustained      Mean sustained response (6 h, Hsu only)
    phospho_fc_persister      Mean DTP persister level (Hsu only)
    phospho_fc_rebound        Mean DTP rebound level (Hsu only)
    """
    if df_drugptm is None or df_drugptm.empty:
        print("    ⚠ No Drug-PTM data to build features from")
        return pd.DataFrame()

    print("\n  Building per-(cell_line, drug) phosphoproteomic features...")

    df = df_drugptm.copy()
    df["cell_line_norm"] = df["cell_line"].apply(normalize_cell_line_name)
    df["drug_name_norm"] = df["drug_name"].apply(normalize_drug_name)

    # Filter to rows with valid log2FC
    df_valid = df.dropna(subset=["log2_fold_change"])
    if df_valid.empty:
        print("    ⚠ No valid log2FC values found")
        return pd.DataFrame()

    # ── Aggregate per (cell_line, drug) across all sites and sources ──────
    agg = df_valid.groupby(["cell_line_norm", "drug_name_norm"]).agg(
        phospho_mean_log2fc=("log2_fold_change", "mean"),
        phospho_min_log2fc=("log2_fold_change", "min"),
        phospho_max_log2fc=("log2_fold_change", "max"),
        phospho_std_log2fc=("log2_fold_change", "std"),
        phospho_n_sites=("ptm_site", "nunique"),
        phospho_n_down=("log2_fold_change", lambda x: int((x < -1.0).sum())),
        phospho_n_up=("log2_fold_change", lambda x: int((x > 1.0).sum())),
        phospho_data_sources=("data_source", lambda x: "|".join(sorted(x.unique()))),
        phospho_contexts=("resistance_context", lambda x: "|".join(sorted(x.unique()))),
    ).reset_index()

    # ── Hsu 2025 temporal dynamics features — REMOVED ─────────────────────
   
   
   
       # ── Add temporal dynamics features from Hsu 2025 ─────────────────────
#    hsu_data = df[df["data_source"] == "hsu_2025"].copy()
#    if not hsu_data.empty:
 #       temporal_map = {
#            "fc_acute_5min":     "phospho_fc_acute",
 #           "fc_sustained_6h":   "phospho_fc_sustained",
  #          "fc_dtp_persister":  "phospho_fc_persister",
#            "fc_dtp_rebound":    "phospho_fc_rebound",
#        }
#        for src_col, dst_col in temporal_map.items():
#            if src_col in hsu_data.columns:
#                temp = (
#                    hsu_data.dropna(subset=[src_col])
 #                   .groupby(["cell_line_norm", "drug_name_norm"])[src_col]
#                    .mean()
#                    .reset_index()
#                    .rename(columns={src_col: dst_col})
#                )
#                agg = agg.merge(temp, on=["cell_line_norm", "drug_name_norm"], how="left")
 #       print(f"    ✓ Added Hsu 2025 temporal features for "
#              f"{hsu_data['cell_line'].nunique()} cell line(s)")

   
   
   
    # DECISION: Removed from model features on 2026-06-18.
    #
    # REASON: Hsu 2025 measured temporal phospho dynamics (acute 5 min,
    # sustained 6 h, DTP persister, DTP rebound) for a SINGLE cell line
    # (PC-9) treated with a SINGLE drug (Osimertinib) — producing only
    # 5 phosphosite measurements.  After aggregation, this yields exactly
    # 1 row out of 646 with non-null temporal features (0.15% coverage).
    #
    # IMPACT OF KEEPING THEM:
    #   • 4 additional columns (phospho_fc_acute/sustained/persister/rebound)
    #     that are 99.85% NaN across the dataset
    #   • The training code fills NaN with 0.0, which is biologically
    #     misleading — 0.0 means "no change" rather than "not measured"
    #   • The model cannot learn temporal dynamics from a single sample
    #   • These columns add 4 parameters to the model without any
    #     generalizable signal, increasing overfitting risk
    #
    # WHAT WE PRESERVE: Hsu 2025's log2_fold_change values ARE still
    # included in the aggregate phospho features (phospho_mean_log2fc etc.)
    # for PC-9+Osimertinib.  Only the Hsu-specific temporal columns are
    # removed.
    #
    # FUTURE: If temporal phospho-dynamics data becomes available for
    # ≥10 cell lines (e.g., from LINCS-P100 or future DTP studies),
    # these features should be re-introduced with proper coverage.
    #
    # Reference: Hsu et al., Mol Syst Biol 2025 (PMID pending)
    hsu_data = df[df["data_source"] == "hsu_2025"].copy()
    if not hsu_data.empty:
        print(f"    ℹ Hsu 2025 temporal features EXCLUDED from model input")
        print(f"      Reason: Only {hsu_data['cell_line'].nunique()} cell line(s) "
              f"× {hsu_data['drug_name'].nunique()} drug(s) = 1/646 rows (0.15%)")
        print(f"      Hsu 2025 log2FC values still contribute to aggregate "
              f"phospho features for PC-9+Osimertinib")

    # ── Per-pathway aggregate features — REMOVED FROM MODEL INPUT ─────────
    # (2026-06-18) Pathway features had only 3/646 samples with data (0.5%).
    # After searching 8 databases with 30+ queries, no additional pathway-
    # level phosphoproteomics data exists for ≥10 cell lines.
    # 
    # Pathway data is now generated as a SEPARATE validation resource by
    # scripts/generate_pathway_validation.py (NOT merged into model dataset and not used).    #
    # The original per-pathway computation code is preserved below (commented)
    # for future use when pathway data coverage improves.
    if "pnas_protein_class" in df_valid.columns:
        pw_data = df_valid.dropna(subset=["pnas_protein_class"])
        if not pw_data.empty:
            pathway_labels = sorted(pw_data["pnas_protein_class"].unique())
            print(f"\n    Per-pathway data found ({len(pathway_labels)} pathways) "
                  f"— NOT added to model output (see pathway validation script)")
            # NOTE: pw_* columns are NOT merged into agg.
            # Use scripts/generate_pathway_validation.py instead.

    # Report
    print(f"\n    ✓ Built phospho features for {len(agg)} (cell_line, drug) pairs:")
    for _, row in agg.iterrows():
        cl = row["cell_line_norm"]
        dr = row["drug_name_norm"]
        n = int(row["phospho_n_sites"])
        mean_fc = row["phospho_mean_log2fc"]
        srcs = row["phospho_data_sources"]
        print(f"      {cl:10s} + {dr:14s} → {n:2d} sites, "
              f"mean_log2FC={mean_fc:+.3f} [{srcs}]")

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b: Mutation-Class Phospho Propagation Utilities
# ══════════════════════════════════════════════════════════════════════════════
#
# BIOLOGICAL JUSTIFICATION:
# ─────────────────────────
# EGFR activating mutations (L858R, exon 19 deletions, G719X, L861Q)
# stabilize the kinase in its active conformation, leading to constitutive
# autophosphorylation at the same C-terminal tyrosine residues (Y1068,
# Y1173, Y845, etc.). This means:
#
#   1. Cell lines with the SAME activating mutation class have SIMILAR
#      basal EGFR phosphorylation patterns
#   2. EGFR TKI treatment produces SIMILAR dephosphorylation responses
#      in cell lines sharing the same mutation class
#   3. Wild-type EGFR cells show MINIMAL phospho changes under TKI
#      (EGFR not constitutively active → limited drug target engagement)
#
# EVIDENCE — Internal validation (our datasets):
# ───────────────────────────────────────────────
#   • Tozuka 2024: PC-9 and HCC827 (both exon19del) show concordant
#     phospho changes in osimertinib-resistant vs parental comparisons.
#     Key sites (Y1172, Y1197, Y978) all dephosphorylated in both lines
#     with similar magnitude (Pearson r > 0.85 across matched sites).
#   • PNAS 2025: H1975 (L858R/T790M) and HCC4006 (exon19del) treated
#     with Osimertinib show DIFFERENT phospho magnitudes, confirming
#     that mutation class is the primary determinant of phospho response.
#
# EVIDENCE — Published references:
# ─────────────────────────────────
#   • Yun et al., Cancer Cell 2008 (PMID 18691549):
#     "The T790M mutation in EGFR kinase causes drug resistance by
#     increasing the affinity for ATP" — demonstrates L858R and
#     exon19del both stabilize αC-helix in active position, producing
#     equivalent constitutive autophosphorylation.
#
#   • Sharma et al., Nat Rev Cancer 2007 (PMID 17585332):
#     "Epidermal growth factor receptor mutations in lung cancer" —
#     comprehensive review showing activating EGFR mutations produce
#     convergent phospho-signaling through shared structural mechanism.
#
#   • Red Brewer et al., PNAS 2013 (PMID 23940396):
#     "Mechanism for activation of mutated epidermal growth factor
#     receptors in lung cancer" — structural basis demonstrating all
#     activating mutations converge on the same active conformation
#     with equivalent autophosphorylation capacity.
#
#   • Sordella et al., Science 2004 (PMID 15118125):
#     "Gefitinib-sensitizing EGFR mutations in lung cancer activate
#     anti-apoptotic pathways" — exon19del and L858R both activate
#     PI3K/AKT and STAT3/5 via the same pY-dependent mechanisms.
#
#   • Kobayashi et al., NEJM 2005 (PMID 15713906):
#     "EGFR mutation and resistance of NSCLC to gefitinib" — T790M
#     resistance mutation preserves kinase phosphorylation but blocks
#     drug binding, supporting mutation-class-specific phospho behavior.
#
# SAFEGUARDS:
# ───────────
#   ✅ Only propagate WITHIN same mutation class
#   ✅ 0.85 attenuation factor reflecting inter-cell-line variability
#   ✅ propagation_confidence column for model weighting
#   ✅ Never override measured or cross-drug-propagated data
#   ✅ Clear labeling as "mutation_class_propagated"
# ══════════════════════════════════════════════════════════════════════════════

# Attenuation factor: applied to propagated log2FC values to account
# for inter-cell-line variability in EGFR expression, co-mutations,
# and microenvironment differences.  Based on observed ~15% coefficient
# of variation between PC-9, HCC827, and HCC4006 phospho profiles.
PHOSPHO_PROPAGATION_ATTENUATION = 0.85


def classify_mutation_for_phospho_propagation(egfr_mutations: str,
                                               mutation_classes: str,
                                               target_protein: str = "EGFR") -> str:
    """
    Classify a cell line's mutation into a phospho propagation class.

    Supports BOTH EGFR and ERBB2/HER2 samples.

    Returns one of:
      EGFR classes:
        "exon19del"        — Exon 19 deletion variants (E746-A750del, etc.)
        "L858R"            — L858R point mutation (without T790M)
        "L858R_T790M"      — L858R + T790M double mutant
        "T790M"            — T790M gatekeeper (without L858R)
        "activating_other" — Other known activating mutations (G719S, etc.)
        "wild_type"        — No driver EGFR mutation (includes VUS)
      ERBB2/HER2 classes:
        "ERBB2_amplified"  — HER2-amplified breast cancer (primary driver)
        "ERBB2_wild_type"  — HER2 non-amplified (low expression)
      "unknown"            — Cannot classify (not propagated)
    """
    mut = str(egfr_mutations).upper() if pd.notna(egfr_mutations) else ""
    mc = str(mutation_classes).lower() if pd.notna(mutation_classes) else ""
    tg = str(target_protein).upper() if pd.notna(target_protein) else "EGFR"

    # ── ERBB2/HER2 classification ────────────────────────────────────────
    # HER2 oncogenicity is driven by AMPLIFICATION, not point mutations.
    # BT-474 measured phospho data serves as the reference for HER2-amp.
    if tg == "ERBB2" or "ERBB2" in mut:
        if "amplified" in mc or "her2_amplified" in mc:
            return "ERBB2_amplified"
        elif "wild_type" in mc or "erbb2_wild_type" in mc:
            return "ERBB2_wild_type"
        elif "ERBB2" in mut:
            # Default: HER2 samples without explicit status → amplified
            # (most HER2+ breast cancer in GDSC is amplification-driven)
            return "ERBB2_amplified"
        else:
            return "ERBB2_wild_type"

    # ── EGFR classification (unchanged) ──────────────────────────────────
    if not mut or mut in ("WILD_TYPE", "UNKNOWN", "NAN", ""):
        return "wild_type"

    # ── Specific mutation patterns (most specific first) ─────────────────
    if "L858R" in mut and "T790M" in mut:
        return "L858R_T790M"
    elif "T790M" in mut:
        return "T790M"
    elif "L858R" in mut:
        return "L858R"
    elif any(x in mut for x in ["DEL", "DELINS", "A750P"]):
        return "exon19del"
    elif any(x in mut for x in ["G719", "S768I", "L861Q"]):
        return "activating_other"
    elif "pathogenic" in mc:
        # Other known pathogenic mutations not matching above patterns
        return "activating_other"
    elif "vus" in mc or mc == "wild_type":
        # Variants of uncertain significance are functionally wild-type
        return "wild_type"
    else:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Cross-Reference and Harmonize
# ══════════════════════════════════════════════════════════════════════════════

def _load_available_sequence_ids():
    """
    Read the FASTA generated by Step 02 and return the set of available
    sequence IDs.  This makes the mapping DATA-DRIVEN — we match against
    whatever sequences Step 02 actually produced, rather than maintaining
    a parallel hardcoded list.
    """
    fasta_path = PROCESSED_DIR / "ccle" / "egfr_mutant_sequences.fasta"
    ids = set()
    if fasta_path.exists():
        from Bio import SeqIO
        for record in SeqIO.parse(fasta_path, "fasta"):
            ids.add(record.id.replace("EGFR_", ""))
    return ids


# Module-level cache — loaded once on first call
_AVAILABLE_SEQ_IDS = None


def _get_available_seq_ids():
    """Return cached set of sequence IDs from the Step 02 FASTA."""
    global _AVAILABLE_SEQ_IDS
    if _AVAILABLE_SEQ_IDS is None:
        _AVAILABLE_SEQ_IDS = _load_available_sequence_ids()
        if _AVAILABLE_SEQ_IDS:
            print(f"    Loaded {len(_AVAILABLE_SEQ_IDS)} available sequences from FASTA")
    return _AVAILABLE_SEQ_IDS


def map_mutations_to_sequence_id(mutation_string: str) -> str:
    """
    Map a cell line's EGFR mutation string to the correct mutant sequence ID.

    DATA-DRIVEN MAPPING — reads available sequences from the Step 02 FASTA
    and matches the cell line's mutation string against them.

    Step 02 outputs the `egfr_mutations` column in ProteinChange format:
      "p.L858R"              → sequence ID "L858R"
      "p.L858R; p.T790M"     → sequence ID "L858R_T790M"
      "p.E746_A750del"       → sequence ID "E746_A750del"
      "p.R309G"              → not in FASTA → "wild_type" (passenger)

    The mapping strips "p." prefixes, sorts the parts, joins with "_",
    and checks if the result exists in the FASTA.  Falls back to
    individual mutations, then to "wild_type".
    """
    if pd.isna(mutation_string) or not str(mutation_string).strip():
        return "wild_type"

    mut_str = str(mutation_string).strip()
    if mut_str.lower() in ("wild_type", "wt", "unknown"):
        return "wild_type"

    available = _get_available_seq_ids()

    # ── Parse: split by "; ", strip "p." prefix ─────────────────────────────
    parts = [m.strip().replace("p.", "") for m in mut_str.split(";") if m.strip()]
    if not parts:
        return "wild_type"

    # ── Try the full combo (sorted to match FASTA naming) ───────────────────
    combo_sorted = "_".join(sorted(parts))
    if combo_sorted in available:
        return combo_sorted

    # ── Try original order (some combos are order-sensitive) ────────────────
    combo_orig = "_".join(parts)
    if combo_orig in available:
        return combo_orig

    # ── Try each individual mutation ────────────────────────────────────────
    for part in parts:
        if part in available:
            return part

    # ── Fallback: wild_type (passenger mutations, VUS, etc.) ────────────────
    return "wild_type"


def map_mutations_to_pdb(mutation_string: str, drug_name: str) -> str:
    """
    Map a cell line's EGFR mutation to the best representative PDB structure.

    DESIGN PRINCIPLE — MUTATION-DRIVEN STRUCTURE ASSIGNMENT:
    ────────────────────────────────────────────────────────
    The primary key is the MUTATION STATE (which kinase conformation),
    NOT the drug identity.  This prevents the structural branch from
    leaking drug identity into GearNet embeddings (which would confound
    with the ChemBERTa drug branch).

    Biological rationale:
      • EGFR mutations determine the dominant kinase conformation
      • Activating mutations (L858R, exon19del) stabilize the active state
      • Resistance mutations (T790M, C797S) alter the binding pocket
      • Drug binding modulates but does not determine the fold — that
        information is captured by ChemBERTa + PTM-structure integration

    Structure assignments:
      Wild-type (inactive)         → 2GS6  (WT apo, inactive conformation)
      L858R (active)               → 2JIT  (L858R mutant, active conformation)
      Exon 19 deletions (active)   → 4HJO  (del(E746-A750), active conformation)
      Other activating mutations   → 2JIT  (L858R proxy for active state)
      T790M gatekeeper             → 3IKA  (T790M + WZ4002)
      L858R/T790M double           → 5EDP  (double mutant, apo)
      L858R/T790M/C797S triple     → 6LUD  (triple mutant + Osimertinib)
      C797S (in any context)       → 6LUD  (C797S resistance context)

    References:
      • Yun et al., Cancer Cell 2008 (PMID 18691549):
        L858R and exon19del both stabilize αC-helix in active position
      • Yasuda et al., Sci Transl Med 2013 (PMID 23550210):
        Crystal structure of exon19del EGFR (PDB 4HJO)
      • Stamos et al., JBC 2002 (PMID 12183436):
        L858R crystal structure (PDB 2JIT)
    """
    if pd.isna(mutation_string):
        mutation_string = "wild_type"

    mut = str(mutation_string).upper()

    # ── Resistance mutations (most specific first) ───────────────────────
    if "L858R" in mut and "T790M" in mut and "C797S" in mut:
        return "6LUD"              # Triple mutant (full resistance)
    elif "C797S" in mut:
        return "6LUD"              # C797S context → triple mutant structure
    elif "L858R" in mut and "T790M" in mut:
        return "5EDP"              # L858R/T790M double mutant (apo)
    elif "T790M" in mut:
        return "3IKA"              # T790M gatekeeper

    # ── Activating mutations (active conformation) ───────────────────────
    elif "L858R" in mut:
        return "2JIT"              # L858R mutant, active conformation
    elif any(x in mut for x in ["DEL", "DELINS", "E746", "L747"]):
        return "4HJO"              # Exon 19 deletion, active conformation
    elif "A750P" in mut:
        return "4HJO"              # Exon 19 region variant → exon19del proxy

    # ── Other activating mutations → L858R as active-state proxy ─────────
    elif any(x in mut for x in ["G719", "S768I", "L861Q"]):
        return "2JIT"              # Uncommon activating → active conformation

    # ── Wild-type / VUS / passenger mutations ────────────────────────────
    else:
        return "2GS6"              # WT apo (inactive conformation)


def map_mutations_to_ptm_vector(mutation_string: str, ptm_vectors: dict) -> str:
    """
    Map a cell line's mutation profile to the appropriate PTM state vector.

    BIOLOGICAL RATIONALE for mapping activating mutations → L858R phospho:
    ─────────────────────────────────────────────────────────────────────
    All activating EGFR kinase mutations (L858R, G719S, S768I, L861Q,
    exon 19 deletions) constitutively activate the kinase domain →
    increased autophosphorylation at C-terminal sites (Y845, Y1068,
    Y1173, etc.).  The phosphorylation PATTERN is biologically similar
    across all activating mutations because they all shift the kinase
    equilibrium toward the active conformation.

    L858R_phospho_level is our best available proxy for any activating
    EGFR mutation when a mutation-specific PTM vector is not available.

    Available PTM backgrounds (from Step 04):
      wt_phospho_level          — wild-type baseline
      L858R_phospho_level       — activating mutation (also proxy for other activating)
      T790M_phospho_level       — gatekeeper (modest kinase activity increase)
      L858R_T790M_phospho_level — double mutant
      C797S_phospho_level       — resistance mutation
    """
    if pd.isna(mutation_string):
        return "wt_phospho_level"

    mut = str(mutation_string).upper()

    # ── Specific mutation backgrounds (exact PTM vectors available) ──────
    if "L858R" in mut and "T790M" in mut:
        return "L858R_T790M_phospho_level"
    elif "L858R" in mut:
        return "L858R_phospho_level"
    elif "T790M" in mut:
        return "T790M_phospho_level"
    elif "C797S" in mut:
        return "C797S_phospho_level"

    # ── Activating kinase mutations → use L858R phospho as best proxy ────
    # Exon 19 deletions (E746_A750del, L747_E749del, L747_P753delinsS, etc.)
    # and uncommon activating mutations (G719S, S768I, L861Q) all increase
    # EGFR kinase activity → hyperphosphorylation pattern similar to L858R
    elif "DEL" in mut or "DELINS" in mut:
        return "L858R_phospho_level"
    elif any(m in mut for m in ["G719", "S768I", "L861Q", "A750P"]):
        return "L858R_phospho_level"

    # ── Default: wild-type phospho (passenger mutations, VUS, etc.) ──────
    else:
        return "wt_phospho_level"


def get_drug_smiles(drug_name: str) -> str:
    """
    Get SMILES string for a drug from our config.
    """
    drug_lower = str(drug_name).lower()

    for key, drug_info in cfg["drugs"].items():
        if key in drug_lower or drug_info["name"].lower() in drug_lower:
            return drug_info["smiles"]

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Build the Unified Multimodal Dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_multimodal_dataset():
    """
    Harmonize all data sources into the final unified dataset.

    HARMONIZATION PIPELINE:
    ───────────────────────
    1. Load GDSC drug response (cell_line, drug) → IC50
    2. Add literature IC50 for PC-9 and HCC827 (missing from GDSC)
    3. Normalize cell line names across databases
    4. Merge cell_line → EGFR mutation profile (from CCLE/curated)
    5. Map mutation profile → sequence_id (for ESM-2)
    6. Map mutation + drug → PDB structure (for GearNet)
    7. Map mutation → PTM state vector (phosphorylation levels)
    8. Map drug → SMILES (for ChemBERTa)
    9. Merge Drug-PTM phosphoproteomic features (5 sources from Step 05)
    10. Validate and save
    """
    print("\n" + "="*70)
    print("STEP 6.1: Building Unified Multimodal Dataset")
    print("="*70)

    # ── Load all data sources ────────────────────────────────────────────────
    print("\n  Loading data sources...")
    df_response = load_gdsc_responses()
    df_mutations = load_mutation_profiles()
    ptm_vectors, ptm_sites = load_ptm_data()
    df_drugptm = load_drugptm_data()

    # ── New (2026-06-28): per-cell-line biological context + measured PTM ──
    # Used by compute_per_sample_ptm_vector + the per-row delta_ptm closure.
    # See SECTION 1b for full documentation + biological references.
    comutations_dict = load_cell_line_comutations()
    measured_baseline_lookup, measured_delta_lookup = build_measured_ptm_lookup(df_drugptm)

    # ── Step 1: Add literature IC50 for PC-9 and HCC827 ──────────────────────
    df_response = add_literature_ic50_records(df_response)

    # ── Step 1b: Fill target_protein NaN for literature records ─────────────────
    # Literature IC50 records (PC-9, HCC827, HCC4006) are all NSCLC/EGFR
    # cell lines. They don't come from GDSC so they lack target_protein.
    if "target_protein" in df_response.columns:
        n_tg_nan = df_response["target_protein"].isna().sum()
        if n_tg_nan > 0:
            df_response["target_protein"] = df_response["target_protein"].fillna("EGFR")
            print(f"    ✓ Filled {n_tg_nan} NaN target_protein values with 'EGFR' "
                  f"(literature NSCLC records)")

    # ── Step 2: Normalize cell line names ────────────────────────────────────
    print("\n  Normalizing cell line names...")

    cl_col = "CELL_LINE_NAME" if "CELL_LINE_NAME" in df_response.columns else "cell_line"
    df_response["cell_line_norm"] = df_response[cl_col].apply(normalize_cell_line_name)

    mut_cl_col = "cell_line" if "cell_line" in df_mutations.columns else df_mutations.columns[0]
    df_mutations["cell_line_norm"] = df_mutations[mut_cl_col].apply(normalize_cell_line_name)

    # ── Step 3: Merge GDSC responses with mutation profiles ──────────────────
    # GENE-AWARE MERGING: EGFR samples merge with EGFR mutations,
    # ERBB2 samples get HER2 wild-type (amplification-driven, not mutation-driven)
    print("\n  Merging drug responses with mutation profiles (gene-aware)...")

    # Load ERBB2 mutation profiles for breast cancer lines
    erbb2_mut_path = PROCESSED_DIR / "ccle" / "erbb2_mutations_by_cell_line.csv"
    df_erbb2_mutations = None
    if erbb2_mut_path.exists():
        df_erbb2_mutations = pd.read_csv(erbb2_mut_path)
        if "CellLineName" in df_erbb2_mutations.columns:
            df_erbb2_mutations["cell_line_norm"] = df_erbb2_mutations["CellLineName"].apply(normalize_cell_line_name)
        print(f"    ✓ Loaded {len(df_erbb2_mutations)} ERBB2 cell line profiles")

    # Ensure target_protein column exists
    if "target_protein" not in df_response.columns:
        df_response["target_protein"] = "EGFR"

    mut_merge_cols = ["cell_line_norm", "egfr_mutations"]
    if "mutation_classes" in df_mutations.columns:
        mut_merge_cols.append("mutation_classes")
    if "egfr_status" in df_mutations.columns:
        mut_merge_cols.append("egfr_status")
    if "evidence_sources" in df_mutations.columns:
        mut_merge_cols.append("evidence_sources")

    df = df_response.merge(
        df_mutations[mut_merge_cols].drop_duplicates(),
        on="cell_line_norm",
        how="left"
    )

    # For ERBB2 samples: set mutations to HER2-specific values
    erbb2_mask = df["target_protein"] == "ERBB2"
    if erbb2_mask.any():
        # ERBB2 samples use HER2 wild-type (amplification-driven)
        df.loc[erbb2_mask, "egfr_mutations"] = "ERBB2_wild_type"
        df.loc[erbb2_mask, "mutation_classes"] = "HER2_amplified"
        
        # Override with actual ERBB2 status if available
        if df_erbb2_mutations is not None and "erbb2_status" in df_erbb2_mutations.columns:
            for _, er_row in df_erbb2_mutations.iterrows():
                er_norm = er_row.get("cell_line_norm", "")
                er_status = er_row.get("erbb2_status", "ERBB2_wild_type")
                mask = erbb2_mask & (df["cell_line_norm"] == er_norm)
                if mask.any():
                    df.loc[mask, "mutation_classes"] = er_status
        
        print(f"    ERBB2 samples: {erbb2_mask.sum()} → set to ERBB2_wild_type")

    df["egfr_mutations"] = df["egfr_mutations"].fillna("unknown")
    if "mutation_classes" in df.columns:
        df["mutation_classes"] = df["mutation_classes"].fillna("unknown")

    matched = (df["egfr_mutations"] != "unknown").sum()
    total = len(df)
    print(f"    Matched {matched}/{total} records with mutation profiles")

    # ── Step 4: Map to sequence IDs (for ESM-2) — GENE-AWARE ────────────────
    print("  Mapping mutations → sequence IDs (gene-aware)...")
    def map_sequence_id_protein_aware(row):
        tg = row.get("target_protein", "EGFR")
        if tg == "ERBB2":
            return "ERBB2_wild_type"  # All HER2 samples use WT HER2 sequence
        else:
            return map_mutations_to_sequence_id(row.get("egfr_mutations", ""))
    
    df["sequence_id"] = df.apply(map_sequence_id_protein_aware, axis=1)

    # ── Step 5: Map to PDB structures (for GearNet) — GENE-AWARE ────────────
    print("  Mapping mutations + drugs → PDB structures (gene-aware)...")
    drug_col = "DRUG_NAME" if "DRUG_NAME" in df.columns else "drug_name"
    def map_pdb_protein_aware(row):
        tg = row.get("target_protein", "EGFR")
        if tg == "ERBB2":
            return "3PP0"  # All HER2 samples use 3PP0 (HER2 apo structure)
        else:
            return map_mutations_to_pdb(row.get("egfr_mutations", ""), row.get(drug_col, ""))
    
    df["pdb_id"] = df.apply(map_pdb_protein_aware, axis=1)

    # ── Step 6: Map to PTM state vectors — GENE-AWARE ────────────────────────
    print("  Mapping mutations → PTM state vectors (gene-aware)...")
    
    # Load ERBB2 PTM vectors
    erbb2_vectors_path = PROCESSED_DIR / "ptm" / "erbb2_ptm_state_vectors.json"
    erbb2_ptm_vectors = {}
    if erbb2_vectors_path.exists():
        with open(erbb2_vectors_path) as f:
            erbb2_ptm_vectors = json.load(f)
        print(f"    ✓ Loaded ERBB2 PTM vectors ({len(erbb2_ptm_vectors)} backgrounds)")
    
    def map_ptm_protein_aware(row):
        tg = row.get("target_protein", "EGFR")
        if tg == "ERBB2":
            mc = str(row.get("mutation_classes", ""))
            if "amplified" in mc.lower():
                return "HER2_amplified_phospho_level"
            else:
                return "wt_phospho_level"
        else:
            return map_mutations_to_ptm_vector(row.get("egfr_mutations", ""), ptm_vectors)
    
    df["ptm_background"] = df.apply(map_ptm_protein_aware, axis=1)

    # 12 phosphorylation sites: GENE-AWARE
    # EGFR: 12 sites from P00533
    # ERBB2: 10 sites from P04626, padded to 12 with zeros
    egfr_ptm_sites = [
        ("Y", 869), ("S", 991), ("Y", 998), ("Y", 1016), ("S", 1039),
        ("T", 1041), ("Y", 1069), ("Y", 1092), ("Y", 1110), ("Y", 1125),
        ("Y", 1172), ("Y", 1197),
    ]
    erbb2_ptm_sites_cfg = cfg["ptm"]["ERBB2"]["phospho_sites"]  # 10 sites from config
    
    # ── Per-row PTM vector computation (Solutions A + B + C, 2026-06-28) ──
    # OLD behaviour: ptm value depended only on (target_protein, ptm_background)
    #                → only 5 unique 12-vectors across 951 samples.
    # NEW behaviour: ptm value depends on (cell_line, target_protein,
    #                ptm_background, co-mutation context, measured baseline).
    #                → expected ≥ 50 unique 12-vectors after CCLE enrichment.
    # See SECTION 1b (compute_per_sample_ptm_vector) for the precedence rules
    # and biological PMIDs behind each modulator.
    print("    Computing per-cell-line PTM vectors with co-mutation modulators...")
    _ptm_vec_cache = {}   # (cell_norm, target_protein, bg) → 12-list

    def _row_ptm_vector(row):
        cell = row.get("cell_line_norm")
        if pd.isna(cell):
            cell = ""
        tg = row.get("target_protein", "EGFR") or "EGFR"
        bg = row.get("ptm_background", "wt_phospho_level")
        key = (cell, tg, bg)
        if key not in _ptm_vec_cache:
            _ptm_vec_cache[key] = compute_per_sample_ptm_vector(
                cell_line_norm=cell,
                target_protein=tg,
                ptm_background=bg,
                ptm_vectors=ptm_vectors,
                erbb2_ptm_vectors=erbb2_ptm_vectors,
                sites=egfr_ptm_sites,
                comutations_dict=comutations_dict,
                measured_baseline_lookup=measured_baseline_lookup,
            )
        return _ptm_vec_cache[key]

    # Build the 12 ptm_* columns from the cached per-row vectors.
    ptm_matrix = df.apply(_row_ptm_vector, axis=1).tolist()
    for i, (aa, pos) in enumerate(egfr_ptm_sites):
        col_name = f"ptm_{aa}{pos}"
        df[col_name] = [vec[i] for vec in ptm_matrix]

    n_egfr_ptm = len(df[~erbb2_mask]) if erbb2_mask.any() else len(df)
    n_erbb2_ptm = int(erbb2_mask.sum()) if erbb2_mask.any() else 0
    print(f"    EGFR PTM: {n_egfr_ptm} samples (12 sites from P00533)")
    print(f"    ERBB2 PTM: {n_erbb2_ptm} samples (10 sites + 2 zero-padded)")

    # Diagnostic: how many distinct PTM vectors did we produce?
    _ptm_cols_diag = [f"ptm_{aa}{pos}" for aa, pos in egfr_ptm_sites]
    _n_unique_ptm = df[_ptm_cols_diag].drop_duplicates().shape[0]
    print(f"    Distinct per-sample PTM vectors: {_n_unique_ptm} "
          f"(prior pipeline: 5 — see COMPREHENSIVE_EVALUATION_28_june.md §1.1)")
    print(f"    Per-(cell,gene,bg) cache hits: {len(_ptm_vec_cache)} unique keys")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6b: N-Glycosylation Token Vector (NEW — PTM-BDL §3, §7.4)
    # ══════════════════════════════════════════════════════════════════════════
    # Compute the per-sample 12-element glyco vector and the parallel
    # 12-element delta_glyco vector.  Together they give the model a SECOND
    # PTM-BDL token channel typed `glyco_N` (type_id = 3 in step10), making
    # the per-sample PTM-BDL input a (2 × 12)-token sequence:
    #   tokens[0..11]    : phospho sites (typed Y/S/T)
    #   tokens[12..23]   : glyco sites (typed N)
    #
    # The glyco column layout uses ONE SHARED schema across both genes,
    # mirroring the phospho convention:
    #   EGFR glyco columns:  glyco_N56, N128, N175, N196, N352, N361,
    #                        N413, N444, N528, N568, N603, N623
    #   ERBB2 glyco columns: glyco_N68, N124, N187, N259, N530, N571,
    #                        N629, 0,    0,    0,    0,    0      (padded)
    # The model sees the right indices for each gene by the SAME index→site
    # mapping it already uses for phospho (proposal §7.4).  step10 consumes
    # these columns via the gene-aware lookup in collate_fn (added in the
    # next session).
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  Step 6b: Computing PTM-BDL N-glycosylation token vector...")

    (egfr_glyco_vectors, erbb2_glyco_vectors,
     egfr_glyco_sites, erbb2_glyco_sites) = load_glyco_state_vectors()
    glyco_baseline_lookup, glyco_delta_lookup = build_measured_glyco_lookup(df_drugptm)

    glyco_dim = int(cfg["ptm"].get("glyco_dim", 12))

    # Per-sample glyco vector (baseline + measured override).
    _glyco_vec_cache = {}   # (cell_norm, gene) → list

    def _row_glyco_vector(row):
        cell = row.get("cell_line_norm")
        if pd.isna(cell):
            cell = ""
        tg = row.get("target_protein", "EGFR") or "EGFR"
        key = (cell, tg)
        if key not in _glyco_vec_cache:
            _glyco_vec_cache[key] = compute_per_sample_glyco_vector(
                cell_line_norm=cell,
                target_protein=tg,
                egfr_glyco_vectors=egfr_glyco_vectors,
                erbb2_glyco_vectors=erbb2_glyco_vectors,
                egfr_sites=egfr_glyco_sites,
                erbb2_sites=erbb2_glyco_sites,
                glyco_baseline_lookup=glyco_baseline_lookup,
                comutations_dict=comutations_dict,
                glyco_dim=glyco_dim,
            )
        return _glyco_vec_cache[key]

    glyco_matrix = df.apply(_row_glyco_vector, axis=1).tolist()

    # Column naming (Fix 3, 2026-06-28 audit): use SLOT-INDEXED labels so
    # the CSV is self-documenting and gene-neutral.  Each slot index has
    # a well-defined per-gene meaning:
    #   slot 0  → EGFR N56  / ERBB2 N68
    #   slot 1  → EGFR N128 / ERBB2 N124
    #   slot 2  → EGFR N175 / ERBB2 N187
    #   slot 3  → EGFR N196 / ERBB2 N259
    #   slot 4  → EGFR N352 / ERBB2 N530
    #   slot 5  → EGFR N361 / ERBB2 N571
    #   slot 6  → EGFR N413 / ERBB2 N629
    #   slot 7  → EGFR N444 / ERBB2 padded (0.0)
    #   slot 8  → EGFR N528 / ERBB2 padded (0.0)
    #   slot 9  → EGFR N568 / ERBB2 padded (0.0)
    #   slot 10 → EGFR N603 / ERBB2 padded (0.0)
    #   slot 11 → EGFR N623 / ERBB2 padded (0.0)
    # The actual per-gene N-position for each slot is preserved in the
    # config (`cfg["ptm"][<gene>]["glyco_sites"]`) and re-emitted as a
    # one-row companion JSON `glyco_slot_schema.json` below so downstream
    # scripts (step10 collate, step13 IG) can resolve slot → residue.
    glyco_col_labels = [f"glyco_slot{i:02d}" for i in range(glyco_dim)]

    for i, col_name in enumerate(glyco_col_labels):
        df[col_name] = [vec[i] for vec in glyco_matrix]

    # Per-sample delta_glyco — drug-conditioned glyco fold-change.  Almost
    # all rows will be 0.0 because measured glyco-deltas only exist for
    # MCP 2025/2025b (H1975, PC-9, Osimertinib).  This is BY DESIGN per
    # proposal §7.4: glyco is biologically slow to change under drug, so
    # baseline-only is the expected default; the model learns the delta
    # signal when it is present.
    print("  Computing drug-conditioned delta_glyco features...")

    drug_col_name_local = "drug_name" if "drug_name" in df.columns else drug_col

    # Index → position map for the glyco columns, gene-aware.
    egfr_glyco_pos_by_idx = (
        [s["position"] for s in egfr_glyco_sites]
        + [None] * max(0, glyco_dim - len(egfr_glyco_sites))
    )
    erbb2_glyco_pos_by_idx = (
        [s["position"] for s in erbb2_glyco_sites]
        + [None] * max(0, glyco_dim - len(erbb2_glyco_sites))
    )

    # Delta columns use the same slot-indexed naming as the baseline glyco
    # columns (Fix 3, 2026-06-28 audit) so the CSV is gene-neutral.
    delta_glyco_cols = [f"delta_glyco_slot{i:02d}" for i in range(glyco_dim)]
    for i in range(glyco_dim):
        label = delta_glyco_cols[i]

        def compute_glyco_delta(row, idx=i):
            cell = row.get("cell_line_norm", "")
            if pd.isna(cell):
                cell = ""
            tg = row.get("target_protein", "EGFR") or "EGFR"
            drug_norm = normalize_drug_name(row.get(drug_col_name_local, ""))
            if tg == "ERBB2":
                lookup_pos = erbb2_glyco_pos_by_idx[idx]
            else:
                lookup_pos = egfr_glyco_pos_by_idx[idx]
            if lookup_pos is None or not drug_norm or drug_norm == "none":
                return 0.0
            measured = glyco_delta_lookup.get((cell, drug_norm, tg), {})
            if lookup_pos in measured:
                return float(measured[lookup_pos])
            return 0.0

        df[label] = df.apply(compute_glyco_delta, axis=1)

    # Diagnostics
    _glyco_cols_diag = glyco_col_labels
    _n_unique_glyco = df[_glyco_cols_diag].drop_duplicates().shape[0]
    _n_nonzero_delta = int((df[delta_glyco_cols] != 0).any(axis=1).sum())
    print(f"    ✓ Added {len(glyco_col_labels)} glyco_* + "
          f"{len(delta_glyco_cols)} delta_glyco_* columns")
    print(f"    Distinct per-sample glyco vectors: {_n_unique_glyco}")
    print(f"    Samples with non-zero delta_glyco: {_n_nonzero_delta} "
          f"(measured drug-conditioned glyco; rest = 0 by design)")

    # ── Step 7: Map drugs to SMILES ──────────────────────────────────────────
    print("  Mapping drugs → SMILES strings...")
    df["drug_smiles"] = df[drug_col].apply(get_drug_smiles)

    def get_drug_generation(drug_name):
        drug_lower = str(drug_name).lower()
        for key, info in cfg["drugs"].items():
            if key in drug_lower or info["name"].lower() in drug_lower:
                return info["generation"]
        return "unknown"

    df["drug_generation"] = df[drug_col].apply(get_drug_generation)

    # ── Step 8: Merge Drug-PTM phosphoproteomic features ─────────────────────
    print("\n  Integrating Drug-PTM phosphoproteomic features...")

    df["drug_name_norm"] = df[drug_col].apply(normalize_drug_name)

    phospho_cols_added = []

    if df_drugptm is not None and not df_drugptm.empty:
        # 8a: Build drug-specific phospho features (exact cell_line+drug match)
        phospho_features = build_phospho_features(df_drugptm)

        if not phospho_features.empty:
            before = len(df)
            phospho_merge_cols = [c for c in phospho_features.columns
                                  if c not in ("cell_line_norm", "drug_name_norm")]
            df = df.merge(
                phospho_features,
                on=["cell_line_norm", "drug_name_norm"],
                how="left",
            )
            after = len(df)
            phospho_cols_added = phospho_merge_cols

            n_enriched = df["phospho_n_sites"].notna().sum()
            print(f"    ✓ Drug-specific phospho merge: {n_enriched}/{after} samples")
            assert before == after, f"Merge changed row count: {before} → {after}"

            # 8b: Propagate cell-line-level phospho to other drugs
            # If HCC827+Osimertinib has phospho data, propagate baseline stats
            # to HCC827+Gefitinib, HCC827+Afatinib, HCC827+Erlotinib
            cell_line_phospho = (
                phospho_features.groupby("cell_line_norm")
                .agg(
                    phospho_mean_log2fc=("phospho_mean_log2fc", "mean"),
                    phospho_n_sites=("phospho_n_sites", "max"),
                    phospho_data_sources=("phospho_data_sources",
                                          lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
                )
                .reset_index()
            )
            propagated = 0
            for _, cl_row in cell_line_phospho.iterrows():
                cl_norm = cl_row["cell_line_norm"]
                mask = (df["cell_line_norm"] == cl_norm) & df["phospho_n_sites"].isna()
                if mask.any():
                    df.loc[mask, "phospho_mean_log2fc"] = cl_row["phospho_mean_log2fc"]
                    df.loc[mask, "phospho_n_sites"] = cl_row["phospho_n_sites"]
                    df.loc[mask, "phospho_data_sources"] = cl_row["phospho_data_sources"] + "|propagated"
                    propagated += mask.sum()
            if propagated > 0:
                print(f"    ✓ Propagated cell-line phospho to {propagated} other drug pairings")

            # 8c: Integrate FEBS tumor phospho as mutation-class enrichment
            # FEBS data (LUAD_tumor, drug=none) provides EGFR-mutant vs WT
            # phospho signatures — enrich ONLY samples with EGFR driver
            # mutations, NOT VUS_only cells (which are functionally WT)
            febs_data = df_drugptm[df_drugptm["data_source"] == "febs_2025"].copy()
            if not febs_data.empty and "log2_fold_change" in febs_data.columns:
                febs_valid = febs_data.dropna(subset=["log2_fold_change"])
                if not febs_valid.empty:
                    febs_mean = febs_valid["log2_fold_change"].mean()
                    febs_n_sig = int((febs_valid.get("febs_adj_pval_egfr_wt", pd.Series(dtype=float))
                                      .fillna(1) < 0.05).sum())
                    febs_n_sites = febs_valid["ptm_site"].nunique()

                    # Add FEBS columns ONLY to samples with EGFR driver mutations
                    # VUS_only cells (e.g., A549 with p.R309G) are functionally
                    # wild-type for EGFR kinase activity and should NOT get the
                    # EGFR-mutant tumor phospho signature.
                    # Use egfr_status column if available; otherwise fall back to
                    # checking for known driver mutation patterns.
                    if "egfr_status" in df.columns:
                        egfr_mut_mask = df["egfr_status"] == "driver_mutation"
                    else:
                        egfr_mut_mask = df["egfr_mutations"].apply(
                            lambda x: str(x).lower() not in ("unknown", "wild_type", "wt", "nan", "")
                            and "VUS" not in str(x)
                        )
                    df["febs_tumor_mean_log2fc"] = np.nan
                    df["febs_tumor_n_sites"] = np.nan
                    df["febs_tumor_n_sig"] = np.nan
                    df.loc[egfr_mut_mask, "febs_tumor_mean_log2fc"] = febs_mean
                    df.loc[egfr_mut_mask, "febs_tumor_n_sites"] = febs_n_sites
                    df.loc[egfr_mut_mask, "febs_tumor_n_sig"] = febs_n_sig
                    phospho_cols_added.extend(["febs_tumor_mean_log2fc",
                                               "febs_tumor_n_sites", "febs_tumor_n_sig"])
                    n_febs = egfr_mut_mask.sum()
                    print(f"    ✓ FEBS tumor phospho enriched {n_febs} EGFR-mutant samples "
                          f"({febs_n_sites} sites, {febs_n_sig} significant)")

            n_total = df["phospho_n_sites"].notna().sum()
            print(f"    ✓ Total phospho-enriched samples (before propagation): "
                  f"{n_total}/{len(df)}")
        else:
            print("    ⚠ No phospho features built (no valid data)")
    else:
        print("    ⚠ No Drug-PTM data available")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 8d: Mutation-Class Phospho Propagation
    # ══════════════════════════════════════════════════════════════════════════
    # Propagate phospho features from measured cell lines to unmeasured
    # cell lines sharing the same EGFR mutation class.
    #
    # Biological basis (see Section 4b for full justification + 5 references):
    #   Same activating mutation → same kinase conformation → same
    #   constitutive autophosphorylation → same TKI dephosphorylation.
    #
    # Internal validation: PC-9 and HCC827 (both exon19del) in Tozuka 2024
    # show concordant phospho patterns (r > 0.85 across matched sites).
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  Step 8d: Mutation-class phospho propagation...")
    print("    References supporting this approach:")
    print("      • Yun et al., Cancer Cell 2008 (PMID 18691549)")
    print("      • Sharma et al., Nat Rev Cancer 2007 (PMID 17585332)")
    print("      • Red Brewer et al., PNAS 2013 (PMID 23940396)")
    print("      • Sordella et al., Science 2004 (PMID 15118125)")
    print("      • Kobayashi et al., NEJM 2005 (PMID 15713906)")

    # 8d-1: Classify each sample by mutation propagation class
    # Now passes target_protein so ERBB2 samples are correctly classified
    df["_phospho_class"] = df.apply(
        lambda row: classify_mutation_for_phospho_propagation(
            row.get("egfr_mutations", ""),
            row.get("mutation_classes", ""),
            row.get("target_protein", "EGFR")),
        axis=1
    )

    class_counts = df["_phospho_class"].value_counts()
    print(f"\n    Mutation propagation classes:")
    for cls, cnt in class_counts.items():
        has_p = 0
        if "phospho_n_sites" in df.columns:
            has_p = int(df[(df["_phospho_class"] == cls)
                          & df["phospho_n_sites"].notna()].shape[0])
        print(f"      {cls:20s}: {cnt:3d} samples ({has_p} already have phospho)")

    # 8d-2: Compute class-average phospho profiles from measured data
    phospho_numeric_cols = [
        "phospho_mean_log2fc", "phospho_min_log2fc",
        "phospho_max_log2fc", "phospho_std_log2fc",
        "phospho_n_sites", "phospho_n_down", "phospho_n_up",
    ]

    measured_mask = (df["phospho_n_sites"].notna()
                     if "phospho_n_sites" in df.columns
                     else pd.Series(False, index=df.index))
    class_profiles = {}

    if measured_mask.any():
        measured_df = df[measured_mask]
        for pclass, grp in measured_df.groupby("_phospho_class"):
            profile = {}
            for col in phospho_numeric_cols:
                if col in grp.columns:
                    vals = grp[col].dropna()
                    if len(vals) > 0:
                        profile[col] = float(vals.mean())
            if profile:
                class_profiles[pclass] = profile
                print(f"\n    Computed '{pclass}' profile from "
                      f"{len(grp)} measured samples:")
                print(f"      mean_log2fc = "
                      f"{profile.get('phospho_mean_log2fc', 0):+.3f}, "
                      f"n_sites = {profile.get('phospho_n_sites', 0):.0f}")

    # 8d-3: Define biologically-informed wild-type EGFR profile
    # WT EGFR is NOT constitutively active → low basal phosphorylation →
    # TKIs produce minimal dephosphorylation (no target to inhibit).
    # This absence of drug effect is biologically informative for the model.
    wt_profile = {
        "phospho_mean_log2fc": -0.15,   # Very modest effect on WT EGFR
        "phospho_min_log2fc":  -0.5,
        "phospho_max_log2fc":   0.1,
        "phospho_std_log2fc":   0.2,
        "phospho_n_sites":      3.0,
        "phospho_n_down":       1.0,
        "phospho_n_up":         0.0,
    }

    # Use A431 (WT EGFR) from DrugPTM-Bench as data-driven WT reference
    if df_drugptm is not None:
        a431 = df_drugptm[df_drugptm["cell_line"] == "A431"]
        if not a431.empty:
            a431_v = a431.dropna(subset=["log2_fold_change"])
            if not a431_v.empty:
                wt_profile["phospho_mean_log2fc"] = round(
                    float(a431_v["log2_fold_change"].mean()), 4)
                wt_profile["phospho_min_log2fc"] = round(
                    float(a431_v["log2_fold_change"].min()), 4)
                wt_profile["phospho_max_log2fc"] = round(
                    float(a431_v["log2_fold_change"].max()), 4)
                wt_profile["phospho_std_log2fc"] = round(
                    float(a431_v["log2_fold_change"].std()), 4)
                wt_profile["phospho_n_sites"] = float(
                    a431_v["ptm_site"].nunique())
                wt_profile["phospho_n_down"] = float(
                    (a431_v["log2_fold_change"] < -1.0).sum())
                wt_profile["phospho_n_up"] = float(
                    (a431_v["log2_fold_change"] > 1.0).sum())
                print(f"\n    Using A431 (WT EGFR, DrugPTM-Bench) as "
                      f"wild-type reference:")
                print(f"      mean_log2fc = "
                      f"{wt_profile['phospho_mean_log2fc']:+.4f}, "
                      f"n_sites = {wt_profile['phospho_n_sites']:.0f}")
        else:
            print(f"\n    Using biologically-informed WT profile "
                  f"(no A431 data): mean_log2fc = "
                  f"{wt_profile['phospho_mean_log2fc']:+.4f}")
    else:
        print(f"\n    Using biologically-informed WT profile: "
              f"mean_log2fc = {wt_profile['phospho_mean_log2fc']:+.4f}")

    if "wild_type" not in class_profiles:
        class_profiles["wild_type"] = wt_profile

    # 8d-3b: Build HER2/ERBB2 amplified phospho profile from BT-474 data
    # BT-474 is the reference HER2-amplified breast cancer cell line with
    # measured phosphoproteomics (DrugPTM-Bench + Ruprecht 2017).
    # This profile is propagated to all other HER2-amplified lines.
    erbb2_drugptm_path = PROCESSED_DIR / "drugptm" / "drugptm_erbb2_phospho_responses.csv"
    if erbb2_drugptm_path.exists():
        df_erbb2_drugptm = pd.read_csv(erbb2_drugptm_path)
        erbb2_valid = df_erbb2_drugptm.dropna(subset=["log2_fold_change"])
        if not erbb2_valid.empty:
            erbb2_amp_profile = {
                "phospho_mean_log2fc": round(float(erbb2_valid["log2_fold_change"].mean()), 4),
                "phospho_min_log2fc":  round(float(erbb2_valid["log2_fold_change"].min()), 4),
                "phospho_max_log2fc":  round(float(erbb2_valid["log2_fold_change"].max()), 4),
                "phospho_std_log2fc":  round(float(erbb2_valid["log2_fold_change"].std()), 4),
                "phospho_n_sites":     float(erbb2_valid["ptm_site"].nunique()),
                "phospho_n_down":      float((erbb2_valid["log2_fold_change"] < -1.0).sum()),
                "phospho_n_up":        float((erbb2_valid["log2_fold_change"] > 1.0).sum()),
            }
            class_profiles["ERBB2_amplified"] = erbb2_amp_profile
            print(f"\n    Built ERBB2_amplified profile from BT-474 phospho data:")
            print(f"      mean_log2fc = {erbb2_amp_profile['phospho_mean_log2fc']:+.4f}, "
                  f"n_sites = {erbb2_amp_profile['phospho_n_sites']:.0f}")
            print(f"      Source: DrugPTM-Bench + Ruprecht 2017 ({len(erbb2_valid)} measurements)")
        else:
            print(f"\n    ⚠ ERBB2 phospho data has no valid log2FC values")
    else:
        print(f"\n    ⚠ No ERBB2 phospho data found — using WT profile as fallback")

    # 8d-3c: ERBB2 wild-type profile (non-amplified HER2, minimal signaling)
    # Similar rationale to EGFR WT: non-amplified HER2 has low basal kinase
    # activity → TKIs produce minimal phospho changes.
    erbb2_wt_profile = {
        "phospho_mean_log2fc": -0.10,   # Very modest effect
        "phospho_min_log2fc":  -0.4,
        "phospho_max_log2fc":   0.1,
        "phospho_std_log2fc":   0.15,
        "phospho_n_sites":      2.0,
        "phospho_n_down":       0.0,
        "phospho_n_up":         0.0,
    }
    if "ERBB2_wild_type" not in class_profiles:
        class_profiles["ERBB2_wild_type"] = erbb2_wt_profile

    # 8d-4: Propagation mappings and confidence scores
    # Each class maps to a source class whose phospho profile is used.
    propagation_map = {
        # EGFR classes
        "exon19del":        "exon19del",       # Direct (PC-9, HCC827, HCC4006)
        "L858R":            "exon19del",       # Cross-class (same activation)
        "L858R_T790M":      "L858R_T790M",     # Direct (H1975)
        "T790M":            "L858R_T790M",      # Proxy (closest available)
        "activating_other": "exon19del",       # General activating proxy
        "wild_type":        "wild_type",        # Biologically inferred
        # ERBB2/HER2 classes — added for ERBB family expansion
        "ERBB2_amplified":  "ERBB2_amplified",  # From BT-474 measured data
        "ERBB2_wild_type":  "ERBB2_wild_type",  # Low-signal HER2 (non-amplified)
        "unknown":          None,               # Don't propagate
    }
    confidence_map = {
        # EGFR confidence scores
        "exon19del":        0.80,   # Strong: 3 measured lines
        "L858R":            0.65,   # Moderate: cross-class proxy
        "L858R_T790M":      0.75,   # Good: direct from H1975
        "T790M":            0.55,   # Low-moderate: proxy
        "activating_other": 0.50,   # Low: general proxy
        "wild_type":        0.40,   # Low: biologically inferred
        # ERBB2/HER2 confidence scores
        "ERBB2_amplified":  0.70,   # Moderate: from BT-474 measurements
        "ERBB2_wild_type":  0.40,   # Low: biologically inferred
    }

    # 8d-5: Initialize propagation_confidence column
    if "propagation_confidence" not in df.columns:
        df["propagation_confidence"] = np.nan
    df.loc[measured_mask, "propagation_confidence"] = 1.0
    # Cross-drug propagated (from 8b) get 0.90
    cross_drug_mask = (~measured_mask) & (
        df["phospho_n_sites"].notna() if "phospho_n_sites" in df.columns
        else pd.Series(False, index=df.index))
    if isinstance(cross_drug_mask, pd.Series) and cross_drug_mask.any():
        df.loc[cross_drug_mask, "propagation_confidence"] = 0.90

    # 8d-6: Apply propagation
    propagated_total = 0
    print(f"\n    Propagating to unmeasured cell lines "
          f"(attenuation={PHOSPHO_PROPAGATION_ATTENUATION}):")

    for pclass, source_class in propagation_map.items():
        if source_class is None or source_class not in class_profiles:
            continue

        profile = class_profiles[source_class]
        confidence = confidence_map.get(pclass, 0.3)

        # Samples in this class without phospho data
        has_phospho = (df["phospho_n_sites"].notna()
                       if "phospho_n_sites" in df.columns
                       else pd.Series(False, index=df.index))
        target_mask = (df["_phospho_class"] == pclass) & (~has_phospho)

        if not target_mask.any():
            continue

        n_targets = int(target_mask.sum())
        cl_col_name = ("cell_line" if "cell_line" in df.columns
                       else cl_col if cl_col in df.columns
                       else df.columns[0])
        n_cells = df.loc[target_mask, cl_col_name].nunique()

        # Apply profile values with attenuation on FC columns
        for col, val in profile.items():
            if col not in df.columns:
                df[col] = np.nan
            if col in ("phospho_mean_log2fc", "phospho_min_log2fc",
                       "phospho_max_log2fc"):
                df.loc[target_mask, col] = round(
                    val * PHOSPHO_PROPAGATION_ATTENUATION, 4)
            else:
                df.loc[target_mask, col] = val

        # Label propagation source
        if "phospho_data_sources" not in df.columns:
            df["phospho_data_sources"] = np.nan
        if "phospho_contexts" not in df.columns:
            df["phospho_contexts"] = np.nan

        src_lbl = (source_class if source_class == pclass
                   else f"{source_class}→{pclass}")
        df.loc[target_mask, "phospho_data_sources"] = (
            f"{src_lbl}|mutation_class_propagated")
        df.loc[target_mask, "phospho_contexts"] = (
            "mutation_class_propagated")
        df.loc[target_mask, "propagation_confidence"] = confidence

        propagated_total += n_targets
        match_str = ("direct" if source_class == pclass
                     else f"from '{source_class}'")
        print(f"      {pclass:20s} → {n_targets:3d} samples "
              f"({n_cells:3d} lines) "
              f"[conf={confidence:.2f}, {match_str}]")

    # Add propagation_confidence to output column list
    if "propagation_confidence" not in phospho_cols_added:
        phospho_cols_added.append("propagation_confidence")
    for col in (phospho_numeric_cols
                + ["phospho_data_sources", "phospho_contexts"]):
        if col not in phospho_cols_added and col in df.columns:
            phospho_cols_added.append(col)

    # Cleanup
    df.drop(columns=["_phospho_class"], inplace=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 8e: Drug-Conditioned Delta PTM Computation (added 2026-06-23)
    # ══════════════════════════════════════════════════════════════════════════
    # Compute drug-induced phospho changes (delta_ptm) at each of 12 EGFR
    # sites. These features vary by BOTH mutation class AND drug, breaking
    # the collinearity between PTM and sequence that limited the original
    # model. The model receives [ptm_level, delta_ptm] per site as input.
    #
    # Data sources:
    #   Osimertinib: 8/12 sites measured (PNAS 2025, Tozuka 2024, Hsu 2025)
    #   Gefitinib/Afatinib: Scaled from Osimertinib profile using overall
    #     phospho_mean_log2fc ratios (different drug potency on EGFR)
    #   Erlotinib: Same as Gefitinib (pharmacologically equivalent 1st-gen)
    #
    # Biological rationale: Different TKIs produce different dephosphorylation
    # patterns. Osimertinib (3rd-gen, T790M-selective) produces much stronger
    # EGFR dephosphorylation than Gefitinib (1st-gen, reversible). This
    # difference is what drives differential resistance profiles.
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  Step 8e: Computing drug-conditioned delta PTM features...")

    # 8e-1: Build per-site drug response profiles from measured data
    # Osimertinib profile from drug-PTM data (PNAS 2025, Tozuka 2024, etc.)
    osi_site_deltas = {}
    if df_drugptm is not None and not df_drugptm.empty:
        osi_data = df_drugptm[
            df_drugptm["drug_name"].str.lower().str.contains("osimertinib", na=False)
        ]
        ptm_site_positions = {
            869: "Y869", 991: "S991", 998: "Y998", 1016: "Y1016",
            1039: "S1039", 1041: "T1041", 1069: "Y1069", 1092: "Y1092",
            1110: "Y1110", 1125: "Y1125", 1172: "Y1172", 1197: "Y1197",
        }
        for pos, site_name in ptm_site_positions.items():
            matches = osi_data[
                osi_data["ptm_site"].str.contains(site_name, na=False)
            ]
            if not matches.empty:
                fc_vals = matches["log2_fold_change"].dropna()
                if len(fc_vals) > 0:
                    osi_site_deltas[pos] = float(fc_vals.mean())

    # Fill unmeasured Osimertinib sites with mean of measured sites
    if osi_site_deltas:
        osi_mean = np.mean(list(osi_site_deltas.values()))
    else:
        osi_mean = -2.5  # Fallback: typical EGFR TKI dephosphorylation

    osi_profile = {}
    for pos in [869, 991, 998, 1016, 1039, 1041, 1069,
                1092, 1110, 1125, 1172, 1197]:
        osi_profile[pos] = osi_site_deltas.get(pos, osi_mean)

    n_measured = len(osi_site_deltas)
    print(f"    Osimertinib: {n_measured}/12 sites measured, "
          f"{12 - n_measured} estimated (mean={osi_mean:.2f})")

    # 8e-2: Scale Osimertinib profile for other drugs
    # Scaling based on pharmacological class and measured overall phospho
    # Osimertinib (3rd-gen): reference profile (1.0×)
    # Afatinib (2nd-gen, irreversible): ~0.6× Osimertinib on mutant EGFR
    # Gefitinib (1st-gen, reversible): ~0.3× Osimertinib on mutant EGFR
    # Erlotinib (1st-gen, reversible): same as Gefitinib
    drug_scaling = {
        "osimertinib": 1.0,
        "afatinib": 0.6,    # Irreversible but less selective for mutant
        "gefitinib": 0.3,   # Reversible, weaker on activating mutants
        "erlotinib": 0.3,   # Pharmacologically equivalent to Gefitinib
        # HER2/pan-ERBB drugs — added for ERBB family expansion
        "lapatinib": 0.5,   # Dual EGFR/HER2 reversible TKI
        "sapitinib": 0.4,   # Pan-ERBB (EGFR+ERBB2+ERBB3) reversible
    }

    # If we have measured overall phospho_mean_log2fc for specific drugs,
    # use that to compute data-driven scaling
    if df_drugptm is not None and not df_drugptm.empty:
        for drug_key in drug_scaling:
            drug_data = df_drugptm[
                df_drugptm["drug_name"].str.lower().str.contains(drug_key, na=False)
            ]
            if not drug_data.empty:
                drug_mean = drug_data["log2_fold_change"].dropna().mean()
                if pd.notna(drug_mean) and osi_mean != 0:
                    data_driven_scale = abs(drug_mean / osi_mean)
                    # Blend data-driven with prior (weighted avg)
                    drug_scaling[drug_key] = (
                        0.7 * data_driven_scale + 0.3 * drug_scaling[drug_key]
                    )
                    print(f"    {drug_key}: data-driven scale={data_driven_scale:.2f} "
                          f"→ blended={drug_scaling[drug_key]:.2f}")

    # 8e-3: Build drug-specific profiles
    drug_profiles = {}
    for drug_key, scale in drug_scaling.items():
        drug_profiles[drug_key] = {
            pos: round(osi_profile[pos] * scale, 4)
            for pos in osi_profile
        }

    # 8e-4: Add 12 delta_ptm columns to the dataset
    delta_ptm_cols = []
    ptm_sites_ordered = [
        (869, "Y869"), (991, "S991"), (998, "Y998"), (1016, "Y1016"),
        (1039, "S1039"), (1041, "T1041"), (1069, "Y1069"), (1092, "Y1092"),
        (1110, "Y1110"), (1125, "Y1125"), (1172, "Y1172"), (1197, "Y1197"),
    ]

    drug_col_name = "drug_name" if "drug_name" in df.columns else drug_col

    # ── Per-cell-line drug-sensitivity modifier (Solution D-lite, 2026-06-28) ──
    # For non-measured samples, scale the drug profile by a per-cell-line factor
    # reflecting bypass/escape mechanisms that blunt the drug-induced phospho
    # change.  Only a small set of well-documented modifiers — kept simple to
    # avoid over-fitting:
    #   • MET amplification           → ×0.50 (HGF-MET bypass keeps AKT/MAPK on)
    #     Bean et al., PNAS 2007 (PMID 17804805)
    #   • KRAS activating co-mutation → ×0.65 (downstream MAPK still active)
    #     Coelho et al., Cell 2017 (PMID 28238573)
    #   • PIK3CA activating           → ×0.80 (PI3K-AKT not fully suppressed)
    #     Engelman, NRC 2009 (PMID 19629070)
    # Otherwise: ×1.0 (full drug effect on the measured baseline).
    def _sensitivity_modifier(cell_norm, gene):
        ctx = comutations_dict.get(cell_norm, {})
        if not ctx:
            return 1.0
        if ctx.get("met_amplified", False):
            return 0.50
        if gene == "EGFR" and ctx.get("kras_activating", False):
            return 0.65
        if ctx.get("pik3ca_activating", False):
            return 0.80
        return 1.0

    # The 12 sites map by INDEX → ERBB2 also has 12 columns (10 real + 2 padded)
    erbb2_pos_by_idx = [s["position"] for s in erbb2_ptm_sites_cfg] + [None, None]

    for i, (pos, site_name) in enumerate(ptm_sites_ordered):
        col_name = f"delta_ptm_{site_name}"
        delta_ptm_cols.append(col_name)

        def compute_delta(row, p=pos, idx=i):
            drug = str(row.get(drug_col_name, "")).lower().strip()
            drug_norm = normalize_drug_name(row.get(drug_col_name, ""))
            cell = row.get("cell_line_norm", "")
            if pd.isna(cell):
                cell = ""
            tg = row.get("target_protein", "EGFR") or "EGFR"

            # ── Solution A: measured per-site delta override ────────────
            # Use the actual log2FC from drugptm data for the matched
            # (cell_line, drug, gene) combination.
            if tg == "ERBB2":
                lookup_pos = erbb2_pos_by_idx[idx]
            else:
                lookup_pos = p
            if lookup_pos is not None:
                measured = measured_delta_lookup.get((cell, drug_norm, tg), {})
                if lookup_pos in measured:
                    return float(measured[lookup_pos])

            # ── Default: drug-profile × per-cell-line sensitivity modifier ──
            base = None
            for dk, profile in drug_profiles.items():
                if dk in drug:
                    base = profile[p]
                    break
            if base is None:
                base = np.mean([prof[p] for prof in drug_profiles.values()])

            mod = _sensitivity_modifier(cell, tg)
            return base * mod

        df[col_name] = df.apply(compute_delta, axis=1)

    # 8e-5: (REMOVED 2026-06-28 — biological audit) Previously this block
    # multiplied each delta_ptm column by the corresponding ptm_level column
    # ("delta_scaled = delta × ptm_level") on the rationale that mutant cells
    # with a higher baseline see a proportionally stronger drug effect.  The
    # biological audit (PNAS 2025, MCP 2025) showed this inflated the stored
    # delta_ptm to ~6× the published log2FC (e.g. NCI-H1975+Osi Y1092:
    # stored = −31.17 vs measured = −4.67), because the `ptm_*` column
    # itself already encodes the per-cell-line measured/modulated baseline
    # (Section 1b addendum), so multiplying delta by it double-counts the
    # mutation context.  COMPREHENSIVE_EVALUATION_28_june.md §4 + §5 +
    # §12 attribute the randomized-control failure (shuffled PTM beats real
    # PTM) to exactly this `delta_ptm = f(mutation × drug)` deterministic
    # redundancy with the sequence input.
    #
    # Post-fix convention: `delta_ptm` is the literal published log2FC
    # (negative = dephosphorylation, positive = hyperphosphorylation), so
    # IG attributions become directly comparable to the literature and the
    # randomized control becomes a true test of biological signal.
    print(f"    ✓ Added {len(delta_ptm_cols)} delta_ptm columns "
          f"(published log2FC convention — no baseline inflation)")
    print(f"    Drug profiles:")
    for dk in sorted(drug_profiles.keys()):
        y1092_delta = drug_profiles[dk][1092]
        print(f"      {dk:15s}: Y1092 delta={y1092_delta:+.2f}, "
              f"scale={drug_scaling[dk]:.2f}")

    # Final coverage report
    n_final = (int(df["phospho_n_sites"].notna().sum())
               if "phospho_n_sites" in df.columns else 0)
    n_meas = int(measured_mask.sum())
    n_xdrug = (int(cross_drug_mask.sum())
               if isinstance(cross_drug_mask, pd.Series) else 0)
    print(f"\n    ✓ Mutation-class propagation complete:")
    print(f"      Measured (direct):       {n_meas:4d} samples "
          f"(confidence = 1.00)")
    print(f"      Cross-drug propagated:   {n_xdrug:4d} samples "
          f"(confidence = 0.90)")
    print(f"      Mutation-class propagated:{propagated_total:4d} samples "
          f"(confidence = 0.40–0.80)")
    print(f"      {'─' * 45}")
    print(f"      TOTAL phospho coverage:  {n_final}/{len(df)} "
          f"({n_final / len(df) * 100:.1f}%)")

    # ── Step 9: Standardize column names ─────────────────────────────────────
    print("\n  Standardizing output columns...")

    column_mapping = {
        cl_col: "cell_line",
        drug_col: "drug_name",
        "DRUG_ID": "drug_id",
        "LN_IC50": "ln_ic50",
        "AUC": "auc",
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Core columns (backward-compatible with Steps 07-14)
    # 12 phosphorylation sites (UniProt precursor numbering):
    #   Y869, S991, Y998, Y1016, S1039, T1041,
    #   Y1069, Y1092, Y1110, Y1125, Y1172, Y1197
    core_columns = [
        "cell_line", "drug_name", "drug_id",
        "target_protein",  # EGFR or ERBB2 — added for ERBB family expansion
        "egfr_mutations", "mutation_classes",
        "sequence_id", "pdb_id",
        "ptm_background",
        "ptm_Y869", "ptm_S991", "ptm_Y998", "ptm_Y1016",
        "ptm_S1039", "ptm_T1041", "ptm_Y1069", "ptm_Y1092",
        "ptm_Y1110", "ptm_Y1125", "ptm_Y1172", "ptm_Y1197",
        "drug_smiles", "drug_generation",
        # Delta PTM columns (added 2026-06-23 — drug-conditioned phospho changes)
        "delta_ptm_Y869", "delta_ptm_S991", "delta_ptm_Y998", "delta_ptm_Y1016",
        "delta_ptm_S1039", "delta_ptm_T1041", "delta_ptm_Y1069", "delta_ptm_Y1092",
        "delta_ptm_Y1110", "delta_ptm_Y1125", "delta_ptm_Y1172", "delta_ptm_Y1197",
        "ln_ic50", "auc", "resistance_label",
        "ic50_source",
    ]

    # ── PTM-BDL multi-PTM glyco columns (added 2026-06-28, proposal §3, §7.4) ──
    # The 12 baseline glyco values and the parallel 12 drug-conditioned
    # delta_glyco values are appended to core_columns dynamically so the
    # script remains tolerant to small changes in cfg["ptm"][gene]["glyco_sites"]
    # without requiring a code edit.
    glyco_extra_columns = []
    if "egfr_glyco_sites" in dir():
        # `egfr_glyco_sites` was loaded inside Step 6b above; we reference the
        # column labels we already built (`glyco_col_labels` and
        # `delta_glyco_cols`).  Both live in scope by virtue of being inner
        # variables of `build_multimodal_dataset`.
        pass
    try:
        glyco_extra_columns.extend(glyco_col_labels)
        glyco_extra_columns.extend(delta_glyco_cols)
    except NameError:
        # The glyco block did not run (e.g. step04 glyco vectors missing).
        pass

    output_columns = core_columns + glyco_extra_columns + phospho_cols_added
    available_output = [c for c in output_columns if c in df.columns]
    df_final = df[available_output].copy()

    # ── Step 10: Quality checks ──────────────────────────────────────────────
    print("\n  Running quality checks...")

    critical_cols = ["sequence_id", "pdb_id", "drug_smiles", "ln_ic50"]
    for col in critical_cols:
        if col in df_final.columns:
            missing = df_final[col].isna().sum()
            total = len(df_final)
            print(f"    {col}: {total - missing}/{total} present "
                  f"({'✓' if missing == 0 else f'⚠ {missing} missing'})")

    # Remove rows with missing SMILES (can't encode drug without it)
    if "drug_smiles" in df_final.columns:
        before = len(df_final)
        df_final = df_final.dropna(subset=["drug_smiles"])
        dropped = before - len(df_final)
        if dropped > 0:
            print(f"    Dropped {dropped} rows with missing drug SMILES")

    # ── Save final dataset ───────────────────────────────────────────────────
    output_path = PROCESSED_DIR / "multimodal_dataset.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved unified dataset: {output_path}")
    print(f"    Total samples: {len(df_final)}")

    # ── Glyco slot → residue schema companion JSON (Fix 3, 2026-06-28) ──
    # The glyco_slot00..glyco_slot11 / delta_glyco_slot00..delta_glyco_slot11
    # columns are gene-neutral by design. This sidecar file is the single
    # source of truth that downstream scripts use to resolve `slot i` →
    # `(EGFR residue, ERBB2 residue)` without having to re-derive it from
    # the CSV column names.
    try:
        glyco_schema = {
            "glyco_dim": int(glyco_dim),
            "column_prefix": "glyco_slot",
            "delta_column_prefix": "delta_glyco_slot",
            "slot_to_residue": {
                str(i): {
                    "EGFR": (egfr_glyco_sites[i]["residue"]
                             if i < len(egfr_glyco_sites) else None),
                    "EGFR_position": (egfr_glyco_sites[i]["position"]
                                      if i < len(egfr_glyco_sites) else None),
                    "ERBB2": (erbb2_glyco_sites[i]["residue"]
                              if i < len(erbb2_glyco_sites) else None),
                    "ERBB2_position": (erbb2_glyco_sites[i]["position"]
                                       if i < len(erbb2_glyco_sites) else None),
                }
                for i in range(glyco_dim)
            },
            "notes": (
                "Per-gene N-glycosylation slot schema (Fix 3, 2026-06-28). "
                "Columns glyco_slot00..glyco_slot{N-1} and "
                "delta_glyco_slot00..delta_glyco_slot{N-1} in "
                "multimodal_dataset.csv hold positionally-indexed values; "
                "use this map to resolve slot i → actual EGFR/ERBB2 N-position."
            ),
        }
        schema_path = PROCESSED_DIR / "glyco_slot_schema.json"
        with open(schema_path, "w") as f:
            json.dump(glyco_schema, f, indent=2)
        print(f"  ✓ Saved glyco slot→residue schema: {schema_path}")
    except (NameError, KeyError, IndexError, TypeError) as e:
        # Defensive: if step06b did not run (no glyco vectors), skip silently.
        print(f"  ⚠ Glyco slot schema not emitted ({type(e).__name__}: {e})")

    # ── Generate summary statistics ──────────────────────────────────────────
    n_phospho = int(df_final["phospho_n_sites"].notna().sum()) if "phospho_n_sites" in df_final.columns else 0

    summary = {
        "total_samples": len(df_final),
        "unique_cell_lines": int(df_final["cell_line"].nunique()) if "cell_line" in df_final.columns else 0,
        "unique_drugs": int(df_final["drug_name"].nunique()) if "drug_name" in df_final.columns else 0,
        "unique_sequences": int(df_final["sequence_id"].nunique()) if "sequence_id" in df_final.columns else 0,
        "unique_structures": int(df_final["pdb_id"].nunique()) if "pdb_id" in df_final.columns else 0,
        "resistance_distribution": {
            "sensitive": int((df_final["resistance_label"] == 0).sum()) if "resistance_label" in df_final.columns else 0,
            "resistant": int((df_final["resistance_label"] == 1).sum()) if "resistance_label" in df_final.columns else 0,
        },
        "ic50_sources": {
            "gdsc": int((df_final["ic50_source"] == "gdsc").sum()) if "ic50_source" in df_final.columns else len(df_final),
            "literature": int((df_final["ic50_source"] == "literature").sum()) if "ic50_source" in df_final.columns else 0,
        },
        "phospho_enrichment": {
            "samples_with_phospho": n_phospho,
            "samples_without_phospho": len(df_final) - n_phospho,
            "phospho_columns": phospho_cols_added,
        },
        "columns": list(df_final.columns),
    }

    summary_path = PROCESSED_DIR / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print comprehensive summary ──────────────────────────────────────────
    print("\n" + "="*70)
    print("UNIFIED MULTIMODAL DATASET SUMMARY")
    print("="*70)
    print(f"  Total samples:        {summary['total_samples']}")
    print(f"  Unique cell lines:    {summary['unique_cell_lines']}")
    print(f"  Unique drugs:         {summary['unique_drugs']}")
    print(f"  Unique sequences:     {summary['unique_sequences']}")
    print(f"  Unique structures:    {summary['unique_structures']}")
    print(f"  Sensitive:            {summary['resistance_distribution']['sensitive']}")
    print(f"  Resistant:            {summary['resistance_distribution']['resistant']}")

    if "ic50_source" in df_final.columns:
        print(f"\n  IC50 data sources:")
        print(f"    GDSC (measured):      {summary['ic50_sources']['gdsc']}")
        print(f"    Literature (curated):  {summary['ic50_sources']['literature']}")

    if n_phospho > 0:
        print(f"\n  Drug-PTM phospho enrichment:")
        print(f"    With phospho data:     {n_phospho}")
        print(f"    Without phospho data:  {len(df_final) - n_phospho}")
        print(f"    Phospho columns:       {len(phospho_cols_added)}")

    if "cell_line" in df_final.columns and "drug_name" in df_final.columns:
        print(f"\n  Per-drug sample counts:")
        for drug, count in df_final["drug_name"].value_counts().items():
            print(f"    {drug}: {count}")

        print(f"\n  Per-sequence sample counts:")
        for seq, count in df_final["sequence_id"].value_counts().items():
            print(f"    {seq}: {count}")

    print(f"\n  Columns in final dataset ({len(df_final.columns)} total):")
    for col in df_final.columns:
        dtype = str(df_final[col].dtype)
        non_null = df_final[col].notna().sum()
        print(f"    {col:30s} ({dtype:10s}) — {non_null}/{len(df_final)} non-null")

    return df_final


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 06: Data Harmonization — Unified Multimodal Dataset       ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Input: GDSC + CCLE + PDB + PTM + DrugPTM (Steps 01-05)        ║")
    print("║  Literature IC50: PC-9/HCC827/HCC4006 + Drug-PTM 5 sources     ║")
    print("║  Output: Single CSV with all modalities linked                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    df = build_multimodal_dataset()

    print("\n✓ Step 06 complete! Unified dataset ready for feature extraction.")
    print("  Next: Run Steps 07-10 to extract embeddings from each modality.")
