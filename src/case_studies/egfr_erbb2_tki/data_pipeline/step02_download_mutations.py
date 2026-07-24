#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 02 — Download ERBB Family Mutation Profiles (CCLE/DepMap)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Download detailed mutation-level annotations for cancer cell lines from   ║
║    the Cancer Cell Line Encyclopedia (CCLE) via DepMap. This tells us the    ║
║    EXACT amino acid changes in EGFR and ERBB2 for each cell line.           ║
║                                                                              ║
║  ERBB FAMILY EXPANSION (v2):                                                 ║
║    This script now extracts BOTH EGFR and ERBB2 mutations from CCLE.        ║
║    • EGFR mutations: L858R, T790M, C797S, exon 19 deletions (NSCLC)        ║
║    • ERBB2 mutations: rare point mutations (breast cancer)                   ║
║    • HER2 amplification is the primary oncogenic mechanism in breast —       ║
║      handled as "wild_type" ERBB2 sequence with amplification flag.         ║
║    Generates mutant sequences for BOTH EGFR and HER2 for ESM-2 encoding.    ║
║                                                                              ║
║  KEY FINDING (Section 7a of HER2_EXPANSION_PLAN.md):                        ║
║    • Neratinib NOT in GDSC2 → replaced with Sapitinib (AZD8931)            ║
║    • ALL EGFR drugs also tested on 52 breast cancer cell lines              ║
║    • GDSC2 tissue = "Breast Carcinoma" (not "BRCA")                        ║
║    • Combined: 943 records (638 EGFR + 305 ERBB2)                          ║
║                                                                              ║
║  OUTPUT FILES:                                                               ║
║    data/processed/ccle/egfr_all_mutations.csv                                ║
║    data/processed/ccle/egfr_mutations_by_cell_line.csv                       ║
║    data/processed/ccle/egfr_mutant_sequences.fasta                           ║
║    data/processed/ccle/erbb2_all_mutations.csv              (NEW)           ║
║    data/processed/ccle/erbb2_mutations_by_cell_line.csv     (NEW)           ║
║    data/processed/ccle/erbb2_mutant_sequences.fasta         (NEW)           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
from pathlib import Path

import pandas as pd

# ── Load Configuration ──────────────────────────────────────────────────────
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ptm_bdl.config import load_config

# Module-level globals — set by run(case_study)
cfg = None
RAW_DIR = None
OUT_DIR = None


def _init(case_study: str):
    """Initialize module globals for a given case study."""
    global cfg, RAW_DIR, OUT_DIR
    cfg = load_config(case_study=case_study)
    RAW_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "ccle"
    OUT_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"] / "ccle"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [Config] case_study = {case_study}")

# ── Rich annotation columns to carry through from the CCLE raw data ────────
# These columns provide real classification power from DepMap's own
# annotation pipeline (VEP, ClinVar, gnomAD, OncoKB, AlphaMissense, etc.)
ANNOTATION_COLS = [
    "ModelID", "ProteinChange", "HugoSymbol", "Exon",
    "VariantType", "VariantInfo", "DNAChange",
    "VepClinSig", "Hotspot", "OncogeneHighImpact",
    "GnomadeAF", "GnomadgAF",
    "Sift", "Polyphen",
    "AMClass", "AMPathogenicity",
    "LikelyLoF", "HessDriver",
    "DbsnpRsID", "VepImpact",
]

MANUAL_DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD REQUIRED — CCLE/DepMap Mutation Data                       ║
║                                                                              ║
║  1. Open: https://depmap.org/portal/data_page/?tab=allData                  ║
║                                                                              ║
║  2. SOMATIC MUTATIONS (required):                                            ║
║     → Search for "OmicsSomaticMutations"  (downloaded 2025Q3)                                 ║
║     → Download the CSV file (~300 MB)                                       ║
║     → Save to: {raw_dir}/ccle_somatic_mutations.csv                          ║
║       (any filename with "mutation" or "somatic" works)                      ║
║                                                                              ║
║  3. MODEL INFO (recommended):                                                ║
║     → Search for "Model.csv" (downloaded 2025Q3)                                       ║
║     → Download the CSV file                                                 ║
║     → Save to: {raw_dir}/ccle_model_info.csv                                ║
║       (any filename with "model" works)                                      ║
║                                                                              ║
║  4. EGFR REFERENCE SEQUENCE (required):                                      ║
║     → Open: https://www.uniprot.org/uniprot/P00533.fasta                    ║
║     → Save as: {raw_dir}/egfr_P00533.fasta                                  ║
║                                                                              ║
║  5. Re-run this script after placing the files.                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def find_file(directory: Path, patterns: list[str], description: str) -> Path | None:
    """Search for a file matching any pattern (case-insensitive)."""
    for f in sorted(directory.iterdir()) if directory.exists() else []:
        fname_lower = f.name.lower()
        if fname_lower.startswith(".") or fname_lower.startswith("~"):
            continue
        for pattern in patterns:
            if pattern.lower() in fname_lower:
                print(f"  ✓ Found {description}: {f.name}")
                return f
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Locate Manually Downloaded CCLE Data
# ══════════════════════════════════════════════════════════════════════════════

def locate_ccle_files():
    """
    Look for manually downloaded CCLE/DepMap files in data/raw/ccle/.
    
    DepMap provides two key files:
    1. OmicsSomaticMutations.csv (~300MB) — all somatic mutations
    2. Model.csv — cell line metadata (ModelID → CellLineName)
    
    MANUAL DOWNLOAD:
      1. Go to https://depmap.org/portal/data_page/?tab=allData
      2. Search for "OmicsSomaticMutations" → download CSV
      3. Search for "Model" → download CSV
      4. Place both in data/raw/ccle/
    """
    print("\n" + "=" * 70)
    print("STEP 2.1: Locating CCLE Mutation Data Files")
    print("=" * 70)
    print(f"  Looking in: {RAW_DIR}")

    if RAW_DIR.exists():
        files = [f.name for f in RAW_DIR.iterdir() if not f.name.startswith(".")]
        print(f"  Files found: {files}")

    mutations_path = find_file(
        RAW_DIR, ["somatic_mutation", "omicssomatic", "ccle_somatic", "mutations"],
        "Somatic mutations"
    )

    model_path = find_file(
        RAW_DIR, ["model_info", "model.csv", "ccle_model"],
        "Model info"
    )

    fasta_path = find_file(
        RAW_DIR, ["egfr_p00533", "p00533.fasta"],
        "EGFR reference FASTA"
    )

    missing = []
    if mutations_path is None:
        missing.append("OmicsSomaticMutations.csv (somatic mutation data)")
    if model_path is None:
        missing.append("Model.csv (cell line metadata)")
    if fasta_path is None:
        missing.append("egfr_P00533.fasta (UniProt EGFR reference)")

    if missing:
        print(f"\n  ✗ Missing {len(missing)} file(s):")
        for m in missing:
            print(f"    • {m}")
        print(MANUAL_DOWNLOAD_INSTRUCTIONS.format(raw_dir=RAW_DIR))
        sys.exit(1)

    return mutations_path, model_path, fasta_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Extract EGFR-Specific Mutations
# ══════════════════════════════════════════════════════════════════════════════

def extract_egfr_mutations(mutations_path: Path):
    """
    Extract all EGFR mutations and map them to cell line names.
    
    PROCESSING STEPS:
    ─────────────────
    1. Load the full CCLE mutation file (millions of rows)
    2. Filter for EGFR gene (HugoSymbol == "EGFR")
    3. Parse protein-level changes (e.g., "p.L858R" → position 858, L→R)
    4. Classify mutations using the rich annotation columns already present
       in the DepMap data (VepClinSig, Hotspot, OncogeneHighImpact, etc.)
    5. Group mutations by cell line (one cell line may have multiple mutations)
    6. Merge with model info to get human-readable cell line names
    
    WHY THIS MATTERS FOR THE MODEL:
    ────────────────────────────────
    Each cell line's EGFR mutation profile determines:
    a) Which mutant protein sequence we feed to ESM-2
    b) Which PDB structure best represents the protein conformation
    c) How we expect the drug to interact with the kinase
    
    For example:
    - H1975 carries L858R + T790M → use PDB 5EDP, sensitive to Osimertinib
    - HCC827 carries exon19del → sensitive to all TKIs
    - A cell line with C797S → resistant to Osimertinib (our key case)
    """
    print("\n" + "=" * 70)
    print("STEP 2.2: Extracting EGFR Mutations")
    print("=" * 70)

    # ── Load mutations (can be very large, so we chunk-read) ─────────────────
    print("  Loading CCLE somatic mutations (this may take a moment)...")

    # Read in chunks to handle the large file size
    egfr_chunks = []
    chunk_size = 100_000

    for chunk in pd.read_csv(mutations_path, chunksize=chunk_size, low_memory=False):
        # Filter for EGFR gene in each chunk
        # Config now uses gene_symbols list: ["EGFR", "ERBB2"]
        egfr_rows = chunk[chunk["HugoSymbol"] == "EGFR"]
        if len(egfr_rows) > 0:
            egfr_chunks.append(egfr_rows)

    if not egfr_chunks:
        print("  ✗ No EGFR mutations found in CCLE data. Cannot proceed.")
        sys.exit(1)

    df_egfr = pd.concat(egfr_chunks, ignore_index=True)
    print(f"  → Found {len(df_egfr)} EGFR mutation records")
    print(f"  → Across {df_egfr['ModelID'].nunique()} cell lines")

    # ── Log which annotation columns are available ──────────────────────────
    available_annot = [c for c in ANNOTATION_COLS if c in df_egfr.columns]
    print(f"  → Retaining {len(available_annot)} rich annotation columns from DepMap")

    # ── Extract exon number from "19/28" format ─────────────────────────────
    if "Exon" in df_egfr.columns:
        df_egfr["exon_number"] = (
            df_egfr["Exon"]
            .astype(str)
            .str.extract(r"^(\d+)", expand=False)
            .astype(float)
        )

    # ── Parse protein changes ────────────────────────────────────────────────
    # ProteinChange format: "p.L858R" means Leucine→Arginine at position 858
    # We need to extract: original AA, position, new AA

    print("\n  Parsing protein-level changes...")

    if "ProteinChange" in df_egfr.columns:
        # Extract position and amino acid change from protein annotation
        df_egfr["protein_change_clean"] = df_egfr["ProteinChange"].str.replace("p.", "", regex=False)

        # Show the most common EGFR mutations found
        print("\n  Most frequent EGFR mutations in CCLE:")
        mutation_counts = df_egfr["protein_change_clean"].value_counts().head(20)
        for mut, count in mutation_counts.items():
            # Show the Exon and VepClinSig from the data for each mutation
            row_match = df_egfr[df_egfr["protein_change_clean"] == mut].iloc[0]
            exon_str = str(row_match.get("Exon", "?"))
            clinsig_str = str(row_match.get("VepClinSig", ""))
            hotspot_str = str(row_match.get("Hotspot", ""))
            print(f"    {mut}: {count} cell lines  |  exon {exon_str}  "
                  f"|  ClinSig={clinsig_str[:40]}  |  Hotspot={hotspot_str[:25]}")

    return df_egfr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Data-Driven Mutation Classification
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=0.0):
    """Convert a value to float, returning *default* on failure."""
    try:
        v = float(val)
        return v if pd.notna(v) else default
    except (ValueError, TypeError):
        return default


def classify_from_annotations(row: pd.Series) -> str:
    """
    Classify an EGFR mutation using the rich annotation columns that DepMap
    already provides.  No hardcoded mutation names — the data speaks for itself.
    
    Uses these columns from the raw CCLE file:
      • VepClinSig   — ClinVar clinical significance (pathogenic, drug_response…)
      • Hotspot       — OncoKB / Oncogene_high_impact flag
      • OncogeneHighImpact — boolean driver flag
      • GnomadeAF     — gnomAD population allele frequency
      • VariantInfo    — VEP consequence type (missense, frameshift, inframe_deletion…)
    
    Priority (highest → lowest):
    ─────────────────────────────
    1. VepClinSig contains 'pathogenic' or 'drug_response'
       → pathogenic_drug_response
    2. Hotspot is 'OncoKB' or 'Oncogene_high_impact', or OncogeneHighImpact=True
       → oncogenic_hotspot
    3. GnomadeAF > 1e-5  (variant found in general population)
       → likely_polymorphism
    4. VariantInfo is frameshift_variant or stop_gained
       → truncating
    5. VariantInfo is inframe_deletion or inframe_insertion
       → inframe_indel
    6. Everything else
       → VUS  (variant of unknown significance)
    """
    clinsig = str(row.get("VepClinSig", "")).lower()
    hotspot = str(row.get("Hotspot", "")).lower()
    onco_hi = str(row.get("OncogeneHighImpact", "")).lower()
    varinfo = str(row.get("VariantInfo", "")).lower()
    gnomad = _safe_float(row.get("GnomadeAF"))

    # 1 — ClinVar / VEP annotation says pathogenic or drug-response
    if "pathogenic" in clinsig or "drug_response" in clinsig:
        return "pathogenic_drug_response"

    # 2 — Recognized oncogenic hotspot by OncoKB or DepMap's own scoring
    if "oncokb" in hotspot or "oncogene_high_impact" in hotspot:
        return "oncogenic_hotspot"
    if onco_hi == "true":
        return "oncogenic_hotspot"

    # 3 — Present in general population → likely a germline polymorphism
    if gnomad > 1e-5:
        return "likely_polymorphism"

    # 4 — Frameshift / nonsense → truncating
    if "frameshift" in varinfo or "stop_gained" in varinfo:
        return "truncating"

    # 5 — Inframe indels without ClinVar annotation
    if "inframe_deletion" in varinfo or "inframe_insertion" in varinfo:
        return "inframe_indel"

    # 6 — Fallback: variant of unknown significance
    return "VUS"


def classify_mutations(df_egfr: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data-driven classification to every EGFR mutation row.
    
    Tag each mutation using the annotation columns from the CCLE raw file
    (VepClinSig, Hotspot, OncogeneHighImpact, GnomadeAF, VariantInfo).
    """
    print("\n" + "=" * 70)
    print("STEP 2.3: Classifying Mutations (data-driven)")
    print("=" * 70)

    df_egfr["mutation_class"] = df_egfr.apply(classify_from_annotations, axis=1)

    print("\n  Mutation classification summary (from CCLE annotation columns):")
    class_counts = df_egfr["mutation_class"].value_counts()
    for cls, count in class_counts.items():
        print(f"    {cls:30s}: {count}")

    # ── Show most common mutations per clinically-relevant class ─────────────
    for cls in ["pathogenic_drug_response", "oncogenic_hotspot"]:
        sub = df_egfr[df_egfr["mutation_class"] == cls]
        if sub.empty:
            continue
        print(f"\n  '{cls}' mutations found:")
        for mut, cnt in sub["protein_change_clean"].value_counts().items():
            exon_vals = sub.loc[sub["protein_change_clean"] == mut, "Exon"]
            exon = exon_vals.iloc[0] if len(exon_vals) else "?"
            varinfo = sub.loc[sub["protein_change_clean"] == mut, "VariantInfo"].iloc[0]
            print(f"    {str(mut):30s}  exon {str(exon):8s}  "
                  f"({cnt} cell line(s))  [{varinfo}]")

    return df_egfr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Merge Model Info & Build Per-Cell-Line Profiles
# ══════════════════════════════════════════════════════════════════════════════

def build_cell_line_profiles(df_egfr: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """
    Merge with Model.csv for cell line names, then create a one-row-per-cell-line
    profile that aggregates mutation info and preserves the key annotation fields.
    
    One cell line can have MULTIPLE EGFR mutations (e.g., H1975 has L858R + T790M).
    We need to create a mutation profile for each cell line.
    """
    print("\n" + "=" * 70)
    print("STEP 2.4: Building Per-Cell-Line Mutation Profiles")
    print("=" * 70)

    # ── Merge with model info for cell line names ────────────────────────────
    print("  Merging with cell line model information...")
    df_model = pd.read_csv(model_path)

    # Merge to get cell line names
    if "ModelID" in df_egfr.columns and "ModelID" in df_model.columns:
        name_cols = ["ModelID", "CellLineName", "StrippedCellLineName",
                     "OncotreeLineage", "OncotreePrimaryDisease"]
        available_name_cols = [c for c in name_cols if c in df_model.columns]
        df_egfr = df_egfr.merge(
            df_model[available_name_cols],
            on="ModelID", how="left"
        )

    # ── Filter for NSCLC cell lines (Lung lineage) ───────────────────────────
    if "OncotreeLineage" in df_egfr.columns:
        before_filter = df_egfr["ModelID"].nunique()
        nsclc_mask = df_egfr["OncotreeLineage"].str.contains(
            "Lung", case=False, na=False
        )
        df_egfr = df_egfr[nsclc_mask].copy()
        after_filter = df_egfr["ModelID"].nunique()
        print(f"\n  Filtered for NSCLC (Lung lineage): {before_filter} → {after_filter} cell lines")
    elif "OncotreePrimaryDisease" in df_egfr.columns:
        before_filter = df_egfr["ModelID"].nunique()
        nsclc_mask = df_egfr["OncotreePrimaryDisease"].str.contains(
            "Non-Small Cell Lung|NSCLC|Lung Adenocarcinoma|Lung Squamous",
            case=False, na=False
        )
        df_egfr = df_egfr[nsclc_mask].copy()
        after_filter = df_egfr["ModelID"].nunique()
        print(f"\n  Filtered for NSCLC: {before_filter} → {after_filter} cell lines")
    else:
        print("\n  ⚠ No lineage column found — cannot filter for NSCLC.")
        print("    Including all cell lines with EGFR mutations.")

    if df_egfr.empty:
        print("  ⚠ No NSCLC cell lines with EGFR mutations found.")
        profiles = pd.DataFrame()
        output_all = OUT_DIR / "egfr_all_mutations.csv"
        profiles.to_csv(output_all, index=False)
        output_profiles = OUT_DIR / "egfr_mutations_by_cell_line.csv"
        profiles.to_csv(output_profiles, index=False)
        return profiles

    # Build mutation profile per cell line
    cell_line_col = "CellLineName" if "CellLineName" in df_egfr.columns else "ModelID"

    # ── Helper: join unique non-null values ─────────────────────────────────
    def join_unique(series):
        return "; ".join(sorted({str(v) for v in series if pd.notna(v) and str(v).strip()}))

    # ── Aggregate all rich columns per cell line ────────────────────────────
    agg_dict = {
        "ProteinChange": join_unique,
        "mutation_class": join_unique,
        "Exon": join_unique,
        "VariantType": join_unique,
        "VariantInfo": join_unique,
        "VepClinSig": join_unique,
        "Hotspot": join_unique,
        "OncogeneHighImpact": join_unique,
    }
    # Only aggregate columns that actually exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_egfr.columns}

    profiles = df_egfr.groupby(cell_line_col).agg(agg_dict).reset_index()
    profiles.rename(columns={
        "ProteinChange": "egfr_mutations",
        "mutation_class": "mutation_classes",
        "Exon": "exons",
        "VariantType": "variant_types",
        "VariantInfo": "variant_info",
        "VepClinSig": "clinical_significance",
        "Hotspot": "hotspot",
        "OncogeneHighImpact": "oncogene_high_impact",
    }, inplace=True)

    # ── Add tissue/lineage info if available ────────────────────────────────
    if "OncotreeLineage" in df_egfr.columns:
        lineage = df_egfr.groupby(cell_line_col)["OncotreeLineage"].first().reset_index()
        profiles = profiles.merge(lineage, on=cell_line_col, how="left")
    if "OncotreePrimaryDisease" in df_egfr.columns:
        disease = df_egfr.groupby(cell_line_col)["OncotreePrimaryDisease"].first().reset_index()
        profiles = profiles.merge(disease, on=cell_line_col, how="left")

    print(f"\n  Cell lines with EGFR mutations: {len(profiles)}")

    # ── Print clinically significant cell lines ─────────────────────────────
    notable = profiles[
        profiles["mutation_classes"].str.contains(
            "pathogenic_drug_response|oncogenic_hotspot", na=False
        )
    ]
    if not notable.empty:
        print(f"\n  Clinically significant EGFR-mutant cell lines ({len(notable)}):")
        for _, row in notable.iterrows():
            print(f"    {row[cell_line_col]:20s}  {row['egfr_mutations']:40s}  "
                  f"exon {row.get('exons', '?'):12s}  [{row['mutation_classes']}]")

    # ── Save results ─────────────────────────────────────────────────────────
    output_all = OUT_DIR / "egfr_all_mutations.csv"
    df_egfr.to_csv(output_all, index=False)
    print(f"\n  ✓ Saved all EGFR mutations: {output_all}")

    output_profiles = OUT_DIR / "egfr_mutations_by_cell_line.csv"
    profiles.to_csv(output_profiles, index=False)
    print(f"  ✓ Saved mutation profiles: {output_profiles}")

    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Load EGFR Reference Sequence
# ══════════════════════════════════════════════════════════════════════════════

def load_egfr_reference_sequence(fasta_path: Path) -> str:
    """
    Load the canonical human EGFR protein sequence from UniProt.
    
    UniProt accession P00533 is the reference sequence for human EGFR.
    We need this to:
    1. Introduce specific mutations at exact positions
    2. Generate mutant sequences for each cell line's EGFR variant
    3. Feed these mutant sequences to ESM-2 for embedding extraction
    
    The EGFR protein has 1210 amino acids:
    - Extracellular domain: residues 1-621
    - Transmembrane domain: residues 622-644  
    - Intracellular kinase domain: residues 645-1186 (our focus)
    - C-terminal tail: residues 1187-1210 (contains phosphorylation sites)
    
    The kinase domain (aa 712-979) is where:
    - TKI drugs bind (ATP-binding pocket)
    - Key mutations occur (L858R at 858, T790M at 790, C797S at 797)
    - Drug resistance is determined
    """
    print("\n" + "=" * 70)
    print("STEP 2.5: Loading EGFR Reference Sequence")
    print("=" * 70)

    print(f"  ✓ Found: {fasta_path.name}")

    # Parse the FASTA sequence
    from Bio import SeqIO
    record = SeqIO.read(fasta_path, "fasta")
    sequence = str(record.seq)

    egfr_cfg = cfg["uniprot"]["EGFR"]
    print(f"  EGFR sequence length: {len(sequence)} amino acids")
    print(f"  Kinase domain: positions {egfr_cfg['kinase_domain_start']}-{egfr_cfg['kinase_domain_end']}")

    # Verify key residues at kinase domain landmarks
    landmarks = [
        egfr_cfg["kinase_domain_start"],
        egfr_cfg["kinase_domain_end"],
        790, 797, 858,
    ]
    print("\n  Reference residues at key kinase domain positions:")
    for pos in sorted(set(landmarks)):
        if pos <= len(sequence):
            print(f"    Position {pos}: {sequence[pos - 1]}")

    return sequence


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Data-Driven Mutant Sequence Generation
# ══════════════════════════════════════════════════════════════════════════════

# ── Regex patterns for parsing ProteinChange strings ────────────────────────
_RE_POINT = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")  # p.L858R
_RE_DEL = re.compile(r"^p\.[A-Z](\d+)_[A-Z](\d+)del$")  # p.E746_A750del
_RE_DELINS = re.compile(r"^p\.[A-Z](\d+)_[A-Z](\d+)delins([A-Z]+)$")  # p.L747_P753delinsS


def _parse_protein_change(pc: str):
    """
    Parse a single ProteinChange string into a machine-usable tuple.
    
    Returns  (mutation_type, details)  or  (None, None)  if unparseable.
    
    Supported formats:
      p.L858R             → point mutation
      p.E746_A750del      → inframe deletion
      p.L747_P753delinsS  → deletion + insertion
    """
    if pd.isna(pc) or not isinstance(pc, str):
        return None, None
    pc = pc.strip()

    # Point mutation (e.g. p.L858R)
    m = _RE_POINT.match(pc)
    if m:
        return "point", {"pos": int(m.group(2)), "ref": m.group(1), "alt": m.group(3)}

    # Inframe deletion (e.g. p.E746_A750del)
    m = _RE_DEL.match(pc)
    if m:
        return "deletion", {"start": int(m.group(1)), "end": int(m.group(2))}

    # Deletion + insertion (e.g. p.L747_P753delinsS)
    m = _RE_DELINS.match(pc)
    if m:
        return "delins", {"start": int(m.group(1)), "end": int(m.group(2)),
                          "insert": m.group(3)}

    return None, None


def _apply_mutations(wt: str, mutations: list[tuple[str, dict]]) -> str | None:
    """
    Apply a list of parsed mutations to the wild-type sequence.
    
    Rules:
      • Deletions / delins are applied first (they shift indices).
      • Point mutations are applied on the (possibly shortened) sequence.
    
    Returns None if any mutation cannot be applied safely.
    """
    seq = list(wt)
    offset = 0  # cumulative index shift from deletions

    # Separate by type
    point_muts = [(t, d) for t, d in mutations if t == "point"]
    del_muts = [(t, d) for t, d in mutations if t in ("deletion", "delins")]

    # Apply deletions first (assumes at most one deletion per combo — typical)
    for mtype, d in del_muts:
        s = d["start"] - 1 + offset  # 0-based inclusive
        e = d["end"] - 1 + offset + 1  # 0-based exclusive
        if mtype == "deletion":
            del seq[s:e]
            offset -= (e - s)
        elif mtype == "delins":
            seq[s:e] = list(d["insert"])
            offset += len(d["insert"]) - (e - s)

    # Apply point mutations
    for _, d in point_muts:
        idx = d["pos"] - 1 + offset
        if 0 <= idx < len(seq):
            seq[idx] = d["alt"]
        else:
            return None  # position out of range after deletion

    return "".join(seq)


def generate_mutant_sequences(wt_sequence: str, df_egfr: pd.DataFrame):
    """
    Generate mutant EGFR sequences from what the CCLE data actually contains.
    No hardcoded mutation combos — everything is derived from the data.
    
    For each cell line's mutation profile, we create a modified protein sequence
    by introducing the specific amino acid changes. This mutant sequence is
    what we feed to ESM-2 to capture how each mutation alters the protein's
    evolutionary/functional context.
    
    EXAMPLE:
    ────────
    Wild-type at position 858: ...KVLGSGAFGTVYK... (L at 858)
    L858R mutant:              ...KVRGSGAFGTVYK... (R at 858)
    
    The ESM-2 protein language model will generate DIFFERENT embeddings for
    L858R vs. wild-type because the model has learned that arginine (R) at
    this position is unusual and alters the protein's functional context.
    
    STRATEGY:
    ─────────
    1. Identify all UNIQUE clinically-significant mutation profiles found in
       the CCLE data (pathogenic_drug_response or oncogenic_hotspot).
    2. FILTER to kinase domain (exons 18–21) — this is where TKIs bind and
       where mutations determine drug sensitivity/resistance. Mutations in
       the extracellular domain (exons 1–15) don't affect TKI binding.
    3. EXCLUDE truncating mutations (frameshift, stop_gained) — these destroy
       the protein and can't be modeled as simple sequence changes.
    4. Parse the ProteinChange strings, apply them to the wild-type sequence,
       and write the result.
    5. Always include wild-type as the baseline.
    """
    print("\n" + "=" * 70)
    print("STEP 2.6: Generating Mutant EGFR Sequences (data-driven)")
    print("=" * 70)

    if wt_sequence is None:
        print("  ✗ No reference sequence available.")
        return

    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    # ── 1. Filter to kinase-domain, non-truncating, clinically significant ──
    # The kinase domain spans exons 18-21 (approximately residues 712-979).
    # Only these mutations affect TKI binding and are relevant for ESM-2.
    # Mutations outside the kinase domain (exons 1-17) don't change drug
    # interaction and would waste ESM-2 compute in step07.

    KINASE_EXONS = {18, 19, 20, 21}
    TRUNCATING_VARIANTS = {"frameshift_variant", "stop_gained"}

    sig = df_egfr[
        (df_egfr["mutation_class"].isin(["pathogenic_drug_response", "oncogenic_hotspot"]))
        & (df_egfr["exon_number"].isin(KINASE_EXONS))
        & (~df_egfr["VariantInfo"].str.lower().str.contains(
            "|".join(TRUNCATING_VARIANTS), na=False
        ))
        ]

    n_total_sig = df_egfr["mutation_class"].isin(
        ["pathogenic_drug_response", "oncogenic_hotspot"]
    ).sum()

    print(f"\n  Clinically significant EGFR mutations: {n_total_sig}")
    print(f"  After kinase-domain filter (exons {sorted(KINASE_EXONS)}): {len(sig)}")
    print(f"  (Excluded: extracellular/non-kinase domain + truncating variants)")

    if sig.empty:
        print("  ⚠ No kinase-domain clinically significant mutations found.")
        print("    Only wild-type sequence will be written.")

    # Group by cell line to get each cell line's set of significant mutations
    combos = set()  # frozenset of ProteinChange tuples
    individual_mutations = set()  # every single ProteinChange string
    cell_line_col = "CellLineName" if "CellLineName" in sig.columns else "ModelID"

    for _, grp in sig.groupby(cell_line_col):
        pcs = sorted(grp["ProteinChange"].dropna().unique())
        if pcs:
            combos.add(tuple(pcs))
            individual_mutations.update(pcs)

    # Also add each individual mutation as its own combo (for single-mutant seqs)
    for pc in list(individual_mutations):
        combos.add((pc,))

    print(f"\n  Unique kinase-domain mutation combos: {len(combos)}")
    print(f"  Individual kinase-domain mutations: {len(individual_mutations)}")

    # ── 2. Parse and apply each combo to the wild-type sequence ─────────────
    sequences = {}
    skipped = []

    for combo in sorted(combos):
        parsed = []
        all_ok = True
        for pc in combo:
            mtype, details = _parse_protein_change(pc)
            if mtype is None:
                all_ok = False
                break
            parsed.append((mtype, details))

        if not all_ok:
            skipped.append(combo)
            continue

        # Build a human-readable name from the ProteinChange strings
        name_parts = []
        for pc in combo:
            clean = pc.replace("p.", "")
            name_parts.append(clean)
        name = "_".join(name_parts)

        # Apply mutations to wild-type
        mutant_seq = _apply_mutations(wt_sequence, parsed)
        if mutant_seq is None:
            skipped.append(combo)
            continue

        sequences[name] = mutant_seq
        delta = len(mutant_seq) - len(wt_sequence)
        delta_str = f" ({delta:+d} AA)" if delta != 0 else ""
        print(f"    ✓ {name:40s} length {len(mutant_seq)}{delta_str}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} unparseable combo(s):")
        for combo in skipped:
            print(f"    • {combo}")

    # ── 3. Add config-driven acquired resistance mutations ──────────────────
    # These mutations emerge during Osimertinib treatment in patients but are
    # not present in any CCLE cell line. They are defined in config.yaml
    # (acquired_resistance_mutations) and are essential for the project.

    config_muts = cfg.get("acquired_resistance_mutations", [])
    if config_muts:
        print(f"\n  Adding {len(config_muts)} config-driven acquired resistance mutation(s):")
        for entry in config_muts:
            name = entry["name"]
            if name in sequences:
                print(f"    ⊘ {name:40s} already generated from CCLE data")
                continue

            # Parse changes from config format into our internal format
            parsed = []
            for ch in entry["changes"]:
                if ch["type"] == "point":
                    parsed.append(("point", {"pos": ch["pos"], "ref": ch["ref"], "alt": ch["alt"]}))
                elif ch["type"] == "deletion":
                    parsed.append(("deletion", {"start": ch["start"], "end": ch["end"]}))

            # Verify reference residues before applying
            ok = True
            for mtype, d in parsed:
                if mtype == "point":
                    actual = wt_sequence[d["pos"] - 1]
                    if actual != d["ref"]:
                        print(f"    ✗ {name}: pos {d['pos']} expected {d['ref']} "
                              f"but found {actual} in P00533 — SKIPPED")
                        ok = False
                        break

            if not ok:
                continue

            mutant_seq = _apply_mutations(wt_sequence, parsed)
            if mutant_seq is None:
                print(f"    ✗ {name}: could not apply mutations — SKIPPED")
                continue

            sequences[name] = mutant_seq
            delta = len(mutant_seq) - len(wt_sequence)
            delta_str = f" ({delta:+d} AA)" if delta != 0 else ""
            print(f"    ✓ {name:40s} length {len(mutant_seq)}{delta_str}"
                  f"  [{entry.get('description', '')}]")

    # ── 4. Always include wild-type as the baseline ─────────────────────────
    sequences["wild_type"] = wt_sequence
    print(f"    ✓ {'wild_type':40s} length {len(wt_sequence)}  (reference)")

    # ── 5. Save all mutant sequences as multi-FASTA ─────────────────────────
    output_path = OUT_DIR / "egfr_mutant_sequences.fasta"

    records = []
    for name, seq in sorted(sequences.items()):
        record = SeqRecord(
            Seq(seq),
            id=f"EGFR_{name}",
            description=f"Human EGFR {name} sequence"
        )
        records.append(record)

    SeqIO.write(records, output_path, "fasta")
    print(f"\n  ✓ Saved {len(records)} sequences: {output_path}")

    return sequences


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Integrate Cell Model Passports (CMP / Sanger) Mutations
# ══════════════════════════════════════════════════════════════════════════════

def _find_cmp_file(pattern: str, description: str) -> Path | None:
    """Search for a Cell Model Passports file in data/raw/cosmic/."""
    CMP_RAW_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "cosmic"
    if not CMP_RAW_DIR.exists():
        return None
    for f in sorted(CMP_RAW_DIR.iterdir()):
        if pattern in f.name.lower() and f.suffix == ".csv":
            print(f"    ✓ Found {description}: {f.name}")
            return f
    return None


def integrate_cmp_mutations():
    """
    Integrate EGFR mutations from Cell Model Passports (Sanger/COSMIC).

    CMP provides an INDEPENDENT mutation calling pipeline (Caveman + Pindel)
    from the CCLE/DepMap pipeline (Mutect2). Cross-referencing both sources:
    1. Confirms EGFR driver mutations found by CCLE
    2. Discovers EGFR mutations that CCLE may have missed
    3. Provides sequencing evidence for wild-type confirmation

    The CMP data uses `model_id` (SIDM format) which directly matches
    GDSC's `SANGER_MODEL_ID` — making cross-referencing reliable.

    Two CMP files are used:
    ─────────────────────
    • mutations_summary (driver mutations):
      Curated cancer driver mutations only. If EGFR appears here,
      it's a confirmed driver mutation with high confidence.

    • mutations_all (all mutations):
      Every somatic variant detected by sequencing. If a cell line
      appears here but has NO EGFR coding/driver entries, that cell
      line was sequenced and confirmed to have no functional EGFR
      mutations → EGFR wild-type with sequencing evidence.

    Returns:
        cmp_egfr_drivers: DataFrame of EGFR driver mutations per model
        cmp_sequenced_sids: set of all model_ids that were sequenced
    """
    print("\n" + "=" * 70)
    print("STEP 2.7: Integrating Cell Model Passports (CMP) Mutations")
    print("=" * 70)

    # ── Locate CMP files ─────────────────────────────────────────────────────
    drv_path = _find_cmp_file("mutations_summary", "CMP driver mutations")
    all_path = _find_cmp_file("mutations_all", "CMP all mutations")

    if drv_path is None and all_path is None:
        print("    ⚠ No CMP mutation files found in data/raw/cosmic/")
        print("    → Download from: https://cellmodelpassports.sanger.ac.uk/downloads")
        print("      'Mutations Summary' (522 kB) and 'Mutations All' (239 MB)")
        return None, set()

    cmp_egfr_drivers = None
    cmp_sequenced_sids = set()

    # ── Process driver mutations ─────────────────────────────────────────────
    if drv_path is not None:
        df_drv = pd.read_csv(drv_path)
        print(f"\n    CMP driver mutations: {len(df_drv):,} rows, "
              f"{df_drv['model_id'].nunique()} models, "
              f"{df_drv['gene_symbol'].nunique()} genes")

        # Extract EGFR drivers
        egfr_drv = df_drv[df_drv["gene_symbol"] == "EGFR"].copy()
        if not egfr_drv.empty:
            print(f"    EGFR driver mutations: {len(egfr_drv)} in "
                  f"{egfr_drv['model_id'].nunique()} models")
            cmp_egfr_drivers = egfr_drv
        else:
            print("    No EGFR driver mutations in CMP driver file")

    # ── Process all mutations (for sequencing evidence) ──────────────────────
    if all_path is not None:
        print(f"\n    Scanning CMP all_mutations for sequenced model IDs...")
        for chunk in pd.read_csv(all_path, chunksize=200_000,
                                 usecols=["model_id"], low_memory=False):
            cmp_sequenced_sids.update(chunk["model_id"].unique())
        print(f"    CMP sequenced models: {len(cmp_sequenced_sids):,}")

    return cmp_egfr_drivers, cmp_sequenced_sids


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Build Comprehensive EGFR Profiles for ALL GDSC NSCLC Cell Lines
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_cell_line_name(name):
    """Standardize cell line names: uppercase, remove NCI- prefix and hyphens."""
    if pd.isna(name):
        return name
    name = str(name).upper().strip()
    for prefix in ["NCI-", "NCI_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("-", "")
    return name


def build_comprehensive_profiles(
        ccle_profiles: pd.DataFrame,
        cmp_egfr_drivers,
        cmp_sequenced_sids: set,
):
    """
    Merge CCLE + CMP mutation data and build a COMPLETE EGFR mutation
    profile for every GDSC NSCLC cell line.

    EVIDENCE-BASED CLASSIFICATION:
    ──────────────────────────────
    For each cell line, we determine EGFR status using ALL available evidence:

    1. EGFR DRIVER MUTATION (from CCLE and/or CMP driver file):
       → Cell line has a confirmed pathogenic/oncogenic EGFR mutation
       → Record the specific mutation(s) and evidence sources
       → egfr_status = "driver_mutation"

    2. EGFR VUS ONLY (from CCLE and/or CMP, no drivers):
       → Cell line has EGFR variants but none classified as drivers
       → These are passenger mutations — don't affect EGFR kinase function
       → egfr_status = "VUS_only" (functionally wild-type for EGFR TKI response)

    3. CONFIRMED WILD-TYPE (sequenced by CCLE and/or CMP, no EGFR coding
       mutations detected at all):
       → Biological evidence: the cell line's EGFR gene was sequenced
         and no somatic mutations were found
       → egfr_status = "confirmed_wild_type"
       → evidence_sources lists which database(s) provided sequencing data

    4. UNKNOWN (not sequenced by either CCLE or CMP):
       → No sequencing data available — EGFR status genuinely unknown
       → egfr_status = "unknown"

    OUTPUT:
    ───────
    Updated egfr_mutations_by_cell_line.csv with ALL GDSC NSCLC cell lines,
    not just those with EGFR mutations.
    """
    print("\n" + "=" * 70)
    print("STEP 2.8: Building Comprehensive EGFR Profiles")
    print("=" * 70)

    # ── Load GDSC NSCLC cell lines ───────────────────────────────────────────
    case_study_name = cfg.get("case_study", "egfr_erbb2_tki")
    gdsc_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "gdsc" / f"gdsc_{case_study_name}_responses.csv"
    if not gdsc_path.exists():
        gdsc_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "gdsc" / "gdsc_nsclc_egfr_tki_responses.csv"
    if not gdsc_path.exists():
        print("    ⚠ GDSC data not found. Run step01 first.")
        return ccle_profiles

    df_gdsc = pd.read_csv(gdsc_path)
    gdsc_cl = df_gdsc.drop_duplicates("CELL_LINE_NAME")[
        ["CELL_LINE_NAME", "SANGER_MODEL_ID"]
    ].copy()
    gdsc_cl["cell_line_norm"] = gdsc_cl["CELL_LINE_NAME"].apply(_normalize_cell_line_name)
    print(f"    GDSC NSCLC cell lines: {len(gdsc_cl)}")

    # ── Also include CCLE cell lines with EGFR driver mutations ──────────────
    # These may not be in GDSC but are added as literature IC50 records in
    # Step 06 (e.g., PC-9, HCC4006). Without including them here, they would
    # get "unknown" mutation status after the merge.
    if ccle_profiles is not None and not ccle_profiles.empty:
        cl_col = "CellLineName" if "CellLineName" in ccle_profiles.columns else "cell_line"
        ccle_driver_mask = ccle_profiles["mutation_classes"].str.contains(
            "pathogenic_drug_response|oncogenic_hotspot", na=False
        )
        ccle_drivers = ccle_profiles[ccle_driver_mask]
        existing_norms = set(gdsc_cl["cell_line_norm"])
        added = 0
        for _, row in ccle_drivers.iterrows():
            norm = _normalize_cell_line_name(row[cl_col])
            if norm not in existing_norms:
                gdsc_cl = pd.concat([gdsc_cl, pd.DataFrame([{
                    "CELL_LINE_NAME": row[cl_col],
                    "SANGER_MODEL_ID": "",
                    "cell_line_norm": norm,
                }])], ignore_index=True)
                existing_norms.add(norm)
                added += 1
        if added:
            print(f"    + Added {added} CCLE driver-mutant cell lines not in GDSC "
                  f"(for literature IC50 matching)")

    # ── Load CCLE model info for cross-referencing ───────────────────────────
    ccle_model_path = RAW_DIR / "ccle_model_info.csv"
    ccle_screened_sids = set()
    if ccle_model_path.exists():
        df_model = pd.read_csv(ccle_model_path, low_memory=False)
        lung_mask = df_model["OncotreeLineage"].str.contains("Lung", case=False, na=False)
        lung_models = df_model[lung_mask]
        ccle_screened_sids = set(
            lung_models["SangerModelID"].dropna().astype(str).str.strip()
        )
        print(f"    CCLE/DepMap Lung models: {len(ccle_screened_sids)}")

    # ── Normalize CCLE profiles for matching ─────────────────────────────────
    ccle_norm = {}
    if ccle_profiles is not None and not ccle_profiles.empty:
        cl_col = "CellLineName" if "CellLineName" in ccle_profiles.columns else "cell_line"
        for _, row in ccle_profiles.iterrows():
            norm = _normalize_cell_line_name(row[cl_col])
            ccle_norm[norm] = row

    # ── Build CMP EGFR driver lookup by model_id ────────────────────────────
    cmp_driver_by_sid = {}
    if cmp_egfr_drivers is not None and not cmp_egfr_drivers.empty:
        for sid, grp in cmp_egfr_drivers.groupby("model_id"):
            mutations = "; ".join(sorted(grp["protein_mutation"].dropna().unique()))
            effects = "; ".join(sorted(grp["effect"].dropna().unique()))
            cmp_driver_by_sid[sid] = {
                "mutations": mutations,
                "effects": effects,
                "model_name": grp["model_name"].iloc[0],
            }

    # ── Build comprehensive profile for each GDSC cell line ─────────────────
    print("\n    Building per-cell-line profiles...")

    records = []
    stats = {"driver": 0, "vus_only": 0, "confirmed_wt": 0, "unknown": 0}

    for _, cl_row in gdsc_cl.iterrows():
        cl_name = cl_row["CELL_LINE_NAME"]
        cl_norm = cl_row["cell_line_norm"]
        sid = str(cl_row["SANGER_MODEL_ID"]).strip() if pd.notna(cl_row.get("SANGER_MODEL_ID")) else ""

        # Collect evidence from both sources
        ccle_data = ccle_norm.get(cl_norm)
        cmp_data = cmp_driver_by_sid.get(sid)
        was_sequenced_ccle = (sid in ccle_screened_sids) or (ccle_data is not None)
        was_sequenced_cmp = sid in cmp_sequenced_sids

        evidence_sources = []
        if was_sequenced_ccle:
            evidence_sources.append("CCLE/DepMap")
        if was_sequenced_cmp:
            evidence_sources.append("CMP/Sanger")

        # ── Determine EGFR status ────────────────────────────────────────
        egfr_mutations = ""
        mutation_classes = ""
        egfr_status = ""

        # Check for CCLE driver mutations (pathogenic/oncogenic)
        has_ccle_driver = False
        if ccle_data is not None:
            mc = str(ccle_data.get("mutation_classes", ""))
            if "pathogenic_drug_response" in mc or "oncogenic_hotspot" in mc:
                has_ccle_driver = True

        # Check for CMP driver mutation
        has_cmp_driver = cmp_data is not None

        if has_ccle_driver or has_cmp_driver:
            # ── DRIVER MUTATION: merge both sources ──────────────────
            parts_mutations = []
            parts_classes = []

            if has_ccle_driver:
                parts_mutations.append(str(ccle_data.get("egfr_mutations", "")))
                parts_classes.append(str(ccle_data.get("mutation_classes", "")))

            if has_cmp_driver:
                # Add CMP mutations not already in CCLE
                cmp_muts = cmp_data["mutations"]
                if cmp_muts not in "; ".join(parts_mutations):
                    parts_mutations.append(cmp_muts)
                parts_classes.append("CMP_driver")

            egfr_mutations = "; ".join(filter(None, parts_mutations))
            mutation_classes = "; ".join(filter(None, parts_classes))
            egfr_status = "driver_mutation"
            stats["driver"] += 1

        elif ccle_data is not None:
            # ── VUS ONLY: CCLE found EGFR variants but no drivers ────
            egfr_mutations = str(ccle_data.get("egfr_mutations", ""))
            mutation_classes = str(ccle_data.get("mutation_classes", ""))
            egfr_status = "VUS_only"
            stats["vus_only"] += 1

        elif evidence_sources:
            # ── CONFIRMED WILD-TYPE: sequenced, no EGFR mutations ────
            egfr_mutations = "wild_type"
            mutation_classes = "wild_type"
            egfr_status = "confirmed_wild_type"
            stats["confirmed_wt"] += 1

        else:
            # ── UNKNOWN: not sequenced by any source ─────────────────
            egfr_mutations = "unknown"
            mutation_classes = "unknown"
            egfr_status = "unknown"
            stats["unknown"] += 1

        # ── Build record ─────────────────────────────────────────────
        record = {
            "CellLineName": cl_name,
            "SANGER_MODEL_ID": sid,
            "egfr_mutations": egfr_mutations,
            "mutation_classes": mutation_classes,
            "egfr_status": egfr_status,
            "evidence_sources": "|".join(evidence_sources) if evidence_sources else "none",
        }

        # Carry forward CCLE annotation columns if available
        if ccle_data is not None:
            for col in ["exons", "variant_types", "variant_info",
                        "clinical_significance", "hotspot", "oncogene_high_impact"]:
                record[col] = ccle_data.get(col, "")

        # Carry forward tissue info
        if ccle_data is not None:
            for col in ["OncotreeLineage", "OncotreePrimaryDisease"]:
                record[col] = ccle_data.get(col, "")

        records.append(record)

    # ── Build DataFrame and save ─────────────────────────────────────────────
    df_comprehensive = pd.DataFrame(records)

    print(f"\n    EGFR Status Summary for {len(df_comprehensive)} GDSC NSCLC cell lines:")
    print(f"      Driver mutations:    {stats['driver']:3d} cell lines")
    print(f"      VUS only:            {stats['vus_only']:3d} cell lines")
    print(f"      Confirmed wild-type: {stats['confirmed_wt']:3d} cell lines "
          f"(sequenced, no EGFR mutations)")
    print(f"      Unknown:             {stats['unknown']:3d} cell lines "
          f"(not sequenced)")

    # Show driver mutations detail
    drivers = df_comprehensive[df_comprehensive["egfr_status"] == "driver_mutation"]
    if not drivers.empty:
        print(f"\n    EGFR driver mutations ({len(drivers)} cell lines):")
        for _, row in drivers.iterrows():
            print(f"      {row['CellLineName']:20s} | {row['egfr_mutations']:40s} "
                  f"| [{row['evidence_sources']}]")

    # Save comprehensive profiles (overwrites the CCLE-only version)
    output_path = OUT_DIR / "egfr_mutations_by_cell_line.csv"
    df_comprehensive.to_csv(output_path, index=False)
    print(f"\n    ✓ Saved comprehensive EGFR profiles: {output_path}")
    print(f"      Coverage: {len(df_comprehensive) - stats['unknown']}/{len(df_comprehensive)} "
          f"cell lines with evidence-backed EGFR status")

    return df_comprehensive


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: ERBB2/HER2 Mutation Extraction & Sequence Generation (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def extract_erbb2_mutations(mutations_path: Path, model_path: Path):
    """
    Extract ERBB2 (HER2) mutations from CCLE and generate HER2 sequences.
    
    BIOLOGICAL CONTEXT:
    ───────────────────
    Unlike EGFR where activating mutations (L858R, exon19del) drive oncogenicity,
    HER2+ breast cancer is primarily driven by GENE AMPLIFICATION (overexpression),
    not point mutations. HER2 mutations are rare in breast cancer.
    
    Therefore, most HER2+ breast cancer cell lines will have:
    - Wild-type HER2 protein sequence (the protein itself is normal)
    - Elevated HER2 expression (amplification → more copies → more protein)
    
    For the model:
    - All HER2+ breast cancer cell lines use the ERBB2 wild-type sequence
    - The model distinguishes them from EGFR samples via the different sequence
      (HER2 is 1255 AA vs EGFR is 1210 AA, only ~83% kinase domain identity)
    - HER2 amplification status is handled as metadata, not sequence change
    
    OUTPUT:
    ───────
    erbb2_all_mutations.csv         — raw ERBB2 mutation records from CCLE
    erbb2_mutations_by_cell_line.csv — per-cell-line ERBB2 mutation profiles
    erbb2_mutant_sequences.fasta    — HER2 wild-type sequence (for ESM-2)
    """
    print("\n" + "=" * 70)
    print("STEP 2.9: Extracting ERBB2 (HER2) Mutations & Sequences")
    print("=" * 70)

    # ── Extract ERBB2 mutations from CCLE ──────────────────────────────────
    print("  Scanning CCLE for ERBB2 mutations...")
    erbb2_chunks = []
    chunk_size = 100_000

    for chunk in pd.read_csv(mutations_path, chunksize=chunk_size, low_memory=False):
        erbb2_rows = chunk[chunk["HugoSymbol"] == "ERBB2"]
        if len(erbb2_rows) > 0:
            erbb2_chunks.append(erbb2_rows)

    if erbb2_chunks:
        df_erbb2 = pd.concat(erbb2_chunks, ignore_index=True)
        print(f"  → Found {len(df_erbb2)} ERBB2 mutation records")
        print(f"  → Across {df_erbb2['ModelID'].nunique()} cell lines")

        # Merge with model info
        df_model = pd.read_csv(model_path)
        if "ModelID" in df_erbb2.columns and "ModelID" in df_model.columns:
            name_cols = ["ModelID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease"]
            available = [c for c in name_cols if c in df_model.columns]
            df_erbb2 = df_erbb2.merge(df_model[available], on="ModelID", how="left")

        # Parse protein changes
        if "ProteinChange" in df_erbb2.columns:
            df_erbb2["protein_change_clean"] = df_erbb2["ProteinChange"].str.replace("p.", "", regex=False)
            print("\n  Most frequent ERBB2 mutations:")
            for mut, count in df_erbb2["protein_change_clean"].value_counts().head(10).items():
                print(f"    {mut}: {count} cell lines")

        # Classify mutations
        df_erbb2["mutation_class"] = df_erbb2.apply(classify_from_annotations, axis=1)

        # Filter for breast cancer cell lines
        if "OncotreeLineage" in df_erbb2.columns:
            breast_mask = df_erbb2["OncotreeLineage"].str.contains("Breast", case=False, na=False)
            n_breast = df_erbb2[breast_mask]["ModelID"].nunique()
            print(f"\n  ERBB2 mutations in breast cancer lines: {breast_mask.sum()} records, {n_breast} cell lines")

        # Save all ERBB2 mutations
        output_all = OUT_DIR / "erbb2_all_mutations.csv"
        df_erbb2.to_csv(output_all, index=False)
        print(f"  ✓ Saved all ERBB2 mutations: {output_all}")

        # Build per-cell-line profiles for breast cancer
        cell_line_col = "CellLineName" if "CellLineName" in df_erbb2.columns else "ModelID"

        def join_unique(series):
            return "; ".join(sorted({str(v) for v in series if pd.notna(v) and str(v).strip()}))

        agg_dict = {"ProteinChange": join_unique, "mutation_class": join_unique}
        agg_dict = {k: v for k, v in agg_dict.items() if k in df_erbb2.columns}

        if agg_dict:
            profiles = df_erbb2.groupby(cell_line_col).agg(agg_dict).reset_index()
            profiles.rename(columns={"ProteinChange": "erbb2_mutations", "mutation_class": "mutation_classes"},
                            inplace=True)
            profiles["target_protein"] = "ERBB2"

            output_profiles = OUT_DIR / "erbb2_mutations_by_cell_line.csv"
            profiles.to_csv(output_profiles, index=False)
            print(f"  ✓ Saved ERBB2 mutation profiles: {output_profiles} ({len(profiles)} cell lines)")
    else:
        print("  → No ERBB2 mutations found in CCLE (expected — HER2 is amplification-driven)")
        # Create empty profiles file
        profiles = pd.DataFrame(columns=["CellLineName", "erbb2_mutations", "mutation_classes", "target_protein"])
        output_profiles = OUT_DIR / "erbb2_mutations_by_cell_line.csv"
        profiles.to_csv(output_profiles, index=False)

    # ── Load HER2 reference sequence and generate FASTA ──────────────────────
    erbb2_fasta = find_file(RAW_DIR, ["erbb2_p04626", "p04626.fasta"], "HER2 reference FASTA")

    if erbb2_fasta is not None:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        record = SeqIO.read(erbb2_fasta, "fasta")
        her2_wt = str(record.seq)

        erbb2_cfg = cfg["uniprot"]["ERBB2"]
        print(f"\n  HER2 sequence length: {len(her2_wt)} amino acids")
        print(f"  Kinase domain: positions {erbb2_cfg['kinase_domain_start']}-{erbb2_cfg['kinase_domain_end']}")

        # Verify key HER2 residues
        key_positions = [
            (1221, "Y", "GRB2 docking (≡EGFR Y1068)"),
            (1248, "Y", "SHC1 docking (≡EGFR Y1173)"),
            (1222, "Y", "GRB2/SHC dual (unique to HER2)"),
        ]
        print("\n  Key HER2 phosphosite residues:")
        for pos, expected_aa, function in key_positions:
            if pos <= len(her2_wt):
                actual = her2_wt[pos - 1]
                match = "✓" if actual == expected_aa else f"✗ (found {actual})"
                print(f"    Position {pos}: {actual} {match} — {function}")

        # Save HER2 wild-type sequence as FASTA for ESM-2
        # HER2+ breast cancer is amplification-driven, so wild-type sequence
        # is used for ALL HER2 samples (no point mutations to apply)
        output_fasta = OUT_DIR / "erbb2_mutant_sequences.fasta"
        records = [SeqRecord(
            Seq(her2_wt),
            id="ERBB2_wild_type",
            description="Human HER2/ERBB2 wild-type sequence (P04626) — used for all HER2+ breast samples"
        )]
        SeqIO.write(records, output_fasta, "fasta")
        print(f"\n  ✓ Saved HER2 wild-type sequence: {output_fasta}")
        print(f"    (HER2+ breast cancer uses WT sequence — oncogenicity is amplification-driven)")
    else:
        print("\n  ⚠ HER2 FASTA (erbb2_P04626.fasta) not found in data/raw/ccle/")
        print("    Download: curl -o data/raw/ccle/erbb2_P04626.fasta https://rest.uniprot.org/uniprotkb/P04626.fasta")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Build ERBB2 Profiles for GDSC Breast Cancer Cell Lines
# ══════════════════════════════════════════════════════════════════════════════

def build_erbb2_comprehensive_profiles():
    """
    Build ERBB2 mutation profiles for all GDSC breast cancer cell lines.
    
    Unlike EGFR (where mutations drive oncogenicity), HER2+ breast cancer is
    driven by gene AMPLIFICATION. Most breast cancer cell lines have wild-type
    HER2 protein sequence but elevated expression.
    
    We assign erbb2_status based on:
    1. HER2-amplified known cell lines → "HER2_amplified" (functionally active)
    2. Cell lines with ERBB2 mutations → "ERBB2_mutant" (rare)
    3. All other breast lines → "ERBB2_wild_type" (normal expression)
    """
    print("\n" + "=" * 70)
    print("STEP 2.10: Building ERBB2 Profiles for Breast Cancer Cell Lines")
    print("=" * 70)

    # Known HER2-amplified breast cancer cell lines (from literature)
    KNOWN_HER2_AMP = {
        "BT-474", "BT474", "SKBR3", "SK-BR-3", "AU565", "HCC1954",
        "MDA-MB-453", "MDA-MB-361", "ZR-75-30", "UACC-812",
        "HCC1569", "HCC202", "JIMT-1", "JIMT1",
    }

    # Load GDSC breast cancer cell lines
    case_study_name2 = cfg.get("case_study", "egfr_erbb2_tki")
    gdsc_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "gdsc" / f"gdsc_{case_study_name2}_responses.csv"
    if not gdsc_path.exists():
        gdsc_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "gdsc" / "gdsc_erbb_tki_responses.csv"
    if not gdsc_path.exists():
        print("    ⚠ GDSC ERBB data not found. Run step01 first.")
        return

    df_gdsc = pd.read_csv(gdsc_path)
    breast_cl = df_gdsc[df_gdsc["target_protein"] == "ERBB2"].drop_duplicates("CELL_LINE_NAME")

    if breast_cl.empty:
        print("    No ERBB2 breast cancer cell lines in GDSC data.")
        return

    print(f"    GDSC breast cancer cell lines: {len(breast_cl)}")

    records = []
    stats = {"amplified": 0, "wild_type": 0}

    for _, row in breast_cl.iterrows():
        cl_name = row["CELL_LINE_NAME"]
        cl_norm = _normalize_cell_line_name(cl_name)

        # Check if this is a known HER2-amplified line
        is_amplified = any(
            _normalize_cell_line_name(known) == cl_norm
            for known in KNOWN_HER2_AMP
        )

        if is_amplified:
            erbb2_status = "HER2_amplified"
            stats["amplified"] += 1
        else:
            erbb2_status = "ERBB2_wild_type"
            stats["wild_type"] += 1

        records.append({
            "CellLineName": cl_name,
            "SANGER_MODEL_ID": row.get("SANGER_MODEL_ID", ""),
            "target_protein": "ERBB2",
            "erbb2_status": erbb2_status,
            "erbb2_mutations": "wild_type",  # All use WT sequence
            "sequence_id": "ERBB2_wild_type",
            "pdb_id": "3PP0",  # HER2 apo structure for all
        })

    df_erbb2_profiles = pd.DataFrame(records)

    print(f"\n    ERBB2 Status Summary:")
    print(f"      HER2-amplified:  {stats['amplified']:3d} cell lines (known HER2+ lines)")
    print(f"      ERBB2 wild-type: {stats['wild_type']:3d} cell lines (other breast lines)")

    # List the amplified lines
    amp_lines = df_erbb2_profiles[df_erbb2_profiles["erbb2_status"] == "HER2_amplified"]
    if not amp_lines.empty:
        print(f"\n    Known HER2-amplified cell lines found:")
        for _, r in amp_lines.iterrows():
            print(f"      {r['CellLineName']}")

    output_path = OUT_DIR / "erbb2_mutations_by_cell_line.csv"
    df_erbb2_profiles.to_csv(output_path, index=False)
    print(f"\n    ✓ Saved ERBB2 profiles: {output_path}")

    return df_erbb2_profiles


def run(case_study: str = "egfr_erbb2_tki"):
    """Main entry point — call from thin wrappers or CLI."""
    _init(case_study)
    # Delegate to __main__ logic below
    _main_logic()


def _main_logic():
    """Core execution logic (separated for run() reuse)."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 02: Download ERBB Family Mutation Profiles           ║")
    print("║  (EGFR + ERBB2/HER2)                                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Sources: CCLE/DepMap + CMP/Sanger + UniProt               ║")
    print("║  Method : Multi-source, evidence-based classification      ║")
    print("║  Output: Comprehensive mutation profiles + sequences       ║")
    print("║  Genes: EGFR (NSCLC) + ERBB2/HER2 (Breast)               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ══════════════════════════════════════════════════════════════════════
    # PART A: EGFR Pipeline (existing, unchanged)
    # ══════════════════════════════════════════════════════════════════════

    # Step 1: Locate manually downloaded CCLE files
    mut_path, model_path, fasta_path = locate_ccle_files()

    # Step 2: Extract EGFR mutations with all annotation columns
    df_mutations = extract_egfr_mutations(mut_path)

    # Step 3: Classify using the data's own rich annotations
    df_mutations = classify_mutations(df_mutations)

    # Step 4: Merge model info & build per-cell-line profiles (CCLE)
    ccle_profiles = build_cell_line_profiles(df_mutations, model_path)

    # Step 5: Load EGFR reference sequence
    wt_sequence = load_egfr_reference_sequence(fasta_path)

    # Step 6: Generate mutant sequences from what the data actually contains
    mutant_seqs = generate_mutant_sequences(wt_sequence, df_mutations)

    # Step 7: Integrate Cell Model Passports (CMP/Sanger) mutations
    cmp_egfr_drivers, cmp_sequenced_sids = integrate_cmp_mutations()

    # Step 8: Build comprehensive profiles for ALL GDSC NSCLC cell lines
    comprehensive_profiles = build_comprehensive_profiles(
        ccle_profiles, cmp_egfr_drivers, cmp_sequenced_sids
    )

    # ══════════════════════════════════════════════════════════════════════
    # PART B: ERBB2/HER2 Pipeline (NEW — ERBB family expansion)
    # ══════════════════════════════════════════════════════════════════════

    # Step 9: Extract ERBB2 mutations + generate HER2 wild-type sequence
    extract_erbb2_mutations(mut_path, model_path)

    # Step 10: Build ERBB2 profiles for breast cancer cell lines
    erbb2_profiles = build_erbb2_comprehensive_profiles()

    print("\n✓ Step 02 complete!")
    print("  EGFR + ERBB2 mutation profiles and sequences ready.")
    print("  Next: Run step03 (structures), then step06 (harmonize).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 02 — Download mutations")
    parser.add_argument("--case-study", default="egfr_erbb2_tki",
                        help="Case study name (default: egfr_erbb2_tki)")
    args, _ = parser.parse_known_args()
    run(case_study=args.case_study)
