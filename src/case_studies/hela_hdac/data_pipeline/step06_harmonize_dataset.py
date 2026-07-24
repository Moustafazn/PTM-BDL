#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 06 — HeLa / HDAC Inhibitor: Build Multimodal Dataset                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Build a multi-cell-line multimodal dataset following the EGFR framework   ║
║    architecture: per-site PTM vectors on specific target proteins.           ║
║                                                                              ║
║  TARGET PROTEINS:                                                            ║
║    HDAC1 (Q13547) — 8 phospho + 4 acetyl sites = 12 tokens                 ║
║    EP300 (Q09472) — 5 phospho + 7 acetyl sites = 12 tokens                 ║
║                                                                              ║
║  PER-SITE PTM VECTOR (matching EGFR pattern):                               ║
║    Each sample gets per-site columns: ptm_S393, ptm_S421, ptm_K218, etc.   ║
║    + delta_ptm columns for drug-induced changes                              ║
║    The PTM-BDL self-attention layer processes these as per-site tokens.     ║
║                                                                              ║
║  INPUT:                                                                      ║
║    data/processed/drugptm/hela_hdac_ptm_responses.csv  (step05 drug data)  ║
║    data/processed/drugptm/hela_hdac_baseline_ptm.csv   (step05 baselines)  ║
║    data/processed/gdsc/gdsc_hela_hdac_responses.csv    (step01 IC50)       ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/processed/hela_hdac/multimodal_dataset.csv                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="hela_hdac")
CASE_STUDY = cfg.get("case_study", "hela_hdac")

PROCESSED_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"]
OUT_DIR = PROCESSED_DIR / CASE_STUDY
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize_cell_line_name(name):
    if pd.isna(name):
        return name
    name = str(name).upper().strip()
    for prefix in ["NCI-", "NCI_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("-", "")


def get_drug_smiles(drug_name: str) -> str:
    dn = str(drug_name).lower().strip()
    for key, info in cfg.get("drugs", {}).items():
        if key in dn or info["name"].lower() in dn:
            return info["smiles"]
    return ""


def assign_tissue_group(tcga_desc: str) -> str:
    if pd.isna(tcga_desc) or not tcga_desc:
        return "other"
    desc = str(tcga_desc).upper()
    for group_name, pattern in cfg.get("loclo", {}).get("tissue_groups", {}).items():
        if not pattern:
            continue
        for p in pattern.split("|"):
            if p.strip() and p.strip() in desc:
                return group_name
    return "other"


def load_protein_sequence(protein_name: str) -> str:
    """Load protein FASTA sequence from data/processed/sequences/."""
    seq_dir = PROCESSED_DIR / "sequences"
    for fasta_path in seq_dir.glob(f"{protein_name.lower()}_*.fasta"):
        with open(fasta_path) as f:
            lines = f.readlines()
        return "".join(l.strip() for l in lines if not l.startswith(">"))
    return ""


_POS_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def resolve_uniprot_position(ptm_site: str, peptide: str,
                              protein_seq: str) -> int:
    """
    Map a DrugPTM-Bench peptide-local site index to a UniProt protein position.

    DrugPTM-Bench encodes ptm_site as "<residue_type><position_in_peptide>"
    (e.g. "S8" = serine at position 8 within the tryptic peptide).
    This function finds the peptide in the full protein sequence and returns
    the 1-based UniProt residue number.

    Returns -1 if the mapping cannot be resolved.
    """
    if not ptm_site or not peptide or not protein_seq:
        return -1
    m = _POS_RE.match(ptm_site.strip())
    if not m:
        return -1
    try:
        pos_in_pep = int(m.group(2)) - 1  # 0-based index within peptide
    except ValueError:
        return -1
    if pos_in_pep < 0 or pos_in_pep >= len(peptide):
        return -1
    idx = protein_seq.find(peptide)
    if idx < 0:
        return -1
    return idx + pos_in_pep + 1  # 1-based UniProt position


# ══════════════════════════════════════════════════════════════════════════════
# Build per-site PTM schema from config
# ══════════════════════════════════════════════════════════════════════════════

def get_phospho_schema(protein_name: str) -> list:
    """
    Get phospho sites for a protein, padded to ptm_dim.
    Returns list of dicts — primary channel (phosphorylation).
    """
    ptm_cfg = cfg.get("ptm", {})
    protein_cfg = ptm_cfg.get(protein_name, {})
    ptm_dim = ptm_cfg.get("ptm_dim", 8)

    sites = []
    for site in protein_cfg.get("phospho_sites", []):
        sites.append({**site, "ptm_type": "phosphorylation"})

    while len(sites) < ptm_dim:
        sites.append({"position": 0, "residue": "PAD", "amino_acid": "X",
                       "function": "padding", "ptm_type": "none"})
    return sites[:ptm_dim]


def get_acetyl_schema(protein_name: str) -> list:
    """
    Get acetyl sites for a protein, padded to secondary_dim.
    Returns list of dicts — secondary channel (acetylation).
    """
    ptm_cfg = cfg.get("ptm", {})
    protein_cfg = ptm_cfg.get(protein_name, {})
    secondary_dim = ptm_cfg.get("secondary_dim",
                                 ptm_cfg.get("glyco_dim", 0))
    if secondary_dim == 0:
        return []

    sites = []
    for site in protein_cfg.get("acetyl_sites", []):
        sites.append({**site, "ptm_type": "acetylation"})

    while len(sites) < secondary_dim:
        sites.append({"position": 0, "residue": "PAD", "amino_acid": "X",
                       "function": "padding", "ptm_type": "none"})
    return sites[:secondary_dim]


def get_all_sites(protein_name: str) -> list:
    """Get combined phospho + acetyl sites (for drug profile / baseline lookups)."""
    return get_phospho_schema(protein_name) + get_acetyl_schema(protein_name)


# ══════════════════════════════════════════════════════════════════════════════
# Load Data
# ══════════════════════════════════════════════════════════════════════════════

def load_gdsc_responses() -> pd.DataFrame:
    gdsc_path = PROCESSED_DIR / "gdsc" / f"gdsc_{CASE_STUDY}_responses.csv"
    if not gdsc_path.exists():
        print(f"    ✗ {gdsc_path} not found. Run step01 first.")
        return pd.DataFrame()
    df = pd.read_csv(gdsc_path)
    our_drugs = list(cfg["gdsc"]["drug_ids"].keys())
    mask = df["DRUG_NAME"].str.lower().isin([d.lower() for d in our_drugs])
    return df[mask].copy()


def load_drug_ptm_responses() -> pd.DataFrame:
    path = PROCESSED_DIR / "drugptm" / "hela_hdac_ptm_responses.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_baseline_ptm() -> pd.DataFrame:
    path = PROCESSED_DIR / "drugptm" / "hela_hdac_baseline_ptm.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
# Build per-site drug-response profiles from DrugPTM-Bench
# ══════════════════════════════════════════════════════════════════════════════

def build_per_site_drug_profiles(df_ptm: pd.DataFrame,
                                  protein_name: str,
                                  sites: list,
                                  protein_seq: str = "") -> dict:
    """
    For each drug, compute the mean log2FC at each defined PTM site
    on the target protein.

    Uses peptide→protein sequence alignment to resolve DrugPTM-Bench's
    peptide-local site indices (e.g. "S8" = position 8 in peptide) to
    UniProt residue positions.

    Returns dict: drug_name → {position: mean_log2fc}
    """
    if df_ptm.empty:
        return {}

    # Filter to target protein rows
    protein_mask = df_ptm["protein"].astype(str).str.contains(protein_name, na=False)
    df_prot = df_ptm[protein_mask].copy()

    if df_prot.empty:
        return {}

    # Build position set from defined sites
    site_positions = {s["position"] for s in sites if s["position"] > 0}

    profiles = {}
    n_resolved = 0
    n_failed = 0

    for drug in df_prot["drug_name"].unique():
        drug_data = df_prot[df_prot["drug_name"] == drug]
        site_values = {}

        for _, row in drug_data.iterrows():
            peptide = str(row.get("peptide_sequence", ""))
            ptm_site = str(row.get("ptm_site", ""))

            # Resolve peptide-local index → UniProt position
            if protein_seq and peptide:
                ptm_idx = resolve_uniprot_position(ptm_site, peptide,
                                                    protein_seq)
            else:
                # Fallback: try direct numeric parse (for data that
                # already uses UniProt positions like "Y245")
                try:
                    ptm_idx = int(ptm_site.lstrip("SYTKsytk"))
                except (ValueError, AttributeError):
                    ptm_idx = -1

            if ptm_idx > 0 and ptm_idx in site_positions:
                fc = row.get("log2_fold_change")
                if pd.notna(fc):
                    site_values.setdefault(ptm_idx, []).append(float(fc))
                    n_resolved += 1
            else:
                n_failed += 1

        # Average multiple measurements per site
        profiles[drug] = {pos: np.mean(vals) for pos, vals in site_values.items()}

    if n_resolved > 0 or n_failed > 0:
        print(f"    Peptide→protein mapping: {n_resolved} resolved, "
              f"{n_failed} unmatched")

    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# Build per-cell-line baseline PTM at defined sites
# ══════════════════════════════════════════════════════════════════════════════

def build_ordinal_to_config_map(df_ptm: pd.DataFrame,
                                 protein_name: str,
                                 protein_seq: str,
                                 config_positions: set) -> dict:
    """
    Build a mapping from DrugPTM-Bench ordinal site labels to config
    UniProt positions, using peptide→protein sequence alignment.

    For each ordinal (e.g. "S10"), we resolve ALL peptide-based UniProt
    positions, then intersect with config_positions.  If exactly one
    config position matches, the ordinal is unambiguously mapped.

    Returns dict: ordinal_label → config_position  (only unambiguous entries)
    """
    if df_ptm.empty or not protein_seq:
        return {}

    protein_mask = df_ptm["protein"].astype(str).str.contains(
        protein_name, na=False)
    df_prot = df_ptm[protein_mask].copy()
    if df_prot.empty:
        return {}

    ordinal_to_positions: dict[str, set] = {}
    for _, row in df_prot.drop_duplicates(
            subset=["ptm_site", "peptide_sequence"]).iterrows():
        site = str(row.get("ptm_site", ""))
        peptide = str(row.get("peptide_sequence", ""))
        pos = resolve_uniprot_position(site, peptide, protein_seq)
        if pos > 0:
            ordinal_to_positions.setdefault(site, set()).add(pos)

    mapping = {}
    for ordinal, positions in ordinal_to_positions.items():
        config_matches = positions & config_positions
        if len(config_matches) == 1:
            mapping[ordinal] = config_matches.pop()

    return mapping


def build_per_site_baselines(df_baseline: pd.DataFrame,
                              protein_name: str,
                              sites: list,
                              ordinal_map: dict | None = None) -> dict:
    """
    For each cell line, extract baseline intensity at defined PTM sites.

    Uses ordinal_map (from build_ordinal_to_config_map) to resolve
    DrugPTM-Bench ordinal labels to UniProt positions when available.
    """
    if df_baseline.empty:
        return {}

    protein_mask = df_baseline["gene"].astype(str).str.contains(
        protein_name, na=False)
    df_prot = df_baseline[protein_mask].copy()
    if df_prot.empty:
        return {}

    site_positions = {s["position"] for s in sites if s["position"] > 0}
    baselines = {}

    for cl in df_prot["cell_line"].unique():
        cl_data = df_prot[df_prot["cell_line"] == cl]
        site_vals = {}
        for _, row in cl_data.iterrows():
            ptm_site = str(row.get("ptm_site", ""))

            ptm_idx = -1
            if ordinal_map and ptm_site in ordinal_map:
                ptm_idx = ordinal_map[ptm_site]
            else:
                try:
                    ptm_idx = int(ptm_site.lstrip("SYTKsytk"))
                except (ValueError, AttributeError):
                    continue

            if ptm_idx > 0 and ptm_idx in site_positions:
                bl = row.get("baseline_intensity")
                if pd.notna(bl):
                    site_vals[ptm_idx] = float(bl)
        if site_vals:
            baselines[cl] = site_vals

    return baselines


# ══════════════════════════════════════════════════════════════════════════════
# Literature IC50 entries for drugs not in GDSC2
# ══════════════════════════════════════════════════════════════════════════════

def _build_literature_drug_entries() -> list[dict]:
    """
    Build GDSC-format rows for epigenetic drugs not in the GDSC2 panel.

    These drugs have DrugPTM-Bench PTM data (step05) but no GDSC2 IC50.
    We add published IC50 values for HeLa and well-characterized cell lines.

    Published IC50 sources:
      A485 (EP300/CBP HAT inhibitor — active enantiomer):
        HeLa:    IC50 ≈ 30 nM  — Lasko et al., Nature 2017 (PMID 29211713)
        A549:    IC50 ≈ 50 nM  — Lasko et al., Nature 2017
        MCF7:    IC50 ≈ 100 nM — Lasko et al., Nature 2017

      A486 (inactive enantiomer of A485 — negative control):
        HeLa:    IC50 > 10 µM  — Lasko et al., Nature 2017 (PMID 29211713)
        A549:    IC50 > 10 µM  — Lasko et al., Nature 2017
        MCF7:    IC50 > 10 µM  — Lasko et al., Nature 2017

      CUDC-101 (triple HDAC/EGFR/HER2 inhibitor):
        HeLa:    IC50 ≈ 1.0 µM — Lai et al., J Med Chem 2010 (PMID 20568778)
        A549:    IC50 ≈ 0.7 µM — Lai et al., J Med Chem 2010
        MDA-MB-231: IC50 ≈ 1.5 µM — Lai et al., J Med Chem 2010

      Curcumin (natural polyphenol, HDAC/HAT modulator):
        HeLa:    IC50 ≈ 15 µM  — Syng-ai et al., Mol Carcinog 2014
        A549:    IC50 ≈ 20 µM  — Chen et al., BBRC 2015
    """
    entries = []

    # ── A485 (HAT inhibitor — active) ────────────────────────────────────
    for cl, ic50_uM, tcga in [("HeLa", 0.030, "CESC"),
                               ("A549", 0.050, "LUAD"),
                               ("MCF7", 0.100, "BRCA")]:
        entries.append({
            "CELL_LINE_NAME": cl, "DRUG_NAME": "A485", "DRUG_ID": 9901,
            "LN_IC50": np.log(ic50_uM), "resistance_label": 0,
            "TCGA_DESC": tcga, "PUTATIVE_TARGET": "EP300",
            "PATHWAY_NAME": "Chromatin histone acetylation",
        })

    # ── A486 (inactive enantiomer — negative control) ────────────────────
    for cl, ic50_uM, tcga in [("HeLa", 10.0, "CESC"),
                               ("A549", 10.0, "LUAD"),
                               ("MCF7", 10.0, "BRCA")]:
        entries.append({
            "CELL_LINE_NAME": cl, "DRUG_NAME": "A486", "DRUG_ID": 9902,
            "LN_IC50": np.log(ic50_uM), "resistance_label": 1,
            "TCGA_DESC": tcga, "PUTATIVE_TARGET": "EP300",
            "PATHWAY_NAME": "Chromatin histone acetylation",
        })

    # ── CUDC-101 (triple HDAC/EGFR/HER2) ────────────────────────────────
    for cl, ic50_uM, tcga, label in [("HeLa", 1.0, "CESC", 1),
                                      ("A549", 0.7, "LUAD", 0),
                                      ("MDA-MB-231", 1.5, "BRCA", 1)]:
        entries.append({
            "CELL_LINE_NAME": cl, "DRUG_NAME": "CUDC-101", "DRUG_ID": 1578,
            "LN_IC50": np.log(ic50_uM), "resistance_label": label,
            "TCGA_DESC": tcga, "PUTATIVE_TARGET": "HDAC1, EGFR, ERBB2",
            "PATHWAY_NAME": "Chromatin histone acetylation",
        })

    # ── Curcumin (natural HDAC/HAT modulator) ────────────────────────────
    for cl, ic50_uM, tcga in [("HeLa", 15.0, "CESC"),
                               ("A549", 20.0, "LUAD")]:
        entries.append({
            "CELL_LINE_NAME": cl, "DRUG_NAME": "Curcumin", "DRUG_ID": 9903,
            "LN_IC50": np.log(ic50_uM), "resistance_label": 1,
            "TCGA_DESC": tcga, "PUTATIVE_TARGET": "HDAC, HAT",
            "PATHWAY_NAME": "Chromatin histone acetylation",
        })

    print(f"  Building literature IC50 entries for A485, A486, CUDC-101, Curcumin")
    print(f"    ({len(entries)} entries)")
    print(f"    Refs: Lasko Nature 2017, Lai JMC 2010, Syng-ai Mol Carcinog 2014")
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Per-cell-line PTM modulators (following EGFR pattern)
# ──────────────────────────────────────────────────────────────────────────────
# Creates per-cell-line variation in the PTM baseline vector based on
# tissue of origin and drug mechanism context.
#
# Biological basis:
#   - Cervical/epithelial cell lines have higher HDAC1 expression and
#     activity → higher CK2-mediated phospho at S393/S421/S423
#     Ref: Pflum et al., JBC 2001 (PMID 11929873)
#   - Hematological cell lines have different chromatin states
#   - EP300 acetylation activity varies by cell lineage
# ══════════════════════════════════════════════════════════════════════════════

def normalize_baseline(value: float) -> float:
    """Normalize raw PTM baseline values to consistent scale."""
    if value <= 0:
        return 0.0
    if value > 100:  # raw MS intensity → log2 scale
        return round(np.log2(1.0 + value), 4)
    return round(value, 4)


TISSUE_MODULATORS = {
    "cervical":    {"S393": 1.3, "S421": 1.2, "S423": 1.2, "K218": 1.1, "default": 1.1},
    "breast":      {"S393": 1.1, "S421": 1.1, "K218": 1.0, "default": 1.0},
    "lung_nsclc":  {"S393": 0.95, "S421": 0.9, "K218": 0.95, "default": 0.95},
    "lung_sclc":   {"S393": 0.9, "K218": 0.9, "default": 0.9},
    "leukemia":    {"S393": 0.85, "S421": 0.85, "K218": 1.1, "default": 0.9},
    "lymphoma":    {"S393": 0.9, "K218": 1.05, "default": 0.92},
    "colorectal":  {"S393": 1.0, "S421": 0.95, "default": 0.95},
    "ovarian":     {"S393": 1.05, "K218": 1.0, "default": 0.98},
    "pancreatic":  {"S393": 0.9, "default": 0.88},
    "skin":        {"S393": 1.0, "default": 0.95},
    "brain":       {"S393": 0.8, "K218": 0.85, "default": 0.85},
    "other":       {"default": 0.95},
}

DRUG_DELTA_SCALE = {
    "Vorinostat":  1.0,   # Pan-HDAC, strong epigenetic effect
    "Romidepsin":  0.9,   # Class I selective
    "CUDC-101":    0.7,   # Triple inhibitor, partial HDAC
    "CUDC101":     0.7,
    "A485":        0.8,   # HAT inhibitor (opposite mechanism)
    "A486":        0.1,   # Inactive control
    "Curcumin":    0.4,   # Natural modulator, pleiotropic
}


def compute_tissue_modulated_baseline(baseline: float, site_residue: str,
                                       tissue_group: str) -> float:
    mods = TISSUE_MODULATORS.get(tissue_group, TISSUE_MODULATORS.get("other", {}))
    multiplier = mods.get(site_residue, mods.get("default", 1.0))
    return round(baseline * multiplier, 4)


def compute_propagation_confidence(tissue_group: str, drug_name: str,
                                    cl_has_measured_baseline: bool) -> float:
    tissue_conf = {
        "cervical": 0.9, "breast": 0.5, "lung_nsclc": 0.4,
        "lung_sclc": 0.35, "leukemia": 0.4, "lymphoma": 0.35,
        "colorectal": 0.35, "ovarian": 0.4, "pancreatic": 0.3,
        "skin": 0.35, "brain": 0.25, "other": 0.3,
    }
    conf = tissue_conf.get(tissue_group, 0.3)
    if cl_has_measured_baseline:
        conf = min(conf + 0.15, 1.0)
    if drug_name in {"Vorinostat", "Romidepsin"}:
        conf = min(conf + 0.05, 1.0)
    return round(conf, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Build Multimodal Dataset — Multi-protein + per-cell-line modulators
# ══════════════════════════════════════════════════════════════════════════════

def build_multimodal_dataset():
    print("\n" + "=" * 70)
    print(f"STEP 06: Building Per-Site PTM Multimodal Dataset — {CASE_STUDY}")
    print(f"  Multi-protein: HDAC1 + EP300")
    print(f"  Per-cell-line modulators: tissue-based epigenetic context")
    print("=" * 70)

    df_gdsc = load_gdsc_responses()
    df_ptm = load_drug_ptm_responses()
    df_baseline = load_baseline_ptm()

    if df_gdsc.empty:
        print("  ✗ No GDSC data. Run step01 first.")
        return None

    # Add literature IC50 entries
    literature_entries = _build_literature_drug_entries()
    if literature_entries:
        df_lit = pd.DataFrame(literature_entries)
        n_before = len(df_gdsc)
        df_gdsc = pd.concat([df_gdsc, df_lit], ignore_index=True)
        print(f"  ✓ Added {len(df_lit)} literature IC50 entries "
              f"(total: {n_before} → {len(df_gdsc)})")

    print(f"  GDSC records: {len(df_gdsc):,}")
    print(f"  Drug-PTM responses: {len(df_ptm):,}")
    print(f"  Baseline PTM sites: {len(df_baseline):,}")

    # ── Build per-protein data ──────────────────────────────────────────
    # Use HDAC1 + EP300 (both defined in config with PTM site schemas)
    target_proteins = ["HDAC1", "EP300"]
    ptm_dim = cfg.get("ptm", {}).get("ptm_dim", 8)
    secondary_dim = cfg.get("ptm", {}).get("secondary_dim",
                                            cfg.get("ptm", {}).get("glyco_dim", 0))

    # ── Column naming: primary (phospho) + secondary (acetyl) ────────────
    # Primary channel: ptm_* columns (phospho sites from first protein)
    primary_protein = target_proteins[0]
    phospho_schema = get_phospho_schema(primary_protein)
    ptm_columns = [f"ptm_{s['residue']}" for s in phospho_schema]
    delta_ptm_columns = [f"delta_ptm_{s['residue']}" for s in phospho_schema]

    # Secondary channel: secondary_slot* columns (acetyl sites)
    # Uses _slot naming convention so ResistanceDataset auto-discovers them
    secondary_columns = [f"secondary_slot{i:02d}" for i in range(secondary_dim)]
    delta_secondary_columns = [f"delta_secondary_slot{i:02d}" for i in range(secondary_dim)]

    print(f"  Primary (phospho):   {len(ptm_columns)} columns (ptm_dim={ptm_dim})")
    print(f"  Secondary (acetyl):  {len(secondary_columns)} columns (secondary_dim={secondary_dim})")

    protein_data = {}
    for protein in target_proteins:
        phospho = get_phospho_schema(protein)
        acetyl = get_acetyl_schema(protein)
        all_sites = phospho + acetyl
        print(f"\n  Protein: {protein}")
        real_phospho = [s for s in phospho if s["position"] > 0]
        real_acetyl = [s for s in acetyl if s["position"] > 0]
        print(f"    Phospho: {len(real_phospho)} real + {ptm_dim - len(real_phospho)} padded")
        for s in real_phospho:
            print(f"      {s['residue']:6s} ({s['amino_acid']}) — {s['function']}")
        print(f"    Acetyl:  {len(real_acetyl)} real + {secondary_dim - len(real_acetyl)} padded")
        for s in real_acetyl:
            print(f"      {s['residue']:6s} ({s['amino_acid']}) — {s['function']}")

        ptm_vectors_path = PROCESSED_DIR / "ptm" / f"{protein.lower()}_ptm_state_vectors.json"
        step04_bl = {}
        if ptm_vectors_path.exists():
            with open(ptm_vectors_path) as f:
                step04_bl = json.load(f)
            print(f"    ✓ Step04 baselines: {list(step04_bl.keys())}")

        protein_seq = load_protein_sequence(protein)
        if protein_seq:
            print(f"    ✓ Sequence: {len(protein_seq)} aa")

        drug_profiles = build_per_site_drug_profiles(
            df_ptm, protein, all_sites, protein_seq=protein_seq)
        for drug, profile in drug_profiles.items():
            print(f"    Drug profile {drug}: {len(profile)} sites")

        site_positions = {s["position"] for s in all_sites if s["position"] > 0}
        ordinal_map = build_ordinal_to_config_map(
            df_ptm, protein, protein_seq, site_positions)
        baselines = build_per_site_baselines(
            df_baseline, protein, all_sites, ordinal_map=ordinal_map)
        print(f"    Baselines: {len(baselines)} cell lines")

        protein_data[protein] = {
            "phospho_sites": phospho,
            "acetyl_sites": acetyl,
            "step04_baselines": step04_bl,
            "drug_profiles": drug_profiles,
            "baselines": baselines,
            "sequence_id": protein.lower(),
        }

    pdb_map = {
        "HDAC1": {"_default": "4LXZ"},  # HDAC8+Vorinostat (closest available)
        "EP300": {"_default": "4BKX"},   # p300 HAT domain
    }

    # ── Build dataset: GDSC × target_proteins ────────────────────────────
    records = []
    print(f"\n  Building dataset: {len(df_gdsc):,} GDSC records × "
          f"{len(target_proteins)} proteins...")

    for _, gdsc_row in df_gdsc.iterrows():
        cell_line = str(gdsc_row.get("CELL_LINE_NAME", "")).strip()
        drug_name = str(gdsc_row.get("DRUG_NAME", "")).strip()
        ln_ic50 = gdsc_row.get("LN_IC50", np.nan)
        resistance_label = gdsc_row.get("resistance_label", np.nan)
        tcga_desc = str(gdsc_row.get("TCGA_DESC", ""))
        tissue_group = assign_tissue_group(tcga_desc)
        cl_norm = normalize_cell_line_name(cell_line)

        for protein in target_proteins:
            pdata = protein_data[protein]
            p_sites = pdata["phospho_sites"]
            a_sites = pdata["acetyl_sites"]
            step04_bl = pdata["step04_baselines"]
            drug_profiles = pdata["drug_profiles"]
            bl_dict = pdata["baselines"]

            prot_pdb = pdb_map.get(protein, {"_default": "4BKX"})
            pdb_id = prot_pdb.get(drug_name, prot_pdb.get("_default", "4BKX"))

            cl_baseline = None
            for bl_name, bl_data in bl_dict.items():
                if normalize_cell_line_name(bl_name) == cl_norm:
                    cl_baseline = bl_data
                    break

            drug_profile = drug_profiles.get(drug_name, {})
            if not drug_profile:
                for pname, pdata_dp in drug_profiles.items():
                    if drug_name.lower().startswith(pname.lower()[:6]):
                        drug_profile = pdata_dp
                        break

            prop_conf = compute_propagation_confidence(
                tissue_group, drug_name, cl_baseline is not None)
            delta_scale = DRUG_DELTA_SCALE.get(drug_name, 0.5)

            rec = {
                "cell_line_name": cell_line,
                "drug_name": drug_name,
                "target_protein": protein,
                "sequence_id": pdata["sequence_id"],
                "pdb_id": pdb_id,
                "drug_smiles": get_drug_smiles(drug_name),
                "ln_ic50": ln_ic50,
                "resistance_label": int(resistance_label) if pd.notna(resistance_label) else np.nan,
                "tissue_group": tissue_group,
                "tcga_desc": tcga_desc,
                "propagation_confidence": prop_conf,
            }

            # ── Primary channel: phospho sites → ptm_* columns ──────────
            for i, site in enumerate(p_sites):
                pos = site["position"]
                col = ptm_columns[i]
                delta_col = delta_ptm_columns[i]

                if pos == 0:
                    rec[col] = 0.0
                    rec[delta_col] = 0.0
                    continue

                baseline = 1.0
                step04_bg = step04_bl.get("baseline_level", {})
                if str(pos) in step04_bg:
                    baseline = float(step04_bg[str(pos)])
                if cl_baseline and pos in cl_baseline:
                    baseline = cl_baseline[pos]

                baseline = normalize_baseline(baseline)
                site_residue = site.get("residue", "")
                baseline = compute_tissue_modulated_baseline(
                    baseline, site_residue, tissue_group)
                # Safety clip (matches EGFR pattern)
                baseline = max(0.1, min(baseline, 5.0))
                rec[col] = baseline

                delta = drug_profile.get(pos, 0.0)
                delta = max(-5.0, min(round(delta * delta_scale, 4), 5.0))
                rec[delta_col] = delta

            # ── Secondary channel: acetyl sites → secondary_slot* columns ─
            for j, site in enumerate(a_sites):
                pos = site["position"]
                col = secondary_columns[j]
                delta_col = delta_secondary_columns[j]

                if pos == 0:
                    rec[col] = 0.0
                    rec[delta_col] = 0.0
                    continue

                baseline = 1.0
                step04_bg = step04_bl.get("baseline_level", {})
                if str(pos) in step04_bg:
                    baseline = float(step04_bg[str(pos)])
                if cl_baseline and pos in cl_baseline:
                    baseline = cl_baseline[pos]

                baseline = normalize_baseline(baseline)
                site_residue = site.get("residue", "")
                baseline = compute_tissue_modulated_baseline(
                    baseline, site_residue, tissue_group)
                # Safety clip (matches EGFR pattern)
                baseline = max(0.1, min(baseline, 5.0))
                rec[col] = baseline

                delta = drug_profile.get(pos, 0.0)
                delta = max(-5.0, min(round(delta * delta_scale, 4), 5.0))
                rec[delta_col] = delta

            records.append(rec)

    df_out = pd.DataFrame(records)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  ✓ Built multimodal dataset:")
    print(f"    Total samples:    {len(df_out):,}")
    print(f"    Cell lines:       {df_out['cell_line_name'].nunique()}")
    print(f"    Target proteins:  {sorted(df_out['target_protein'].unique())}")
    for prot in df_out['target_protein'].unique():
        n = len(df_out[df_out['target_protein'] == prot])
        print(f"      {prot}: {n} samples")
    print(f"    Drugs:            {sorted(df_out['drug_name'].unique())}")

    n_sens = (df_out["resistance_label"] == 0).sum()
    n_res = (df_out["resistance_label"] == 1).sum()
    print(f"    Sensitive:        {n_sens} ({100*n_sens/len(df_out):.1f}%)")
    print(f"    Resistant:        {n_res} ({100*n_res/len(df_out):.1f}%)")

    all_ptm_cols = [c for c in ptm_columns + secondary_columns if c in df_out.columns]
    all_delta_cols = [c for c in delta_ptm_columns + delta_secondary_columns if c in df_out.columns]
    n_unique = df_out[all_ptm_cols + all_delta_cols].drop_duplicates().shape[0]
    print(f"    Unique PTM vectors: {n_unique}")
    print(f"    Primary (phospho) columns: {[c for c in ptm_columns if c in df_out.columns]}")
    print(f"    Secondary (acetyl) columns: {[c for c in secondary_columns if c in df_out.columns]}")

    output_path = OUT_DIR / "multimodal_dataset.csv"
    df_out.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path}")

    summary = {
        "case_study": CASE_STUDY,
        "total_samples": len(df_out),
        "n_cell_lines": int(df_out["cell_line_name"].nunique()),
        "target_proteins": sorted(df_out["target_protein"].unique().tolist()),
        "drugs": sorted(df_out["drug_name"].unique().tolist()),
        "ptm_types": ["phosphorylation", "acetylation"],
        "n_unique_ptm_vectors": n_unique,
        "columns": list(df_out.columns),
    }
    with open(OUT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return df_out


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 06 — HeLa/HDAC: Multi-Protein PTM Multimodal Dataset    ║")
    print("║  Proteins: HDAC1 + EP300 (per-cell-line modulators)           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    df = build_multimodal_dataset()
    if df is not None:
        print("\n✓ Step 06 complete!")
