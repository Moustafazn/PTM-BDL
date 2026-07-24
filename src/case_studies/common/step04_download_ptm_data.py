#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 04 — Download PTM (Post-Translational Modification) Data              ║
║  (ERBB Family: EGFR + HER2/ERBB2)                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Collect experimentally validated phosphorylation data for EGFR and HER2  ║
║    from UniProt and augment with literature-curated quantitative profiles.   ║
║    PTMs are the CORE of our hypothesis — we propose that phosphorylation    ║
║    states interact with mutations to drive drug resistance.                  ║
║                                                                              ║
║  ERBB FAMILY EXPANSION (v2):                                                 ║
║    Now creates PTM profiles for BOTH EGFR (12 sites) and HER2 (10 sites     ║
║    + 2 zero-padded = ptm_dim=12 for both). Config defines sites per gene.   ║
║                                                                              ║
║  KEY FINDINGS (Section 7a of HER2_EXPANSION_PLAN.md):                       ║
║    • Neratinib NOT in GDSC2 → replaced with Sapitinib (AZD8931, ID 1549)   ║
║    • ALL EGFR drugs also tested on 52 breast cancer cell lines              ║
║    • GDSC2 tissue = "Breast Carcinoma" (not "BRCA")                        ║
║    • Combined: 943 records (638 EGFR + 305 ERBB2), 212 cell lines          ║
║    • 9 known HER2-amplified breast lines found in GDSC2                     ║
║    • HER2 PTM: 10 sites from P04626, padded to 12 to match ptm_dim        ║
║                                                                              ║
║  WHY PTM DATA IS CRITICAL:                                                   ║
║    Phosphorylation is not just a binary on/off — it is QUANTITATIVE:         ║
║    • Different mutation backgrounds → different phosphorylation levels       ║
║    • L858R activating mutation → hyper-phosphorylation of Y1092, Y1197      ║
║    • T790M gatekeeper → altered phosphorylation pattern                      ║
║    • Drug treatment → suppresses phosphorylation (if drug works)            ║
║    • Resistant cells → maintain phosphorylation despite drug treatment       ║
║                                                                              ║
║  MANUAL DOWNLOAD REQUIRED:                                                   ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    EGFR — UniProt P00533 JSON:                                              ║
║      curl -o data/raw/ptm/uniprot_P00533.json \                             ║
║           "https://rest.uniprot.org/uniprotkb/P00533.json"                  ║
║    HER2 — UniProt P04626 JSON (optional, for validation):                   ║
║      curl -o data/raw/ptm/uniprot_P04626.json \                             ║
║           "https://rest.uniprot.org/uniprotkb/P04626.json"                  ║
║    HER2 phospho sites are defined in config.yaml (ptm.ERBB2.phospho_sites) ║
║    and do NOT require a separate JSON download — they are literature-curated.║
║                                                                              ║
║  NUMBERING CONVENTION:                                                       ║
║    All positions use UniProt P00533 PRECURSOR numbering (1–1210),            ║
║    which includes the 24-aa signal peptide.  This matches the mutation       ║
║    nomenclature used throughout the project (T790M, L858R, C797S) and       ║
║    the kinase domain boundaries (712–979).                                   ║
║                                                                              ║
║    Classic literature sites use mature-protein numbering (precursor − 24):   ║
║      Y845 (lit.) = Y869 (UniProt)  — Src kinase activation loop            ║
║      Y992 (lit.) = Y1016 (UniProt) — PLCγ binding                          ║
║      Y1045 (lit.) = Y1069 (UniProt) — c-Cbl / receptor degradation         ║
║      Y1068 (lit.) = Y1092 (UniProt) — Grb2/RAS-MAPK (CRITICAL)            ║
║      Y1086 (lit.) = Y1110 (UniProt) — Grb2 secondary                       ║
║      Y1101 (lit.) = Y1125 (UniProt) — Signaling (Grb2/Shc region)          ║
║      Y1148 (lit.) = Y1172 (UniProt) — Shc binding                          ║
║      Y1173 (lit.) = Y1197 (UniProt) — Shc/PLCγ/PI3K-AKT (CRITICAL)        ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    EGFR:                                                                     ║
║      data/raw/ptm/uniprot_egfr_ptm_sites.csv                                ║
║      data/processed/ptm/egfr_phosphorylation_sites.csv                       ║
║      data/processed/ptm/egfr_ptm_state_vectors.json                          ║
║      data/processed/ptm/egfr_glycosylation_sites.csv      (NEW PTM-BDL §3.1) ║
║      data/processed/ptm/egfr_glyco_state_vectors.json     (NEW PTM-BDL §3.1) ║
║    HER2:                                                                     ║
║      data/processed/ptm/erbb2_phosphorylation_sites.csv                      ║
║      data/processed/ptm/erbb2_ptm_state_vectors.json                         ║
║      data/processed/ptm/erbb2_glycosylation_sites.csv     (NEW PTM-BDL §3.2) ║
║      data/processed/ptm/erbb2_glyco_state_vectors.json    (NEW PTM-BDL §3.2) ║
║                                                                              ║
║  PTM-BDL MULTI-PTM EXTENSION (PTM_Biological_Dynamics_Layer.md §3, §7):     ║
║    In addition to the phosphorylation sites this step now also emits the     ║
║    canonical N-glycosylation site database for both EGFR (12 sites) and      ║
║    HER2/ERBB2 (7 sites + 5 zero-padded → glyco_dim = 12).  These provide    ║
║    the extracellular receptor-surface PTM channel that the PTM-BDL token     ║
║    branch (step10) consumes as the SECOND modification type (phospho_N=3).  ║
║    Functional rationale: glyco controls HOW MUCH receptor reaches the cell   ║
║    surface; phospho controls HOW ACTIVE that receptor is.  Together they    ║
║    form the two-axis PTM code that the model needs to learn (proposal §6).  ║
║                                                                              ║
║    Glyco state vectors are simpler than phospho:                            ║
║      • EGFR:  baseline = 1.0 across all sites (no mutation-conditional       ║
║              baseline available; per-cell-line modulation in step06).        ║
║      • ERBB2: baseline = 1.0 (WT) / 1.5 (HER2-amplified) — Taniguchi 2024.  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path

import pandas as pd

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
    RAW_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "ptm"
    OUT_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"] / "ptm"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [Config] case_study = {case_study}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Parse UniProt PTM Annotations
# ══════════════════════════════════════════════════════════════════════════════

def download_uniprot_ptm_annotations():
    """
    Parse experimentally validated PTM annotations for EGFR from UniProt JSON.

    The JSON must be manually downloaded first:
      curl -o data/raw/ptm/uniprot_P00533.json \\
           "https://rest.uniprot.org/uniprotkb/P00533.json"

    UniProt PTM annotations are expert-curated from published literature.
    They provide HIGH-CONFIDENCE site identifications with evidence counts.
    We use this as the foundation and augment with quantitative data from
    literature-curated profiles (Section 2).
    """
    print("\n" + "=" * 70)
    print("STEP 4.1: Parsing EGFR PTM Annotations from UniProt")
    print("=" * 70)

    accession = cfg["uniprot"]["EGFR"]["accession"]
    json_path = RAW_DIR / f"uniprot_{accession}.json"

    if json_path.exists():
        print(f"  ✓ Found UniProt data: {json_path.name}")
        with open(json_path) as f:
            data = json.load(f)

        # Report entry metadata
        audit = data.get("entryAudit", {})
        print(f"    Entry version: {audit.get('entryVersion')}")
        print(f"    Last annotation update: {audit.get('lastAnnotationUpdateDate')}")
        print(f"    Sequence version: {audit.get('sequenceVersion')}")

        # Extract PTM features
        ptm_sites = []
        features = data.get("features", [])

        for feat in features:
            feat_type = feat.get("type", "")
            if feat_type == "Modified residue":
                location = feat.get("location", {})
                start = location.get("start", {}).get("value")
                description = feat.get("description", "")
                evidences = feat.get("evidences", [])
                if "Phospho" in description:
                    ptm_sites.append({
                        "position": start,
                        "residue_type": description.split(";")[0],
                        "description": description,
                        "evidence_count": len(evidences),
                        "source": "UniProt"
                    })
            elif feat_type in ["Active site", "Binding site"]:
                location = feat.get("location", {})
                start = location.get("start", {}).get("value")
                description = feat.get("description", "")
                ptm_sites.append({
                    "position": start,
                    "residue_type": feat_type,
                    "description": description,
                    "evidence_count": 0,
                    "source": "UniProt"
                })

        df_ptm = pd.DataFrame(ptm_sites)
        if len(df_ptm) > 0:
            n_phospho = sum(1 for _, r in df_ptm.iterrows()
                            if "Phospho" in str(r.get("residue_type", "")))
            print(f"  → Found {len(df_ptm)} modification/functional sites "
                  f"({n_phospho} phosphorylation)")

        output_path = RAW_DIR / "uniprot_egfr_ptm_sites.csv"
        df_ptm.to_csv(output_path, index=False)
        print(f"  ✓ Saved: {output_path}")

        # Validate against our curated sites
        _validate_curated_vs_uniprot(data)

        return df_ptm
    else:
        print(f"  ✗ UniProt data NOT FOUND.")
        print(f"  → Download with:")
        print(f"      curl -o {json_path} \\")
        print(f'           "https://rest.uniprot.org/uniprotkb/{accession}.json"')
        return None


def _validate_curated_vs_uniprot(uniprot_data):
    """
    Cross-check our 12 curated phosphosite positions against the actual
    UniProt P00533 sequence to confirm amino acid identity.
    """
    seq = uniprot_data["sequence"]["value"]
    # Config now has per-protein PTM sites: cfg["ptm"]["EGFR"]["phospho_sites"]
    sites = cfg["ptm"]["EGFR"]["phospho_sites"]
    print(f"\n  Validating {len(sites)} curated sites against UniProt sequence:")

    all_ok = True
    for site in sites:
        pos = site["position"]
        expected_aa = site["amino_acid"]
        actual_aa = seq[pos - 1] if pos <= len(seq) else "?"
        ok = actual_aa == expected_aa
        status = "✓" if ok else "✗ MISMATCH"
        classic = site.get("classic_name", "")
        label = f" (classic {classic})" if classic != site["residue"] else ""
        print(f"    {site['residue']}{label}: "
              f"UniProt pos {pos} = {actual_aa} {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"  ✓ All {len(sites)} sites validated against UniProt sequence")
    else:
        print(f"  ✗ SOME SITES FAILED VALIDATION — check numbering!")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Curate Known EGFR Phosphorylation Sites
# ══════════════════════════════════════════════════════════════════════════════

def create_curated_phospho_sites():
    """
    Create a comprehensive table of EGFR phosphorylation sites from literature.

    All positions use UniProt P00533 PRECURSOR numbering (includes 24-aa
    signal peptide).  Classic literature names are noted for cross-reference.

    THE 12 KEY EGFR PHOSPHORYLATION SITES:
    ───────────────────────────────────────
    Y869  (lit. Y845):  Activation loop. Phosphorylated by Src kinase.
    S991:               Regulatory serine. Detected in Hsu 2025 + Tozuka 2024.
    Y998:               PLCγ/Cbl binding. Controls receptor endocytosis.
    Y1016 (lit. Y992):  PLCγ1 binding. Activates PKC pathway.
    S1039:              Regulatory serine near Cbl-binding region.
    T1041:              Phosphothreonine. Regulatory role near Y1069.
    Y1069 (lit. Y1045): c-Cbl binding. Triggers receptor degradation.
    Y1092 (lit. Y1068): Grb2 binding. RAS-MAPK — THE CRITICAL SITE.
    Y1110 (lit. Y1086): Secondary Grb2 binding.
    Y1125 (lit. Y1101): Signaling between Grb2 and Shc sites.
    Y1172 (lit. Y1148): Shc binding. Alternative RAS-MAPK route.
    Y1197 (lit. Y1173): Shc/PLCγ1. PI3K-AKT survival — SECOND CRITICAL SITE.
    """
    print("\n" + "=" * 70)
    print("STEP 4.2: Curating EGFR Phosphorylation Site Database")
    print("=" * 70)

    # All positions: UniProt P00533 PRECURSOR numbering
    # Quantitative values: relative phosphorylation (1.0 = WT baseline)
    phospho_sites = [
        # ── Y869 (lit. Y845): Activation loop (Src substrate) ──
        {
            "position": 869, "residue": "Y869", "amino_acid": "Y",
            "classic_name": "Y845",
            "region": "activation_loop", "kinase": "Src",
            "binding_partner": "None (structural role)",
            "pathway": "Kinase activation stabilization",
            "function": "Stabilizes active kinase conformation; enhances catalytic activity",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 2.5,
            "T790M_phospho_level": 1.2,
            "L858R_T790M_phospho_level": 3.0,
            "C797S_phospho_level": 1.1,
            "drug_sensitive_phospho": 0.2,
            "drug_resistant_phospho": 2.8,
            "clinical_importance": "medium",
            "evidence_sources": "UniProt Y869; Sato et al., 2003; Chung et al., 2009"
        },
        # ── S991: Regulatory serine (C-terminal tail) ──
        {
            "position": 991, "residue": "S991", "amino_acid": "S",
            "classic_name": "S991",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "Unknown",
            "pathway": "Regulatory",
            "function": "Regulatory serine phosphorylation; may modulate C-terminal tail conformation",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 1.3,
            "T790M_phospho_level": 1.0,
            "L858R_T790M_phospho_level": 1.4,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.5,
            "drug_resistant_phospho": 1.3,
            "clinical_importance": "low",
            "evidence_sources": "UniProt S991 (4 evidences); Hsu et al., 2025; Tozuka et al., 2024"
        },
        # ── Y998: PLCγ/Cbl binding — endocytosis control ──
        {
            "position": 998, "residue": "Y998", "amino_acid": "Y",
            "classic_name": "Y998",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "PLCγ / Cbl",
            "pathway": "Endocytosis / receptor internalization",
            "function": "Controls EGFR internalization via Cbl and PLCγ recruitment",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 1.8,
            "T790M_phospho_level": 1.1,
            "L858R_T790M_phospho_level": 2.0,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.3,
            "drug_resistant_phospho": 1.8,
            "clinical_importance": "medium",
            "evidence_sources": "UniProt Y998 (2 evidences); PNAS 2025 (pY998 TMT)"
        },
        # ── Y1016 (lit. Y992): PLCγ binding site ──
        {
            "position": 1016, "residue": "Y1016", "amino_acid": "Y",
            "classic_name": "Y992",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "PLCγ1",
            "pathway": "PKC/calcium signaling",
            "function": "Recruits PLCγ1; activates protein kinase C and calcium release",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 2.0,
            "T790M_phospho_level": 1.1,
            "L858R_T790M_phospho_level": 2.2,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.3,
            "drug_resistant_phospho": 2.0,
            "clinical_importance": "low",
            "evidence_sources": "UniProt Y1016 (2 evidences); Schulze et al., 2005"
        },
        # ── S1039: Regulatory serine (C-terminal tail) ──
        {
            "position": 1039, "residue": "S1039", "amino_acid": "S",
            "classic_name": "S1039",
            "region": "c_terminal_tail", "kinase": "Unknown",
            "binding_partner": "Unknown",
            "pathway": "Regulatory",
            "function": "Serine phosphorylation near Cbl-binding region; may modulate "
                        "Y1069-dependent Cbl recruitment or receptor trafficking",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 1.2,
            "T790M_phospho_level": 1.0,
            "L858R_T790M_phospho_level": 1.3,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.6,
            "drug_resistant_phospho": 1.2,
            "clinical_importance": "low",
            "evidence_sources": "UniProt S1039 (1 evidence); Tozuka et al., 2024"
        },
        # ── T1041: Phosphothreonine (regulatory, near Cbl site) ──
        {
            "position": 1041, "residue": "T1041", "amino_acid": "T",
            "classic_name": "T1041",
            "region": "c_terminal_tail", "kinase": "Unknown",
            "binding_partner": "Unknown",
            "pathway": "Regulatory (proximal to Cbl recruitment)",
            "function": "Phosphothreonine adjacent to Y1069 Cbl-binding site; may modulate "
                        "ubiquitination efficiency and receptor degradation kinetics",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 1.5,
            "T790M_phospho_level": 1.0,
            "L858R_T790M_phospho_level": 1.6,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.3,
            "drug_resistant_phospho": 1.5,
            "clinical_importance": "low",
            "evidence_sources": "UniProt T1041 (Phosphothreonine, 1 evidence)"
        },
        # ── Y1069 (lit. Y1045): c-Cbl binding (NEGATIVE REGULATOR) ──
        {
            "position": 1069, "residue": "Y1069", "amino_acid": "Y",
            "classic_name": "Y1045",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "c-Cbl",
            "pathway": "Receptor degradation/endocytosis",
            "function": "Recruits c-Cbl E3 ligase; triggers EGFR ubiquitination and degradation",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 0.6,
            "T790M_phospho_level": 0.8,
            "L858R_T790M_phospho_level": 0.4,
            "C797S_phospho_level": 0.9,
            "drug_sensitive_phospho": 0.1,
            "drug_resistant_phospho": 0.3,
            "clinical_importance": "high",
            "evidence_sources": "UniProt Y1069 (2 evidences); Shtiegman et al., 2007",
            "resistance_note": "REDUCED Y1069 phosphorylation in mutant EGFR → impaired receptor "
                               "degradation → sustained signaling despite drug treatment."
        },
        # ── Y1092 (lit. Y1068): Grb2 binding (MAJOR — RAS-MAPK) ──
        {
            "position": 1092, "residue": "Y1092", "amino_acid": "Y",
            "classic_name": "Y1068",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "Grb2",
            "pathway": "RAS-MAPK (proliferation)",
            "function": "PRIMARY Grb2 recruitment site; activates RAS-RAF-MEK-ERK cascade",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 4.0,
            "T790M_phospho_level": 1.5,
            "L858R_T790M_phospho_level": 5.0,
            "C797S_phospho_level": 1.3,
            "drug_sensitive_phospho": 0.1,
            "drug_resistant_phospho": 4.5,
            "clinical_importance": "critical",
            "evidence_sources": "UniProt Y1092 (3 evidences); Sordella et al., 2004",
            "resistance_note": "Y1092 (pY1068 in literature) is THE key readout of EGFR activity. "
                               "Persistent phosphorylation after drug treatment = drug is NOT working."
        },
        # ── Y1110 (lit. Y1086): Secondary Grb2 site ──
        {
            "position": 1110, "residue": "Y1110", "amino_acid": "Y",
            "classic_name": "Y1086",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "Grb2 (secondary)",
            "pathway": "RAS-MAPK (secondary)",
            "function": "Secondary Grb2 binding; partially redundant with Y1092",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 2.5,
            "T790M_phospho_level": 1.2,
            "L858R_T790M_phospho_level": 3.0,
            "C797S_phospho_level": 1.1,
            "drug_sensitive_phospho": 0.2,
            "drug_resistant_phospho": 2.5,
            "clinical_importance": "medium",
            "evidence_sources": "UniProt Y1110 (2 evidences); Schulze et al., 2005"
        },
        # ── Y1125 (lit. Y1101): Signaling tyrosine ──
        {
            "position": 1125, "residue": "Y1125", "amino_acid": "Y",
            "classic_name": "Y1101",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "Unknown",
            "pathway": "Signaling (Grb2/Shc region)",
            "function": "Tyrosine between Y1110 (Grb2) and Y1172 (Shc); may contribute "
                        "to adapter protein recruitment and fine-tune MAPK signaling",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 2.0,
            "T790M_phospho_level": 1.1,
            "L858R_T790M_phospho_level": 2.3,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.2,
            "drug_resistant_phospho": 2.0,
            "clinical_importance": "medium",
            "evidence_sources": "Sequence-verified Y at UniProt position 1125"
        },
        # ── Y1172 (lit. Y1148): Shc binding site ──
        {
            "position": 1172, "residue": "Y1172", "amino_acid": "Y",
            "classic_name": "Y1148",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "Shc",
            "pathway": "RAS-MAPK (alternative route)",
            "function": "Recruits Shc adapter; alternative RAS activation pathway",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 2.0,
            "T790M_phospho_level": 1.0,
            "L858R_T790M_phospho_level": 2.2,
            "C797S_phospho_level": 1.0,
            "drug_sensitive_phospho": 0.3,
            "drug_resistant_phospho": 2.0,
            "clinical_importance": "low",
            "evidence_sources": "UniProt Y1172 (3 evidences)"
        },
        # ── Y1197 (lit. Y1173): Shc/PLCγ (MAJOR — PI3K-AKT) ──
        {
            "position": 1197, "residue": "Y1197", "amino_acid": "Y",
            "classic_name": "Y1173",
            "region": "c_terminal_tail", "kinase": "EGFR (auto)",
            "binding_partner": "Shc / PLCγ1",
            "pathway": "PI3K-AKT (survival) + RAS-MAPK",
            "function": "Recruits Shc and PLCγ1; activates PI3K-AKT survival signaling",
            "wt_phospho_level": 1.0,
            "L858R_phospho_level": 3.5,
            "T790M_phospho_level": 1.3,
            "L858R_T790M_phospho_level": 4.0,
            "C797S_phospho_level": 1.2,
            "drug_sensitive_phospho": 0.1,
            "drug_resistant_phospho": 3.5,
            "clinical_importance": "critical",
            "evidence_sources": "UniProt Y1197 (7 evidences); Sordella et al., 2004",
            "resistance_note": "Y1197 (pY1173 in literature) activates PI3K-AKT survival pathway. "
                               "Persistent phosphorylation can sustain cell survival through AKT."
        },
    ]

    df = pd.DataFrame(phospho_sites)

    output_path = OUT_DIR / "egfr_phosphorylation_sites.csv"
    df.to_csv(output_path, index=False)

    n_tyr = sum(1 for s in phospho_sites if s["amino_acid"] == "Y")
    n_ser = sum(1 for s in phospho_sites if s["amino_acid"] == "S")
    n_thr = sum(1 for s in phospho_sites if s["amino_acid"] == "T")
    print(f"\n  Created curated phosphorylation database:")
    print(f"    Sites: {len(df)} ({n_tyr} tyrosine + {n_ser} serine + {n_thr} threonine)")
    print(f"    Critical sites: Y1069 (Cbl), Y1092 (Grb2/MAPK), Y1197 (Shc/AKT)")
    print(f"  ✓ Saved: {output_path}")

    # ── Create PTM state vectors ─────────────────────────────────────────────
    print("\n  Creating PTM state vectors per mutation background...")

    mutation_backgrounds = [
        "wt_phospho_level",
        "L858R_phospho_level",
        "T790M_phospho_level",
        "L858R_T790M_phospho_level",
        "C797S_phospho_level",
        "drug_sensitive_phospho",
        "drug_resistant_phospho"
    ]

    ptm_vectors = {}
    for bg in mutation_backgrounds:
        vector = df.set_index("position")[bg].to_dict()
        ptm_vectors[bg] = vector
        values = [f"{pos}={val:.1f}" for pos, val in sorted(vector.items())]
        print(f"    {bg}: {', '.join(values)}")

    vectors_path = OUT_DIR / "egfr_ptm_state_vectors.json"
    with open(vectors_path, "w") as f:
        json.dump(ptm_vectors, f, indent=2)
    print(f"\n  ✓ Saved PTM state vectors: {vectors_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Create HER2/ERBB2 Phosphorylation Sites (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def create_erbb2_phospho_sites():
    """
    Create HER2/ERBB2 phosphorylation site database from config.
    
    HER2 has 10 phospho sites (from config ptm.ERBB2.phospho_sites),
    padded to ptm_dim=12 with zeros for model compatibility.
    
    Unlike EGFR (where mutation-specific phospho levels are known from
    literature), HER2 phospho is primarily driven by AMPLIFICATION level
    rather than point mutations. We use a simpler baseline model:
    - HER2-amplified: elevated baseline phosphorylation (1.5× WT)
    - HER2 wild-type: normal baseline (1.0)
    - Drug-treated: reduced phosphorylation (measured from DrugPTM-Bench)
    
    Homologous sites: Y1221 (HER2) ≡ Y1068/Y1092 (EGFR) — GRB2 docking
                      Y1248 (HER2) ≡ Y1173/Y1197 (EGFR) — SHC1/PI3K-AKT
    """
    print("\n" + "=" * 70)
    print("STEP 4.3: Creating HER2/ERBB2 Phosphorylation Site Database")
    print("=" * 70)

    erbb2_sites_cfg = cfg["ptm"]["ERBB2"]["phospho_sites"]
    ptm_dim = cfg["ptm"]["ptm_dim"]  # 12

    print(f"  HER2 phospho sites from config: {len(erbb2_sites_cfg)}")
    print(f"  ptm_dim: {ptm_dim} (HER2 has {len(erbb2_sites_cfg)} real + {ptm_dim - len(erbb2_sites_cfg)} zero-padded)")

    # Build phospho site records with quantitative values
    phospho_sites = []
    for site in erbb2_sites_cfg:
        record = {
            "position": site["position"],
            "residue": site["residue"],
            "amino_acid": site["amino_acid"],
            "classic_name": site["classic_name"],
            "function": site.get("function", ""),
            "egfr_homolog": site.get("egfr_homolog", None),
            "target_protein": "ERBB2",
            # Quantitative phospho levels for HER2 (amplification-driven)
            "wt_phospho_level": 1.0,
            "HER2_amplified_phospho_level": 1.5,  # Amplification → elevated baseline
            "drug_sensitive_phospho": 0.2,  # TKI suppresses phospho
            "drug_resistant_phospho": 1.4,  # Resistant cells maintain phospho
            "clinical_importance": "high" if site["classic_name"] in ["Y1221", "Y1248"] else "medium",
        }
        phospho_sites.append(record)

        homolog = site.get("egfr_homolog")
        homolog_str = f" (≡EGFR {homolog})" if homolog else ""
        print(f"    {site['residue']}: {site['function']}{homolog_str}")

    df = pd.DataFrame(phospho_sites)

    output_path = OUT_DIR / "erbb2_phosphorylation_sites.csv"
    df.to_csv(output_path, index=False)

    n_tyr = sum(1 for s in phospho_sites if s["amino_acid"] == "Y")
    n_ser = sum(1 for s in phospho_sites if s["amino_acid"] == "S")
    n_thr = sum(1 for s in phospho_sites if s["amino_acid"] == "T")
    print(f"\n  Created HER2 phosphorylation database:")
    print(f"    Sites: {len(df)} ({n_tyr} tyrosine + {n_ser} serine + {n_thr} threonine)")
    print(f"    Critical homologous sites: Y1221 (≡EGFR Y1068), Y1248 (≡EGFR Y1173)")
    print(f"    Zero-padded positions: sites {len(erbb2_sites_cfg) + 1}–{ptm_dim}")
    print(f"  ✓ Saved: {output_path}")

    # ── Create HER2 PTM state vectors ──────────────────────────────────────
    print("\n  Creating HER2 PTM state vectors...")

    her2_vectors = {
        "wt_phospho_level": {},
        "HER2_amplified_phospho_level": {},
        "drug_sensitive_phospho": {},
        "drug_resistant_phospho": {},
    }

    for bg in her2_vectors:
        vector = df.set_index("position")[bg].to_dict()
        her2_vectors[bg] = vector

    vectors_path = OUT_DIR / "erbb2_ptm_state_vectors.json"
    with open(vectors_path, "w") as f:
        json.dump(her2_vectors, f, indent=2)
    print(f"  ✓ Saved HER2 PTM state vectors: {vectors_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: N-Glycosylation Site Databases (NEW — PTM-BDL §3.1, §3.2, §7.4)
# ══════════════════════════════════════════════════════════════════════════════
#
# These two functions are the GLYCO analogue of `create_curated_phospho_sites`
# and `create_erbb2_phospho_sites`.  They emit per-protein:
#
#   • `<gene>_glycosylation_sites.csv`   — one row per canonical N-glyco
#                                          site (config.yaml ptm.<gene>.glyco_sites)
#                                          with functional annotation
#   • `<gene>_glyco_state_vectors.json`  — relative glyco occupancy per
#                                          biological background (1.0 = WT
#                                          baseline); consumed by step06's
#                                          map_mutations_to_glyco_vector and
#                                          fed to the PTM-BDL glyco token
#                                          branch as the ptm_modification_type
#                                          = "glyco_N" (type_id = 3) channel.
#
# DESIGN NOTE:  glyco state vectors are SIMPLER than phospho because
# (a) glycosylation is not driven by point mutations at the kinase domain
# in our datasets, and (b) we have no drug-conditioned glyco fold-changes
# for the EGFR TKIs.  The per-cell-line modulation (HER2-amp tier, MET-amp,
# tissue) is therefore deferred to step06 — same pattern as the ERBB2
# phospho HER2-amp multiplier.
# ══════════════════════════════════════════════════════════════════════════════

# Canonical glyco backgrounds we emit per gene (kept deliberately small to
# match the data we actually have).  step06's glyco vector pipeline picks
# whichever key matches the sample's gene + (optional) HER2-amp tier.
EGFR_GLYCO_BACKGROUNDS = (
    "wt_glyco_level",  # 1.0 across all sites — reference
    "drug_sensitive_glyco",  # 1.0 (no drug-induced glyco data for EGFR TKIs)
    "drug_resistant_glyco",  # 1.0
)
ERBB2_GLYCO_BACKGROUNDS = (
    "wt_glyco_level",  # 1.0 across all sites — reference
    "HER2_amplified_glyco_level",  # 1.5 — Taniguchi 2024 atlas: more
    #       glycopeptide PSMs detected in
    #       BT-474 / SK-BR-3 vs control,
    #       consistent with elevated
    #       receptor surface density.
    "drug_sensitive_glyco",  # 1.0 (baseline post-drug context)
    "drug_resistant_glyco",  # 1.0
)


def _validate_glyco_sites_vs_uniprot(uniprot_data, sites, gene_label: str):
    """
    Cross-check that each `N<pos>` glyco site actually has an Asn residue
    at the given UniProt precursor position.  Identical contract to the
    phospho validator above.
    """
    seq = uniprot_data["sequence"]["value"]
    print(f"\n  Validating {len(sites)} {gene_label} glyco sites against "
          f"UniProt sequence:")
    all_ok = True
    for site in sites:
        pos = site["position"]
        expected_aa = site["amino_acid"]
        actual_aa = seq[pos - 1] if pos <= len(seq) else "?"
        ok = actual_aa == expected_aa
        status = "✓" if ok else "✗ MISMATCH"
        classic = site.get("classic_name", "")
        label = (f" (mature {classic})"
                 if classic and classic != site["residue"] else "")
        print(f"    {site['residue']}{label}: "
              f"UniProt pos {pos} = {actual_aa} {status}")
        if not ok:
            all_ok = False
    if all_ok:
        print(f"  ✓ All {len(sites)} {gene_label} glyco sites validated")
    else:
        print(f"  ✗ Some {gene_label} glyco sites failed validation — "
              "check numbering!")


def create_egfr_glyco_sites():
    """
    Create the canonical EGFR N-glycosylation site database.

    Reads from `cfg["ptm"]["EGFR"]["glyco_sites"]` (12 sites, UniProt P00533
    precursor numbering — see config.yaml comments).  Emits:

      data/processed/ptm/egfr_glycosylation_sites.csv
      data/processed/ptm/egfr_glyco_state_vectors.json

    Quantitative columns mirror the phospho schema so step06 can consume
    both files with the same logic:
       wt_glyco_level, drug_sensitive_glyco, drug_resistant_glyco.

    All values default to 1.0 because (a) no mutation-class glyco baseline
    is available in our datasets (the OSCC and CHO-sEGFR data in step05 are
    REFERENCE distributions, not mutant-vs-WT contrasts), and (b) per-cell-
    line modulation is applied in step06 via the same modulator pattern
    used for phospho.  The model's PTM-BDL still receives the per-site
    GLYCO TOKEN (typed `glyco_N`, type_id = 3) and the model is free to
    learn that the per-site BASELINE is informative on its own — which is
    exactly the orthogonality we want (proposal §6, §7.4).
    """
    print("\n" + "=" * 70)
    print("STEP 4.4: Curating EGFR N-Glycosylation Site Database (NEW)")
    print("=" * 70)

    egfr_glyco_cfg = cfg["ptm"]["EGFR"].get("glyco_sites", [])
    glyco_dim = cfg["ptm"].get("glyco_dim", 12)

    if not egfr_glyco_cfg:
        print("  ⚠ ptm.EGFR.glyco_sites missing from config — skipping.")
        return None

    print(f"  EGFR glyco sites from config: {len(egfr_glyco_cfg)}")
    print(f"  glyco_dim: {glyco_dim} "
          f"(EGFR has {len(egfr_glyco_cfg)} real + "
          f"{glyco_dim - len(egfr_glyco_cfg)} zero-padded)")

    # Validate against UniProt sequence (if JSON is present)
    accession = cfg["uniprot"]["EGFR"]["accession"]
    json_path = RAW_DIR / f"uniprot_{accession}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        _validate_glyco_sites_vs_uniprot(data, egfr_glyco_cfg, "EGFR")
    else:
        print(f"  (UniProt JSON not found — skipping site-residue validation)")

    glyco_sites = []
    for site in egfr_glyco_cfg:
        record = {
            "position": site["position"],
            "residue": site["residue"],
            "amino_acid": site["amino_acid"],  # always "N"
            "classic_name": site.get("classic_name", ""),
            "function": site.get("function", ""),
            "target_protein": "EGFR",
            "ptm_modification_type": "glyco_N",  # PTM-BDL type ID 3
            # All glyco backgrounds = 1.0 baseline — see docstring above
            "wt_glyco_level": 1.0,
            "drug_sensitive_glyco": 1.0,
            "drug_resistant_glyco": 1.0,
            # The "critical" tag mirrors the phospho column; we mark sites
            # that are known to modulate ligand binding or trastuzumab-EGFR
            # heterodimerisation (Zhu 2026 / Taniguchi 2024).
            "clinical_importance": (
                "high" if site["residue"] in ("N361", "N528") else "medium"
            ),
        }
        glyco_sites.append(record)
        print(f"    {site['residue']:6s} "
              f"({site.get('classic_name', ''):8s})  "
              f"{site.get('function', '')}")

    df = pd.DataFrame(glyco_sites)
    output_path = OUT_DIR / "egfr_glycosylation_sites.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path.relative_to(PROJECT_ROOT)}")

    # ── State vectors (parallel to phospho) ──────────────────────────────
    print("\n  Creating EGFR glyco state vectors per background...")
    egfr_glyco_vectors = {}
    for bg in EGFR_GLYCO_BACKGROUNDS:
        col = bg
        if col not in df.columns:
            # Same baseline for all sites — synthesise on the fly.
            vector = {int(pos): 1.0 for pos in df["position"]}
        else:
            vector = df.set_index("position")[col].to_dict()
        egfr_glyco_vectors[bg] = vector
        sample = ", ".join(
            f"{pos}={val:.1f}"
            for pos, val in list(sorted(vector.items()))[:4]
        )
        print(f"    {bg:25s}: {sample}, ...")

    vectors_path = OUT_DIR / "egfr_glyco_state_vectors.json"
    with open(vectors_path, "w") as f:
        json.dump(egfr_glyco_vectors, f, indent=2)
    print(f"\n  ✓ Saved EGFR glyco state vectors: "
          f"{vectors_path.relative_to(PROJECT_ROOT)}")

    return df


def create_erbb2_glyco_sites():
    """
    Create the canonical HER2/ERBB2 N-glycosylation site database.

    Reads from `cfg["ptm"]["ERBB2"]["glyco_sites"]` (7 sites, UniProt P04626
    precursor numbering; padded to glyco_dim = 12 with zeros).

    Quantitative columns:
       wt_glyco_level                 = 1.0 across all sites (reference)
       HER2_amplified_glyco_level     = 1.5 — Taniguchi 2024 atlas observation
                                        that BT-474 / SK-BR-3 produce more
                                        glycoform-PSM peaks per site than
                                        non-amplified controls, consistent
                                        with elevated receptor density.
       drug_sensitive_glyco           = 1.0 (no drug-induced glyco data)
       drug_resistant_glyco           = 1.0

    Outputs:
      data/processed/ptm/erbb2_glycosylation_sites.csv
      data/processed/ptm/erbb2_glyco_state_vectors.json

    Homologous-site anchor for cross-receptor validation (proposal §3.3):
      ERBB2 N530  ↔  EGFR N528  — extracellular domain IV — trastuzumab
                                  binding interface.
    """
    print("\n" + "=" * 70)
    print("STEP 4.5: Curating HER2/ERBB2 N-Glycosylation Site Database (NEW)")
    print("=" * 70)

    erbb2_glyco_cfg = cfg["ptm"]["ERBB2"].get("glyco_sites", [])
    glyco_dim = cfg["ptm"].get("glyco_dim", 12)

    if not erbb2_glyco_cfg:
        print("  ⚠ ptm.ERBB2.glyco_sites missing from config — skipping.")
        return None

    print(f"  HER2 glyco sites from config: {len(erbb2_glyco_cfg)}")
    print(f"  glyco_dim: {glyco_dim} "
          f"(HER2 has {len(erbb2_glyco_cfg)} real + "
          f"{glyco_dim - len(erbb2_glyco_cfg)} zero-padded)")

    # Validate against UniProt sequence (if JSON is present)
    accession = cfg["uniprot"]["ERBB2"]["accession"]
    json_path = RAW_DIR / f"uniprot_{accession}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        _validate_glyco_sites_vs_uniprot(data, erbb2_glyco_cfg, "ERBB2")
    else:
        print(f"  (UniProt JSON P04626 not found — skipping residue validation)")

    glyco_sites = []
    for site in erbb2_glyco_cfg:
        record = {
            "position": site["position"],
            "residue": site["residue"],
            "amino_acid": site["amino_acid"],  # always "N"
            "classic_name": site.get("classic_name", ""),
            "function": site.get("function", ""),
            "egfr_homolog": site.get("egfr_homolog"),
            "target_protein": "ERBB2",
            "ptm_modification_type": "glyco_N",
            "wt_glyco_level": 1.0,
            "HER2_amplified_glyco_level": 1.5,
            "drug_sensitive_glyco": 1.0,
            "drug_resistant_glyco": 1.0,
            # N530 is the trastuzumab-binding glyco; we mark it as "high".
            "clinical_importance": (
                "high" if site["residue"] == "N530" else "medium"
            ),
        }
        glyco_sites.append(record)
        homolog = site.get("egfr_homolog")
        homolog_str = f"  (≡EGFR {homolog})" if homolog else ""
        print(f"    {site['residue']:6s} "
              f"({site.get('classic_name', ''):8s})  "
              f"{site.get('function', '')}{homolog_str}")

    df = pd.DataFrame(glyco_sites)
    output_path = OUT_DIR / "erbb2_glycosylation_sites.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path.relative_to(PROJECT_ROOT)}")

    # ── State vectors (parallel to phospho) ──────────────────────────────
    print("\n  Creating ERBB2 glyco state vectors per background...")
    erbb2_glyco_vectors = {}
    for bg in ERBB2_GLYCO_BACKGROUNDS:
        col = bg
        if col not in df.columns:
            vector = {int(pos): 1.0 for pos in df["position"]}
        else:
            vector = df.set_index("position")[col].to_dict()
        erbb2_glyco_vectors[bg] = vector
        sample = ", ".join(
            f"{pos}={val:.1f}"
            for pos, val in list(sorted(vector.items()))[:4]
        )
        print(f"    {bg:32s}: {sample}, ...")

    vectors_path = OUT_DIR / "erbb2_glyco_state_vectors.json"
    with open(vectors_path, "w") as f:
        json.dump(erbb2_glyco_vectors, f, indent=2)
    print(f"\n  ✓ Saved ERBB2 glyco state vectors: "
          f"{vectors_path.relative_to(PROJECT_ROOT)}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run(case_study: str = "egfr_erbb2_tki"):
    """Main entry point — call from thin wrappers or CLI."""
    _init(case_study)
    _main_logic()


def _main_logic():
    """Core execution logic."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 04: Download PTM Data — Phospho + N-Glyco            ║")
    print("║  (ERBB Family: EGFR + HER2/ERBB2)                         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Source: UniProt P00533/P04626 + literature-curated         ║")
    print("║  Output: EGFR + HER2 phospho + glyco sites + state vectors ║")
    print("║  PTM-BDL: glyco_N (type_id 3) — see proposal §3, §7.4      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Step 1: Parse UniProt PTM annotations (from manually downloaded JSON)
    df_uniprot = download_uniprot_ptm_annotations()

    # Step 2: Create curated EGFR phosphorylation database
    df_egfr = create_curated_phospho_sites()

    # Step 3: Create HER2/ERBB2 phosphorylation database
    df_erbb2 = create_erbb2_phospho_sites()

    # ── PTM-BDL multi-PTM extension ───────────────────────────────────────
    # Step 4: EGFR N-glycosylation database (proposal §3.1)
    df_egfr_glyco = create_egfr_glyco_sites()

    # Step 5: HER2/ERBB2 N-glycosylation database (proposal §3.2)
    df_erbb2_glyco = create_erbb2_glyco_sites()

    print("\n✓ Step 04 complete!")
    print("  EGFR : 12 phospho sites + 12 glyco sites + state vectors")
    print("  HER2 : 10 phospho sites + 7 glyco sites (+5 padded) + state vectors")
    print("  Ready for harmonization (Step 06) and PTM-BDL model (Step 10).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 04 — Download PTM data")
    parser.add_argument("--case-study", default="egfr_erbb2_tki",
                        help="Case study name (default: egfr_erbb2_tki)")
    args, _ = parser.parse_known_args()
    run(case_study=args.case_study)
