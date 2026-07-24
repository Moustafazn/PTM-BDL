"""
K562 / CML (BCR-ABL) — Application-specific biological knowledge.

ALL application-specific labels, PTM sites, drug classifications, and
biological validation targets live here. The framework packages (ptm_bdl.*)
are protein-agnostic.

Biological context:
  K562 — chronic myeloid leukemia (CML), BCR-ABL fusion (Ph+ chromosome)
    • BCR-ABL fusion kinase: constitutively active tyrosine kinase
    • Drives proliferation via RAS-MAPK, PI3K-AKT, STAT5 pathways
    • 5 drugs in DrugPTM-Bench: Dasatinib, Imatinib, Cytarabine, Paclitaxel,
      Methotrexate — spanning TKI and non-TKI mechanisms
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# Drug Classification
# ══════════════════════════════════════════════════════════════════════════════

DRUGS = {
    "Dasatinib": {
        "mechanism": "BCR-ABL/SRC multi-kinase TKI (2nd-gen)",
        "targets": ["ABL1", "SRC", "LCK", "YES1", "FYN"],
        "gdsc_id": 1066,
        "class": "Multi-kinase TKI",
        "rows_in_drugptm": 1_081_759,
    },
    "Imatinib": {
        "mechanism": "BCR-ABL TKI (1st-gen, Gleevec)",
        "targets": ["ABL1", "KIT", "PDGFRA"],
        "gdsc_id": 1003,
        "class": "Kinase inhibitor",
        "rows_in_drugptm": 241_582,
    },
    "Cytarabine": {
        "mechanism": "Nucleoside analog (Ara-C) — DNA synthesis inhibitor",
        "targets": ["DNA_polymerase"],
        "gdsc_id": 1006,
        "class": "Chemotherapy (antimetabolite)",
        "rows_in_drugptm": 98_860,
    },
    "Paclitaxel": {
        "mechanism": "Microtubule stabilizer (taxane) — mitotic arrest",
        "targets": ["TUBB", "microtubules"],
        "gdsc_id": 1080,
        "class": "Chemotherapy (antimicrotubule)",
        "rows_in_drugptm": 98_590,
    },
    "Methotrexat": {
        "mechanism": "Antifolate — DHFR inhibitor, blocks DNA synthesis",
        "targets": ["DHFR"],
        "gdsc_id": 1007,
        "class": "Chemotherapy (antifolate)",
        "rows_in_drugptm": 87_630,
        "note": "Spelled 'Methotrexat' in DrugPTM-Bench (without final 'e')",
    },
}

TKI_DRUGS = ["Dasatinib", "Imatinib"]
CHEMO_DRUGS = ["Cytarabine", "Paclitaxel", "Methotrexat"]

# ══════════════════════════════════════════════════════════════════════════════
# Biological Validation Targets
# ══════════════════════════════════════════════════════════════════════════════

VALIDATION_TARGETS = {
    "imatinib_dephosphorylates_abl_substrates": {
        "description": "Imatinib should reduce phospho of BCR-ABL substrates",
        "expected": "Negative delta for CRKL-Y207, STAT5-Y694 under Imatinib",
    },
    "dasatinib_stronger_than_imatinib": {
        "description": "Dasatinib (325x more potent) should show larger deltas",
        "expected": "Larger |delta| for Dasatinib vs Imatinib at same targets",
    },
    "chemo_different_phospho_pattern": {
        "description": "Non-TKI drugs should show DIFFERENT phospho patterns",
        "expected": "Cytarabine/Paclitaxel affect DNA damage and mitotic sites",
    },
    "tki_vs_chemo_discrimination": {
        "description": "Model should learn TKI and chemo drugs work differently",
        "expected": "Attention patterns differ between Imatinib vs Cytarabine",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PDB Structures
# ══════════════════════════════════════════════════════════════════════════════

ABL_PDB_STRUCTURES = [
    {"id": "1IEP", "description": "ABL kinase + Imatinib", "drug": "Imatinib"},
    {"id": "2GQG", "description": "ABL kinase + Dasatinib", "drug": "Dasatinib"},
    {"id": "2HYY", "description": "ABL kinase (apo, DFG-in)", "drug": None},
]

CRKL_PDB_STRUCTURES = [
    {"id": "2EYZ", "description": "CRKL SH2+SH3 domains (Birge 2009)", "drug": None},
]

STAT5A_PDB_STRUCTURES = [
    {"id": "1Y1U", "description": "STAT5A core domain (Neculai 2005)", "drug": None},
]

# Combined list for step08 feature extraction
ALL_PDB_STRUCTURES = ABL_PDB_STRUCTURES + CRKL_PDB_STRUCTURES + STAT5A_PDB_STRUCTURES
