"""
EGFR/ERBB2 TKI Resistance — Application-specific biological knowledge.

ALL application-specific labels, homology pairs, mutation groups, and drug
comparisons live here. The tool packages (ptm_bdl.*) are protein-agnostic.

Biological context:
  EGFR (P00533) — receptor tyrosine kinase, NSCLC driver
    • 12 phosphosites: Y869(Y845), S991, Y998, Y1016(Y992), S1039, T1041,
      Y1069(Y1045), Y1092(Y1068), Y1110(Y1086), Y1125(Y1101), Y1172(Y1148), Y1197(Y1173)
    • 12 N-glycosites: N56, N128, N175, N196, N352, N361, N413, N444, N528, N568, N603, N623
    • Key mutations: L858R (activating), exon19del, T790M (gatekeeper), C797S (resistance)

  ERBB2/HER2 (P04626) — receptor tyrosine kinase, breast cancer driver
    • 10 phosphosites + 2 pad: T686, Y1005, S1054, T1099, Y1139, S1151,
      Y1196, Y1221(≡Y1068), Y1222, Y1248(≡Y1173)
    • 7 N-glycosites + 5 pad: N68, N124, N187, N259, N530(↔N528), N571, N629
    • Oncogenicity: HER2 amplification-driven (Hudis NEJM 2007, Citri & Yarden NRMCB 2006)

Cross-receptor homology:
  Phospho: Y1092(EGFR) ≡ Y1221(ERBB2) — both GRB2 docking → RAS-MAPK
           Y1197(EGFR) ≡ Y1248(ERBB2) — both SHC1 docking → PI3K-AKT
  Glyco:   N528(EGFR) ↔ N530(ERBB2) — extracellular domain IV anchor
           (ERBB2 site overlaps trastuzumab binding interface)
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# Site Labels (per-protein, per-mod-type)
# ══════════════════════════════════════════════════════════════════════════════
# EGFR: UniProt P00533 (precursor numbering; mature/classic name in parens)
# ERBB2: UniProt P04626
# Pad slots are zero-padded to match ptm_dim=12 / glyco_dim=12

PHOSPHO_LABELS_EGFR = [
    "Y869(Y845)", "S991", "Y998", "Y1016(Y992)",
    "S1039", "T1041", "Y1069(Y1045)", "Y1092(Y1068)",
    "Y1110(Y1086)", "Y1125(Y1101)", "Y1172(Y1148)", "Y1197(Y1173)",
]
PHOSPHO_LABELS_ERBB2 = [
    "T686", "Y1005", "S1054", "T1099",
    "Y1139", "S1151", "Y1196", "Y1221(≡Y1068)",
    "Y1222", "Y1248(≡Y1173)", "pad_11", "pad_12",
]
GLYCO_LABELS_EGFR = [
    "N56", "N128", "N175", "N196", "N352", "N361",
    "N413", "N444", "N528(↔HER2-N530)", "N568", "N603", "N623",
]
GLYCO_LABELS_ERBB2 = [
    "N68", "N124", "N187", "N259",
    "N530(↔EGFR-N528)", "N571", "N629",
    "gpad_07", "gpad_08", "gpad_09", "gpad_10", "gpad_11",
]

# ══════════════════════════════════════════════════════════════════════════════
# Per-slot phospho subtype maps
# ══════════════════════════════════════════════════════════════════════════════
# 0 = Y (tyrosine), 1 = S (serine), 2 = T (threonine)
# Must match the registry subtype IDs: phospho_Y=0, phospho_S=1, phospho_T=2

PHOSPHO_TYPE_EGFR = [0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
PHOSPHO_TYPE_ERBB2 = [2, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0]

# ══════════════════════════════════════════════════════════════════════════════
# Real/pad masks
# ══════════════════════════════════════════════════════════════════════════════
# ERBB2 has 10 real phospho slots + 2 pad, 7 real glyco slots + 5 pad

PHOSPHO_REAL_EGFR = [True] * 12
PHOSPHO_REAL_ERBB2 = [True] * 10 + [False, False]
GLYCO_REAL_EGFR = [True] * 12
GLYCO_REAL_ERBB2 = [True] * 7 + [False] * 5

# ══════════════════════════════════════════════════════════════════════════════
# Cross-receptor homology slot indices
# ══════════════════════════════════════════════════════════════════════════════
# Phospho-Y (slot 7): EGFR Y1092(Y1068) ≡ ERBB2 Y1221 — primary GRB2 docking site
# Glyco-N: EGFR N528 at slot 8; ERBB2 N530 at slot 4

PHOSPHO_Y_HOMOLOGY_SLOT = 7
GLYCO_HOMOLOGY_SLOT_EGFR = 8
GLYCO_HOMOLOGY_SLOT_ERBB2 = 4
GRB2_PHOSPHO_INDEX = 7
EGFR_N528_INDEX = 8
ERBB2_N530_INDEX = 4

# ══════════════════════════════════════════════════════════════════════════════
# Valid effector slots for biological validation
# ══════════════════════════════════════════════════════════════════════════════
# The model may prioritize DIFFERENT top sites for different proteins because
# the dominant resistance pathway differs by tissue context:
#   EGFR (NSCLC): RAS-MAPK dominant → Y1068/GRB2 (slot 7) expected #1
#   HER2 (breast): PI3K-AKT dominant → Y1248/SHC1 (slot 9) also valid #1
# Refs: Arteaga & Engelman 2014 (Cancer Cell), Razavi et al. 2020 (Nat Cancer),
#       Scaltriti et al. 2011 (PNAS), Citri & Yarden 2006 (Nat Rev Mol Cell Biol)

EGFR_VALID_TOP_EFFECTOR_SLOTS = {
    7: {"site": "Y1092(Y1068)", "pathway": "GRB2 → RAS-MAPK", "context": "NSCLC primary"},
    11: {"site": "Y1197(Y1173)", "pathway": "SHC1 → PI3K-AKT", "context": "survival signaling"},
}
ERBB2_VALID_TOP_EFFECTOR_SLOTS = {
    7: {"site": "Y1221", "pathway": "GRB2 → RAS-MAPK", "context": "pan-ERBB"},
    9: {"site": "Y1248", "pathway": "SHC1 → PI3K-AKT", "context": "breast cancer resistance"},
    1: {"site": "Y1005", "pathway": "c-Cbl → degradation", "context": "receptor turnover"},
}

# ══════════════════════════════════════════════════════════════════════════════
# Drug classification
# ══════════════════════════════════════════════════════════════════════════════
# Cross-protein drugs target BOTH EGFR and ERBB2 in GDSC2
# HER2-only drugs target only ERBB2

CROSS_PROTEIN_DRUGS = ["Afatinib", "Erlotinib", "Gefitinib", "Osimertinib"]
HER2_ONLY_DRUGS = ["Lapatinib", "Sapitinib"]

# Drug generation classification for biological analysis:
#   1st-gen (reversible): Gefitinib, Erlotinib
#   2nd-gen (irreversible, pan-ERBB): Afatinib
#   3rd-gen (covalent, T790M-selective): Osimertinib
#   HER2-targeted: Lapatinib (dual EGFR/HER2), Sapitinib (pan-ERBB)
DRUG_GENERATIONS = {
    "Gefitinib": "1st-gen reversible",
    "Erlotinib": "1st-gen reversible",
    "Afatinib": "2nd-gen irreversible pan-ERBB",
    "Osimertinib": "3rd-gen covalent T790M-selective",
    "Lapatinib": "1st-gen reversible dual EGFR/HER2",
    "Sapitinib": "2nd-gen reversible pan-ERBB",
}
