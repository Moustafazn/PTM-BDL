"""
EGFR/ERBB2 TKI Resistance Case Study — First application instance of PTM-BDL.

Biological context:
  - EGFR-mutant NSCLC: L858R, exon19del, T790M gatekeeper → sequential TKI generations
  - HER2-positive breast cancer: amplification-driven → HER2-targeted TKIs
  - 951 samples: 646 EGFR (NSCLC) + 305 ERBB2 (breast cancer)
  - 6 drugs: Gefitinib, Erlotinib (1st-gen), Afatinib (2nd-gen pan-ERBB),
             Osimertinib (3rd-gen T790M-selective), Lapatinib, Sapitinib (HER2)
  - 2 PTM types: phosphorylation (3 subtypes: Y/S/T) + N-glycosylation (1 subtype: N)
  - 24 PTM tokens per sample (12 phospho + 12 glyco)

Key biological findings:
  - Y1068 (EGFR) ≡ Y1221 (ERBB2): GRB2 docking site homology across receptors
  - N528 (EGFR) ↔ N530 (ERBB2): extracellular DIV glyco anchor homology
  - Tissue-specific pathway discovery: EGFR→MAPK vs HER2→PI3K-AKT dominance
  - Cross-type phospho↔glyco attention reveals intracellular-extracellular crosstalk
"""
