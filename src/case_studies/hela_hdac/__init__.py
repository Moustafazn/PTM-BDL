"""
HeLa / HDAC Inhibitor Case Study — Epigenetic drug response with acetylation.

Biological context:
  - HeLa (cervical carcinoma, HPV18+): widely used epigenetics model
  - 6 drugs: Vorinostat/SAHA (pan-HDAC), Romidepsin (HDAC class I),
             CUDC-101 (HDAC/EGFR/HER2 triple), A485 (p300/CBP HAT inhibitor),
             A486 (inactive control), Curcumin (natural HDAC inhibitor)
  - 2 PTM types: phosphorylation (3 subtypes: S/T/Y) + acetylation (1 subtype: K)
    → acetylation is a NEW PTM type not seen in Case Study 1
  - DrugPTM-Bench: 980,608 rows (921K phospho + 59K acetylation)
  - 15 dose points, 6,892 unique genes

Key biological rationale:
  - HDAC inhibitors block deacetylation → acetylation INCREASES at histone marks
  - This is fundamentally different from TKIs (which block phosphorylation)
  - Top acetylation targets: EP300, CREBBP, HIST1H4A, NCL, histones
  - Proves PTM-BDL handles a NEW PTM type (acetyl_K) with ZERO framework changes

What this case study proves for generalization:
  1. NEW PTM type works: acetylation tokens attend to phospho tokens → cross-type
     crosstalk is NOT TKI-specific
  2. Different drug mechanism: HDAC inhibitors ≠ kinase inhibitors
  3. Different cancer type: cervical carcinoma ≠ NSCLC/breast
  4. Config-only extension: adding acetyl_K requires ZERO changes to src/ptm_bdl/
"""
