"""
K562 / CML (BCR-ABL) Case Study — Hematological cancer with diverse drug mechanisms.

Biological context:
  - K562: chronic myeloid leukemia (CML), BCR-ABL fusion (Philadelphia chromosome)
  - 5 drugs spanning 3 different mechanisms:
      • Dasatinib, Imatinib (BCR-ABL TKIs — different kinase from EGFR/HER2)
      • Cytarabine (nucleoside analog), Paclitaxel (taxane), Methotrexate (antifolate)
  - PTM type: phosphorylation (S/T/Y)
  - DrugPTM-Bench: 1,608,421 rows, 14 dose points, 6,751 unique genes

What this case study proves:
  1. Different cancer type: CML (hematological) ≠ NSCLC/breast (epithelial)
  2. Different kinase target: BCR-ABL ≠ EGFR/HER2
  3. Non-TKI drugs: Cytarabine, Paclitaxel, Methotrexate are chemotherapy
  4. Same tool code with different config
"""
