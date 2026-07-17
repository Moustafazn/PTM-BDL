# Expansion Feasibility Analysis: Beyond EGFR/NSCLC
## Toward a Cross-Kinase PTM-Driven Drug Resistance Framework

**Date:** 2026-06-19  
**Status:** Final Decision Document  
**Decision:** Option B+ (EGFR + HER2, with priority on deepening EGFR phosphoproteomics)

---

## Executive Summary

**Question:** Can we expand beyond 4 EGFR TKIs in NSCLC?

**Answer:** Yes — but the right expansion is **surgical, not broad**. The critical insight is:

```
Sample count ≠ Biological evidence
Propagation ≠ Measurement
Scale ≠ Depth
```

The current limitation is NOT 646 samples. It is that only **~8 unique (cell_line, drug) combinations** have REAL phosphoproteomic measurements — everything else is propagated via mutation-class biological priors. Adding more cancer types doesn't automatically fix this.

**The best plan has three priorities:**
1. **Deepen EGFR phosphoproteomics** — find +2 published papers (highest scientific value per hour)
2. **Add HER2 (ERBB2)** — same receptor family, biologically coherent, helps class balance
3. **Defer BCR-ABL** — biological distance is too large, single cell line, save for future paper

---

## 1. Current State: The Real Numbers

### Dataset Summary
| Metric | Current Value |
|--------|--------------|
| Total samples | 646 |
| Cell lines | 163 (NSCLC only) |
| Drugs | 4 (Gefitinib, Afatinib, Erlotinib, Osimertinib) |
| Target gene | EGFR only |
| Cancer type | NSCLC only |
| Drug-PTM phospho rows | 2,109 |
| Unique PTM sites | 960 |
| Cell lines with measured phospho | 6 (A431, H1975, HCC4006, HCC827, PC-9, PC9GR) |
| Phospho data sources | 8 published studies |
| Class balance | 48 sensitive / 598 resistant (7.4% / 92.6%) — **severely imbalanced** |

### The Core Problem: Measured vs Propagated

| Category | Samples | % of Dataset | What It Means |
|----------|---------|-------------|---------------|
| **Directly measured phospho** | ~16 | 2.5% | Real experimental data |
| **Cross-drug propagated** | ~12 | 1.9% | Same cell line, different drug |
| **Mutation-class propagated** | ~618 | 95.6% | Biological inference |
| **Total** | 646 | 100% | |

**95.6% of PTM features are biologically inferred, not measured.** This is the number reviewers will challenge.

### The Novelty Bottleneck: Step 05 Drug-PTM Data

Current drug-PTM coverage (what's actually measured):

| Cell Line | Drug | Source | Rows | Context |
|-----------|------|--------|------|---------|
| A431 (WT EGFR) | Gefitinib | DrugPTM-Bench | ~33 | dose_response |
| A431 (WT EGFR) | Afatinib | DrugPTM-Bench | ~34 | dose_response |
| H1975 (L858R/T790M) | Osimertinib | PNAS 2025, CancerRes 2021, MCP 2025 | ~556 | tki_phosphoproteome, parental_vs_resistant |
| H1975 (L858R/T790M) | Rociletinib | CancerRes 2021 | ~50 | parental_vs_resistant |
| HCC4006 (Exon19del) | Osimertinib | PNAS 2025 | ~444 | tki_phosphoproteome |
| HCC827 (Exon19del) | Osimertinib | Tozuka 2024 | ~15 | parental_vs_resistant |
| PC-9 (Exon19del) | Osimertinib | Tozuka 2024, Hsu 2025, MCP 2025 | ~36 | parental_vs_resistant, temporal |
| PC9GR (Exon19del) | Osimertinib | Remsing Rix 2022 | ~780 | cell_line_phosphoproteome |
| LUAD tumors | none | FEBS 2025 | ~211 | tumor_phospho_signatures |

Only **8 unique (cell_line, drug) combinations** have REAL phosphoproteomic measurements.

---

## 2. Priority 1: Deepen EGFR Phosphoproteomics (HIGHEST IMPACT)

### Why This Matters Most

| Action | Measured Conditions Added | Propagated Samples Added | Scientific Value |
|--------|--------------------------|--------------------------|-----------------|
| **Find 2 more EGFR phospho papers** | **+4-6** | 0 | ★★★★★ |
| Add HER2 expansion | +2-4 | +200-600 | ★★★★ |
| Add BCR-ABL expansion | +2 | +150-240 | ★★★ |
| Add NSCLC pathway drugs | 0 | +800-1000 | ★★ |

**+2 papers with real phosphoproteomics > +500 propagated samples**

### What To Search For

| Target | Why | Databases to Search |
|--------|-----|---------------------|
| Osimertinib-resistant derivatives of H1975 | Direct resistance phospho, our exact cell line | PubMed, PRIDE, ProteomeXchange |
| PC-9 osimertinib long-term adaptation | Temporal phospho dynamics beyond Hsu 2025 | PubMed, PRIDE |
| H3255 (L858R) + EGFR TKI phospho | L858R is underrepresented in our phospho data | PubMed, MassIVE |
| EGFR-mutant PDX + phosphoproteomics | In vivo context that validates cell-line findings | CPTAC, PubMed |
| Any 2024-2026 EGFR TKI phosphoproteomics | Recent publications not yet captured | PubMed, bioRxiv, ProteomeXchange |
| NSCLC clinical sample phosphoproteomics | Patient-derived validation | CPTAC LUAD cohort |

### Search Queries

```
PubMed:
  "EGFR" AND "phosphoproteomics" AND ("osimertinib" OR "resistance") AND 2023:2026[dp]
  "NSCLC" AND "phosphoproteome" AND "TKI" AND "resistant"
  "EGFR" AND "drug-tolerant persister" AND "phospho"

ProteomeXchange/PRIDE:
  EGFR NSCLC phospho osimertinib
  EGFR kinase inhibitor phosphoproteome

CPTAC:
  LUAD phosphoproteomics
```

### What Each New Paper Would Add

Each phosphoproteomics paper typically provides:
- 1-3 new (cell_line, drug) conditions
- 5-20 EGFR phosphosite measurements per condition
- 100-500 signaling network phosphosites (for pathway validation)
- A new resistance context (temporal, dose-response, parental-vs-resistant)

Finding even **2 papers** would increase our measured conditions from 8 to 12-14, which is a **50-75% increase in real biological evidence**.

---

## 3. Priority 2: HER2 (ERBB2) Expansion — The Natural Extension

### Why HER2 Is The Right Expansion

EGFR (ERBB1) and HER2 (ERBB2) are **paralogs** — they share:
- 81.3% kinase domain sequence identity (218/268 residues, computed from UniProt P00533 aa 712–979 vs P04626 aa 720–987)
- Same alphaC-helix activation mechanism  
- Same C-terminal phosphorylation sites (homologous Y positions)
- Same downstream pathways (MAPK, PI3K-AKT, SRC)
- Same resistance mechanisms (bypass pathway activation, secondary mutations)
- ERBB1/ERBB2 heterodimerization is core ERBB biology

**References:**
- Citri & Yarden, *Nat Rev Mol Cell Biol* 2006: "ERBB receptors form heterodimers and share signaling cascades"
- Arteaga & Engelman, *Cancer Cell* 2014: "ERBB receptors: from oncogene discovery to basic science to mechanism-based cancer therapeutics"

### HER2 Solves the Class Imbalance Problem

The current dataset has a severe imbalance: 48 sensitive / 598 resistant (7.4%).

HER2+ breast cancer lines are **genuinely sensitive** to HER2-targeted therapy:
- BT-474 + Lapatinib: IC50 ~30 nM (very sensitive)
- SKBR3 + Lapatinib: IC50 ~50 nM (very sensitive)
- AU565 + Lapatinib: IC50 ~100 nM (sensitive)

Adding HER2+ breast cancer data naturally adds sensitive samples, improving balance:
```
Current:  48 sensitive / 598 resistant = 7.4% sensitive
With HER2: ~100-150 sensitive / ~650-700 resistant = ~15-18% sensitive
```

This is a structural improvement, not just more data.

### HER2 Data Already Available

| Component | Status | What Exists |
|-----------|--------|-------------|
| Drug-PTM phospho | Already downloaded | BT-474: 510 ERBB2 rows, 17 unique sites (Trastuzumab, Pertuzumab) |
| Drug-PTM phospho | Already downloaded | MDA-MB-175: Lapatinib, Trastuzumab (ERBB2 pathway) |
| GDSC IC50 | Available in GDSC2 | Lapatinib (~200+ cell lines), Neratinib available |
| Mutations (CCLE) | In DepMap | ERBB2 mutations across breast cancer lines |
| PDB structures | Rich | 3PP0 (HER2 kinase), 1N8Z (HER2+Trastuzumab), 3RCD (HER2+Lapatinib) |
| PTM sites | Well-characterized | Y1221, Y1222, Y1248 (analogous to EGFR Y1068, Y1086, Y1173) |
| Sequence | UniProt P04626 | Ready for ESM-2 |

### HER2 Drug-PTM Quality Assessment

| Metric | Value | vs EGFR Benchmark |
|--------|-------|-------------------|
| Cell lines with phospho | 2 (BT-474, MDA-MB-175) | EGFR: 6 — fewer but acceptable |
| Drugs | 3 (Trastuzumab, Pertuzumab, Lapatinib) | EGFR: 4 — comparable |
| ERBB2 phospho rows | 510+ | EGFR: 2,109 — fewer but substantial |
| ERBB2 unique sites | 17 | EGFR: 960 — fewer (expected: HER2 is less studied) |
| Resistance context | dose_response only | EGFR: 5 contexts — need published HER2 resistance phospho |

### Gap: HER2 Resistance Phosphoproteomics

The main gap for HER2 is lack of **parental vs resistant** phosphoproteomics (we only have dose-response from DrugPTM-Bench). Published HER2 resistance phospho studies to search for:

| Candidate Paper | What It Would Add |
|-----------------|-------------------|
| Rexer et al., *Cancer Res* 2011 | Lapatinib-resistant BT-474 phosphoproteomics |
| Chandarlapaty et al., *Cancer Cell* 2012 | PI3K reactivation in Trastuzumab-resistant HER2+ cells |
| Stuhlmiller et al., *Cell Rep* 2015 | Kinome reprogramming in Lapatinib-resistant cells |
| Any 2023-2026 HER2-TKI resistance phospho | Recent resistant cell line derivatives |

### Estimated Impact of HER2 Expansion

| Metric | Before | After HER2 |
|--------|--------|------------|
| Total samples | 646 | ~900-1,200 |
| Drugs | 4 | 6-7 (+ Lapatinib, Neratinib, possibly Trastuzumab) |
| Cancer types | 1 (NSCLC) | 2 (NSCLC + Breast) |
| Target genes | 1 (EGFR) | 2 (EGFR + ERBB2) |
| Sensitive samples | 48 (7.4%) | ~150 (~15%) |
| Measured phospho conditions | ~8 | ~10-14 |
| Drug-PTM rows | 2,109 | ~2,600-3,100 |

### Pipeline Changes Required for HER2

| Step | Change Needed | Effort |
|------|--------------|--------|
| step01 | Add breast cancer tissue filter + HER2 drugs (Lapatinib, Neratinib) | Small |
| step02 | Add ERBB2 gene to mutation extraction | Small |
| step03 | Add ~3-4 HER2 PDB structures | Small |
| step04 | Add HER2 phospho sites (Y1221, Y1222, Y1248, etc.) | Small |
| step05 | Extract ERBB2 from BT-474 and MDA-MB-175 | Medium |
| step06 | Multi-gene harmonization (gene_id column, gene-specific PTM vectors) | Medium |
| step07 | Add HER2 sequences to ESM-2 extraction | Small |
| step08 | Add HER2 structures to GearNet extraction | Small |
| step09 | Add new drug SMILES (Lapatinib, Neratinib) | Small |
| step10 | Update model for variable PTM vector length + gene identity | Medium |

**Total effort: ~2-3 weeks**

---

## 4. What NOT To Do (Deferred)

### BCR-ABL Expansion — Save for Future Paper

**Reason for deferral:**

EGFR and ABL1 are biologically much further apart than EGFR and HER2:

1. **Different tissues** — NSCLC (epithelial) vs CML (hematopoietic)
2. **Different signaling networks** — EGFR->MAPK/PI3K vs ABL->STAT5/CrkL
3. **Single cell line** — K562 is THE CML gold standard, but having only 1 cell line with phospho is weaker than EGFR (6 lines)
4. **Model confound risk** — Model may learn `cancer_type` instead of `universal PTM biology`
5. **Story complexity** — ERBB1+ERBB2 is a clean family story; adding ABL dilutes the narrative

**Data preserved for future use:**
- K562: 4,324 ABL1/BCR phospho rows, 20+ unique sites (Imatinib, Dasatinib)
- GDSC has Imatinib (1003), Dasatinib (1079), Nilotinib (1013)
- PDB: 1IEP (Imatinib), 2GQG (Dasatinib), 3OXZ (Ponatinib)

**Future Direction language for the paper:**
> "The PTM-driven resistance framework generalizes to the ERBB receptor family (demonstrated with EGFR and HER2). The same mechanistic chain — genotype -> structure -> PTM state -> drug response — applies to other oncogenic kinases. BCR-ABL in CML represents a natural extension, with published phosphoproteomic data under Imatinib/Dasatinib treatment available in the DrugPTM-Bench dataset (4,324 ABL1 phosphosite measurements in K562 cells). We leave cross-family kinase generalization as future work."

### Pathway Drug Expansion — Also Deferred

Adding PI3K/MEK/mTOR inhibitors from A549:
- These are downstream pathway drugs, not RTK inhibitors
- Different mechanism of action complicates the PTM->resistance story
- 0 new measured phospho conditions (just propagated)
- Save for Phase 2 / second paper

### "Foundation Model" Framing — Do Not Use

With 900-1,200 samples across 2 kinases, this is NOT a foundation model. Foundation models in this space have 100K-500K samples.

**Instead use:**
> "Cross-Kinase PTM-Driven Drug Resistance Framework"
> or
> "Generalizable PTM-Aware Resistance Prediction Across the ERBB Receptor Family"

---

## 5. DrugPTM-Bench Data Inventory (Raw Data Already Downloaded)

For reference — what sits in `data/raw/drugptm/30394195/`:

| Cell Line | Cancer | Target | Drugs | Raw Rows | Status |
|-----------|--------|--------|-------|----------|--------|
| **A431** | Epidermoid (WT EGFR) | EGFR | Afatinib, Gefitinib, Dasatinib, Imatinib | 3.5M | Currently used (EGFR only) |
| **A549** | NSCLC (KRAS-mut) | Pathway drugs | AZD8055, Dactolisib, Dasatinib, MK2206, Nintedanib, PD325901, Pictilisib, Refametinib, Staurosporin, Tideglusib | 3.0M | Deferred |
| **BT-474** | HER2+ Breast | ERBB2 | Pertuzumab, Trastuzumab | 270K | **Priority 2: Extract ERBB2** |
| **K562** | CML | BCR-ABL | Cytarabine, Dasatinib, Imatinib, Methotrexate, Paclitaxel | 1.6M | Deferred (future paper) |
| **MDA-MB-175** | Breast | ERBB2 pathway | Lapatinib, Trastuzumab | 224K | **Priority 2: Extract ERBB2** |
| **HeLa** | Cervical | HDAC/epigenetic | A485, A486, CUDC101, Romidepsin, Vorinostat | 981K | Not applicable (different mechanism) |
| **RPMI8226** | Myeloma | Proteasome | Bortezomib, Carfilzomib | 1.1M | Not applicable (different mechanism) |

### Key numbers from DrugPTM-Bench for HER2:
- **BT-474:** 510 ERBB2 phospho rows, 17 unique ERBB2 sites (S, T, Y residues)
- **MDA-MB-175:** ERBB2 pathway coverage under Lapatinib treatment
- **25 SMILES** available in `all_SMILES.csv` for all drugs

---

## 6. Available Structural Data for HER2 Expansion

| PDB ID | Description | Mutations | Drug | Resolution |
|--------|-------------|-----------|------|------------|
| 3PP0 | HER2 kinase domain (active) | WT | Apo | 2.25 A |
| 3RCD | HER2 kinase + Lapatinib | WT | Lapatinib | 2.40 A |
| 1N8Z | HER2 ECD + Trastuzumab | WT | Trastuzumab (Fab) | 2.50 A |
| 3BE1 | HER2 kinase domain | WT | SYR127063 | 2.05 A |
| 7JXH | HER2 + Neratinib | WT | Neratinib | 2.45 A |

For mutation-driven assignment (same philosophy as EGFR):
- WT HER2 -> 3PP0 (apo, active conformation)
- HER2 + Lapatinib context -> 3RCD (reference only)

---

## 7. GDSC Data for HER2 Expansion

GDSC2 contains breast cancer cell lines tested with:

| Drug | GDSC ID | Target | Expected Breast Lines |
|------|---------|--------|-----------------------|
| **Lapatinib** | 1558 | EGFR/HER2 dual TKI | ~40-60 breast lines |
| **Neratinib** | 2097 | Pan-ERBB irreversible | ~40-60 breast lines |
| **Afatinib** | 1032 | Pan-ERBB (already in dataset) | Can extend to breast |

Note: **Trastuzumab is NOT in GDSC** (antibody, not small molecule). Options:
1. Use Lapatinib/Neratinib only (small molecules with GDSC IC50) — recommended
2. Add literature IC50 for Trastuzumab (same approach as PC-9/HCC827)
3. Exclude Trastuzumab initially

---

## 8. Model Architecture Changes for ERBB Family Extension

### What Changes

| Component | Current (EGFR-only) | ERBB Family Model |
|-----------|--------------------|--------------------|
| `target_gene` | Implicit (always EGFR) | Explicit input: "EGFR" or "ERBB2" |
| PTM sites | Fixed 12 EGFR positions | Gene-specific: EGFR (12) + ERBB2 (~10) |
| `PTMFeatureModulator` | 12-dim fixed vector | Max(12, 10) = 12-dim with masking |
| Sequence | EGFR variants only | EGFR + ERBB2 variants (ESM-2 handles natively) |
| Structure | EGFR PDBs only | + HER2 PDBs (GearNet handles natively) |
| Drug | 4 SMILES | + Lapatinib, Neratinib SMILES (ChemBERTa handles natively) |
| Pathway groups | EGFR-specific | Shared (MAPK, PI3K work for both ERBB1 and ERBB2) |

### Key Insight: Minimal Architecture Change

ESM-2, GearNet, and ChemBERTa are **already foundation models** — they generalize across genes/drugs natively. The only EGFR-specific components are:
1. PTM site positions -> add ERBB2 sites (trivial)
2. Pathway classification -> already shared (MAPK, PI3K-AKT, SRC)
3. Mutation-class propagation -> build ERBB2 mutation classes (same logic)

---

## 9. Revised Project Framing

### Paper Title Options

**If EGFR-only (current):**
> "Integrative Multimodal Learning Reveals PTM-Dependent Mechanisms of Osimertinib Resistance in EGFR-Mutant Lung Cancer"

**If EGFR + HER2 (Option B+):**
> "Cross-Receptor PTM-Driven Drug Resistance Prediction Across the ERBB Family: From EGFR-Mutant Lung Cancer to HER2-Positive Breast Cancer"

**Novelty claim (strengthened with HER2):**
> "The first multimodal AI system that demonstrates PTM-driven resistance mechanisms are shared across the ERBB receptor family, with a joint model achieving better predictions than single-receptor models for both EGFR TKI resistance in NSCLC and HER2-targeted therapy resistance in breast cancer."

---

## 10. Final Action Plan

### Phase 1: Deepen EGFR (1-2 weeks, do FIRST)
1. Systematic literature search for published EGFR phosphoproteomics 2023-2026
2. Check PRIDE/ProteomeXchange for deposited EGFR TKI phospho datasets
3. For each paper found: download supplementary data, integrate into step05
4. Target: +2 papers = +4-6 new measured (cell_line, drug) conditions

### Phase 2: HER2 Expansion (2-3 weeks, do SECOND)
1. Modify step05 to extract ERBB2 from BT-474 and MDA-MB-175
2. Search for 1-2 published HER2 resistance phosphoproteomics studies
3. Modify step01: add breast cancer tissue filter + Lapatinib/Neratinib
4. Add HER2 sequence (P04626), structures (3PP0, 3RCD), PTM sites
5. Modify step06 for multi-gene harmonization (gene_id column)
6. Update PTMFeatureModulator for gene-specific PTM vectors

### Phase 3: Present to Professor
- Show current EGFR work + any new phospho papers found
- Present HER2 expansion as "implemented" or "ready to implement"
- Let professor decide: include in first paper or save for follow-up

### Deferred to Future Paper
- BCR-ABL expansion (K562, 4,324 ABL1 phospho rows ready)
- NSCLC pathway drugs (A549, 10 drugs in DrugPTM-Bench)
- "Foundation model" framing (wait for 5,000+ samples)

---

## 11. Decision Matrix (Final)

| Option | Samples | Measured Conditions | Class Balance | Biological Coherence | Effort | Recommendation |
|--------|---------|--------------------|--------------|--------------------|--------|----------------|
| **A: EGFR only** | 646 | ~8 | 7.4% sensitive | Excellent | None | Baseline |
| **B+: EGFR + HER2** | ~1,000 | ~12-14 | ~15% sensitive | Excellent | 2-3 wks | **RECOMMENDED** |
| C: EGFR + HER2 + BCR-ABL | ~1,200 | ~14-16 | ~18% sensitive | Moderate | 4-5 wks | Future paper |
| D: + Pathway drugs | ~2,200 | ~14-16 | ~20% sensitive | Low-Moderate | 6-8 wks | Phase 2 |

**Final decision: Option B+ with Priority 1 = deepen EGFR phospho first.**
