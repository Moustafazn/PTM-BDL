# HER2 (ERBB2) Expansion Plan — Step-by-Step Implementation Guide

**Date:** 2026-06-25  
**Status:** Implementation roadmap  
**Decision:** Expand from EGFR-only to EGFR + HER2 (ERBB family)

---

## Table of Contents
1. [Why HER2? Biological Justification](#1-why-her2-biological-justification)
2. [What HER2 Adds to the Paper](#2-what-her2-adds-to-the-paper)
3. [Step-by-Step Implementation Plan](#3-step-by-step-implementation-plan)
4. [Data Sources Already Available](#4-data-sources-already-available)
5. [Model Architecture Considerations](#5-model-architecture-considerations)
6. [Timeline Estimate](#6-timeline-estimate)

---

## 1. Why HER2? Biological Justification

### EGFR and HER2 Are Biological Paralogs

EGFR (ERBB1/HER1) and HER2 (ERBB2) are members of the **same receptor tyrosine kinase family** (ERBB/HER). They share:

| Feature | EGFR (ERBB1) | HER2 (ERBB2) | Shared |
|---------|-------------|-------------|--------|
| **Kinase domain identity** | — | — | **81.3% amino acid identity** (218/268 residues, UniProt P00533 vs P04626 kinase domains) |
| **Activation mechanism** | αC-helix rotation | αC-helix rotation | **Same conformational switch** |
| **C-terminal phosphosites** | Y1068, Y1173, Y1086, Y845 | Y1221, Y1222, Y1248 | **Homologous positions** |
| **Downstream pathways** | RAS-MAPK, PI3K-AKT, SRC | RAS-MAPK, PI3K-AKT, SRC | **Same signaling cascades** |
| **Resistance mechanisms** | Bypass (MET, AXL), secondary mutations | Bypass (MET, IGF1R), truncation | **Convergent escape** |
| **Heterodimerization** | EGFR-HER2, EGFR-HER3 | HER2-EGFR, HER2-HER3 | **Obligate partners** |

### Key References Supporting Biological Coherence

1. **Citri & Yarden, Nat Rev Mol Cell Biol 2006** (PMID: 16829981)
   > "ERBB receptors form heterodimers and share signaling cascades. The four ERBB receptors constitute a layered signaling network."

2. **Arteaga & Engelman, Cancer Cell 2014** (PMID: 24651011)
   > "ERBB receptors: from oncogene discovery to basic science to mechanism-based cancer therapeutics." Shows EGFR and HER2 share resistance mechanisms.

3. **Yarden & Sliwkowski, Nat Rev Mol Cell Biol 2001** (PMID: 11252954)
   > Definitive review showing ERBB family members use the same phosphorylation-dependent signaling architecture.

4. **Graus-Porta et al., EMBO J 1997** (PMID: 9065234)
   > "ErbB-2, the preferred heterodimerization partner of all ErbB receptors, is a mediator of lateral signaling." HER2 is the CENTRAL hub of ERBB signaling.

### Why This Expansion Is NOT a Stretch

- EGFR and HER2 **physically heterodimerize** — they are not independent proteins. An EGFR-mutant tumor's drug response depends on HER2 expression and vice versa.
- The same PTM machinery (kinases, phosphatases, adaptors) controls both receptors — phosphorylation at Y1068 (EGFR) and Y1221 (HER2) both recruit GRB2 → RAS-MAPK.
- Drugs like **Afatinib** and **Neratinib** target BOTH EGFR and HER2 — they are already pan-ERBB inhibitors in our dataset.
- The same **resistance biology** applies: bypass pathway activation, secondary mutations, PTM rewiring.

### What This Means for the Model

The model should learn that:
- Y1068 (EGFR) and Y1221 (HER2) are **functionally equivalent** phosphosites (both recruit GRB2)
- Afatinib resistance in NSCLC and Lapatinib resistance in breast cancer share **convergent PTM signatures**
- A cross-receptor model provides **stronger biological evidence** than a single-receptor model

---

## 2. What HER2 Adds to the Paper

### Quantitative Impact

| Metric | EGFR Only (Current) | EGFR + HER2 (Expanded) |
|--------|---------------------|------------------------|
| **Total samples** | 646 | ~900–1,200 |
| **Drugs** | 4 | 6–7 (+Lapatinib, +Neratinib) |
| **Cancer types** | 1 (NSCLC) | 2 (NSCLC + Breast) |
| **Target genes** | 1 (EGFR) | 2 (EGFR + ERBB2) |
| **Sensitive samples** | 48 (7.4%) | ~100–150 (~15%) |
| **Measured phospho conditions** | ~8 | ~12–14 |
| **Class balance** | Severe (92.6% resistant) | Improved (~85% resistant) |

### Publication Impact

| Without HER2 | With HER2 |
|--------------|-----------|
| "EGFR-only proof of concept" | "Cross-receptor ERBB family framework" |
| PTM can't help (95.6% identical) | PTM has real variance (different receptor biology) |
| Single cancer type | Built-in cross-cancer validation |
| "Model may not generalize" | "Model generalizes across ERBB family" |
| Oxford Bioinformatics (maybe) | Oxford Bioinformatics (strong) or Briefings in Bioinformatics |

### The Key Paper Narrative Upgrade

**Before:**
> "We built a multimodal resistance predictor for EGFR TKIs in NSCLC."

**After:**
> "We demonstrate that PTM-driven resistance mechanisms are shared across the ERBB receptor family, with a joint model achieving cross-receptor predictions for both EGFR-mutant lung cancer and HER2-positive breast cancer. Integrated Gradients analysis reveals that homologous phosphosites (EGFR Y1068 / HER2 Y1221) are independently identified as the most important predictors in both contexts."

---

## 3. Step-by-Step Implementation Plan

### Step 01 — Download GDSC Drug Response Data

**Current:** Filters for NSCLC + 4 EGFR TKIs only.

**Changes needed:**
- [ ] Add breast cancer tissue filter: `"breast|BRCA"` alongside NSCLC
- [ ] Add HER2-targeted drugs to `gdsc.drug_ids` in config.yaml:
  - Lapatinib (GDSC ID: 1558) — EGFR/HER2 dual TKI
  - Neratinib (GDSC ID: 2097) — pan-ERBB irreversible
- [ ] Add `target_gene` column to output CSV:
  - NSCLC samples → `target_gene = "EGFR"`
  - Breast cancer samples → `target_gene = "ERBB2"`
- [ ] Update tissue filter logic to be gene-aware
- [ ] Output: `gdsc_erbb_tki_responses.csv` (replaces `gdsc_nsclc_egfr_tki_responses.csv`)

**Key decision:** Trastuzumab is NOT in GDSC (it's an antibody, not small molecule). Use Lapatinib and Neratinib only.

---

### Step 02 — Download Mutations

**Current:** Extracts EGFR mutations only from CCLE/DepMap.

**Changes needed:**
- [ ] Add ERBB2 gene to mutation extraction (alongside EGFR)
- [ ] Generate HER2 mutant sequences for ESM-2:
  - HER2 wild-type (UniProt P04626)
  - Common HER2 mutations (if any in CCLE — HER2 mutations are rare compared to amplification)
- [ ] Add `target_gene` column to mutation profiles
- [ ] Download HER2 reference FASTA: `https://www.uniprot.org/uniprot/P04626.fasta`
- [ ] Handle HER2 amplification: Most HER2+ breast cancer is driven by **amplification** (not point mutations). The model should distinguish HER2-amplified (overexpression) vs HER2-mutant vs HER2-WT.

**Biological note:** Unlike EGFR where driver mutations (L858R, exon19del) are the primary oncogenic mechanism, HER2 oncogenicity is primarily driven by gene amplification/overexpression. The model needs to handle this difference.

---

### Step 03 — Download PDB Structures

**Current:** 9 EGFR PDB structures.

**Changes needed:**
- [ ] Download HER2 kinase domain structures:
  - `3PP0` — HER2 kinase domain (apo, active conformation, 2.25Å)
  - `3RCD` — HER2 kinase + Lapatinib (2.40Å)
  - `7JXH` — HER2 + Neratinib (2.45Å) — optional reference
- [ ] Add HER2 structure entries to `config.yaml` `pdb.structures`
- [ ] Define mutation-to-PDB mapping for HER2:
  - WT HER2 → `3PP0` (apo, active)
  - HER2 amplified → `3PP0` (same structure, different expression level)
- [ ] Update `structure_catalog.csv` format to include `target_gene`

---

### Step 04 — Download PTM Data

**Current:** 12 EGFR phosphorylation sites from UniProt P00533.

**Changes needed:**
- [ ] Add HER2 phosphorylation sites from UniProt P04626:
  - **Y1221** (equivalent to EGFR Y1068) — GRB2 docking → RAS-MAPK
  - **Y1222** (unique to HER2) — GRB2/SHC dual docking
  - **Y1248** (equivalent to EGFR Y1173) — SHC1 → PI3K-AKT
  - **Y1139** — adaptor recruitment
  - **Y1196** — c-Cbl → receptor degradation
  - **Y1005** — c-Cbl direct binding
  - **S1054** — regulatory (Ser/Thr)
  - **T686** — regulatory (Thr)
  - Additional sites as available from UniProt/PhosphoSitePlus
- [ ] Create `config.yaml` entry for HER2 phospho sites (separate from EGFR)
- [ ] Output: `erbb2_phosphorylation_sites.csv` alongside existing EGFR file
- [ ] Create HER2 PTM state vectors (per mutation/amplification class)

**Key decision: PTM vector length.**
- EGFR: 12 sites → ptm_dim=12
- HER2: ~10 sites → ptm_dim=10
- Options:
  - (A) Pad HER2 to 12 with zeros → keep ptm_dim=12 for all
  - (B) Use max(12, 10) = 12 with masking
  - (C) Gene-specific PTM encoding (more complex)
- **Recommended: Option A** (pad to 12, simplest, backward compatible)

---

### Step 05 — Download Drug-PTM Data ✅ COMPLETED

**Current:** 8 phosphoproteomic datasets for EGFR + 2 new HER2 sources.

#### 5a. Extract ERBB2 data from DrugPTM-Bench ✅ DONE
- [x] `extract_erbb2_data()` + `build_erbb2_summaries()` added to step05
- [x] BT-474: 510 raw ERBB2 rows → **44 dose-response summaries**, 17 unique sites
  - Drugs: Pertuzumab, Trastuzumab (antibodies)
  - ⚠ LIMITATION: BT-474 phospho measured under antibodies, but GDSC IC50 uses Lapatinib/Afatinib (TKIs)
- [x] MDA-MB-175: 10 raw ERBB2 rows → sparse but has Lapatinib + Trastuzumab
- [x] Output: `data/processed/drugptm/drugptm_erbb2_phospho_responses.csv` (44 rows, 17 sites)

#### 5b. Search + Download HER2 Resistance Phosphoproteomics ✅ DONE
- [x] `scripts/step05b_search_her2_papers.py` — searches Europe PMC + PRIDE + Google Scholar
  - Updated with SSL fix (certifi), fixed sort param, added Scholar scraping
  - Queries focus on 2023-2026 cell-line-level phosphoproteomics
- [x] **Ruprecht et al. 2017** (Cancer Research, PMID 28209619) — DOWNLOADED
  - `process_ruprecht_2017()` added to step05
  - 10 ERBB2 phosphosites from BT-474 lapatinib-resistant vs parental (SILAC)
  - Drug: Lapatinib (matches GDSC!)
  - Context: `parental_vs_resistant`
  - Key finding: Y1233 log2FC = -2.94 (strongest dephosphorylation in resistant cells)
- [x] Papers evaluated and rejected (not phosphoproteomics data):
  - Jaehnig 2025: patient-level pathway data, wrong format → deleted
  - Hunt 2026: RPPA data (not MS), no drug context → deleted
  - Zecha 2023: IS the DrugPTM-Bench source paper, redundant → deleted
  - Steggall 2025: ATAC-seq/RNA-seq/proteome, not phosphoproteomics
  - Cui 2025: glycosylation methodology paper, not clean phospho dataset

#### 5c. HER2 Phospho Data Summary ✅
- [x] Output: `drugptm_erbb2_phospho_responses.csv` with `target_gene = "ERBB2"`

| Source | Rows | Cell Line | Drugs | Sites | Context |
|--------|------|-----------|-------|-------|---------|
| DrugPTM-Bench | 44 | BT-474, MDA-MB-175 | Pertuzumab, Trastuzumab, Lapatinib | 17 | dose_response |
| Ruprecht 2017 | 10 | BT-474 | Lapatinib | 10 | parental_vs_resistant |
| **Total** | **54** | **2 lines** | **3 drugs** | **~27** | **2 contexts** |

#### Key Architecture Note (from step06 analysis):
> **The pathway/signaling network data from PNAS/Remsing (1,964 rows) is NOT used in the model.**
> Step06 line 602: "Per-pathway aggregate features — REMOVED FROM MODEL INPUT"
> The model only uses **12 delta_ptm values per drug** (specific phosphosite fold-changes),
> NOT the full network rows. So the 54 ERBB2 rows are **sufficient** — they map to the
> equivalent delta_ptm slots. The "2109 vs 54" comparison is misleading because 93% of
> EGFR data (1,964 rows) is unused pathway context. Fair comparison: 145 EGFR-protein vs 54 ERBB2-protein.

#### GDSC HER2 Drug Response (from step01 analysis):
| Drug | Breast Cancer Samples | Sensitive | Resistant |
|------|----------------------|-----------|-----------|
| Lapatinib | 51 | 6 (11.8%) | 45 (88.2%) |
| Afatinib | 50 | 9 (18.0%) | 41 (82.0%) |
| **Total** | **101** | **15 (14.9%)** | **86 (85.1%)** |

> Combined with EGFR: 638 + 101 = **739 total samples**
> Class balance improves: 6.3% → 7.4% sensitive

---

### Step 06 — Harmonize Dataset

**Current:** Merges GDSC + CCLE + PTM into `multimodal_dataset.csv` for EGFR only.

**Changes needed:**
- [ ] Add `target_gene` column (EGFR or ERBB2) to every row
- [ ] Multi-gene harmonization: same cell line can appear with different `target_gene` if it has both EGFR and HER2 data
- [ ] Gene-specific mutation-to-sequence mapping:
  - EGFR mutations → EGFR sequences (existing)
  - HER2 mutations/amplification → HER2 sequences (new)
- [ ] Gene-specific PDB mapping:
  - EGFR mutation classes → EGFR PDB structures (existing)
  - HER2 status → HER2 PDB structures (new: 3PP0)
- [ ] Gene-specific PTM vector assignment:
  - EGFR: 12 sites (existing)
  - HER2: 10 sites padded to 12 (new)
- [ ] Gene-specific delta_ptm computation:
  - Use HER2-specific drug-PTM data for HER2 samples
- [ ] Add HER2 drug SMILES to config:
  - Lapatinib: `ClC1=CC=C(C=C1Cl)C1=CC2=C(C=C1)N=CN=C2NC1=CC(=C(C=C1)OCC1=CC=CC=N1)OCC1=CC=CC=N1` (verify)
  - Neratinib: `CC(=O)NC1=CC=C(C=C1)OC1=C(C=C(C=C1)NC1=NC=C(C=C1)C#N)NC1=CC=CC=C1OC` (verify)
- [ ] Phospho propagation for HER2:
  - HER2-amplified class → propagated from BT-474 measurements
  - HER2-WT class → WT prior (similar to EGFR WT)
- [ ] Output: `multimodal_dataset.csv` with both EGFR and HER2 rows, `target_gene` column

**Biological note on HER2 propagation:**
- Unlike EGFR (where mutations determine phospho), HER2 phospho depends on **amplification level** and **co-receptor availability**
- BT-474 is HER2-amplified, SKBR3 is HER2-amplified — these should share phospho profiles
- HER2-low breast cancer lines should get a different (lower) phospho prior

---

### Step 07 — Extract ESM-2 Embeddings

**Current:** Generates ESM-2 embeddings for 17 EGFR sequence variants.

**Changes needed:**
- [ ] Add HER2 sequences to ESM-2 extraction:
  - HER2 wild-type (from UniProt P04626 FASTA)
  - HER2 mutant variants (if any found in CCLE)
- [ ] ESM-2 handles any protein sequence natively — no architecture change needed
- [ ] Save embeddings with `target_gene` prefix or in separate subdirectory:
  - `data/features/esm2/ERBB2_wild_type_per_residue.npy`
  - `data/features/esm2/ERBB2_wild_type_pooled.npy`
- [ ] Update `embedding_metadata.json` with HER2 entries

---

### Step 08 — Extract GearNet Structural Embeddings

**Current:** Extracts GearNet embeddings for 9 EGFR PDB structures.

**Changes needed:**
- [ ] Add HER2 PDB structures to GearNet extraction:
  - `3PP0` → `3PP0_residue_embeddings.npy` + `3PP0_coords.npy`
  - `3RCD` → (optional, Lapatinib-bound reference)
- [ ] GearNet handles any PDB natively — no architecture change needed
- [ ] Update `structural_embedding_metadata.json` with HER2 entries

---

### Step 09 — Extract ChemBERTa Drug Embeddings

**Current:** Extracts embeddings for 4 drugs (Gefitinib, Afatinib, Erlotinib, Osimertinib).

**Changes needed:**
- [ ] Add new drug SMILES to config.yaml and extraction:
  - Lapatinib SMILES
  - Neratinib SMILES
- [ ] ChemBERTa handles any SMILES natively — no architecture change needed
- [ ] Save: `lapatinib_per_token.npy`, `lapatinib_pooled.npy`, etc.
- [ ] Update `drug_embedding_metadata.json`

**Note:** Afatinib is already a pan-ERBB inhibitor in both EGFR and HER2 contexts. No change needed for Afatinib.

---

### Step 10 — Build Model

**Current:** `MultimodalResistancePredictor` with `ptm_dim=12`.

**Changes needed:**
- [ ] **Minimal change:** Keep `ptm_dim=12`. HER2 samples have 10 PTM sites padded to 12 with zeros.
  - The `PTMFeatureModulator` and `PTMTokenEncoder` both accept `ptm_dim=12` — no change
  - Zero-padded sites contribute zero signal (correct behavior)
  - The model learns that the last 2 sites are always zero for HER2 samples

- [ ] **Optional enhancement:** Add `gene_embedding` as an additional input:
  - Learnable embedding for EGFR vs ERBB2 (2-dimensional one-hot → shared_dim)
  - Added as an extra token in the attention sequence
  - Helps the model distinguish EGFR-context from HER2-context
  - This is OPTIONAL — the sequence/structure embeddings already distinguish the two proteins

- [ ] Verify model handles variable sequence lengths (ESM-2 EGFR: 1210 residues, HER2: 1255 residues) — already handled by padding in `collate_fn`

- [ ] Update `architecture_info.json` to document multi-gene support

---

### Step 11 — Training

**Current:** Trains on EGFR-only dataset.

**Changes needed:**
- [ ] No architecture changes needed (model accepts any dataset)
- [ ] Stratified splits should account for BOTH `resistance_label` AND `target_gene`:
  - `StratifiedShuffleSplit` on combined label: `"EGFR_resistant"`, `"EGFR_sensitive"`, `"ERBB2_resistant"`, `"ERBB2_sensitive"`
  - Ensures each split has proportional EGFR and HER2 samples
- [ ] Class-balanced sampling should handle the improved balance (~15% sensitive with HER2)
- [ ] Report per-gene training distribution
- [ ] Focal loss alpha may need adjustment (0.25 → recalculate based on new class ratio)

---

### Step 11b — Ablation Study

**Current:** 4 ablation models on EGFR-only.

**Changes needed:**
- [ ] Same 4 ablation modes work unchanged
- [ ] Add **per-gene ablation reporting**: does PTM help more for EGFR or HER2?
- [ ] LODO validation: add Lapatinib and Neratinib to the drug rotation
- [ ] Stability analysis: check if IG rankings are stable for BOTH EGFR sites AND HER2 sites

---

### Step 11c — Cross-Validation

**Current:** Already built with multi-cancer support (target_gene stratification).

**Changes needed:**
- [ ] Already handles `target_gene` column in stratification ✅
- [ ] Already reports per-gene metrics ✅
- [ ] May need to adjust IG to handle variable PTM site labels (EGFR 12 sites vs HER2 10+2 padded)
- [ ] Add per-gene IG analysis: which sites matter most for EGFR vs HER2?

---

### Step 12 — Evaluation

**Current:** Evaluates on EGFR-only test set.

**Changes needed:**
- [ ] Add per-gene evaluation section:
  - EGFR samples: BAcc, AUROC, RMSE, R for EGFR drugs only
  - HER2 samples: BAcc, AUROC, RMSE, R for HER2 drugs only
  - Cross-gene: does the model transfer across receptor types?
- [ ] Drug-specific analysis: add Lapatinib and Neratinib
- [ ] Mutation-stratified analysis: add HER2-amplified groups
- [ ] Confidence analysis: separate measured vs propagated for each gene

---

### Step 13 — Explainability

**Current:** IG on 12 EGFR phosphosites.

**Changes needed:**
- [ ] Run IG separately for EGFR samples and HER2 samples
- [ ] Report EGFR site importance ranking (Y1068, Y1173, ...) — should match June 23 results
- [ ] Report HER2 site importance ranking (Y1221, Y1222, Y1248, ...)
- [ ] **KEY ANALYSIS:** Compare EGFR Y1068 rank vs HER2 Y1221 rank:
  - If both rank #1 in their respective gene → model learns homologous function
  - This is the strongest biological validation for cross-receptor generalization
- [ ] Cross-modal attention: compare EGFR vs HER2 attention patterns
- [ ] Pathway validation: use HER2-specific pathway profiles

---

## 4. Data Sources Already Available

### Already Downloaded (in data/raw/)
| Source | File | HER2 Content |
|--------|------|-------------|
| DrugPTM-Bench | `PTM_CellLine_BT-474.csv` | 510 ERBB2 phospho rows, 17 sites (Pertuzumab, Trastuzumab) |
| GDSC | `GDSC2_fitted_dose_response_*.xlsx` | Contains Lapatinib (1558), Neratinib (2097) for breast cancer |
| DepMap/CCLE | `ccle_somatic_mutations.csv` | Contains ERBB2 mutations (rare) |

### Need to Download
| Source | What | How |
|--------|------|-----|
| UniProt P04626 | HER2 reference FASTA + PTM annotations | `curl https://www.uniprot.org/uniprot/P04626.fasta` |
| PDB 3PP0 | HER2 kinase domain (apo) | `curl https://files.rcsb.org/download/3PP0.pdb` |
| PDB 3RCD | HER2 + Lapatinib (optional) | `curl https://files.rcsb.org/download/3RCD.pdb` |
| HER2 resistance phospho papers | Via step05b search script | Manual download after paper review |

---

## 5. Model Architecture Considerations

### How to Differentiate Cancer Types

The model differentiates EGFR from HER2 through **three natural channels**:

1. **Sequence (ESM-2):** EGFR and HER2 have different amino acid sequences → different per-residue embeddings. ESM-2 captures evolutionary context, so HER2 embeddings are naturally distinct.

2. **Structure (GearNet):** EGFR PDBs (2GS6, 4HJO, etc.) produce different structural embeddings than HER2 PDBs (3PP0). The kinase domain fold is similar but not identical.

3. **PTM vector:** EGFR has 12 phosphosites at specific positions; HER2 has 10 sites at different positions (padded to 12). The site identity embeddings in `PTMTokenEncoder` distinguish site 7 (Y1092/Y1068 for EGFR) from site 7 (Y1221 for HER2) through the learned embedding.

**Optional: Explicit gene identity token**
- A learnable `gene_embedding` (EGFR=0, ERBB2=1) added as an extra attention token
- Helps if the above three channels don't provide sufficient discrimination
- Can be added later if needed — start without it

### Stability Across Cancer Types

**Concern:** Adding HER2 might destabilize EGFR predictions if the model tries to learn a single representation for both.

**Mitigation:**
1. The PhosphoContextEncoder's `has_activating_mutation` indicator can be extended to `mutation_type` (amplification vs point mutation)
2. Cross-validation with per-gene stratification ensures each fold tests both EGFR and HER2
3. Per-gene metrics in step12/step13 detect if one gene's performance degrades

---

## 6. Timeline Estimate

| Phase | Steps | Effort | Dependencies |
|-------|-------|--------|-------------|
| **Phase A: Data acquisition** | 01, 02, 03, 04, 05 | 3–4 days | User downloads manual data |
| **Phase B: Data harmonization** | 06 | 2–3 days | Phase A complete |
| **Phase C: Feature extraction** | 07, 08, 09 | 1 day (mostly automated) | Phase B complete |
| **Phase D: Model & training** | 10, 11, 11b, 11c | 1–2 days code + runtime | Phase C complete |
| **Phase E: Evaluation & XAI** | 12, 13 | 1 day | Phase D complete |
| **Total** | — | **~2–3 weeks** | — |

### Order of Implementation
1. Config.yaml updates (add HER2 drugs, structures, PTM sites)
2. Step 05b: Search for HER2 phospho papers → user reviews → downloads data
3. Steps 01–04: Download/process GDSC breast cancer, HER2 mutations, PDB, PTM
4. Step 05: Process HER2 Drug-PTM data
5. Step 06: Multi-gene harmonization (biggest code change)
6. Steps 07–09: Feature extraction (minimal code changes)
7. Step 10: Verify model handles expanded dataset
8. Steps 11–13: Train, evaluate, interpret on expanded dataset

---

## Appendix: HER2 Phosphorylation Sites (P04626)

| Position (P04626) | Residue | Equivalent EGFR Site | Known Function |
|-------------------|---------|---------------------|----------------|
| Y1005 | Y1005 | — | c-Cbl direct binding → receptor degradation |
| T1099 | T1099 | — | Regulatory |
| Y1139 | Y1139 | Y1069 (Y1045) | Adaptor recruitment |
| S1151 | S1151 | — | Regulatory |
| Y1196 | Y1196 | Y1110 (Y1086) | c-Cbl → receptor downregulation |
| Y1221 | Y1221 | **Y1092 (Y1068)** | **GRB2 → RAS-MAPK (primary)** |
| Y1222 | Y1222 | — | GRB2/SHC dual docking (unique to HER2) |
| Y1248 | Y1248 | **Y1197 (Y1173)** | **SHC1 → PI3K-AKT (survival)** |
| S1054 | S1054 | S1039 | Regulatory (Ser) |
| T686 | T686 | T693 | Regulatory (Thr, juxtamembrane) |

**Mapping to padded 12-site vector:**
Sites 1–10: HER2 phosphosites in order
Sites 11–12: Zero-padded (no biological site)

---

*This expansion plan was developed based on the EXPANSION_FEASIBILITY_ANALYSIS.md (Option B+) and refined for step-by-step implementation.*

---

## 7. Implementation Notes — Critical Findings (Updated During Implementation)

> **IMPORTANT:** These findings were discovered during implementation and affect ALL subsequent steps.
> Every step script must account for these changes.

### 7a. GDSC2 Drug Availability (Step 01 — 2026-06-25)

**Neratinib (ID 2097) is NOT in GDSC2 (Oct 2023 release).** No records found under any name variant (Neratinib, HKI-272, or drug ID 2097). Neratinib has been **replaced with Sapitinib** (AZD8931, ID 1549):

| Drug | GDSC2 ID | Target | GDSC2 Status |
|------|----------|--------|-------------|
| Neratinib | 2097 | pan-ERBB | ❌ **NOT IN GDSC2** — removed |
| Sapitinib | 1549 | EGFR, ERBB2, ERBB3 | ✅ 966 records — added as replacement |

**ALL EGFR drugs are also tested on breast cancer cell lines.** The GDSC2 dataset tests every drug on ~52 breast cancer cell lines. This allows dual-context mapping where EGFR drugs appear in BOTH NSCLC (target_gene=EGFR) and breast (target_gene=ERBB2) contexts:

| Drug | NSCLC (EGFR) | Breast (ERBB2) | Breast Sensitive |
|------|-------------|---------------|-----------------|
| Osimertinib | 159 lines | 51 lines | 13 (25.5%) |
| Gefitinib | 159 lines | 51 lines | 1 (2.0%) |
| Afatinib | 161 lines | 50 lines | 9 (18.0%) |
| Erlotinib | 159 lines | 51 lines | 0 (0.0%) |
| Lapatinib | — | 51 lines | 6 (11.8%) |
| Sapitinib | — | 51 lines | 1 (2.0%) |
| **Total** | **638** | **305** | **30 (9.8%)** |

**Combined dataset: 943 records** (638 EGFR + 305 ERBB2), 212 unique cell lines, 70 sensitive (7.4%).

**GDSC2 tissue type is `"Breast Carcinoma"`** — not `"BRCA"`. The tissue filter regex must include `"Breast Carcinoma"` alongside `"BRCA|breast"`.

**Known HER2+ breast cancer cell lines found (7/8):**
BT-474, AU565, HCC1954, MDA-MB-453, MDA-MB-361, ZR-75-30, UACC-812
(SKBR3 not found — may use different name in GDSC2)

### 7b. Impact on All Steps

These findings affect every subsequent step:

| Step | Impact |
|------|--------|
| **Step 02** | Must extract ERBB2 mutations from CCLE alongside EGFR. Must download HER2 FASTA (P04626). Breast cancer lines need ERBB2 mutation profiles. |
| **Step 03** | Must download HER2 PDB structures (3PP0, 3RCD). No new structures needed for Sapitinib. |
| **Step 04** | Must create HER2 phospho sites file. 10 ERBB2 sites + 2 zero-padded = ptm_dim=12. |
| **Step 05** | Already done — ERBB2 phospho data from DrugPTM-Bench + Ruprecht 2017. Sapitinib has no drug-specific phospho data (use pan-ERBB prior from Afatinib). |
| **Step 06** | Must handle dual-context drug_gene_mapping (list-based). Must assign target_gene per row. Must map breast cancer lines to HER2 sequences/structures/PTM. |
| **Step 07** | Must generate ESM-2 embeddings for HER2 wild-type sequence (1255 AA vs 1210 for EGFR). |
| **Step 08** | Must generate GearNet embeddings for 3PP0 (HER2 apo structure). |
| **Step 09** | Must generate ChemBERTa embeddings for Lapatinib and Sapitinib SMILES. |
| **Step 10** | ptm_dim=12 unchanged. Model handles variable sequence lengths via padding. |
| **Step 11** | Stratified splits must account for target_gene (EGFR vs ERBB2). |
| **Step 12** | Must report per-gene metrics. Must include Lapatinib and Sapitinib in drug analysis. |

### 7c. Drug SMILES Updates

| Drug | Source | SMILES |
|------|--------|--------|
| Lapatinib | PubChem CID 208908 | `CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1` |
| Sapitinib | PubChem CID 11476171 | `COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCOCC1` |

### 7d. Config Changes Summary

All changes are reflected in `config/config.yaml`:
- `project.tissue_filters.ERBB2`: `"BRCA|breast|Breast Carcinoma"`
- `project.target_drugs`: Neratinib → Sapitinib
- `gdsc.drug_ids`: Added Lapatinib (1558), Sapitinib (1549); removed Neratinib (2097)
- `gdsc.drug_gene_mapping`: List-based (e.g., `Afatinib: ["EGFR", "ERBB2"]`)
- `uniprot.ERBB2`: Added P04626 reference
- `ptm.ERBB2`: 10 phospho sites defined
- `drugs.sapitinib`: SMILES added (replaced neratinib)
- `pdb.structures`: Added 3PP0 (HER2 apo) and 3RCD (HER2+Lapatinib)
