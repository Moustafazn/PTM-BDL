# PROJECT SUMMARY
## PTM-Driven EGFR & HER2 Drug Resistance Predictor
### Integrative Multimodal Learning Reveals PTM-Dependent Mechanisms of Drug Resistance Across EGFR-Mutant Lung Cancer and HER2-Positive Breast Cancer

---

## 🎯 What This Project Does

This project builds an **AI system that predicts drug resistance** across **EGFR-mutant NSCLC** and **HER2-positive breast cancer** — by simultaneously analyzing four biological modalities:

1. **Protein Sequence** (mutations) — via ESM-2 (EGFR: 1210 AA, HER2: 1255 AA)
2. **3D Protein Structure** (binding pocket shape) — via pretrained GearNet-Edge (EGFR + HER2 PDBs)
3. **Post-Translational Modifications** (phosphorylation states) — via PTM modulation (12 sites per protein)
4. **Drug Chemistry** (molecular structure) — via ChemBERTa (6 ERBB-targeting TKIs)

The core innovation is an **Early-Correlation Hybrid Fusion** architecture that allows all modalities to interact through joint self-attention, enabling the discovery of cross-modal resistance mechanisms that single-modality models miss.

### Receptor Coverage (EGFR + HER2)

| Feature | EGFR (NSCLC) | HER2/ERBB2 (Breast) |
|---------|-------------|---------------------|
| **Samples** | 646 | 305 |
| **Drugs** | Osimertinib, Gefitinib, Afatinib, Erlotinib | + Lapatinib, Sapitinib |
| **Cross-protein drugs** | All 4 EGFR drugs also tested on breast cancer | ✓ |
| **Protein sequence** | P00533 (1210 AA) | P04626 (1255 AA) |
| **PDB structures** | 2GS6, 2JIT, 4HJO, 5EDP | 3PP0 |
| **PTM sites** | 12 EGFR phosphosites | 10 HER2 sites + 2 zero-padded |
| **Phospho data** | 8 sources (DrugPTM-Bench, Tozuka, Hsu, PNAS, FEBS...) | BT-474 (DrugPTM-Bench + Ruprecht 2017) |

---

## 📁 Project Structure

```
PTM-BDL-Framework/
├── config/
│   └── config.yaml              # All parameters, drug SMILES, PTM sites, PDB structures
├── scripts/
│   ├── step01_download_gdsc.py      # Drug response data (IC50) — EGFR + HER2
│   ├── step02_download_mutations.py # EGFR + ERBB2 mutation profiles + sequences
│   ├── step03_download_structures.py # PDB structures (EGFR: 9 + HER2: 2)
│   ├── step04_download_ptm_data.py  # Phosphorylation sites (EGFR + HER2)
│   ├── step05_download_drugptm.py   # Drug-induced PTM changes (EGFR + HER2)
│   ├── step06_harmonize_dataset.py  # ★ MERGE all sources → unified dataset (951 samples)
│   ├── step07_extract_esm2.py       # ESM-2 protein embeddings (EGFR + HER2 sequences)
│   ├── step08_extract_gearnet.py    # GearNet structural embeddings (all PDBs incl. 3PP0)
│   ├── step09_extract_chemberta.py  # ChemBERTa drug embeddings (6 drugs)
│   ├── step10_build_model.py        # Build & verify model architecture
│   ├── step11_train.py              # Training pipeline (stratified by protein + resistance)
│   ├── step11b_ablation.py          # ★ PTM Ablation Study (proves core thesis)
│   ├── step11c_crossval.py          # 5-fold cross-validation with per-protein metrics
│   ├── step12_evaluate.py           # Evaluation (per-protein, per-drug, cross-protein)
│   └── step13_explainability.py     # XAI: attention + IG + per-protein PTM attribution
├── src/
│   └── models/
│       └── multimodal_predictor.py  # ★ CORE MODEL ARCHITECTURE
├── data/                            # Created by scripts
│   ├── raw/                         # Downloaded data
│   ├── processed/                   # Harmonized dataset (multimodal_dataset.csv)
│   └── features/                    # Extracted embeddings (ESM-2, GearNet, ChemBERTa)
├── results/                         # Evaluation reports & figures
├── research/
│   └── HER2_EXPANSION_PLAN.md       # HER2 expansion documentation
├── pyproject.toml                   # Dependencies
├── Scientific Explanation.md        # Full technical documentation
└── How to run the project.md        # This file
```

---

## 📊 Dataset Selection & Rationale

| Dataset | What It Provides | Why It's Critical |
|---------|-----------------|-------------------|
| **GDSC** | IC50 for >1000 cell lines × >400 drugs | **Ground-truth labels** for 6 ERBB TKIs across NSCLC + breast cancer |
| **CCLE/DepMap** | EGFR + ERBB2 mutations per cell line | **Links mutations to response** — L858R, T790M, HER2 amplification |
| **PDB** | 3D structures: 5 EGFR + 2 HER2 (3PP0, 3RCD) | **Structural context** for both receptor proteins |
| **UniProt** | Phosphosites: 12 EGFR + 10 HER2 | **PTM biology** — the CORE hypothesis |
| **DrugPTM-Bench** | Dose-response phosphoproteomics | **Drug-PTM link** for EGFR (A431, H3255) + HER2 (BT-474) |
| **Ruprecht 2017** | BT-474 lapatinib-resistant phospho | **HER2 resistance signatures** |
| **Tozuka 2024** | PC-9/HCC827 osimertinib-resistant | **EGFR resistance phospho** |
| **PNAS 2025** | H1975/HCC4006 pY phosphoproteome | **Direct pY data under Osimertinib** |

---

## 🏗️ Model Architecture Summary

```
INPUT MODALITIES:
  Protein Sequence (ESM-2)          → (L × 1280) tokens  [EGFR: L=1210, HER2: L=1255]
  3D Structure (GearNet + PTM)      → (M × 512) tokens   [+ PTMFeatureModulator gating]
  PTM Site Tokens (12 × [p, Δp])    → (12 × 512) tokens  [ptm_level + drug-conditioned delta]
  PTM Rewiring Context              → (1 × 512) token    [7 phospho features + 2 indicators]
  Drug Chemistry (ChemBERTa)        → (N × 384) tokens

                    ↓ Project to shared dimension (512) ↓

JOINT SELF-ATTENTION TRANSFORMER (4 layers × 8 heads):
  All tokens concatenated: [seq ; struct+PTM ; ptm_sites ; context ; drug]
  Cross-modal attention: protein residues ↔ PTM sites ↔ drug atoms
  AttentionPooling (learned weights, not mean)
  → Protein-PTM representation (512-dim)

BILINEAR FUSION:
  protein_rep ⊙ drug_rep → prediction vector
  → IC50 regression head + Resistance classification head
```

---

## 🚀 How to Run

### Prerequisites
```bash
# Install core dependencies
pip install -e .

# For pretrained structural embeddings:
pip install fair-esm biotite    # ESM-IF1 (best)
# OR: pip install torch-geometric  # PyG fallback
```

### Full Pipeline
```bash
# ═══════════════════════════════════════════════════
# Phase 1: Data Acquisition (Steps 01-05)
# ═══════════════════════════════════════════════════
python scripts/step01_download_gdsc.py       # GDSC IC50: EGFR + HER2 drugs
python scripts/step02_download_mutations.py  # EGFR + ERBB2 mutation profiles
python scripts/step03_download_structures.py # PDB: EGFR structures + 3PP0 (HER2)
python scripts/step04_download_ptm_data.py   # PTM sites: EGFR (12) + HER2 (10)
python scripts/step05_download_drugptm.py    # Drug-PTM phospho: EGFR + HER2

# ═══════════════════════════════════════════════════
# Phase 2: Data Harmonization (Step 06)
# ═══════════════════════════════════════════════════
python scripts/step06_harmonize_dataset.py   # → 951 samples, 100% phospho coverage

# ═══════════════════════════════════════════════════
# Phase 3: Feature Extraction (Steps 07-09)
# ═══════════════════════════════════════════════════
python scripts/step07_extract_esm2.py        # ESM-2: EGFR + HER2 sequences
python scripts/step08_extract_gearnet.py     # GearNet: all PDBs incl. 3PP0
python scripts/step09_extract_chemberta.py   # ChemBERTa: 6 drugs (incl. Lapatinib, Sapitinib)

# ═══════════════════════════════════════════════════
# Phase 4: Model Build, Training & Analysis (Steps 10-13)
# ═══════════════════════════════════════════════════
python scripts/step10_build_model.py         # Verify model shapes
python scripts/step11_train.py               # Train (stratified by protein + resistance)
python scripts/step11b_ablation.py           # ★ PTM Ablation + Stability + Randomized Control
python scripts/step11c_crossval.py           # 5-fold CV with per-protein metrics
python scripts/step12_evaluate.py            # Per-protein, per-drug, cross-protein analysis
python scripts/step13_explainability.py      # XAI: attention + IG (EGFR + HER2 site labels)
```

---

## 🔬 Biological Questions This System Answers

1. **Do PTM features improve resistance prediction?**
   → Step 11b ablation: Full model vs No-PTM baseline (4 metrics)

2. **Are homologous phosphosites independently important across receptors?**
   → IG analysis: EGFR Y1068 vs HER2 Y1221 (both recruit GRB2 → RAS-MAPK)

3. **Do cross-protein drugs (Afatinib, Osimertinib, Gefitinib, Erlotinib) show consistent behavior across EGFR and HER2?**
   → Step 12 cross-protein drug analysis

4. **Which PTM sites drive resistance predictions for each receptor?**
   → Step 13 per-protein IG with EGFR-specific and HER2-specific site labels

5. **Does the model generalize across EGFR and HER2?**
   → Per-protein BAcc/AUROC in step 12 (EGFR vs HER2 test performance)

---

## 📊 Benchmarking & Publication (Steps 14–15)

After training and evaluation, run the benchmarking suite to compare against external methods and generate publication-quality outputs:

```bash
# ═══════════════════════════════════════════════════
# Phase 5: Benchmarking (Steps 14a-d)
# ═══════════════════════════════════════════════════
python scripts/step14a_ml_baselines.py        # Tier 0: RF, XGBoost, Ridge, Elastic Net
python scripts/step14b_external_baselines.py  # Tier 1-2: DIPK, HiDRA, GraTransDRP, etc.
python scripts/step14c_statistical_tests.py   # Bootstrap CIs, DeLong, Wilcoxon, BH correction
python scripts/step14d_loclo.py               # Cell-blind LOCLO generalization test

# ═══════════════════════════════════════════════════
# Phase 6: Publication Figures & Tables (Steps 15a-b)
# ═══════════════════════════════════════════════════
python scripts/step15a_paper_figures.py       # Camera-ready PDF figures (300 DPI)
python scripts/step15b_paper_tables.py        # CSV + LaTeX tables (booktabs style)

# Or use Make targets:
make benchmark    # Runs steps 14a-d
make figures      # Runs steps 15a-b
make all          # Runs entire pipeline including benchmarking
```

### Benchmarking Outputs

| Output | Location | Description |
|--------|----------|-------------|
| ML baseline metrics | `results/ml_baselines.json` | RF, XGBoost, Ridge, Elastic Net on same features |
| External method results | `results/external_baselines/` | Per-method JSON with metrics + integration status |
| Statistical tests | `results/statistical_tests.json` | Bootstrap CIs, DeLong, Wilcoxon, BH correction |
| LOCLO generalization | `results/loclo_results.json` | Cell-blind split by mutation class |
| Publication figures | `results/publication/figures/*.pdf` | Nature Methods format (300 DPI, colorblind) |
| Publication tables | `results/publication/tables/*.csv` + `*.tex` | Main text + supplementary tables |

### Comparison Methods

| Tier | Methods | Purpose |
|------|---------|---------|
| **Tier 0** | RF, XGBoost, Ridge, Elastic Net | Simple ML baselines on same 2224-d features |
| **Tier 1** | DIPK (2024), HiDRA (2023), GraTransDRP (2023), TransCDR (2023), PathDSP (2024) | Recent state-of-the-art DRP methods |
| **Tier 2** | GraphDRP (2022), DrugCell (2020), DeepCDR (2020) | Established baselines |

See [`docs/BENCHMARKING_PLAN.md`](BENCHMARKING_PLAN.md) for the full benchmarking strategy, metric selection rationale, and statistical testing protocol.

---

## 📈 Expected Outcomes

1. A **cross-receptor PTM-aware resistance predictor** for EGFR and HER2
2. **Per-protein phosphosite importance rankings** (EGFR: Y1068 #1? HER2: Y1221 #1?)
3. **Cross-protein drug consistency** — same drug, different receptor, comparable predictions
4. **100% phospho coverage** (vs 68.6% before ERBB2 propagation fix)
5. A **reusable multimodal framework** applicable to other kinase families
6. **Comprehensive benchmarking** against 12 methods with full statistical rigor
7. **Publication-ready figures and tables** formatted for Nature Methods
