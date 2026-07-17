# Multimodal Self-Attention with PTM Biological Dynamics Layer: A PTM Foundational Framework for Drug Response Prediction

**A foundational multimodal deep learning framework that combines cross-modal self-attention over protein sequence, 3D structure, and drug chemistry with a PTM Biological Dynamics Layer (PTM-BDL) — a typed self-attention module that encodes the dynamic post-translational modification signaling state of the cell. The framework accepts one or more PTM types (e.g., phosphorylation, glycosylation), each with 1-to-N subtypes, learning how their combinatorial patterns drive drug resistance. Demonstrated on EGFR (NSCLC) and ERBB2/HER2 (breast cancer) TKI resistance across two proteins, two PTM types, four subtypes, and six drugs.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

---

## 1. Overview

### The Problem

Drug resistance is the primary cause of treatment failure in targeted cancer therapy. Predicting resistance requires integrating multiple layers of biological information — protein sequence (mutations), 3D structure (binding pocket changes), drug chemistry (molecular properties), and dynamic post-translational modification (PTM) states (signaling rewiring) — that no single modality can capture alone.

Current drug response prediction (DRP) methods either ignore PTMs entirely, or treat them as flat features in an MLP. In reality, drug resistance is determined by the **combinatorial PTM signaling code** — the pattern of which sites are phosphorylated, glycosylated, or otherwise modified, how drugs change those patterns, and how different modification types interact. No existing model captures this multi-layered biological process.

### Two Contributions

This framework addresses both problems through a **two-stage architecture**:

#### Contribution 1: Multimodal Cross-Modal Self-Attention

Four biological modalities are projected into a shared latent space and processed through **joint self-attention**, enabling the model to discover cross-modal interaction patterns that single-modality or late-fusion models cannot learn:

| Modality | Encoder | What it captures | What it misses alone |
|----------|---------|------------------|---------------------|
| Protein sequence | ESM-2 (650M params) | Mutations exist | Their 3D structural effect |
| 3D structure | GearNet-Edge | Binding pocket shape | Dynamic PTM signaling |
| Drug chemistry | ChemBERTa-77M | Molecular properties | Biological context |
| **PTM state** | **PTM-BDL** | **Dynamic signaling** | Drug chemistry |

Sequence residues attend to structural features, structural features attend to drug atoms, and vice versa — the model can discover patterns like "mutation at residue 790 changes the binding pocket shape for this drug's warhead."

#### Contribution 2: PTM Biological Dynamics Layer (PTM-BDL)

A typed self-attention module that encodes the dynamic PTM signaling state of the cell. PTM-BDL is designed as a **general-purpose PTM encoder** that handles:

- **One or more PTM types** (e.g., phosphorylation, glycosylation, acetylation, ubiquitination)
- **1-to-N subtypes per type**, each with its own learned embedding:

```
PTM Type: phosphorylation
  └── Subtypes: phospho_Y (subtype 0), phospho_S (subtype 1), phospho_T (subtype 2)

PTM Type: glycosylation
  └── Subtypes: glyco_N (subtype 3)

PTM Type: acetylation (future — zero architecture changes)
  └── Subtypes: acetyl_K (subtype 4), acetyl_Nt (subtype 5)
```

Each PTM site becomes a **typed token** encoded as `[level, delta, ratio]`:
- `level` — baseline modification occupancy
- `delta` — drug-induced change in occupancy
- `ratio = delta / (level + ε)` — fractional drug efficacy at that site

The typed self-attention then:
1. **Type-gates** each token — a learnable gate controlled by the subtype embedding determines which information passes (the model learns that phospho-Y has a different biological role than phospho-S or glyco-N)
2. **Adds protein identity** — protein-indexed embeddings let the model distinguish which protein a site belongs to
3. **Discovers inter-site dependencies** — self-attention over all PTM tokens finds which site combinations predict resistance (e.g., Y1068 queries Y1173: "Is the survival pathway also shut down?")
4. **Discovers cross-type interactions** — phospho and glyco tokens attend to each other in the same attention space, learning crosstalk between intracellular signaling and receptor surface biology
5. **Applies a residual gate** — α·attended + (1−α)·independent, so sites that don't benefit from context keep their independent signal

The architecture handles proteins with **different numbers of real vs. padded sites** through attention masking — adding a new protein with different PTM coverage requires only adding its site definitions and padding configuration.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL INPUT LAYER                        │
│  ESM-2 (L×1280)  GearNet (M×512)  ChemBERTa (N×384)           │
│       ↓                ↓                 ↓                      │
│    Modality Projection → shared_dim (512) each                  │
│       ↓                ↓                 ↓                      │
│  + modality embeddings (learned: seq / struct / drug)           │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              JOINT SELF-ATTENTION TRANSFORMER                   │
│  [seq_tokens ; struct_tokens ; drug_tokens] concatenated        │
│  → 4 layers × 8 heads cross-modal self-attention                │
│  → Attention pooling (Ilse et al., ICML 2018)                  │
│  → S_rep: static protein-drug representation (512-dim)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           PTM BIOLOGICAL DYNAMICS LAYER (PTM-BDL)         │  │
│  │                                                           │  │
│  │  PTM sites → [level, delta, ratio] per token              │  │
│  │  → Type-gated projection (per modification subtype)       │  │
│  │  → + type_emb + protein_emb + slot_emb                    │  │
│  │  → Typed self-attention (inter-site dependencies)         │  │
│  │  → Residual gate: α·attended + (1−α)·independent          │  │
│  │  → Mask-aware mean pool → P_rep (64-dim)                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  BILINEAR LATE FUSION: S_rep ⊙ P_rep                           │
│  "Given the drug CAN bind (S_rep), does PTM say it WORKS?"    │
│                                                                  │
│  → IC50 regression head                                         │
│  → Resistance classification head                               │
└─────────────────────────────────────────────────────────────────┘
```

**Why two-stage fusion?** Drug identity enters through S_rep (early joint attention), not through PTM-BDL. This prevents the model from learning drug→prediction shortcuts that bypass PTM modulation. The delta_ptm input already encodes drug-induced changes, so PTM-BDL IS drug-conditioned through its *input features*, not through a fusion shortcut.

---

## 2. Demonstration: EGFR & ERBB2 TKI Resistance

The framework is demonstrated on a challenging real-world application: predicting tyrosine kinase inhibitor resistance across two receptor tyrosine kinases in two cancer types.

### Dataset

| Property | Value |
|----------|-------|
| Total samples | 951 |
| EGFR (NSCLC) | 646 samples |
| ERBB2/HER2 (breast) | 305 samples |
| Drugs | 6 TKIs (Osimertinib, Gefitinib, Afatinib, Erlotinib, Lapatinib, Sapitinib) |
| PTM sites per protein | 12 phospho + 12 glyco = 24 typed tokens |
| PTM subtypes | 4 (phospho-Y, phospho-S, phospho-T, glyco-N) |
| Class distribution | ~92% resistant, ~8% sensitive |
| PTM data sources | 8 independent phosphoproteomic + 5 glycoproteomic studies |

### Why Two Proteins Prove Generalizability

| Design dimension | EGFR | ERBB2/HER2 |
|-----------------|------|------------|
| Cancer type | Non-small cell lung cancer | Breast cancer |
| Real PTM sites | 24 (12 phospho + 12 glyco) | 17 (10 phospho + 7 glyco) + 7 padded |
| Drugs tested | Osimertinib, Gefitinib, Afatinib, Erlotinib | + Lapatinib, Sapitinib |
| Dominant resistance pathway | RAS-MAPK (Y1068/GRB2) | PI3K-AKT (Y1248/SHC1) |

The architecture handles both proteins through the **same encoder** — different padding masks, different type assignments, different pathway biology — proving the framework is protein-agnostic.

### Data Integration

| Source | What | Reference |
|--------|------|-----------|
| GDSC2 | IC50 drug response | Iorio et al., Cell 2016 |
| DepMap/CCLE | Somatic mutations | Ghandi et al., Nature 2019 |
| PDB | Crystal structures (EGFR + HER2) | wwPDB Consortium |
| UniProt | PTM site annotations | UniProt Consortium 2025 |
| DrugPTM-Bench | Drug→PTM phosphoproteomics | Badkul et al., 2026 |
| Tozuka 2024, Hsu 2025, PNAS 2025, MCP 2025, Cancer Res 2021, FEBS 2025, Ruprecht 2017 | Site-level phospho + glyco quantitation | See [`config/config.yaml`](config/config.yaml) |

### Key Findings

- **Cross-receptor homology**: The model independently discovers that EGFR Y1068 ≡ HER2 Y1221 (both GRB2→RAS-MAPK docking sites) — learning biological **function**, not protein identity
- **Tissue-specific pathway discovery**: EGFR resistance is MAPK-driven (Y1068 top site) while HER2 resistance is PI3K-AKT-driven (Y1248 top site) — the model learns tissue-specific pathway hierarchies without explicit supervision
- **Reproducible attributions**: Integrated Gradients across 3 seeds produces identical site rankings (std_rank = 0.0 for top sites)
- **Cross-type attention**: Non-trivial phospho↔glyco off-diagonal attention mass — the model USES crosstalk between intracellular signaling and receptor surface biology

---

## 3. Quick Start

### Installation

#### Option A: pip (local development)

```bash
git clone https://github.com/Moustafazn/PTM-BDL-Framework.git
cd PTM-BDL-Framework

python -m venv .venv
source .venv/bin/activate  # Linux/macOS

pip install -e ".[structural]"

# For GPU acceleration (optional)
pip install -e ".[gpu]"
```

#### Option B: Docker Compose (reproducible — recommended for reviewers)

```bash
git clone https://github.com/Moustafazn/PTM-BDL-Framework.git
cd PTM-BDL-Framework

docker compose up                         # Full pipeline
docker compose --profile train up         # Training + ablation + CV
docker compose --profile benchmark up     # Benchmarking suite
docker compose --profile figures up       # Publication figures & tables
docker compose --profile test up          # Test suite
docker compose run --rm shell             # Interactive shell
docker compose --profile gpu up           # GPU support (NVIDIA)
```

### Data Download

All raw data files are available as a single archive:

> **📦 [Download all raw data](https://github.com/Moustafazn/PTM-BDL-Framework/releases)**
>
> ```bash
> tar -xzf data_raw.tar.gz
> ```
> This creates the full `data/raw/` directory with all required files.

For manual download instructions, see the [Data Download Guide](docs/DATA_DOWNLOAD_GUIDE.md).

### Running the Full Pipeline

```bash
make data         # Phase 1: Data Acquisition (Steps 01–05)
make harmonize    # Phase 2: Data Harmonization (Step 06)
make features     # Phase 3: Feature Extraction (Steps 07–09)
make train        # Phase 4: Training & Evaluation (Steps 10–13)
make benchmark    # Phase 5: Benchmarking (Steps 14a–d)
make figures      # Phase 6: Publication Figures & Tables (Steps 15a–b)

# Or run everything end-to-end:
make all
```

<details>
<summary><b>Individual Steps (click to expand)</b></summary>

```bash
# ── Data Acquisition ──
python scripts/step01_download_gdsc.py       # GDSC IC50 drug response data
python scripts/step02_download_mutations.py   # EGFR + ERBB2 mutation profiles
python scripts/step03_download_structures.py  # PDB crystal structures
python scripts/step04_download_ptm_data.py    # PTM site annotations (UniProt)
python scripts/step05_download_drugptm.py     # Drug-induced PTM changes (8 studies)

# ── Harmonization ──
python scripts/step06_harmonize_dataset.py    # → multimodal_dataset.csv (951 samples)

# ── Feature Extraction ──
python scripts/step07_extract_esm2.py         # ESM-2 protein language model
python scripts/step08_extract_gearnet.py      # GearNet structural encoder
python scripts/step09_extract_chemberta.py    # ChemBERTa drug encoder

# ── Training & Analysis ──
python scripts/step10_build_model.py          # Verify model architecture
python scripts/step11_train.py                # Train multimodal model
python scripts/step11b_ablation.py            # Ablation + stability + randomized control
python scripts/step11c_crossval.py            # 5-fold cross-validation
python scripts/step12_evaluate.py             # Comprehensive evaluation
python scripts/step13_explainability.py       # XAI: IG + attention analysis

# ── Benchmarking ──
python scripts/step14a_ml_baselines.py        # ML baselines (RF, XGBoost, Ridge, Elastic Net)
python scripts/step14b_external_baselines.py  # External DRP methods (DIPK, HiDRA, etc.)
python scripts/step14c_statistical_tests.py   # Bootstrap CIs, DeLong, Wilcoxon, BH correction
python scripts/step14d_loclo.py               # Cell-blind LOCLO generalization

# ── Publication Outputs ──
python scripts/step15a_paper_figures.py       # Camera-ready figures (PDF, 300 DPI)
python scripts/step15b_paper_tables.py        # LaTeX + CSV tables
```
</details>

---

## 4. Model Components

### 4.1 Multimodal Input Encoders

| Modality | Pretrained Model | Output |
|----------|-----------------|--------|
| Protein sequence | [ESM-2](https://github.com/facebookresearch/esm) (650M params) | Per-residue embeddings (L × 1280) |
| 3D structure | [GearNet-Edge](https://github.com/DeepGraphLearning/GearNet) | Per-residue structural features (M × 512) |
| Drug chemistry | [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR) | Per-token + pooled drug features (N × 384) |

### 4.2 Joint Cross-Modal Self-Attention

All modality tokens are projected to a shared 512-dimensional space and concatenated. A 4-layer, 8-head Transformer encoder performs **cross-modal self-attention** — sequence residues attend to structural features, structural features attend to drug atoms, and vice versa. Attention pooling (Ilse et al., ICML 2018) produces the static representation **S_rep**.

### 4.3 PTM Biological Dynamics Layer (PTM-BDL)

The core architectural contribution — a typed self-attention module that encodes the dynamic PTM signaling state:

1. **Value projection (§7.4)**: Each PTM site → `[level, delta, ratio]` → linear projection → d_model
2. **Type-gated projection (§7.5)**: `gate = σ(W·[projected; type_emb])`, then `token = gate ⊙ projected` — the modification subtype controls which information passes through
3. **Embeddings**: `token += type_emb + protein_emb + slot_emb` — each token knows its PTM subtype, which protein it belongs to, and its positional slot
4. **Typed self-attention (§7.6)**: Standard Transformer encoder with padding-aware masking — PTM sites attend to each other, discovering inter-site signaling dependencies
5. **Residual gate (§7.7)**: `α = σ(W·[attended; pre_attn])`, then `out = α·attended + (1−α)·pre_attn` — sites that don't benefit from inter-site context keep their independent representation
6. **Mask-aware mean pool**: Only real (non-padded) tokens contribute to the final **P_rep**

### 4.4 Bilinear Late Fusion

S_rep (static: "what is the protein-drug system?") and P_rep (dynamic: "what does the PTM state say?") are fused via element-wise bilinear interaction: `tanh(W_s · S_rep) ⊙ tanh(W_p · P_rep)`.

### 4.5 Prediction Heads

- **Regression**: Predicts ln(IC50) drug sensitivity
- **Classification**: Predicts P(resistance) via focal loss with class-conditional α

### Training Strategy

- **Multi-task loss**: λ₁·Huber(IC50) + λ₂·FocalLoss(resistance), λ₂=2.0 to boost classification signal
- **Class imbalance**: WeightedRandomSampler + focal loss (α=0.25, γ=2.0) → 3× up-weight on minority
- **Early stopping**: On max(AUROC, BAcc) — threshold-independent for 92:8 imbalanced data
- **Regularization**: Gradient clipping (1.0), cosine annealing LR, dropout (0.1)

> For the detailed mathematical formulation, see [`docs/ARCHITECTURE.md`](docs/PTM_Biological_Dynamics_Layer.md).

---

## 5. Configuration

All parameters are centralized in [`config/config.yaml`](config/config.yaml):

### Model Architecture

```yaml
model:
  shared_dim: 512                  # Shared embedding dimension across modalities
  num_joint_attention_layers: 4    # Cross-modal self-attention depth
  num_attention_heads: 8           # Attention heads per layer
  dropout: 0.1
  learning_rate: 1.0e-4
  batch_size: 16
  num_epochs: 100
  early_stopping_patience: 15

ptm_bdl:
  d_model: 64       # PTM token embedding dimension
  n_heads: 4        # Self-attention heads in PTM-BDL
  n_layers: 2       # Transformer layers in PTM-BDL

training:
  device: "cpu"      # "cpu", "cuda", "mps", "auto"
  seed: 42
  train_ratio: 0.7
  val_ratio: 0.15
```

### PTM Site Definitions

The config defines per-protein PTM sites with amino acid types that determine subtype embeddings:

```yaml
ptm:
  ptm_dim: 12       # Phospho sites per protein
  glyco_dim: 12     # Glyco sites per protein

  EGFR:
    phospho_sites:
      - {position: 869,  residue: "Y869",  amino_acid: "Y", function: "SRC substrate"}
      - {position: 991,  residue: "S991",  amino_acid: "S", function: "regulatory"}
      # ... 12 sites total
    glyco_sites:
      - {position: 56,   residue: "N56",   amino_acid: "N", function: "domain I"}
      # ... 12 sites total

  ERBB2:
    phospho_sites:    # 10 real + 2 zero-padded to match ptm_dim=12
      - {position: 686,  residue: "T686",  amino_acid: "T", function: "regulatory"}
      # ... 10 real sites
    glyco_sites:      # 7 real + 5 zero-padded to match glyco_dim=12
      # ...
```

### Drug Definitions

```yaml
drugs:
  osimertinib:
    name: "Osimertinib"
    smiles: "C=CC(=O)Nc1cc(OC)c(Nc2nccc(-c3cn(C)c4ccccc34)n2)cc1N(C)CCN(C)C"
    generation: "3rd"
    binding_type: "covalent (C797)"
  # ... 6 drugs total
```

> The full config includes phospho propagation rules, HER2 amplification tiers, per-cell-line PTM modulators, and drug-protein mappings. See [`config/config.yaml`](config/config.yaml).

---

## 6. Extensibility

The PTM-BDL architecture supports extension to new proteins, PTM types, and drugs. The typed token system, protein embeddings, and padding masks are designed to be configurable.

### Adding a New Protein

To add a third protein (e.g., BRAF), you would:

1. Add site definitions in `config/config.yaml` under `ptm.BRAF`
2. Add protein ID constant and buffer arrays in the model
3. Add a PDB structure for GearNet
4. Provide PTM quantitation data

The architecture handles variable padding (different proteins can have different numbers of real vs. padded sites) through the existing `is_real_table` mechanism.

### Adding a New PTM Type

To add acetylation alongside phospho and glyco:

1. Define new subtype IDs (e.g., `acetyl_K = 4`)
2. Increment `N_PTM_TYPES` (4 → 5)
3. Add acetylation site data to the config

The type embedding table (`nn.Embedding(N_PTM_TYPES, d_model)`) scales automatically. The self-attention mechanism, type gate, and residual gate all work unchanged — they operate on abstract typed tokens, not specific modification types.

### Adding a New Drug

Add the drug's SMILES string to `config/config.yaml` under `drugs`, add its GDSC drug ID, and run the feature extraction pipeline.

---

## 7. Evaluation & Ablation

### Ablation Study (5 Arms)

| Model | Description | What it tests |
|-------|-------------|---------------|
| A: No PTM | All PTM features zeroed | Is PTM signal needed at all? |
| E: No glyco | Phospho only | Glycosylation marginal value |
| F: Glyco only | Glyco only | Phosphorylation marginal value |
| G: No typed attention | MLP replaces self-attention in PTM-BDL | Do inter-site dependencies matter? |
| D: Full model | All features + typed self-attention | Production model |

### Validation Protocol

- **Randomized PTM control**: Inference-only permutation test (Breiman 2001, Fisher et al. 2019) — shuffle PTM features at test time, same trained model. If real PTM carries signal, shuffled must perform worse on AUROC and AUPRC-sensitive.
- **Multi-seed stability**: Train 3× with different seeds, run Integrated Gradients on each → tests whether site importance rankings are reproducible across initializations.
- **Cross-receptor homology**: EGFR Y1068 and HER2 Y1221 are homologous GRB2-docking sites. Both should independently rank as top effector sites if the model learns biological function rather than protein identity.

---

## 8. Benchmarking

Comprehensive benchmarking suite following Nature Methods 2026 standards:

```bash
make benchmark    # Run ML baselines + external methods + statistical tests + cell-blind split
make figures      # Generate publication-quality figures + tables (PDF + LaTeX)
```

### Methods Compared

| Tier | Methods | Purpose |
|------|---------|---------|
| Tier 0 | RF, XGBoost, Ridge, Elastic Net | If simple ML matches DL, architecture adds no value |
| Tier 1 | DIPK (2024), HiDRA (2023), GraTransDRP (2023), TransCDR (2023), PathDSP (2024) | Recent state-of-the-art DRP methods |
| Tier 2 | GraphDRP (2022), DrugCell (2020), DeepCDR (2020) | Established baselines |

### Statistical Rigor

- Bootstrap 95% confidence intervals (1,000 resamples)
- DeLong paired AUROC tests (PTM-BDL vs. each baseline)
- Wilcoxon signed-rank tests on per-drug comparisons
- Benjamini-Hochberg correction across K baselines
- Cell-blind LOCLO generalization (leave-one-mutation-class-out)

### Benchmarking Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `step14a_ml_baselines.py` | Tier 0 ML baselines on same 2224-d features | `results/ml_baselines.json` |
| `step14b_external_baselines.py` | Tier 1–2 external DRP methods | `results/external_baselines/` |
| `step14c_statistical_tests.py` | Bootstrap CIs, DeLong, Wilcoxon, BH | `results/statistical_tests.json` |
| `step14d_loclo.py` | Cell-blind LOCLO generalization | `results/loclo_results.json` |
| `step15a_paper_figures.py` | Camera-ready figures (300 DPI, colorblind-friendly) | `results/publication/figures/*.pdf` |
| `step15b_paper_tables.py` | LaTeX + CSV tables | `results/publication/tables/*.tex` |

---

## 9. Explainability (XAI)

The framework provides three complementary XAI analyses:

### Per-Mod-Type Integrated Gradients

IG attributions bucketed by modification subtype (phospho_Y, phospho_S, phospho_T, glyco_N) and partitioned per protein. Integrates along all 4 input channels simultaneously (level + delta for both phospho and glyco). Per-site importance = |grad_level × Δlevel| + |grad_delta × Δdelta|.

Reference: Sundararajan, Taly & Yan, "Axiomatic Attribution for Deep Networks", ICML 2017.

### Cross-Type Attention Analysis

Post-softmax attention weights from the final PTM-BDL transformer layer, decomposed into four quadrants: phospho→phospho, phospho→glyco, glyco→phospho, glyco→glyco. Non-trivial off-diagonal (phospho↔glyco) attention mass is evidence that the model uses crosstalk between the two PTM types.

### Cross-Receptor Homology Check

Two independent biological validations:
- **Phospho-Y**: EGFR Y1068 (precursor Y1092) ≡ ERBB2 Y1221 — both are GRB2 docking sites driving RAS-MAPK signaling. If the model learns function (not protein identity), both should rank as top effector sites.
- **Glyco-N**: EGFR N528 ↔ ERBB2 N530 — extracellular domain IV membrane-proximal anchors (ERBB2 site overlaps trastuzumab-binding interface).

---

## 10. Project Structure

```
PTM-BDL-Framework/
├── config/
│   └── config.yaml                 # All parameters: PTM sites, drugs, model, training
├── scripts/
│   ├── step01–05_*.py              # Data acquisition (GDSC, CCLE, PDB, UniProt, DrugPTM)
│   ├── step06_harmonize_dataset.py # Data harmonization → 951-sample dataset
│   ├── step07–09_*.py              # Feature extraction (ESM-2, GearNet, ChemBERTa)
│   ├── step10_build_model.py       # Architecture verification
│   ├── step11_train.py             # Training pipeline
│   ├── step11b_ablation.py         # 5-arm ablation + stability + randomized control
│   ├── step11c_crossval.py         # 5-fold cross-validation
│   ├── step12_evaluate.py          # Per-protein, per-drug, mutation-stratified evaluation
│   ├── step13_explainability.py    # IG attributions + cross-type attention + homology
│   ├── step14a–d_*.py              # Benchmarking (ML baselines, external, stats, LOCLO)
│   └── step15a–b_*.py              # Publication figures & tables
├── src/models/
│   └── multimodal_predictor.py     # Core model: PTMBDLEncoder, StaticJointTransformer,
│                                   #   BilinearLateFusion, MultimodalResistancePredictor
├── results/                        # Evaluation outputs (JSON + figures)
│   └── publication/                # Camera-ready figures (PDF) + tables (CSV/LaTeX)
├── tests/                          # pytest test suite
├── docs/                           # Technical documentation
│   ├── PTM_Biological_Dynamics_Layer.md  # Detailed architecture with equations
│   ├── PAPER_REFERENCES.md               # Bibliography
│   └── DATA_DOWNLOAD_GUIDE.md            # Manual data download instructions
├── data/                           # Created by pipeline (gitignored, ~1 GB)
├── benchmarks/                     # Cloned external method repos (gitignored)
├── Makefile                        # Pipeline automation
├── pyproject.toml                  # Dependencies
├── Dockerfile                      # Reproducible environment
├── docker-compose.yml              # Multi-profile orchestration
├── LICENSE                         # MIT License
└── CITATION.cff                    # Citation metadata
```

---

## 11. Dependencies

- Python 3.11 (required by torch-geometric)
- PyTorch 2.1+
- transformers 5.11+ (for ESM-2, ChemBERTa)
- fair-esm 2.0+ (for ESM-2)
- torch-geometric 2.8+ (optional, for GearNet)
- biotite 1.6+ (for PDB parsing)

See `pyproject.toml` and `requirements.txt` for the full list.

---

## 12. Authors

- **Moustafa Zein** — Lead developer
- **Prof. Aboul Ella Hassanien** — Co-author

---

## 13. Citation

```bibtex
@software{zein2026ptmbdl,
  title     = {Multimodal Self-Attention with {PTM} Biological Dynamics Layer:
               A {PTM} Foundational Framework for Drug Response Prediction},
  author    = {Zein, Moustafa and Hassanien, Aboul Ella},
  year      = {2026},
  url       = {https://github.com/Moustafazn/PTM-BDL-Framework},
}
```

---

## 14. License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
