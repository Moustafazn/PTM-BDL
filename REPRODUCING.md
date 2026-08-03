# Reproducing Results — Code Ocean Guide for Reviewers

## Quick Start (Default: Evaluation Only)

Click **"Reproducible Run"** → the capsule evaluates all 3 case studies using pre-trained models
and generates the paper's figures, tables, and statistical tests.

This is the **default mode** because pre-trained model checkpoints are included in the `data/` folder.
The pipeline automatically detects what data is available and runs the appropriate stages.

---

## What the Default Run Produces

| Output | Description |
|--------|-------------|
| `test_results.txt` | Framework test suite results (verifies code correctness) |
| `egfr_evaluate.txt` | EGFR/ERBB2 TKI resistance — evaluation metrics (AUROC, BAcc, R², etc.) |
| `hela_evaluate.txt` | HeLa/HDAC inhibitors — evaluation metrics |
| `k562_evaluate.txt` | K562/CML BCR-ABL — evaluation metrics |
| `*_explain.txt` | XAI results (Integrated Gradients + attention analysis) |
| `*_baselines.txt` | ML baseline comparisons (XGBoost, Random Forest, etc.) |
| `*_statistics.txt` | Bootstrap confidence intervals and statistical tests |
| `*_figures.txt` | Publication figures (saved as PNG/PDF in results/) |
| `*_tables.txt` | Publication tables (LaTeX + CSV) |

---

## Running the Full Pipeline from Scratch

The capsule supports 3 execution modes depending on what data is available:

### Mode 1: Evaluation Only (default)
**When**: Pre-trained models exist in `data/models/`
**Runs**: evaluate → explain → baselines → statistics → figures → tables

### Mode 2: Train + Evaluate
**When**: Processed data exists but NO pre-trained models
**How to trigger**: Remove the `models/` folder from the capsule's `data/` directory
**Runs**: train → ablation → crossval → evaluate → explain → figures

### Mode 3: Full Pipeline
**When**: Only raw data exists (no processed data or models)
**How to trigger**: Remove both `models/` AND `processed/` from `data/`
**Runs**: download → harmonize → features (ESM-2, GearNet, ChemBERTa) → train → evaluate → benchmark → figures

---

## Data Structure

```
data/
├── raw/          → Raw input files (GDSC, PDB structures, PTM annotations, DrugPTM)
├── processed/    → Harmonized multimodal datasets (Step 06 output)
├── features/     → Pre-computed embeddings:
│   ├── esm2/         ESM-2 protein sequence embeddings (Step 07)
│   ├── gearnet/      Structural embeddings via ESM-IF1 (Step 08)
│   └── chemberta/    ChemBERTa drug embeddings (Step 09)
└── models/       → Trained model checkpoints:
    ├── egfr_erbb2_tki/   CS1: EGFR/ERBB2 TKI Resistance
    ├── hela_hdac/        CS2: HeLa/HDAC Inhibitors
    └── k562_cml/         CS3: K562/CML BCR-ABL
```

## Case Studies

| # | Case Study | Proteins | PTM Types | Drugs | Cancer |
|---|-----------|----------|-----------|-------|--------|
| 1 | EGFR/ERBB2 TKI Resistance | EGFR, HER2 | phospho (Y/S/T) + glyco (N) | 6 TKIs | NSCLC + Breast |
| 2 | HeLa/HDAC Inhibitors | HDAC1, EP300 | phospho (S/T/Y) + acetyl (K) | 6 drugs | Cervical |
| 3 | K562/CML BCR-ABL | ABL1, CRKL, STAT5A | phospho (S/T/Y) | 5 drugs | CML |

## Environment

- Python 3.11 (via Miniconda)
- PyTorch 2.12.0
- Transformers 5.10.2
- ESM-2 (facebook/esm2_t33_650M_UR50D) — protein language model
- ESM-IF1 — structural encoder
- ChemBERTa (DeepChem/ChemBERTa-77M-MTR) — drug encoder
- PyTorch Geometric 2.8.0 — GearNet structural GNN

## Citation

```bibtex
@software{zein2026ptmbdl,
  title  = {Multimodal Self-Attention with {PTM} Biological Dynamics Layer:
            A {PTM} Framework for Drug Response Prediction},
  author = {Zein, Moustafa and Hassanien, Aboul Ella},
  year   = {2026},
  url    = {https://github.com/Moustafazn/PTM-BDL-Framework},
}
```
