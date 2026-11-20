# PTM-BDL: A site-resolved post-translational modification framework for drug-response analysis

PTM-BDL is a research framework for evaluating the contribution of post-translational modification profiles to drug response and interpreting predictions at individual modification sites. This repository accompanies the manuscript **A site-resolved post-translational modification framework for drug-response analysis**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## Scientific contribution

PTM states describe regulatory information that fixed protein and compound features do not specify, motivating a framework that evaluates their contribution to response prediction while preserving protein, modification type, site identity, baseline state and treatment-associated change for interpretation.

Retraining models with and without PTM inputs assesses their additional predictive contribution, while integrated gradients and input-replacement analyses describe the reliance of a fitted model on particular inputs. Repeated-seed analysis then assesses the stability of site rankings, connecting evaluation of the PTM panel with selection of candidate pharmacodynamic measurements for independent experimental validation.

### Main findings

| Case study | Internal AUROC | Mean AUROC gain after retraining with PTM inputs | Folds with positive gain |
| --- | ---: | ---: | ---: |
| K562/CML | 0.785 | 0.0268 | 5 of 5 |
| HeLa/HDAC | 0.946 | 0.0011 | 3 of 5 |
| EGFR/ERBB2 | 0.875 | 0.0057 | 3 of 5 |

The internal AUROCs describe the recorded context splits, while the gains come from separate five-fold comparisons of Full and No-PTM models, with both sets of values taken directly from the saved evaluation and retraining JSON files in `results/`. The biologically distinct case studies showed a consistent incremental contribution in K562/CML and more modest incremental gains in HeLa/HDAC and EGFR/ERBB2.

Within K562/CML, CRKL Y207 and STAT5A Y694 retained the leading attribution ranks for their proteins across three seeds, while greater PTM attribution in dasatinib than in cytotoxic drug contexts identified a focused setting for pharmacodynamic investigation. These findings motivate matched measurements under defined drug concentrations and exposure times, with the less stable rankings in the other systems indicating that site prioritisation depends on the evaluated context.

Classical models remained competitive or better in the predictive comparisons, and the internal splits include repeated or related endpoint contexts. The separate leave-one-drug-out evaluations showed limited performance on drugs excluded during training, which addresses a different question from applying the framework across the three biological settings. Because PTM inputs combine exact source matches where available with propagated profiles, attributions and cross-type attention describe fitted-model behaviour, supporting retrospective hypothesis generation and prioritisation of experiments to examine candidate PTM effects.

## Architecture

<p align="center">
  <img src="docs/figures/ptm_bdl_site_framework.png" alt="PTM-BDL architecture showing molecular inputs, typed PTM site representation, response prediction and site attribution" width="100%">
</p>

**Figure 1.** PTM-BDL combines fixed sequence, structure-derived and chemical features with a separate PTM representation. Sequence and chemical features are obtained from pretrained ESM-2 and ChemBERTa encoders. The reported structural features come from a graph encoder without pretrained weights. Each PTM token retains its protein, site and modification type together with baseline, treatment-associated change and relative-change values. Type-conditioned projection, masked self-attention and gated pooling form the PTM representation. Bilinear late fusion combines the molecular and PTM representations for lnIC50 and resistance prediction. Integrated gradients quantify contributions of the configured site inputs to fitted resistance predictions.

### Case studies

| Case study | Proteins | PTM channels | Drugs | Model contexts | Cell–drug endpoints |
| --- | --- | --- | --- | ---: | ---: |
| K562/CML | ABL1, CRKL, STAT5A | Phosphorylation | Two kinase inhibitors and three cytotoxic drugs | 10,656 | 3,552 |
| HeLa/HDAC | HDAC1, EP300 | Phosphorylation and acetylation | Six epigenetic modulators | 3,374 | 1,687 |
| EGFR/ERBB2 | EGFR, ERBB2 | Phosphorylation and glycosylation | Six ERBB-family inhibitors | 951 | 951 |

Contexts are cell-line–drug–target-protein rows. HeLa/HDAC and K562/CML expand cell–drug endpoints over two and three proteins, respectively. These are the processed modelling cohorts reported in the manuscript. The three case studies characterise the framework in different biological settings, with generalisation assessed separately using held-out cells and drugs.

## Manuscript and results

The current submission materials are maintained in the local `paper/` directory. The previous `manuscript/` directory is preserved as an archive. Both directories are excluded from version control. The current manuscript is a draft and has not been published as a journal article.

The saved JSON outputs in `results/<case_study>/` are the numerical sources for the manuscript tables, quantitative figures and reported results. Editorial revisions do not alter these outputs. The local plotting helper `paper/scripts/render_result_figures.py` reads the original JSON files to generate the manuscript attribution and supplementary calibration figures. Existing case-study plotting commands below remain available for the wider analysis outputs.

The final processed cohorts, fixed feature arrays, saved production splits,
classification thresholds, and numerical result records are versioned with the
repository. The three production checkpoints are tracked through Git LFS under
`data/models/<case_study>/best_model.pt`. Raw source downloads and
non-production checkpoints remain excluded; acquisition and harmonisation
instructions are provided in the [data download guide](docs/DATA_DOWNLOAD_GUIDE.md).

## Project Structure

```
PTM-BDL/
│
├── src/ptm_bdl/                        # CORE tool (protein-agnostic)
│   ├── registry.py                     # Config-driven PTM type/subtype system
│   ├── model/                          # encoder, ablation, static, fusion, predictor
│   ├── data/                           # dataset, collate, splits
│   ├── training/                       # loss, trainer, metrics, factory, checkpoint, device, sampler
│   ├── evaluation/                     # evaluator, baselines, statistical, loclo, loader
│   └── xai/                            # integrated_gradients, attention, homology
│
├── src/case_studies/                   # CASE STUDY INSTANCES
│   ├── common/                         # Shared data pipeline utilities (GDSC, structures, PTM)
│   ├── egfr_erbb2_tki/                # CS1: EGFR/ERBB2 TKI Resistance (complete)
│   │   ├── biology.py                  # Application-specific biological constants
│   │   ├── data_pipeline/              # Steps 01-06: data acquisition + harmonization
│   │   ├── features/                   # Steps 07-09: ESM-2, GearNet, ChemBERTa
│   │   └── scripts/                    # Steps 10-15: train, evaluate, explain, benchmark
│   ├── hela_hdac/                      # CS2: HeLa/HDAC Inhibitors (complete)
│   │   ├── biology.py                  # HDAC/HAT drug classifications + PDB structures
│   │   ├── data_pipeline/              # Steps 01-06 (phospho + NEW acetyl_K PTM type)
│   │   ├── features/                   # Steps 07-09: ESM-2, GearNet, ChemBERTa
│   │   └── scripts/                    # train, evaluate, explain, ablation, crossval, etc.
│   └── k562_cml/                       # CS3: K562/CML BCR-ABL (complete)
│       ├── biology.py                  # BCR-ABL substrates + TKI/chemo classifications
│       ├── data_pipeline/              # Steps 01-06 (5 drugs: 2 TKI + 3 chemo)
│       ├── features/                   # Steps 07-09: ESM-2, GearNet, ChemBERTa
│       └── scripts/                    # train, evaluate, explain, ablation, crossval, etc.
│
├── tests/                              # pytest test suite
├── docs/                               # Architecture docs, evaluation, data guide
└── results/                            # Evaluation outputs + publication figures
```

---

## Quick Start

### Prerequisites

- **Python 3.11** (required — tested with 3.11.9; must be ≥ 3.11 and < 3.13)
- **Git**
- **~4 GB disk space** for pretrained model weights (downloaded on first run)
- **~1 GB disk space** for raw data files (see [Data Download](#4-download-raw-data) below)
- **HuggingFace account** — Steps 07 and 09 download pretrained models (ESM-2, ChemBERTa) from HuggingFace Hub

### 1. Clone the Repository

```bash
git clone https://github.com/Moustafazn/PTM-BDL.git
cd PTM-BDL
```

### 2. Create Virtual Environment (Python 3.11)

```bash
# Create venv with Python 3.11
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
# Install all core dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (requires matching torch version — must be installed separately)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.12.0+cpu.html

# Install the project in editable mode
pip install -e ".[dev]"
```

> **Why is `torch-geometric` installed separately?**
> PyTorch Geometric requires a wheel that matches your exact PyTorch version and platform.
> It cannot be installed via a standard `pip install -r requirements.txt` — the `-f` flag
> points pip to the correct version-matched wheel index.
>
> **If `pip install -r requirements.txt` tries to upgrade torch and times out**, install
> the missing packages individually instead:
> ```bash
> pip install fair-esm biotite xgboost
> pip install torch-geometric -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
> ```

### 4. Download Raw Data

Raw data files must be downloaded manually before running the pipeline.
See [`docs/DATA_DOWNLOAD_GUIDE.md`](docs/DATA_DOWNLOAD_GUIDE.md) for complete instructions.

**Required data sources:**

| Source | Files | Size | Place In |
|--------|-------|------|----------|
| [GDSC2 (Sanger)](https://cellmodelpassports.sanger.ac.uk/downloads) | IC50 dose-response + model list | ~22 MB | `data/raw/gdsc/` |
| [DepMap (Broad)](https://depmap.org/portal/data_page/?tab=allData) | Somatic mutations + model info | ~900 MB | `data/raw/ccle/` |
| [UniProt](https://www.uniprot.org) | EGFR/HER2 FASTA + PTM JSON | ~1 MB | `data/raw/ccle/`, `data/raw/ptm/` |
| [RCSB PDB](https://www.rcsb.org) | Crystal structures (11 PDB files) | ~10 MB | `data/raw/pdb/` |
| [DrugPTM-Bench](https://github.com/Xie-lab/DrugPTM-Bench) | Cell line phosphoproteomics | ~3 GB | `data/raw/drugptm/` |

### 5. HuggingFace Setup (for pretrained model downloads)

Steps 07 (ESM-2) and 09 (ChemBERTa) download pretrained models from HuggingFace Hub.
To avoid rate limits and enable faster downloads:

1. **Create a HuggingFace account**: https://huggingface.co/join
2. **Accept model licenses** (visit each page and click "Agree"):
   - ESM-2: https://huggingface.co/facebook/esm2_t33_650M_UR50D
   - ChemBERTa: https://huggingface.co/DeepChem/ChemBERTa-77M-MTR (public, no gate)
3. **Create an access token**: https://huggingface.co/settings/tokens → "New token" → "Read"
4. **Login** (run once):
   ```bash
   hf auth login
   # Paste your token when prompted
   ```

> **Note**: If you skip this step, the pipeline will still run but with unauthenticated
> requests (slower downloads, rate limits). Steps 07 and 09 will NOT crash.

### 6. Pre-Download Pretrained Models (optional but recommended)

Steps 07 and 09 automatically download models on first run, but you can pre-cache them
to avoid delays during pipeline execution:

```bash
# Pre-download ESM-2 (protein language model, ~2.5 GB)
hf download facebook/esm2_t33_650M_UR50D

# Pre-download ChemBERTa (drug embedding model, ~300 MB)
hf download DeepChem/ChemBERTa-77M-MTR
```


> The structural extraction scripts support ESM-IF1 and graph-feature fallbacks, depending on the available packages. The results reported in the manuscript use fixed features from graph encoders without pretrained weights. Running a different backend produces different input features and does not reproduce the recorded evaluation unchanged.

### Docker (recommended for reproducibility)

```bash
docker compose up                        # Run ALL case studies
docker compose --profile egfr up         # EGFR/ERBB2 only
docker compose --profile hela up         # HeLa/HDAC only
docker compose --profile k562 up         # K562/CML only
docker compose --profile test up         # Test suite
docker compose run shell                 # Interactive shell
```

---

## Running Case Studies

### Using Make (simplest)

```bash
make egfr        # CS1: EGFR/ERBB2 TKI Resistance (NSCLC + breast)
make hela        # CS2: HeLa/HDAC Inhibitors (phospho + acetyl)
make k562        # CS3: K562/CML BCR-ABL (TKI + chemo)
make all         # All three case studies
```

Run individual phases for a specific case study:

```bash
make data CASE=hela         # Steps 01-05: download + process data
make harmonize CASE=hela    # Step 06: build multimodal dataset
make features CASE=hela     # Steps 07-09: extract ESM-2, GearNet, ChemBERTa
make train CASE=hela        # Train + ablation + cross-validation
make evaluate CASE=hela     # Evaluation + XAI
make benchmark CASE=hela    # ML baselines + external methods + LOCLO
make figures CASE=hela      # Publication figures + tables
```

### Running Step-by-Step

Each case study has its own complete pipeline. Replace `<cs>` with the case study module name:
- `egfr_erbb2_tki` — EGFR/ERBB2 TKI resistance
- `hela_hdac` — HeLa/HDAC inhibitors
- `k562_cml` — K562/CML BCR-ABL

**Data Pipeline (Steps 01-06):**

```bash
python -m src.case_studies.<cs>.data_pipeline.step01_download_gdsc
python -m src.case_studies.<cs>.data_pipeline.step02_download_sequences
python -m src.case_studies.<cs>.data_pipeline.step03_download_structures
python -m src.case_studies.<cs>.data_pipeline.step04_download_ptm_data
python -m src.case_studies.<cs>.data_pipeline.step05_download_drugptm
python -m src.case_studies.<cs>.data_pipeline.step06_harmonize_dataset
```

**Feature Extraction (Steps 07-09):**

```bash
python -m src.case_studies.<cs>.features.step07_extract_esm2       # ESM-2 protein embeddings (uses transformers)
python -m src.case_studies.<cs>.features.step08_extract_gearnet    # Structural embeddings (uses fair-esm/biotite/PyG)
python -m src.case_studies.<cs>.features.step09_extract_chemberta  # ChemBERTa drug embeddings (uses transformers)
```

**Analysis & Evaluation:**

```bash
python -m src.case_studies.<cs>.scripts.train              # Train PTM-BDL model
python -m src.case_studies.<cs>.scripts.evaluate           # Comprehensive evaluation
python -m src.case_studies.<cs>.scripts.explain            # XAI (IG + attention)
python -m src.case_studies.<cs>.scripts.ablation           # Modality ablation study
python -m src.case_studies.<cs>.scripts.crossval           # K-fold cross-validation
python -m src.case_studies.<cs>.scripts.ml_baselines       # ML baseline comparison
python -m src.case_studies.<cs>.scripts.external_baselines # Published methods
python -m src.case_studies.<cs>.scripts.statistical_tests  # Bootstrap CIs
python -m src.case_studies.<cs>.scripts.loclo              # LOCLO generalization
python -m src.case_studies.<cs>.scripts.paper_figures      # Publication figures
python -m src.case_studies.<cs>.scripts.paper_tables       # Publication tables
```

---

## Using PTM-BDL as a Package

PTM-BDL can be installed directly from this source repository.

### Installation

```bash
pip install git+https://github.com/Moustafazn/PTM-BDL.git
```

### Model Input/Output Specification

**Inputs** (all tensors, batch dimension B):

| Input                | Shape              | Type    | Description                                       |
|----------------------|--------------------|---------|---------------------------------------------------|
| `seq_embeddings`     | (B, L, 1280)       | float32 | ESM-2 per-residue protein embeddings              |
| `struct_embeddings`  | (B, M, 512)        | float32 | Fixed structure-derived graph features         |
| `drug_pooled`        | (B, 384)           | float32 | ChemBERTa pooled drug embedding                   |
| `drug_embeddings`    | (B, N, 384)        | float32 | ChemBERTa per-token drug embeddings (optional)    |
| `ptm_vector`              | (B, n_tokens)          | float32 | Baseline PTM state (all types concatenated, 1.0 is the model reference state) |
| `delta_ptm_vector`        | (B, n_tokens)          | float32 | Treatment-associated PTM change (all types concatenated, 0.0 is the no-change reference)  |
| `target_protein`          | (B,)                   | long    | Protein ID index (0, 1, 2, ...)                           |

**Outputs**:

| Output              | Shape  | Type    | Description                                       |
|---------------------|--------|---------|---------------------------------------------------|
| `ic50_pred`         | (B, 1) | float32 | Predicted ln(IC50) drug sensitivity               |
| `resistance_logits` | (B, 1) | float32 | Resistance logits (apply sigmoid for probability) |

### Minimal Example — Build, Train, Predict

```python
import torch
import yaml
from src.ptm_bdl.training import build_model_from_cfg, FocalLoss, train_epoch, validate
from src.ptm_bdl.evaluation.evaluator import collect_predictions, compute_full_metrics

# 1. Load config (tool + case study settings)
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Build model — all PTM types, subtypes, and embeddings auto-configured
model = build_model_from_cfg(cfg)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# 3. Prepare a single sample (example dimensions)
#    All PTM types are concatenated into a single flat ptm_vector.
#    E.g., 12 phospho + 12 glyco = 24 tokens total.
n_tokens = 24  # from registry.n_tokens
sample = {
    "seq_emb": torch.randn(1, 100, 1280),       # ESM-2 protein embeddings
    "struct_emb": torch.randn(1, 80, 512),       # Fixed structure-derived features
    "drug_pooled": torch.randn(1, 384),           # ChemBERTa pooled
    "drug_emb": torch.randn(1, 20, 384),          # ChemBERTa per-token
    "ptm_vector": torch.ones(1, n_tokens),        # Flat PTM baseline (all types)
    "delta_ptm_vector": torch.zeros(1, n_tokens), # Drug-induced PTM change (all types)
    "target_protein": torch.tensor([0]),           # Protein ID (0=first protein)
}

# 4. Forward pass
model.eval()
with torch.no_grad():
    ic50_pred, resistance_logits = model(
        seq_embeddings=sample["seq_emb"],
        struct_embeddings=sample["struct_emb"],
        drug_pooled=sample["drug_pooled"],
        drug_embeddings=sample["drug_emb"],
        ptm_vector=sample["ptm_vector"],
        delta_ptm_vector=sample["delta_ptm_vector"],
        target_protein=sample["target_protein"],
    )

print(f"Predicted ln(IC50): {ic50_pred.item():.3f}")
print(f"P(resistance): {torch.sigmoid(resistance_logits).item():.3f}")

# 5. Training loop (with your DataLoader)
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# train_epoch handles one full epoch:
train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                         focal_loss, lambda_reg=1.0, lambda_cls=2.0, device="cpu")

# validate returns comprehensive metrics:
val_metrics = validate(model, val_loader, focal_loss, 1.0, 2.0, "cpu")
print(f"Val AUROC: {val_metrics['auroc']:.3f}, BAcc: {val_metrics['balanced_acc']:.3f}")
```

### tool Utilities — Training & Evaluation

The tool provides shared utilities that ensure consistent behavior across all case studies:

```python
from src.ptm_bdl.training import (
    compute_optimal_threshold,
    save_checkpoint, load_checkpoint,
    resolve_device, create_balanced_sampler,
)
from src.ptm_bdl.evaluation.evaluator import (
    collect_predictions, compute_full_metrics,
    load_threshold, make_eval_loader,
)

# ── Device selection (auto-detects CUDA > MPS > CPU) ──
device = resolve_device(cfg)

# ── Class-balanced sampling (prevents majority-class collapse) ──
sampler = create_balanced_sampler(dataset, train_idx)
train_loader = DataLoader(train_set, batch_size=16, sampler=sampler, collate_fn=collate_fn)

# ── Save/load checkpoints (always uses weights_only=True) ──
save_checkpoint(model, "data/models/best_model.pt")
load_checkpoint(model, "data/models/best_model.pt", device)  # loads + sets eval mode

# After training — compute optimal classification threshold (Youden's J)
# Finds the probability threshold that maximizes (sensitivity + specificity)
# on the validation set. Essential when focal loss shifts probabilities away from 0.5.
# Ref: Youden WJ (1950) Cancer 3:32-35.
threshold_info = compute_optimal_threshold(model, val_loader, device)
# Returns: {"optimal_threshold": 0.38, "method": "Youden_J", "reference": "..."}

# During evaluation — load the saved threshold
threshold = load_threshold(model_dir)  # Loads optimal_threshold.json, falls back to 0.5

# Create evaluation DataLoaders from index arrays (consistent across all scripts)
loader = make_eval_loader(dataset, test_idx, batch_size=32, collate_fn=collate_fn)

# Collect predictions and compute metrics with calibrated threshold
y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls = collect_predictions(model, loader)
regression, classification = compute_full_metrics(
    y_true_ic50, y_pred_ic50, y_true_cls, y_prob_cls,
    threshold=threshold,
)
```

### Using the PTMTypeRegistry Directly

```python
from src.ptm_bdl.registry import PTMTypeRegistry

# Build registry from config
registry = PTMTypeRegistry.from_config(cfg)

# Query the registry
print(f"PTM subtypes: {registry.n_subtypes}")    # e.g. 4
print(f"Tokens per sample: {registry.n_tokens}")  # e.g. 24 (12 primary + 12 secondary)
print(f"Proteins: {registry.protein_names}")       # e.g. ['PROTEIN_A', 'PROTEIN_B']
print(f"Subtype names: {registry.subtype_names}")  # e.g. {0: 'primary_Y', 1: 'primary_S', ...}

# Get site labels for XAI reporting
labels = registry.get_flat_site_labels(registry.protein_names[0])
print(f"Sites: {labels[:5]}")

# Get buffer tensors (used by the encoder)
print(f"Type ID table shape: {registry.type_id_table.shape}")  # (2, 24)
print(f"Is-real table shape: {registry.is_real_table.shape}")  # (2, 24)
```

### Adding a New Case Study (Separate Project)

To use PTM-BDL in your own project for a different biological system:

```python
# your_project/train_my_model.py
import yaml
from src.ptm_bdl.training import build_model_from_cfg, FocalLoss, train_epoch, validate
from src.ptm_bdl.data import ResistanceDataset, collate_fn

# Your config defines YOUR proteins, PTM types, and drugs
my_config = {
    "model": {"shared_dim": 512, "num_joint_attention_layers": 4,
              "num_attention_heads": 8, "dropout": 0.1,
              "learning_rate": 1e-4, "weight_decay": 1e-5,
              "batch_size": 16, "num_epochs": 100, "early_stopping_patience": 15},
    "ptm_bdl": {"d_model": 64, "n_heads": 4, "n_layers": 2},
    "ptm": {
        "ptm_dim": 10,  # YOUR number of primary PTM sites
        "secondary_dim": 8,  # YOUR number of secondary PTM sites (0 or omit if none)
        # Also accepts "glyco_dim" (legacy alias for secondary_dim)
        "YOUR_PROTEIN": {
            "phospho_sites": [
                {"position": 100, "residue": "Y100", "amino_acid": "Y", "function": "kinase substrate"},
                # ... your primary PTM sites (first *_sites key → primary channel)
            ],
            # Optional secondary channel — ANY additional *_sites key:
            # "acetyl_sites": [   # acetylation
            #     {"position": 200, "residue": "K200", "amino_acid": "K", "function": "regulatory"},
            # ],
            # "glyco_sites": [    # glycosylation
            #     {"position": 50, "residue": "N50", "amino_acid": "N", "function": "extracellular"},
            # ],
        },
    },
}

# Build model — auto-configures for YOUR proteins and PTM types
model = build_model_from_cfg(my_config)

# Train on YOUR data using the same training infrastructure
# ... (same API as above)
```

**The tool supports new biological systems through its typed-token architecture.**

---

## Configuration

tool-level settings are in `src/ptm_bdl/default_config.yaml` (shipped with the package).
Case-study-specific settings are in each case study's `config.yaml`.

```python
from src.ptm_bdl.config import load_config

# Load merged config (base tool + EGFR/ERBB2 case study)
cfg = load_config(case_study="egfr_erbb2_tki")

# Load base tool config only
cfg = load_config(case_study=None)
```

**tool defaults** (`src/ptm_bdl/default_config.yaml`):

```yaml
model:
  shared_dim: 512
  num_joint_attention_layers: 4
  num_attention_heads: 8
  learning_rate: 1.0e-4
  batch_size: 16

ptm_bdl:
  d_model: 64
  n_heads: 4
  n_layers: 2

training:
  seed: 42
  train_ratio: 0.7
  val_ratio: 0.15
  device: "auto"  # "auto" selects best available: cuda > mps > cpu
```

**Case-study-specific settings** (proteins, drugs, PTM sites, tissue filters) are in
`src/case_studies/egfr_erbb2_tki/config.yaml`. The `load_config` function deep-merges both configs automatically.

---

## Dependencies

All dependencies are required for the full pipeline. Install with `pip install -r requirements.txt`.

| Package | Version | Used By | Purpose |
|---------|---------|---------|---------|
| Python | ≥ 3.11, < 3.13 | All | Language runtime |
| PyTorch | ≥ 2.1 | Steps 07-09, training | Deep learning backend |
| transformers | ≥ 4.36 | Steps 07, 09 | ESM-2 protein + ChemBERTa drug model loading |
| fair-esm | ≥ 2.0 | Step 08 | ESM-IF1 pretrained structural encoder |
| biotite | ≥ 1.0 | Step 08 | PDB parsing for ESM-IF1 |
| torch-geometric | ≥ 2.4 | Step 08 | PyG GearNet structural GNN (separate install) |
| biopython | ≥ 1.84 | Steps 02-03, 08 | FASTA/PDB parsing |
| scikit-learn | ≥ 1.3 | Evaluation | Metrics, data splits, ML baselines |
| xgboost | ≥ 2.0 | ML baselines | Gradient boosting comparison |
| pandas | ≥ 2.0 | Steps 01-06 | Data processing |
| openpyxl | ≥ 3.1 | Steps 01, 04-05 | Excel file reading |

See [`pyproject.toml`](pyproject.toml) for the full list.

> **Note on `torch-geometric`**: This package must be installed separately after PyTorch because
> it requires a version-matched wheel. See [installation step 3](#3-install-dependencies) above.

---

## Documentation

| Document                                                       | Description                                      |
|----------------------------------------------------------------|--------------------------------------------------|
| [`docs/DATA_DOWNLOAD_GUIDE.md`](docs/DATA_DOWNLOAD_GUIDE.md)  | Manual data download instructions (per-step, per-source) |

---

## Tests

```bash
make test
# or
python -m pytest tests/ -v --tb=short
```

---

## Citation

The manuscript is unpublished. Use the repository citation below until a publication record is available.

```bibtex
@software{zein2026ptmbdl,
  title  = {A site-resolved post-translational modification framework for drug-response analysis},
  author = {Zein, Moustafa and Hassanien, Aboul Ella},
  year   = {2026},
  url    = {https://github.com/Moustafazn/PTM-BDL},
}
```

## License

MIT License — see [LICENSE](LICENSE).
