# CLAUDE.md — Project Instructions for AI Assistants

## Project Overview

**PTM-BDL** is a typed self-attention framework for learning how post-translational modification (PTM) patterns drive drug response. The core architecture uses **cross-modal self-attention** over four biological modalities (protein sequence, 3D structure, drug chemistry, PTM signaling state) with a dedicated **PTM Biological Dynamics Layer** that treats each PTM site as a typed token.

The framework is demonstrated on EGFR (NSCLC) and ERBB2/HER2 (breast cancer) TKI resistance — this is the first **application instance**, not the framework's scope.

## Key Architecture

```
Stage 1 (Static): ESM-2 + GearNet + ChemBERTa → Joint Self-Attention → S_rep
Stage 2 (Dynamic): PTM tokens → Type-Gated → Typed Self-Attention → P_rep
Fusion: S_rep ⊙ P_rep → IC50 regression + Resistance classification
```

**PTM-BDL typed token system:**
- 4 subtypes: phospho_Y (0), phospho_S (1), phospho_T (2), glyco_N (3)
- Subtypes are children of PTM types: phospho → {Y, S, T}, glyco → {N}
- 24 tokens per sample (12 phospho + 12 glyco), with protein-specific padding
- Self-attention discovers inter-site and cross-type (phospho↔glyco) dependencies

## Repository Structure

- `config/config.yaml` — All hyperparameters, PTM sites, drug SMILES, data sources
- `src/models/multimodal_predictor.py` — Core model (PTMBDLEncoder, MultimodalResistancePredictor)
- `scripts/step01-06` — Data pipeline (EGFR/ERBB2 application instance)
- `scripts/step07-09` — Feature extraction (ESM-2, GearNet, ChemBERTa)
- `scripts/step10-13` — Training, ablation, evaluation, explainability
- `scripts/step14a-d` — Benchmarking (ML baselines, external methods, stats, LOCLO)
- `scripts/step15a-b` — Publication figures & tables
- `results/` — JSON evaluation outputs + figures
- `tests/` — pytest test suite
- `docs/` — Technical documentation

## How to Run

```bash
source venv/bin/activate
python scripts/step11_train.py        # Train
python scripts/step11b_ablation.py    # Ablation study
python scripts/step12_evaluate.py     # Evaluate
python scripts/step13_explainability.py  # XAI
python -m pytest tests/ -v            # Tests
make all                              # Full pipeline
```

## Key Files to Understand

1. **`src/models/multimodal_predictor.py`** — PTMBDLEncoder (typed self-attention), PTMBDLMlpAblation (MLP ablation), StaticJointTransformer, BilinearLateFusion, MultimodalResistancePredictor
2. **`scripts/step11_train.py`** — `build_model_from_cfg()`, `ResistanceDataset`, `FocalLoss`, training loop
3. **`scripts/step11b_ablation.py`** — 5-arm ablation + multi-seed stability + randomized PTM control
4. **`config/config.yaml`** — PTM sites, drug SMILES, model hyperparameters, PTM modulators

## Common Tasks

### "Add a new PTM site"
→ Edit `config/config.yaml` under `ptm.EGFR.phospho_sites` or `ptm.ERBB2.phospho_sites`, then update buffer tables in `src/models/multimodal_predictor.py`

### "Add a new drug"
→ Add SMILES in `config/config.yaml` under `drugs`, add GDSC drug ID under `gdsc.drug_ids`, run steps 01, 06, 09

### "Add a new ablation arm"
→ Add entry in `scripts/step11b_ablation.py:ABLATION_CONFIGS` and `ABLATION_ORDER`

### "Run benchmarks"
→ `make benchmark` (steps 14a-d) then `make figures` (steps 15a-b)

### "Understand results"
→ Check `results/*.json` for raw metrics

## Important Conventions

- **PTM site numbering**: UniProt PRECURSOR numbering (includes signal peptide). Classic names differ by +24 for EGFR.
- **Proteins**: EGFR (P00533) and ERBB2/HER2 (P04626). Both ERBB family members.
- **PTM tokens**: 24 per sample (12 phospho + 12 glyco). ERBB2 has pad slots (10+7 real, 2+5 pad).
- **Ablation modes**: "full", "no_ptm", "no_glyco", "glyco_only", "no_typed_attention"
- **Early stopping**: On max(AUROC, BAcc) — AUROC is threshold-independent for 92:8 imbalance.

## Dependencies

- Python 3.11, PyTorch 2.1+, transformers 5.11+, fair-esm 2.0+
- See `pyproject.toml` and `requirements.txt`

## Authors

- Moustafa Zein (lead developer)
- Prof. Aboul Ella Hassanien (co-author)
