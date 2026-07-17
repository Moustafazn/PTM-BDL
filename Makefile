# ============================================================================
# Makefile — PTM-Driven Multimodal Self-Attention Pipeline
# ============================================================================
# Usage:
#   make all         — Run full pipeline end-to-end
#   make data        — Download all raw data (Steps 01–05)
#   make harmonize   — Build multimodal dataset (Step 06)
#   make features    — Extract embeddings (Steps 07–09)
#   make train       — Train + ablation + cross-validation (Steps 10–11)
#   make evaluate    — Evaluate + explainability (Steps 12–13)
#   make benchmark   — Run benchmarking suite (Steps 14a–d)
#   make figures     — Generate publication figures + tables (Steps 15a–b)
#   make test        — Run test suite
#   make clean       — Remove generated data and models
# ============================================================================

PYTHON ?= python
SCRIPTS = scripts

.PHONY: all data harmonize features train evaluate benchmark figures test clean help

help:
	@echo "PTM-Driven Multimodal Self-Attention Pipeline"
	@echo "============================================="
	@echo ""
	@echo "  make all         Run full pipeline end-to-end"
	@echo "  make data        Download raw data (Steps 01-05)"
	@echo "  make harmonize   Build multimodal dataset (Step 06)"
	@echo "  make features    Extract embeddings (Steps 07-09)"
	@echo "  make train       Train + ablation + CV (Steps 10-11)"
	@echo "  make evaluate    Evaluate + XAI (Steps 12-13)"
	@echo "  make benchmark   Benchmarking suite (Steps 14a-d)"
	@echo "  make figures     Publication figures + tables (Steps 15a-b)"
	@echo "  make test        Run test suite"
	@echo "  make clean       Remove generated data and models"
	@echo ""

# ── Phase 1: Data Acquisition ────────────────────────────────────────────────
data:
	$(PYTHON) $(SCRIPTS)/step01_download_gdsc.py
	$(PYTHON) $(SCRIPTS)/step02_download_mutations.py
	$(PYTHON) $(SCRIPTS)/step03_download_structures.py
	$(PYTHON) $(SCRIPTS)/step04_download_ptm_data.py
	$(PYTHON) $(SCRIPTS)/step05_download_drugptm.py

# ── Phase 2: Data Harmonization ──────────────────────────────────────────────
harmonize:
	$(PYTHON) $(SCRIPTS)/step06_harmonize_dataset.py

# ── Phase 3: Feature Extraction ──────────────────────────────────────────────
features:
	$(PYTHON) $(SCRIPTS)/step07_extract_esm2.py
	$(PYTHON) $(SCRIPTS)/step08_extract_gearnet.py
	$(PYTHON) $(SCRIPTS)/step09_extract_chemberta.py

# ── Phase 4: Training ────────────────────────────────────────────────────────
train:
	$(PYTHON) $(SCRIPTS)/step10_build_model.py
	$(PYTHON) $(SCRIPTS)/step11_train.py
	$(PYTHON) $(SCRIPTS)/step11b_ablation.py
	$(PYTHON) $(SCRIPTS)/step11c_crossval.py

# ── Phase 5: Evaluation ──────────────────────────────────────────────────────
evaluate:
	$(PYTHON) $(SCRIPTS)/step12_evaluate.py
	$(PYTHON) $(SCRIPTS)/step13_explainability.py

# ── Phase 6: Benchmarking ────────────────────────────────────────────────────
benchmark:
	$(PYTHON) $(SCRIPTS)/step14a_ml_baselines.py
	$(PYTHON) $(SCRIPTS)/step14b_external_baselines.py
	$(PYTHON) $(SCRIPTS)/step14c_statistical_tests.py
	$(PYTHON) $(SCRIPTS)/step14d_loclo.py

# ── Phase 7: Publication Figures & Tables ────────────────────────────────────
figures:
	$(PYTHON) $(SCRIPTS)/step15a_paper_figures.py
	$(PYTHON) $(SCRIPTS)/step15b_paper_tables.py

# ── Full Pipeline ────────────────────────────────────────────────────────────
all: data harmonize features train evaluate benchmark figures

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/processed data/features data/models
	@echo "Cleaned generated data. Raw data preserved in data/raw/"
