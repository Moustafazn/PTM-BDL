# ============================================================================
# Makefile — PTM-BDL tool Pipeline
# ============================================================================
# Supports ALL case studies independently:
#
#   make egfr        — Run EGFR/ERBB2 TKI case study (full pipeline)
#   make hela        — Run HeLa/HDAC inhibitor case study (full pipeline)
#   make k562        — Run K562/CML BCR-ABL case study (full pipeline)
#   make all         — Run ALL case studies sequentially
#
# Per-phase targets (specify CASE=egfr|hela|k562):
#   make data CASE=hela       — Download data for HeLa
#   make train CASE=k562      — Train K562 model
#   make evaluate CASE=egfr   — Evaluate EGFR model
#
# Default: CASE=egfr (original case study)
# ============================================================================

PYTHON ?= python

# Case study selector (default: egfr)
CASE ?= egfr

# Map short names to full paths
ifeq ($(CASE),egfr)
  CS = src/case_studies/egfr_erbb2_tki
else ifeq ($(CASE),hela)
  CS = src/case_studies/hela_hdac
else ifeq ($(CASE),k562)
  CS = src/case_studies/k562_cml
else
  $(error Unknown CASE=$(CASE). Use: egfr, hela, or k562)
endif

DATA_PIPELINE = $(CS)/data_pipeline
FEATURES = $(CS)/features
SCRIPTS = $(CS)/scripts

.PHONY: all egfr hela k562 data harmonize features train evaluate benchmark figures test clean help

help:
	@echo "PTM-BDL tool Pipeline"
	@echo "======================================="
	@echo ""
	@echo "  Run a complete case study:"
	@echo "    make egfr        EGFR/ERBB2 TKI resistance (NSCLC + breast)"
	@echo "    make hela        HeLa/HDAC inhibitors (phospho + acetyl)"
	@echo "    make k562        K562/CML BCR-ABL (TKI + chemo)"
	@echo "    make all         All case studies sequentially"
	@echo ""
	@echo "  Run individual phases (use CASE=egfr|hela|k562):"
	@echo "    make data CASE=hela       Download raw data"
	@echo "    make harmonize CASE=hela  Build multimodal dataset"
	@echo "    make features CASE=hela   Extract embeddings"
	@echo "    make train CASE=hela      Train model"
	@echo "    make evaluate CASE=hela   Evaluate + XAI"
	@echo "    make benchmark CASE=hela  Benchmarking suite"
	@echo "    make figures CASE=hela    Publication figures"
	@echo ""
	@echo "  Utilities:"
	@echo "    make test        Run test suite"
	@echo "    make clean       Remove generated data"
	@echo ""

# ── Phase 1: Data Acquisition (Steps 01-05) ──────────────────────────────────
data:
	$(PYTHON) -m $(subst /,.,$(DATA_PIPELINE)).step01_download_gdsc
	$(PYTHON) -m $(subst /,.,$(DATA_PIPELINE)).step02_download_sequences
	$(PYTHON) -m $(subst /,.,$(DATA_PIPELINE)).step03_download_structures
	$(PYTHON) -m $(subst /,.,$(DATA_PIPELINE)).step04_download_ptm_data
	$(PYTHON) -m $(subst /,.,$(DATA_PIPELINE)).step05_download_drugptm

# ── Phase 2: Data Harmonization (Step 06) ────────────────────────────────────
harmonize:
	$(PYTHON) -m $(subst /,.,$(DATA_PIPELINE)).step06_harmonize_dataset

# ── Phase 3: Feature Extraction (Steps 07-09) ────────────────────────────────
features:
	$(PYTHON) -m $(subst /,.,$(FEATURES)).step07_extract_esm2
	$(PYTHON) -m $(subst /,.,$(FEATURES)).step08_extract_gearnet
	$(PYTHON) -m $(subst /,.,$(FEATURES)).step09_extract_chemberta

# ── Phase 4: Training ────────────────────────────────────────────────────────
train:
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).train
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).ablation
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).crossval

# ── Phase 5: Evaluation ──────────────────────────────────────────────────────
evaluate:
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).evaluate
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).explain

# ── Phase 6: Benchmarking ────────────────────────────────────────────────────
benchmark:
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).ml_baselines
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).external_baselines
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).statistical_tests
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).loclo

# ── Phase 7: Publication Figures & Tables ────────────────────────────────────
figures:
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).paper_figures
	$(PYTHON) -m $(subst /,.,$(SCRIPTS)).paper_tables

# ── Complete case study pipelines ────────────────────────────────────────────
egfr:
	$(MAKE) data harmonize features train evaluate benchmark figures CASE=egfr

hela:
	$(MAKE) data harmonize features train evaluate benchmark figures CASE=hela

k562:
	$(MAKE) data harmonize features train evaluate benchmark figures CASE=k562

# ── All case studies ─────────────────────────────────────────────────────────
all: egfr hela k562

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/processed data/features data/models
	@echo "Cleaned generated data. Raw data preserved in data/raw/"
