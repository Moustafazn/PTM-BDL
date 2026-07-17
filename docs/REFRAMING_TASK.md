# Refactoring Task: Multimodal Self-Attention + PTM-BDL as a Config-Driven Extensible Framework
## Comprehensive Refactoring Plan (2026-07-04)

---

## 1. PURPOSE

This project has **two architectural contributions**:

1. **Multimodal Cross-Modal Self-Attention** — jointly learning from protein sequence (ESM-2), 3D structure (GearNet), and drug chemistry (ChemBERTa) through 4-layer × 8-head joint self-attention, producing a static protein-drug representation (S_rep)
2. **PTM Biological Dynamics Layer (PTM-BDL)** — a typed self-attention encoder that accepts **1-to-N PTM types**, each with **1-to-N subtypes**, encoding the dynamic post-translational modification signaling state as typed tokens and producing a dynamic PTM representation (P_rep)

Both are fused via bilinear late fusion (S_rep ⊙ P_rep) for drug response prediction. The current code proves generalizability with two proteins (EGFR, ERBB2), two PTM types (phospho with 3 subtypes, glyco with 1 subtype), and six drugs. However, the implementation has hardcoded constants, flat script structure, and duplicated application-specific labels that make extending to new proteins, drugs, cell lines, PTM types, or subtypes require code changes instead of config changes.

**This task refactors the codebase to match the architecture's intent:**
- Adding a new protein = adding a section in config
- Adding a new PTM type with 1-N subtypes = adding entries in config → registry auto-assigns subtype IDs
- Adding a new drug = adding SMILES in config
- Adding a new modality encoder = implementing one class with the same projection interface
- The core framework code never changes

**Constraint:** Existing results (ablation_study.json, stability_analysis.json, randomized_ptm_control.json, xai_report.json, evaluation_report.json) must remain reproducible. The EGFR/ERBB2 case study is the first application instance of the framework.

---

## 2. CURRENT PROBLEMS — What Must Change

### 2a. Hardcoded Constants in Model (`src/models/multimodal_predictor.py`)

| What | Lines | Problem |
|------|-------|---------|
| `_TYPE_Y=0, _TYPE_S=1, _TYPE_T=2, _TYPE_N=3` | 56–59 | PTM subtypes hardcoded as module-level constants |
| `N_PTM_TYPES = 4` | 60 | Fixed count — adding acetylation requires code change |
| `_TYPE_PHOSPHO_EGFR = [...]` | 62–65 | Per-protein subtype arrays hardcoded |
| `_TYPE_PHOSPHO_ERBB2 = [...]` | 66–69 | Same — every new protein needs a new array |
| `_TYPE_GLYCO = [_TYPE_N] * 12` | 70 | Hardcoded glyco mapping |
| `_PAD_EGFR = [False] * 24` | 72 | Per-protein padding hardcoded |
| `_PAD_ERBB2 = (...)` | 73–76 | Same |
| `PROTEIN_ID_EGFR = 0, PROTEIN_ID_ERBB2 = 1` | 78–79 | Hardcoded protein ID mapping |
| `PTM_TYPE_NAMES = {0: "phospho_Y", ...}` | 80 | Hardcoded name→id mapping |
| `nn.Embedding(2, d_model)` | 137 | Protein embedding hardcoded to 2 proteins |
| `assert phospho_dim == 12 and glyco_dim == 12` | 124 | Fixed dimension assert blocks extension |
| `self.n_tokens = 24` | 127 | Hardcoded token count (should be sum of all PTM slots) |

### 2b. Hardcoded Dataset Columns (`scripts/step11_train.py`)

| What | Lines | Problem |
|------|-------|---------|
| `ptm_cols = ["ptm_Y869", "ptm_S991", ...]` | 249–253 | EGFR-specific phospho column names hardcoded |
| `delta_ptm_cols = [...]` | 262–266 | Same for delta |
| `glyco_cols = [f"glyco_slot{i:02d}" for i in range(12)]` | 276 | Hardcoded to 12 glyco slots |
| `target_protein = 1 if tp_str == "ERBB2" else 0` | 293 | Hardcoded protein→id mapping |
| Ablation zeroing with `np.ones(12)` | 306–320 | Hardcoded to 12-dim per channel |

### 2c. Duplicated Application-Specific Labels

The same site labels, type maps, and pad masks are **duplicated** in three places:

| Data | Appears in |
|------|-----------|
| `PHOSPHO_LABELS_EGFR/ERBB2` | `step13_explainability.py:158–167`, `step11b_ablation.py:78–87` |
| `GLYCO_LABELS_EGFR/ERBB2` | `step13_explainability.py:168–176`, `step11b_ablation.py:88–96` |
| `PHOSPHO_TYPE_EGFR/ERBB2` | `step13_explainability.py:180–181` |
| `PHOSPHO_REAL_EGFR/ERBB2`, `GLYCO_REAL_*` | `step13_explainability.py:184–187` |
| Homology slot indices | `step13_explainability.py:193–195`, `step11b_ablation.py:98–101` |

Any new protein requires updating ALL of these files identically.

### 2d. Flat Script Structure with Import Hacks

All scripts live in `scripts/` as flat files. Other scripts import from `step11_train.py` using `sys.path.insert`:

| Script | Imports from `step11_train` |
|--------|----------------------------|
| `step11b_ablation.py` | `ResistanceDataset, collate_fn, FocalLoss, train_epoch, validate, build_model_from_cfg` |
| `step11c_crossval.py` | `ResistanceDataset, collate_fn, FocalLoss` |
| `step12_evaluate.py` | `ResistanceDataset, collate_fn, build_model_from_cfg, create_stratified_splits` |
| `step13_explainability.py` | `ResistanceDataset, build_model_from_cfg` |
| `step14d_loclo.py` | `ResistanceDataset, collate_fn, FocalLoss, train_epoch, validate, build_model_from_cfg` |

Additionally, scripts import constants from `multimodal_predictor.py`:

| Script | Imports |
|--------|---------|
| `step10_build_model.py` | `MultimodalResistancePredictor` |
| `step11b_ablation.py` | `PROTEIN_ID_EGFR, PROTEIN_ID_ERBB2` |
| `step11c_crossval.py` | `PROTEIN_ID_ERBB2` |
| `step13_explainability.py` | `PROTEIN_ID_EGFR, PROTEIN_ID_ERBB2, PTM_TYPE_NAMES` |

Tests also use `importlib` hacks to import from `step11_train.py` (`tests/test_training.py:23–31`).

### 2e. PTM Subtypes Are Not Hierarchical

Currently subtypes are flat constants (`_TYPE_Y=0`). The relationship "phospho_Y is a subtype of phospho" exists only in human understanding, not in the code. When adding acetylation with subtypes (acetyl_K, acetyl_Nt), there's no mechanism to:
- Auto-discover subtypes from a PTM type definition
- Auto-assign subtype IDs
- Auto-generate subtype names
- Group IG attributions by parent PTM type

---

## 3. TARGET ARCHITECTURE — Package Structure

### 3a. New `src/` Layout

```
src/
├── __init__.py
│
├── ptm_bdl/                          # CORE FRAMEWORK (protein/PTM-agnostic)
│   ├── __init__.py                   # Public API exports
│   │
│   ├── registry.py                   # PTMTypeRegistry — dynamic subtype system
│   │   • PTMType(name, subtypes: dict[amino_acid → subtype_id])
│   │   • PTMTypeRegistry.from_config(cfg) → builds type_ids, pad_masks, names
│   │   • ProteinPTMConfig(protein_name, sites_per_ptm_type, pad_masks)
│   │   • Computes: n_subtypes, n_tokens, type_id_table, is_real_table
│   │
│   ├── model/                        # MODEL PACKAGE (clear input/output)
│   │   ├── __init__.py
│   │   ├── encoder.py                # PTMBDLEncoder — takes registry, not hardcoded arrays
│   │   ├── static.py                 # StaticJointTransformer, AttentionPooling, ModalityProjection
│   │   ├── fusion.py                 # BilinearLateFusion
│   │   ├── predictor.py              # MultimodalResistancePredictor
│   │   └── ablation.py               # PTMBDLMlpAblation
│   │
│   ├── data/                         # DATA PACKAGE
│   │   ├── __init__.py
│   │   ├── dataset.py                # ResistanceDataset (config-driven columns)
│   │   ├── collate.py                # collate_fn
│   │   └── splits.py                 # create_stratified_splits
│   │
│   ├── training/                     # TRAINING PACKAGE
│   │   ├── __init__.py
│   │   ├── loss.py                   # FocalLoss
│   │   ├── trainer.py                # train_epoch, validate, _train_loop, early stopping
│   │   ├── metrics.py                # compute_metrics
│   │   └── factory.py                # build_model_from_cfg (model factory)
│   │
│   ├── evaluation/                   # EVALUATION PACKAGE
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Full evaluation (collect_predictions, compute_full_metrics)
│   │   ├── baselines.py              # ML baseline framework (RF, XGBoost, Ridge, ElasticNet)
│   │   ├── statistical.py            # Bootstrap CIs, DeLong, Wilcoxon, BH correction
│   │   └── loclo.py                  # Leave-One-Class-Line-Out framework
│   │
│   └── xai/                          # EXPLAINABILITY PACKAGE
│       ├── __init__.py
│       ├── integrated_gradients.py   # Generic per-token IG (any protein, any PTM type)
│       ├── attention.py              # Cross-type attention analysis (any PTM type pair)
│       ├── homology.py               # Cross-protein homology check (config-driven)
│       └── reporter.py               # XAI report assembly
│
├── case_studies/                     # APPLICATION INSTANCES
│   ├── __init__.py
│   └── egfr_erbb2_tki/             # Current case study
│       ├── __init__.py
│       ├── biology.py                # Site labels, homology slots, mutation groups,
│       │                             # drug comparisons — ALL application-specific biology
│       ├── data_pipeline/            # Steps 01–06 (data download + harmonize)
│       │   ├── __init__.py
│       │   ├── download_gdsc.py
│       │   ├── download_mutations.py
│       │   ├── download_structures.py
│       │   ├── download_ptm_data.py
│       │   ├── download_drugptm.py
│       │   └── harmonize_dataset.py
│       ├── features/                 # Steps 07–09 (feature extraction)
│       │   ├── __init__.py
│       │   ├── extract_esm2.py
│       │   ├── extract_gearnet.py
│       │   └── extract_chemberta.py
│       └── scripts/                  # Entry points that call framework packages
│           ├── train.py              # → ptm_bdl.training
│           ├── ablation.py           # → ptm_bdl.training + ptm_bdl.xai
│           ├── crossval.py           # → ptm_bdl.training
│           ├── evaluate.py           # → ptm_bdl.evaluation
│           ├── explain.py            # → ptm_bdl.xai + case_studies.egfr_erbb2_tki.biology
│           ├── ml_baselines.py       # → ptm_bdl.evaluation.baselines
│           ├── external_baselines.py # → ptm_bdl.evaluation
│           ├── statistical_tests.py  # → ptm_bdl.evaluation.statistical
│           ├── loclo.py              # → ptm_bdl.evaluation.loclo
│           ├── paper_figures.py      # → publication-specific
│           └── paper_tables.py       # → publication-specific
│
└── models/                           # LEGACY (keep for backward compat during transition)
    ├── __init__.py                   # Re-exports from ptm_bdl.model
    └── multimodal_predictor.py       # → ptm_bdl.model (deprecated wrapper)
```

### 3b. Model Input/Output Contract

The model package must have a clear, documented I/O interface:

**Inputs (all config-driven, no hardcoded dimensions):**
```python
@dataclass
class PTMBDLInput:
    # Static modalities (from pretrained encoders)
    seq_embeddings: Tensor       # (B, L, seq_dim)     — protein language model
    struct_embeddings: Tensor    # (B, M, struct_dim)   — structural encoder
    drug_pooled: Tensor          # (B, drug_dim)        — drug encoder pooled
    drug_embeddings: Tensor      # (B, N, drug_dim)     — drug encoder per-token (optional)

    # Dynamic PTM channels (per PTM type, from config)
    ptm_channels: dict[str, Tensor]        # {"phospho": (B, n_phospho_sites), "glyco": (B, n_glyco_sites)}
    delta_ptm_channels: dict[str, Tensor]  # {"phospho": (B, n_phospho_sites), "glyco": (B, n_glyco_sites)}

    # Protein identity
    protein_id: Tensor           # (B,) long — index into protein registry

@dataclass
class PTMBDLOutput:
    ic50_pred: Tensor            # (B, 1)
    resistance_logits: Tensor    # (B, 1)
    # Optional extras (for XAI)
    ptm_bdl_tokens: Tensor       # (B, n_tokens, d_model) — if return_ptm_bdl
    ptm_bdl_mask: Tensor         # (B, n_tokens)          — if return_ptm_bdl
    attention_maps: list         # static attention       — if return_attention
```

---

## 4. DYNAMIC PTM TYPE/SUBTYPE REGISTRY

### 4a. The Core Design

PTM types and subtypes are defined hierarchically in config. Subtypes are derived from (ptm_type, amino_acid):

```yaml
# In config.yaml (NEW section replacing hardcoded constants)
ptm_type_registry:
  phospho:
    description: "Phosphorylation"
    subtypes:
      Y: {description: "phospho-tyrosine — direct TKI target"}
      S: {description: "phospho-serine — downstream indicator"}
      T: {description: "phospho-threonine — regulatory feedback"}
  glyco:
    description: "N-linked glycosylation"
    subtypes:
      N: {description: "N-glycosylation — receptor surface biology"}

  # ── FUTURE EXTENSIONS (add here, zero code changes) ──────────
  # acetyl:
  #   description: "Acetylation"
  #   subtypes:
  #     K:  {description: "lysine acetylation"}
  #     Nt: {description: "N-terminal acetylation"}
  # ubiq:
  #   description: "Ubiquitination"
  #   subtypes:
  #     K48: {description: "degradation signal"}
  #     K63: {description: "signaling signal"}
```

### 4b. The Registry Class

```python
class PTMTypeRegistry:
    """
    Dynamic PTM type/subtype system built from config.

    Responsibilities:
      1. Assign contiguous subtype IDs automatically
      2. Build per-protein type_id_table and is_real_table (buffer tensors)
      3. Provide name↔id mappings for XAI reporting
      4. Compute n_subtypes (= N_PTM_TYPES) and n_tokens_per_protein
    """

    @classmethod
    def from_config(cls, cfg: dict) -> "PTMTypeRegistry":
        """
        Build registry from config.yaml ptm_type_registry section.

        Assigns subtype IDs in order of (ptm_type, amino_acid):
          phospho.Y → 0, phospho.S → 1, phospho.T → 2, glyco.N → 3
          (matches current hardcoded _TYPE_Y=0, _TYPE_S=1, _TYPE_T=2, _TYPE_N=3)

        Adding acetyl.K → 4, acetyl.Nt → 5 requires only config change.
        """
        ...

    @property
    def n_subtypes(self) -> int:
        """Total number of distinct subtypes (= size of type embedding)."""
        ...

    @property
    def subtype_names(self) -> dict[int, str]:
        """Map subtype_id → human-readable name (e.g., 0 → 'phospho_Y')."""
        ...

    @property
    def parent_type(self) -> dict[int, str]:
        """Map subtype_id → parent PTM type (e.g., 0 → 'phospho')."""
        ...

    def build_protein_buffers(self, protein_configs: dict) -> tuple[Tensor, Tensor]:
        """
        Build type_id_table and is_real_table for all proteins.

        Returns:
          type_id_table: (n_proteins, n_tokens) — subtype IDs per slot
          is_real_table: (n_proteins, n_tokens) — True for real sites, False for padding
        """
        ...
```

### 4c. Per-Protein Site Definition (Config-Driven)

```yaml
# In config.yaml — replaces hardcoded _TYPE_PHOSPHO_EGFR etc.
proteins:
  EGFR:
    id: 0
    uniprot: "P00533"
    ptm_sites:
      phospho:
        # Each site specifies its amino_acid → determines the subtype
        - {position: 869,  residue: "Y869",  amino_acid: "Y", function: "SRC substrate"}
        - {position: 991,  residue: "S991",  amino_acid: "S", function: "regulatory"}
        - {position: 998,  residue: "Y998",  amino_acid: "Y", function: "regulatory"}
        # ... (12 total)
        max_slots: 12    # padded to this if fewer real sites
      glyco:
        - {position: 56,   residue: "N56",   amino_acid: "N", function: "domain I"}
        - {position: 128,  residue: "N128",  amino_acid: "N", function: "domain II"}
        # ... (12 total)
        max_slots: 12

  ERBB2:
    id: 1
    uniprot: "P04626"
    ptm_sites:
      phospho:
        - {position: 686,  residue: "T686",  amino_acid: "T", function: "regulatory"}
        - {position: 1005, residue: "Y1005", amino_acid: "Y", function: "c-Cbl"}
        # ... (10 real + 2 pad → max_slots: 12)
        max_slots: 12
      glyco:
        - {position: 68,   residue: "N68",   amino_acid: "N", function: "domain I"}
        # ... (7 real + 5 pad → max_slots: 12)
        max_slots: 12
```

The registry reads this config and automatically:
1. Computes the subtype ID for each site from `amino_acid` + parent PTM type
2. Builds `type_id_table[protein_id]` per slot
3. Builds `is_real_table[protein_id]` (real sites vs padding)
4. Sets `n_tokens = sum(max_slots for each ptm_type)` per protein
5. Provides `site_labels[protein_id][slot]` for XAI reporting

**Result:** The model's `PTMBDLEncoder.__init__` receives a `PTMTypeRegistry` instead of hardcoded constants:

```python
class PTMBDLEncoder(nn.Module):
    def __init__(self, registry: PTMTypeRegistry, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_tokens = registry.n_tokens
        self.type_emb = nn.Embedding(registry.n_subtypes, d_model)
        self.protein_emb = nn.Embedding(registry.n_proteins, d_model)
        self.slot_emb = nn.Embedding(self.n_tokens, d_model)
        # Register pre-built buffers from registry
        self.register_buffer("type_id_table", registry.type_id_table, persistent=False)
        self.register_buffer("is_real_table", registry.is_real_table, persistent=False)
        # ... rest unchanged
```

---

## 5. DETAILED REFACTORING STEPS

### Phase 1: Create `PTMTypeRegistry` + Config Migration (Foundation)

**Files to create:**
- `src/ptm_bdl/__init__.py`
- `src/ptm_bdl/registry.py` — `PTMTypeRegistry`, `ProteinPTMConfig`

**Files to modify:**
- `config/config.yaml` — Add `ptm_type_registry` and `proteins` sections (structured version of existing data)

**Validation:**
- Registry produces IDENTICAL `type_id_table` and `is_real_table` tensors as current hardcoded constants
- Add unit test asserting exact match

### Phase 2: Refactor Model to Use Registry

**Files to create:**
- `src/ptm_bdl/model/__init__.py`
- `src/ptm_bdl/model/encoder.py` — `PTMBDLEncoder` (registry-driven)
- `src/ptm_bdl/model/static.py` — `StaticJointTransformer`, `AttentionPooling`, `ModalityProjection`
- `src/ptm_bdl/model/fusion.py` — `BilinearLateFusion`
- `src/ptm_bdl/model/predictor.py` — `MultimodalResistancePredictor`
- `src/ptm_bdl/model/ablation.py` — `PTMBDLMlpAblation` (registry-driven)

**Files to deprecate (keep as wrappers):**
- `src/models/multimodal_predictor.py` — Thin wrapper importing from `ptm_bdl.model`

**Key changes in encoder:**
- Remove all `_TYPE_*`, `_PAD_*`, `PROTEIN_ID_*`, `N_PTM_TYPES` constants
- Remove `assert phospho_dim == 12 and glyco_dim == 12`
- Replace `nn.Embedding(2, d_model)` with `nn.Embedding(registry.n_proteins, d_model)`
- Replace `self.n_tokens = 24` with `self.n_tokens = registry.n_tokens`
- Buffer tables come from registry, not hardcoded arrays

**Validation:**
- Load existing `best_model.pt` weights into refactored model
- Assert identical forward pass output on same input (bit-exact or <1e-6 tolerance)

### Phase 3: Extract Data Package

**Files to create:**
- `src/ptm_bdl/data/__init__.py`
- `src/ptm_bdl/data/dataset.py` — `ResistanceDataset` (reads PTM columns from config)
- `src/ptm_bdl/data/collate.py` — `collate_fn` (handles variable PTM channel counts)
- `src/ptm_bdl/data/splits.py` — `create_stratified_splits`

**Key changes in ResistanceDataset:**
- PTM column names read from `registry.get_column_names(protein, ptm_type)` not hardcoded lists
- `target_protein` mapping from `registry.protein_name_to_id` not `if/else`
- Ablation zeroing uses `registry.n_sites_per_type` not hardcoded `12`
- Dataset accepts arbitrary PTM channels (dict of tensors, keyed by PTM type name)

**Validation:**
- `ResistanceDataset.__getitem__` produces identical tensors for existing EGFR/ERBB2 data

### Phase 4: Extract Training Package

**Files to create:**
- `src/ptm_bdl/training/__init__.py`
- `src/ptm_bdl/training/loss.py` — `FocalLoss`
- `src/ptm_bdl/training/trainer.py` — `train_epoch`, `validate`, early stopping loop
- `src/ptm_bdl/training/metrics.py` — `compute_metrics`
- `src/ptm_bdl/training/factory.py` — `build_model_from_cfg` (uses registry)

**Key change in `build_model_from_cfg`:**
```python
def build_model_from_cfg(cfg, use_typed_attention=True):
    registry = PTMTypeRegistry.from_config(cfg)
    return MultimodalResistancePredictor(
        registry=registry,
        seq_dim=1280, struct_dim=512, drug_dim=384,
        shared_dim=cfg["model"]["shared_dim"],
        # ... rest from config
    )
```

### Phase 5: Extract Evaluation Package

**Files to create:**
- `src/ptm_bdl/evaluation/__init__.py`
- `src/ptm_bdl/evaluation/evaluator.py` — `collect_predictions`, `compute_full_metrics`
- `src/ptm_bdl/evaluation/baselines.py` — ML baseline framework
- `src/ptm_bdl/evaluation/statistical.py` — Bootstrap, DeLong, Wilcoxon, BH
- `src/ptm_bdl/evaluation/loclo.py` — LOCLO framework (generic group-based leave-out)

**Key change:** Evaluation code becomes protein/drug/mutation-agnostic. Grouping keys come from config:
```python
def evaluate_by_groups(predictions, df, group_columns: list[str]):
    """Generic group-based evaluation — works for any protein/drug/mutation grouping."""
```

### Phase 6: Extract XAI Package

**Files to create:**
- `src/ptm_bdl/xai/__init__.py`
- `src/ptm_bdl/xai/integrated_gradients.py` — Generic per-token IG with config-driven bucketing
- `src/ptm_bdl/xai/attention.py` — Cross-type attention analysis for any PTM type pair
- `src/ptm_bdl/xai/homology.py` — Cross-protein homology check (slot pairs from config)
- `src/ptm_bdl/xai/reporter.py` — Assemble XAI report

**Key changes:**
- IG bucketing by subtype uses `registry.parent_type` and `registry.subtype_names` — no hardcoded `PHOSPHO_TYPE_EGFR` arrays
- Cross-type attention quadrants derived from `registry.ptm_type_slot_ranges` — not hardcoded `[:12, 12:24]`
- Homology checks read slot pairs from config — not hardcoded `PHOSPHO_Y_HOMOLOGY_SLOT = 7`
- Site labels come from `registry.site_labels[protein_id]` — no duplicated `PHOSPHO_LABELS_*` arrays

### Phase 7: Create Case Study Package

**Files to create:**
- `src/case_studies/__init__.py`
- `src/case_studies/egfr_erbb2_tki/__init__.py`
- `src/case_studies/egfr_erbb2_tki/biology.py` — ALL application-specific knowledge:
  - Homology site pairs (Y1068≡Y1221, N528↔N530)
  - Valid effector slots per protein (EGFR_VALID_TOP_EFFECTOR_SLOTS, ERBB2_VALID_TOP_EFFECTOR_SLOTS)
  - Mutation group assignment logic (exon19del, L858R, T790M, etc.)
  - Drug comparison logic (Afatinib vs Osimertinib, cross-protein drugs)
  - HER2 amplification tier assignment
- `src/case_studies/egfr_erbb2_tki/scripts/` — Entry point scripts that wire together framework packages with application biology

**Key principle:** The case study scripts are THIN wrappers:
```python
# src/case_studies/egfr_erbb2_tki/scripts/explain.py
from ptm_bdl.xai import IntegratedGradients, CrossTypeAttention, HomologyCheck
from ptm_bdl.training.factory import build_model_from_cfg
from case_studies.egfr_erbb2_tki.biology import (
    HOMOLOGY_PAIRS, VALID_EFFECTOR_SLOTS, mutation_group_analysis
)

def explain():
    registry = PTMTypeRegistry.from_config(cfg)
    model = build_model_from_cfg(cfg)
    # Framework XAI (generic)
    ig = IntegratedGradients(model, registry)
    cross_attn = CrossTypeAttention(model, registry)
    homology = HomologyCheck(registry, HOMOLOGY_PAIRS)  # Application-specific pairs
    # ...
```

### Phase 8: Update Config Schema

**New sections in `config/config.yaml`:**

```yaml
# ── PTM Type Registry (replaces hardcoded constants) ─────────────
ptm_type_registry:
  phospho:
    description: "Phosphorylation"
    subtypes:
      Y: {description: "phospho-tyrosine — direct TKI target"}
      S: {description: "phospho-serine — downstream indicator"}
      T: {description: "phospho-threonine — regulatory feedback"}
  glyco:
    description: "N-linked glycosylation"
    subtypes:
      N: {description: "N-glycosylation — receptor surface biology"}

# ── Protein Registry (replaces hardcoded PROTEIN_ID_*, _PAD_*, _TYPE_*) ───
protein_registry:
  EGFR:
    id: 0
    uniprot: "P00533"
    sites:
      phospho:    # → subtype auto-assigned from amino_acid
        max_slots: 12
        entries: [...]    # existing ptm.EGFR.phospho_sites
      glyco:
        max_slots: 12
        entries: [...]    # existing ptm.EGFR.glyco_sites
  ERBB2:
    id: 1
    uniprot: "P04626"
    sites:
      phospho:
        max_slots: 12
        entries: [...]
      glyco:
        max_slots: 12
        entries: [...]

# ── XAI Configuration (replaces hardcoded homology slots) ─────────
xai:
  homology_pairs:
    - name: "GRB2 docking site"
      protein_a: {name: "EGFR",  ptm_type: "phospho", slot: 7, site: "Y1092(Y1068)"}
      protein_b: {name: "ERBB2", ptm_type: "phospho", slot: 7, site: "Y1221"}
      pathway: "GRB2 → RAS-MAPK"
    - name: "Extracellular DIV anchor"
      protein_a: {name: "EGFR",  ptm_type: "glyco", slot: 8, site: "N528"}
      protein_b: {name: "ERBB2", ptm_type: "glyco", slot: 4, site: "N530"}
      pathway: "Trastuzumab binding interface"
  valid_top_effector_slots:
    EGFR:
      - {slot: 7,  site: "Y1092(Y1068)", pathway: "GRB2 → RAS-MAPK"}
      - {slot: 11, site: "Y1197(Y1173)", pathway: "SHC1 → PI3K-AKT"}
    ERBB2:
      - {slot: 7,  site: "Y1221", pathway: "GRB2 → RAS-MAPK"}
      - {slot: 9,  site: "Y1248", pathway: "SHC1 → PI3K-AKT"}
      - {slot: 1,  site: "Y1005", pathway: "c-Cbl → degradation"}
```

### Phase 9: Update Tests

**New test structure:**
- `tests/test_registry.py` — Test PTMTypeRegistry builds correct tables from config
- `tests/test_ptm_bdl.py` — Updated to use registry, not hardcoded constants
- `tests/test_model.py` — Updated imports from `ptm_bdl.model`
- `tests/test_training.py` — Updated imports from `ptm_bdl.training`
- `tests/test_backward_compat.py` — Assert registry produces SAME buffers as old hardcoded values

**Critical backward-compat test:**
```python
def test_registry_matches_legacy_constants():
    """The registry MUST produce identical buffers to the hardcoded constants."""
    registry = PTMTypeRegistry.from_config(load_config())
    legacy_type_ids = torch.tensor([
        _TYPE_PHOSPHO_EGFR + _TYPE_GLYCO,      # EGFR row
        _TYPE_PHOSPHO_ERBB2 + _TYPE_GLYCO,      # ERBB2 row
    ])
    legacy_is_real = torch.tensor([
        [not x for x in _PAD_EGFR],
        [not x for x in _PAD_ERBB2],
    ])
    assert torch.equal(registry.type_id_table, legacy_type_ids)
    assert torch.equal(registry.is_real_table, legacy_is_real)
```

### Phase 10: Documentation Consolidation

The current `docs/` folder has 14 files, many of which are historical (dated evaluations, expansion plans, migration notes). After refactoring, documentation should reflect the **current state only** — no mention of old architectures, migration steps, or historical evaluations.

**Target: 2 top-level files + 2 reference docs.**

#### 10a. `README.md` — The Single Source of Truth (Long, Comprehensive)

The README becomes the primary documentation, containing everything a user, reviewer, or contributor needs:

```
README.md
├── 1. Overview
│   ├── What is PTM-BDL (framework description)
│   ├── Why typed self-attention over PTM tokens
│   └── Key results summary
│
├── 2. Architecture
│   ├── Two-stage fusion diagram (static + dynamic)
│   ├── PTM-BDL encoder: typed tokens, type gate, self-attention, residual gate
│   ├── Input/Output contract (what the model expects and produces)
│   └── Config-driven extensibility (how type registry works)
│
├── 3. The EGFR/ERBB2 Case Study (Application Instance)
│   ├── Biological problem (TKI resistance in NSCLC + breast cancer)
│   ├── Dataset (951 samples, 6 drugs, 2 proteins, 2 PTM types)
│   ├── Data sources table (GDSC, DepMap, UniProt, DrugPTM-Bench, etc.)
│   └── Key biological findings (Y1068≡Y1221 homology, cross-type attention)
│
├── 4. Quick Start
│   ├── Installation (pip + Docker)
│   ├── Data download (link to archive + manual guide)
│   ├── Running the full pipeline (make all)
│   └── Individual steps (expandable)
│
├── 5. Configuration
│   ├── PTM type registry (how subtypes are defined)
│   ├── Protein registry (how sites are defined)
│   ├── Model hyperparameters
│   ├── Training settings
│   └── XAI configuration (homology pairs, effector slots)
│
├── 6. How to Extend
│   ├── Adding a new protein (config example)
│   ├── Adding a new PTM type + subtypes (config example)
│   ├── Adding a new drug (config example)
│   └── Creating a new case study
│
├── 7. Evaluation & Benchmarking
│   ├── Ablation study (5 arms)
│   ├── Benchmarking against 8 external methods
│   ├── Statistical rigor (Bootstrap CIs, DeLong, Wilcoxon, BH)
│   ├── Cell-blind LOCLO generalization
│   └── Randomized PTM control
│
├── 8. Explainability (XAI)
│   ├── Per-mod-type Integrated Gradients
│   ├── Cross-type attention analysis
│   ├── Cross-receptor homology check
│   └── Biological validation summary
│
├── 9. Project Structure
│   ├── Package diagram (ptm_bdl/, case_studies/)
│   └── File reference table
│
├── 10. Citation
├── 11. Authors
└── 12. License
```

**Content sources** (merge and rewrite, not copy-paste):

| Current file | Where it goes in README | Action |
|-------------|------------------------|--------|
| `How_to_Run.md` | §4 Quick Start, §9 Project Structure | Merge |
| `Scientific_Explanation.md` | §2 Architecture, §3 Case Study | Merge |
| `PTM-BDL_One_Page_Summary.md` | §1 Overview, §2 Architecture | Merge |
| `DATA_DOWNLOAD_GUIDE.md` | §4 Quick Start (data download) | Merge |
| `DATA_BIOLOGICAL_VALIDATION.md` | §8 Explainability | Merge relevant parts |
| Current `README.md` | Foundation — rewrite in framework-first framing | Rewrite |

#### 10b. `CLAUDE.md` — Concise AI Assistant Instructions

Rewrite to reflect new package structure. Should be SHORT (under 150 lines):

```
CLAUDE.md
├── Project Overview (2-3 sentences: "PTM-BDL framework + EGFR/ERBB2 case study")
├── Package Structure (ptm_bdl.model, ptm_bdl.data, ptm_bdl.training, etc.)
├── How to Run (venv activation + key commands)
├── Key Files to Understand First (registry.py, encoder.py, predictor.py, dataset.py)
├── Common Tasks (add protein, add PTM type, add drug, run evaluation)
├── Important Conventions (numbering, tokens, subtypes)
└── Dependencies
```

#### 10c. `docs/` — Only 2 Reference Documents

| Keep | Purpose |
|------|---------|
| `docs/ARCHITECTURE.md` | **Deep technical reference** — Detailed PTM-BDL architecture with equations, §7.4–§7.7 design decisions, type gate math, residual gate math. Consolidates content from `PTM_Biological_Dynamics_Layer.md`. Linked from README §2. |
| `docs/PAPER_REFERENCES.md` | **Bibliography** — All literature references organized by topic. Linked from README. |

#### 10d. Files to Remove (Historical / Superseded)

| File | Reason |
|------|--------|
| `docs/COMPREHENSIVE_EVALUATION_23_june.md` | Historical evaluation — results are in `results/*.json` |
| `docs/COMPREHENSIVE_EVALUATION_24_june.md` | Same — superseded by 28 June version |
| `docs/COMPREHENSIVE_EVALUATION_28_june.md` | Same — results live in JSON files |
| `docs/EXPANSION_FEASIBILITY_ANALYSIS.md` | Historical — HER2 expansion is now complete |
| `docs/HER2_EXPANSION_PLAN.md` | Historical — HER2 is fully integrated |
| `docs/BENCHMARKING_PLAN.md` | Merge essential parts into README §7, remove |
| `docs/PTM_Biological_Dynamics_Layer.md` | Superseded by `docs/ARCHITECTURE.md` |
| `docs/PTM-BDL_One_Page_Summary.md` | Merged into README §1–§2 |
| `docs/Scientific_Explanation.md` | Merged into README §2–§3 |
| `docs/How_to_Run.md` | Merged into README §4 |
| `docs/DATA_DOWNLOAD_GUIDE.md` | Merged into README §4 |
| `docs/DATA_BIOLOGICAL_VALIDATION.md` | Merged into README §8 |
| `docs/REFRAMING_TASK.md` | Remove after refactoring is complete |

#### 10e. Writing Principles for New Documentation

1. **Framework-first framing:** "PTM-BDL is a typed self-attention framework for learning how PTMs drive drug response" — not "a tool for EGFR resistance prediction"
2. **No historical references:** Don't mention "previously we had...", "this was changed from...", dated evaluations, or migration steps
3. **Config examples over code examples:** Show how to extend via config, not by editing Python files
4. **One canonical example:** The EGFR/ERBB2 case study is the DEMONSTRATION, not the PRODUCT
5. **Extensibility is the headline:** "Adding a new protein requires zero code changes" should be prominent

### Phase 11: Update Build/CI

| File | Changes |
|------|---------|
| `Makefile` | Update targets to call case study scripts |
| `pyproject.toml` | Add `ptm_bdl` and `case_studies` as packages |
| `docker-compose.yml` | Update entry points |
| `.gitignore` | No changes needed |

---

## 6. MIGRATION ORDER (Dependency-Safe)

The phases must be executed in this order to avoid breaking the pipeline at any step:

```
Phase 1: Registry + Config     ← No existing code changes yet, only new files
Phase 2: Model refactor        ← Model uses registry; legacy wrapper preserves imports
Phase 3: Data package          ← Dataset uses registry; legacy imports still work
Phase 4: Training package      ← Extracts from step11_train.py
Phase 5: Evaluation package    ← Extracts from step12, step14a-c
Phase 6: XAI package           ← Extracts from step13
Phase 7: Case study package    ← Moves application biology out of framework
Phase 8: Config migration      ← Restructure config.yaml
Phase 9: Tests                 ← Update imports + add backward-compat tests
Phase 10: Documentation        ← Rewrite docs
Phase 11: Build/CI             ← Update Makefile, pyproject.toml
```

At each phase: existing scripts continue to work via backward-compatible imports from `src/models/` and `scripts/step11_train.py`.

After all phases: old scripts become thin entry points that delegate to the framework packages.

---

## 7. EXTENSIBILITY PROOF — Adding BRAF with Acetylation

After refactoring, adding a third protein (BRAF) with a new PTM type (acetylation) requires ONLY config changes:

```yaml
# In ptm_type_registry:
acetyl:
  description: "Acetylation"
  subtypes:
    K: {description: "lysine acetylation — blocks ubiquitination"}

# In protein_registry:
BRAF:
  id: 2
  uniprot: "P15056"
  sites:
    phospho:
      max_slots: 8
      entries:
        - {position: 599, residue: "S599", amino_acid: "S", function: "activation loop"}
        - {position: 601, residue: "T601", amino_acid: "T", function: "activation loop"}
        # ... 6 real + 2 pad → max_slots: 8
    acetyl:
      max_slots: 4
      entries:
        - {position: 601, residue: "K601", amino_acid: "K", function: "acetylation site"}
        # ... 2 real + 2 pad → max_slots: 4
```

**Zero code changes.** The registry auto-assigns `acetyl_K = subtype_id 4`, builds 12-token BRAF config (8 phospho + 4 acetyl), extends `type_id_table` to 3 rows, and the model's `nn.Embedding(n_subtypes=5, d_model)` and `nn.Embedding(n_proteins=3, d_model)` adjust automatically.

---

## 8. WHAT STAYS THE SAME

These parts of the codebase are **already framework-level** and need minimal or no changes:

| Component | Why it's already general |
|-----------|------------------------|
| `StaticJointTransformer` | Cross-modal attention for any seq/struct/drug tokens |
| `BilinearLateFusion` | S_rep ⊙ P_rep — agnostic to what's inside each |
| `AttentionPooling` | Ilse et al. — general attention pooling |
| `FocalLoss` | Class-conditional focal loss — general |
| Early stopping on max(AUROC, BAcc) | General training strategy |
| ESM-2, GearNet, ChemBERTa extractors | Pretrained encoders — already protein/drug-agnostic |
| Bootstrap CIs, DeLong, Wilcoxon, BH | Statistical tests — fully general |
| `_stitch()` method | `[level, delta, ratio]` construction — works for any PTM values |
| Residual gate, type gate mechanisms | Architectural components — already generic |

---

## 9. ACCEPTANCE CRITERIA

1. **Backward compatibility:** `python scripts/step12_evaluate.py` produces identical `evaluation_report.json` with refactored code
2. **Config extensibility:** Adding a new protein section to config.yaml (with sites) and running `build_model_from_cfg()` produces a working model with the correct embedding sizes — no code changes
3. **PTM type extensibility:** Adding a new PTM type with subtypes in `ptm_type_registry` and running `build_model_from_cfg()` produces a model with the correct `N_PTM_TYPES` — no code changes
4. **No duplicate labels:** Site labels, type maps, and pad masks exist in exactly ONE place (the config or the registry)
5. **Clean imports:** No `sys.path.insert` hacks — all imports are proper package imports
6. **Tests pass:** All existing tests pass, plus new registry backward-compat tests
7. **Weight loading:** Existing `best_model.pt` can be loaded into refactored model (same parameter names) and produces identical outputs

---

## 10. SECOND CASE STUDY — ABL1/BCR-ABL TKI Resistance in CML (3 PTM Types)

### 10a. Purpose

A second, independent case study that proves the framework is truly general by applying it to:
- A **different protein family** (non-receptor tyrosine kinase vs. receptor TKI)
- A **different cancer** (CML vs. NSCLC/breast)
- **Three PTM types** (phospho + acetylation + ubiquitination) — extending beyond the 2-type EGFR/ERBB2 demonstration

This is the ultimate extensibility proof: the SAME PTM-BDL architecture, trained on a completely different biological system with a third PTM type, using ONLY config changes to the refactored framework.

### 10b. Why ABL1/BCR-ABL (Recommended)

| Dimension | EGFR/ERBB2 (Case Study 1) | ABL1/BCR-ABL (Case Study 2) |
|-----------|--------------------------|----------------------------|
| Protein family | Receptor tyrosine kinase (ERBB) | Non-receptor tyrosine kinase (ABL) |
| Cancer | NSCLC + Breast | Chronic Myeloid Leukemia (CML) |
| Gatekeeper mutation | T790M (EGFR) | T315I (ABL1) — same resistance mechanism, different protein |
| Drug generations | 1st→2nd→3rd gen EGFR TKIs | 1st→2nd→3rd gen BCR-ABL TKIs |
| PTM types | 2 (phospho, glyco) | **3 (phospho, acetylation, ubiquitination)** |

**The T790M/T315I parallel is a compelling story:** Both are gatekeeper mutations that block drug binding, both drive sequential generations of TKIs, and both involve PTM signaling rewiring as a resistance mechanism. If PTM-BDL learns this in TWO independent protein families, it proves the architecture captures fundamental biology, not protein-specific patterns.

### 10c. Three PTM Types for ABL1

| PTM Type | Subtypes | Sites | Biological Role in CML Resistance |
|----------|----------|-------|----------------------------------|
| **Phosphorylation** | phospho_Y, phospho_S, phospho_T | Y245 (SH2-kinase linker), Y412 (activation loop), Y393 (catalytic), S69, T315 | Direct TKI target — imatinib blocks Y412 autophosphorylation; T315I restores it |
| **Acetylation** | acetyl_K | K282 (SH2-kinase linker) | HDAC-regulated; acetylation at K282 modulates ABL1 kinase activity independent of phosphorylation (Dai et al., Genes Dev 2004). HDAC inhibitors synergize with imatinib via this site — a known combination therapy mechanism |
| **Ubiquitination** | ubiq_K48, ubiq_K63 | K117, K135, K24 | Controls BCR-ABL protein stability via c-Cbl (K48-linked → degradation) and CHIP. K63-linked ubiquitin at K135 serves as a signaling scaffold. Resistance can emerge from decreased degradation (Grossmann et al., Cell 2004) |

**This extends the type registry to 7 subtypes:**
```
phospho_Y  = 0     phospho_S = 1     phospho_T  = 2    ← from Case Study 1
glyco_N    = 3                                          ← from Case Study 1
acetyl_K   = 4                                          ← NEW in Case Study 2
ubiq_K48   = 5     ubiq_K63  = 6                        ← NEW in Case Study 2
```

### 10d. Data Availability (Verified 2026-07-04)

#### DrugPTM-Bench Verification Results

DrugPTM-Bench contains 7 cell lines. Acetylation data exists **ONLY in HeLa** (58,985 rows), and exclusively on histones/chromatin proteins — NOT on kinase drug targets:

| Cell Line | Cancer | Phospho Rows | Acetylation Rows |
|-----------|--------|-------------|-----------------|
| A431 | Skin (EGFR WT) | 3,533,035 | 0 |
| A549 | NSCLC | 3,012,097 | 0 |
| BT-474 | Breast (HER2+) | 270,170 | 0 |
| **HeLa** | Cervical | 921,623 | **58,985** (histones/EP300/CREBBP only) |
| **K562** | **CML (BCR-ABL)** | **1,608,421** | 0 |
| MDA-MB-175 | Breast | 223,630 | 0 |
| RPMI8226 | Myeloma | 1,105,560 | 0 |

**ABL1 phospho in K562: ✅ CONFIRMED**
- 1,832 ABL1 rows (UniProt P00519-2;P00519)
- 459 additional ABL1;ABL2 shared peptide rows
- ~29 unique phosphosites (Y, S, T residues)
- Drugs: **Dasatinib (1,246 rows)**, **Imatinib (286 rows)**, Methotrexat (130), Cytarabine (100), Paclitaxel (70)
- Full dose-response curves with EC50, pEC50, R², curve effect size

**ABL1 acetylation in DrugPTM-Bench: ❌ ZERO rows** in all 7 cell lines.

**ABL1 ubiquitination in DrugPTM-Bench: ❌ Not measured** (DrugPTM-Bench only has phospho + acetylation).

#### Data Sourcing Strategy: Parallels Current Case Study

This mirrors exactly how the EGFR/ERBB2 case study was built — phospho from DrugPTM-Bench, glycosylation from 5 external publications (MCP 2025, Taniguchi 2024, Garnham 2021, Sethi 2020, Ruprecht 2017). The ABL1 case study follows the same pattern:

| PTM Type | Source | Data Type | Status |
|----------|--------|-----------|--------|
| **Phosphorylation** | DrugPTM-Bench (K562) | Dose-response (Dasatinib + Imatinib) | ✅ Verified: 1,832 rows |
| **Phosphorylation** | GDSC2 | IC50 drug response | ✅ Available: Imatinib, Dasatinib, Nilotinib, Bosutinib, Ponatinib |
| **Acetylation** | Dai et al., Genes Dev 2004 (PMID 14701881) | K282 acetylation quantitation, imatinib-sensitive vs resistant | **Published**: Shows HDAC inhibitors restore K282ac + synergize with imatinib |
| **Acetylation** | Nimmanapalli et al., Cancer Res 2003 (PMID 14633664) | LAQ824 (HDAC inhibitor) induces BCR-ABL acetylation → degradation in K562 | **Published**: Dose-response acetylation in K562 specifically |
| **Acetylation** | PhosphoSitePlus | Curated ABL1 acetylation sites + literature references | **Public database** |
| **Ubiquitination** | Grossmann et al., Cell 2004 (PMID 15382145) | c-Cbl ubiquitinates BCR-ABL → proteasomal degradation | **Published**: Mechanism + site identification |
| **Ubiquitination** | Mao et al., Blood 2010 (PMID 20068224) | CHIP E3 ligase ubiquitinates BCR-ABL; HSP90 inhibitors trigger degradation | **Published**: K48-linked ubiq quantitation |
| **Ubiquitination** | PhosphoSitePlus | Curated ABL1 ubiquitination sites (K117, K135, K24, K1070) | **Public database** |
| **Structure** | PDB: 1IEP (WT apo), 1OPJ (imatinib), 2GQG (dasatinib), 3CS9 (T315I) | Crystal structures | ✅ Available |
| **Sequence** | UniProt P00519 | ABL1 full sequence + PTM annotations | ✅ Available |

### 10e. Config-Only Extension (After Refactoring)

With the refactored framework from Phases 1–11, adding this case study requires:

```yaml
# In ptm_type_registry — add 2 new PTM types:
acetyl:
  description: "Acetylation"
  subtypes:
    K: {description: "lysine acetylation — HDAC-regulated kinase modulation"}

ubiq:
  description: "Ubiquitination"
  subtypes:
    K48: {description: "K48-linked — proteasomal degradation signal"}
    K63: {description: "K63-linked — signaling scaffold"}

# In protein_registry — add ABL1:
ABL1:
  id: 2
  uniprot: "P00519"
  sites:
    phospho:
      max_slots: 8
      entries:
        - {position: 245, residue: "Y245", amino_acid: "Y", function: "SH2-kinase linker autophosphorylation"}
        - {position: 412, residue: "Y412", amino_acid: "Y", function: "activation loop — imatinib target"}
        - {position: 393, residue: "Y393", amino_acid: "Y", function: "catalytic loop"}
        - {position: 69,  residue: "S69",  amino_acid: "S", function: "SH3 domain regulatory"}
        - {position: 315, residue: "T315", amino_acid: "T", function: "gatekeeper — T315I resistance mutation"}
        # ... (5 real + 3 pad → max_slots: 8)
    acetyl:
      max_slots: 4
      entries:
        - {position: 282, residue: "K282", amino_acid: "K", function: "SH2-kinase linker — HDAC target"}
        # ... (1 real + 3 pad → max_slots: 4)
    ubiq:
      max_slots: 4
      entries:
        - {position: 117, residue: "K117", amino_acid: "K48", function: "c-Cbl degradation signal"}
        - {position: 135, residue: "K135", amino_acid: "K63", function: "signaling scaffold"}
        - {position: 24,  residue: "K24",  amino_acid: "K48", function: "CHIP-mediated degradation"}
        # ... (3 real + 1 pad → max_slots: 4)
```

**n_tokens for ABL1 = 8 (phospho) + 4 (acetyl) + 4 (ubiq) = 16 tokens**
**n_subtypes = 7** (phospho_Y, phospho_S, phospho_T, glyco_N, acetyl_K, ubiq_K48, ubiq_K63)
**n_proteins = 3** (EGFR, ERBB2, ABL1)

The registry auto-sizes all embeddings. Zero code changes.

### 10f. Alternative Candidate: BRAF (Melanoma)

If ABL1 data proves insufficient, BRAF V600E melanoma is the backup:

| Dimension | BRAF |
|-----------|------|
| Protein family | Serine/Threonine kinase (RAF family) — different catalytic mechanism |
| Cancer | Melanoma |
| Drugs (GDSC) | Vemurafenib, Dabrafenib, Trametinib (MEK inhibitor for combination) |
| PTM types | Phospho (S445, T599, S602) + Acetylation + potentially SUMOylation |
| Gatekeeper | V600E (constitutive activation, not gatekeeper — different mechanism) |

BRAF is less ideal than ABL1 because: (a) it's a Ser/Thr kinase not a Tyr kinase, so phospho_Y subtypes would be absent, and (b) the V600E mechanism is constitutive activation rather than gatekeeper resistance, making the biological parallel to EGFR T790M weaker.

### 10g. Implementation Steps

This case study is a **separate task** executed AFTER Phases 1–11:

1. **Verify DrugPTM-Bench data for ABL1** — Check K562 CSV for ABL1 phospho rows with imatinib/dasatinib dose-response
2. **Curate ABL1 PTM site data** — Gather phospho (PhosphoSitePlus), acetylation (Dai 2004), ubiquitination (Grossmann 2004, PhosphoSitePlus) quantitation
3. **Download GDSC2 CML data** — Filter for K562, KCL22, LAMA84, KU812 etc. × 5 BCR-ABL TKIs
4. **Add ABL1 config** — ptm_type_registry + protein_registry + drugs + XAI homology pairs
5. **Build data pipeline** — `case_studies/abl1_cml_tki/data_pipeline/` (download, harmonize, extract features)
6. **Train and evaluate** — Using the SAME framework packages (ptm_bdl.training, ptm_bdl.evaluation, ptm_bdl.xai)
7. **Cross-case-study comparison** — Train on both case studies separately, compare PTM-BDL's ability to learn resistance biology in two independent protein families
8. **Publication figure** — Side-by-side: EGFR/ERBB2 (2 PTM types, receptor TK) vs ABL1 (3 PTM types, non-receptor TK) → same architecture, same biological discovery capability
