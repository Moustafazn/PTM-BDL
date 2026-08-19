"""
ResistanceDataset — Generic PyTorch Dataset for PTM-BDL training.

Auto-discovers PTM columns and protein mappings from the CSV.
No hardcoded column names, protein IDs, or PTM types.

All PTM sites (regardless of modification type) are concatenated into a
single flat ``ptm_vector`` of size ``n_tokens``.  The PTMTypeRegistry's
``type_id_table`` and type embeddings in the encoder handle per-type
differentiation — every PTM type is treated equally at the data level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ResistanceDataset(Dataset):
    """
    Generic PyTorch Dataset for PTM-BDL training.

    Auto-discovers PTM columns from CSV headers and builds a single flat
    ``ptm_vector`` containing all PTM sites across all modification types.
    Type differentiation is handled by the registry + type embeddings in
    the encoder — not at the data level.

    Column discovery:
      • ``ptm_*`` columns  → PTM site values (named by residue)
      • ``*_slot*`` columns → additional PTM site values (positional naming)
      Both groups are concatenated into a single vector in discovery order.

    Ablation modes (tool-level):
      • ``"full"``          — all features active
      • ``"no_ptm"``        — zero ALL PTM features (static-only baseline)
      • ``"no_drug"``       — zero drug embeddings (drug_emb + drug_pooled)
      • ``"no_structure"``  — zero structural embeddings (struct_emb)
      • ``"baseline_only"`` — keep PTM levels, zero all deltas (prospective scenario)
      • ``"delta_only"``    — set PTM levels to 1.0, keep deltas (dynamic signal only)
      • ``"measured_only"`` — zero PTM for propagated samples (confidence < 0.90)

    Per-PTM-type ablation:
      Pass ``zero_slot_range=(start, end)`` to zero a specific PTM type's
      slots.  The range is computed from the registry at the case-study
      level using ``registry.get_ptm_type_slot_range(ptm_type)``.
    """

    def __init__(self, dataset_csv: Path | str, features_dir: Path | str,
                 ablation_mode: str = "full",
                 zero_slot_range: tuple[int, int] | None = None):
        self.df = pd.read_csv(dataset_csv)
        self.features_dir = Path(features_dir)
        self.ablation_mode = ablation_mode
        self.zero_slot_range = zero_slot_range

        self._discover_columns()
        self._discover_proteins()
        self._load_embeddings()

    # ── Column & protein auto-discovery ──────────────────────────────────

    def _discover_columns(self):
        """Auto-discover all PTM columns from CSV headers.

        Two naming conventions are supported and concatenated:
          • ``ptm_{residue}``      — named by biological residue
          • ``{type}_slot{i}``     — positional naming for additional types

        Both are merged into ``_all_level_cols`` / ``_all_delta_cols``
        so that every PTM type is part of one flat vector.
        """
        all_cols = self.df.columns.tolist()

        # Named PTM columns: ptm_* (excluding delta_ptm_* and non-numeric)
        self._ptm_cols = [c for c in all_cols
                          if c.startswith("ptm_")
                          and not c.startswith("ptm_pad")
                          and self.df[c].dtype in ("float64", "float32", "int64", "int32")]
        self._delta_ptm_cols = [c for c in all_cols if c.startswith("delta_ptm_")]

        # Positional PTM columns: *_slot* (any additional PTM types)
        self._secondary_cols = [c for c in all_cols
                                if '_slot' in c
                                and not c.startswith('delta_')
                                and not c.startswith('ptm_')]
        self._delta_secondary_cols = [c for c in all_cols
                                      if c.startswith('delta_')
                                      and '_slot' in c]

        # Track dimensions (useful for case-study-level per-type ablation)
        self._ptm_dim = len(self._ptm_cols)
        self._secondary_dim = len(self._secondary_cols)

        # Flat column lists: ALL PTM columns concatenated
        self._all_level_cols = self._ptm_cols + self._secondary_cols
        self._all_delta_cols = self._delta_ptm_cols + self._delta_secondary_cols

    def _discover_proteins(self):
        """Build protein_name → integer ID mapping from data."""
        if "target_protein" in self.df.columns:
            unique_proteins = sorted(self.df["target_protein"].dropna().unique())
            self._protein_map = {name: idx for idx, name in enumerate(unique_proteins)}
        else:
            self._protein_map = {}

    # ── Embedding loading ────────────────────────────────────────────────

    def _load_embeddings(self):
        """Load all pre-extracted embeddings from data/features/.

        Handles two naming conventions produced by different case studies:
          Convention 1: *_per_residue.npy, *_residue_embeddings.npy, *_per_token.npy
          Convention 2: *_esm2.npy, *_gearnet.npy, *_tokens.npy
        """
        esm2_dir = self.features_dir / "esm2"
        gearnet_dir = self.features_dir / "gearnet"
        chemberta_dir = self.features_dir / "chemberta"

        self.seq_embeddings = {}
        self.struct_embeddings = {}
        self.drug_embeddings = {}
        self.drug_pooled = {}

        if esm2_dir.exists():
            for f in esm2_dir.glob("*_per_residue.npy"):
                seq_id = f.stem.replace("_per_residue", "")
                self.seq_embeddings[seq_id] = np.load(f)
            for f in esm2_dir.glob("*_esm2.npy"):
                seq_id = f.stem.replace("_esm2", "")
                if seq_id not in self.seq_embeddings:
                    self.seq_embeddings[seq_id] = np.load(f)

        if gearnet_dir.exists():
            for f in gearnet_dir.glob("*_residue_embeddings.npy"):
                pdb_id = f.stem.replace("_residue_embeddings", "")
                self.struct_embeddings[pdb_id] = np.load(f)
            for f in gearnet_dir.glob("*_gearnet.npy"):
                pdb_id = f.stem.replace("_gearnet", "")
                if pdb_id not in self.struct_embeddings:
                    self.struct_embeddings[pdb_id] = np.load(f)

        if chemberta_dir.exists():
            for f in chemberta_dir.glob("*_per_token.npy"):
                drug_key = f.stem.replace("_per_token", "")
                self.drug_embeddings[drug_key] = np.load(f)
                pooled_path = chemberta_dir / f"{drug_key}_pooled.npy"
                if pooled_path.exists():
                    self.drug_pooled[drug_key] = np.load(pooled_path)
            for f in chemberta_dir.glob("*_tokens.npy"):
                drug_key = f.stem.replace("_tokens", "")
                if drug_key not in self.drug_embeddings:
                    self.drug_embeddings[drug_key] = np.load(f)
                if drug_key not in self.drug_pooled:
                    pooled_path = chemberta_dir / f"{drug_key}_pooled.npy"
                    if pooled_path.exists():
                        self.drug_pooled[drug_key] = np.load(pooled_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Sequence embedding (ESM-2) ────────────────────────────────────
        seq_id = row.get("sequence_id", "wild_type")
        seq_emb = self.seq_embeddings.get(seq_id)
        if seq_emb is None:
            seq_emb = np.random.randn(100, 1280).astype(np.float32)

        # ── Structural embedding (GearNet) ────────────────────────────────
        pdb_id = row.get("pdb_id", "default")
        struct_emb = self.struct_embeddings.get(pdb_id)
        if struct_emb is None:
            struct_emb = np.random.randn(200, 512).astype(np.float32)

        # ── Drug embedding (ChemBERTa) ────────────────────────────────────
        drug_name = str(row.get("drug_name", "unknown")).lower()
        drug_key = drug_name.split()[0] if drug_name else "unknown"
        drug_emb = self.drug_embeddings.get(drug_key)
        drug_pool = self.drug_pooled.get(drug_key)
        if drug_emb is None:
            drug_emb = np.random.randn(20, 384).astype(np.float32)
            drug_pool = np.random.randn(384).astype(np.float32)
        if drug_pool is None:
            drug_pool = drug_emb.mean(axis=0)

        # ── PTM vector (all types, single flat vector) ────────────────────
        level_values = []
        for col in self._all_level_cols:
            val = row.get(col, 1.0)
            level_values.append(float(val) if pd.notna(val) else 1.0)
        ptm_vector = np.array(level_values, dtype=np.float32) if level_values else np.ones(1, dtype=np.float32)

        delta_values = []
        for col in self._all_delta_cols:
            val = row.get(col, 0.0)
            delta_values.append(float(val) if pd.notna(val) else 0.0)
        delta_ptm_vector = np.array(delta_values, dtype=np.float32) if delta_values else np.zeros(len(ptm_vector), dtype=np.float32)

        # ── target_protein → integer ID ───────────────────────────────────
        tp_str = str(row.get("target_protein", "unknown")).upper()
        target_protein = self._protein_map.get(tp_str, 0)

        # ── propagation_confidence ────────────────────────────────────────
        prop_conf = float(row.get("propagation_confidence", 0.5))
        if pd.isna(prop_conf):
            prop_conf = 0.5

        # ── ABLATION: zero features based on mode ─────────────────────────
        if self.ablation_mode == "no_ptm":
            ptm_vector = np.ones_like(ptm_vector)
            delta_ptm_vector = np.zeros_like(delta_ptm_vector)
        elif self.ablation_mode == "no_drug":
            drug_emb = np.zeros_like(drug_emb)
            drug_pool = np.zeros_like(drug_pool)
        elif self.ablation_mode == "no_structure":
            struct_emb = np.zeros_like(struct_emb)
        elif self.ablation_mode == "baseline_only":
            delta_ptm_vector = np.zeros_like(delta_ptm_vector)
        elif self.ablation_mode == "delta_only":
            ptm_vector = np.ones_like(ptm_vector)
        elif self.ablation_mode == "measured_only":
            if prop_conf < 0.90:
                ptm_vector = np.ones_like(ptm_vector)
                delta_ptm_vector = np.zeros_like(delta_ptm_vector)

        # ── Per-PTM-type ablation (zero specific slot range) ──────────────
        # Used by case studies to ablate individual PTM types, e.g.:
        #   registry.get_ptm_type_slot_range("glyco") → (12, 24)
        #   dataset = ResistanceDataset(..., zero_slot_range=(12, 24))
        if self.zero_slot_range is not None:
            s, e = self.zero_slot_range
            ptm_vector[s:e] = 1.0       # reset levels to WT baseline
            delta_ptm_vector[s:e] = 0.0  # zero drug-induced changes

        # ── Targets ───────────────────────────────────────────────────────
        ln_ic50 = float(row.get("ln_ic50", 0.0))
        resistance = int(row.get("resistance_label", 0))

        return {
            "seq_emb": torch.from_numpy(seq_emb.astype(np.float32)),
            "struct_emb": torch.from_numpy(struct_emb.astype(np.float32)),
            "drug_emb": torch.from_numpy(drug_emb.astype(np.float32)),
            "drug_pooled": torch.from_numpy(drug_pool.astype(np.float32)),
            "ptm_vector": torch.from_numpy(ptm_vector),
            "delta_ptm_vector": torch.from_numpy(delta_ptm_vector),
            "target_protein": torch.tensor(target_protein, dtype=torch.long),
            "propagation_confidence": torch.tensor([prop_conf], dtype=torch.float32),
            "ln_ic50": torch.tensor([ln_ic50], dtype=torch.float32),
            "resistance_label": torch.tensor([resistance], dtype=torch.float32),
        }
