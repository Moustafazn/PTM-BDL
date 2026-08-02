"""
ResistanceDataset — Generic PyTorch Dataset for PTM-BDL training.

Auto-discovers PTM columns and protein mappings from the CSV.
No hardcoded column names, protein IDs, or PTM types.
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

    Auto-discovers:
      • ``ptm_*`` columns → ptm_vector (primary PTM channel)
      • ``*_slot*`` columns → secondary_vector (secondary channel, if present)
      • ``target_protein`` unique values → sequential integer IDs

    Ablation modes:
      • "full"            — all features active
      • "no_ptm"          — zero ALL PTM features (static baseline)
      • "no_secondary"    — zero secondary channel only
      • "secondary_only"  — zero primary channel, keep secondary
      • "no_drug"         — zero drug embeddings (drug_emb + drug_pooled)
      • "no_structure"    — zero structural embeddings (struct_emb)
      • "baseline_only"   — keep PTM levels, zero all deltas (no drug-induced changes)
                            Tests prospective DRP scenario: can we predict IC50 from
                            baseline PTM state alone (no drug-PTM interaction data)?
      • "delta_only"      — set PTM levels to 1.0 (WT), keep deltas (purely dynamic)
                            Isolates pharmacodynamic signal: does the drug-induced PTM
                            change carry predictive value beyond baseline state?
                            Note: ratio channel becomes delta/1.0 ≈ delta; this is
                            intentional — it represents the correct ratio when baseline
                            is assumed WT (unknown patient baseline scenario).
    """

    def __init__(self, dataset_csv: Path | str, features_dir: Path | str,
                 ablation_mode: str = "full"):
        self.df = pd.read_csv(dataset_csv)
        self.features_dir = Path(features_dir)
        self.ablation_mode = ablation_mode

        self._discover_columns()
        self._discover_proteins()
        self._load_embeddings()

    # ── Column & protein auto-discovery ──────────────────────────────────

    def _discover_columns(self):
        """Auto-discover primary and secondary PTM columns from CSV headers."""
        all_cols = self.df.columns.tolist()

        # Primary channel: ptm_* columns (excluding delta_ptm_* and non-numeric metadata)
        self._ptm_cols = [c for c in all_cols
                          if c.startswith("ptm_")
                          and not c.startswith("ptm_pad")
                          and self.df[c].dtype in ("float64", "float32", "int64", "int32")]
        self._delta_ptm_cols = [c for c in all_cols if c.startswith("delta_ptm_")]

        # Secondary channel: *_slot* columns (any additional PTM types)
        self._secondary_cols = [c for c in all_cols
                                if '_slot' in c
                                and not c.startswith('delta_')
                                and not c.startswith('ptm_')]
        self._delta_secondary_cols = [c for c in all_cols
                                      if c.startswith('delta_')
                                      and '_slot' in c]

        self._ptm_dim = len(self._ptm_cols)
        self._secondary_dim = len(self._secondary_cols)

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
          EGFR:       *_per_residue.npy, *_residue_embeddings.npy, *_per_token.npy
          K562/HeLa:  *_esm2.npy,        *_gearnet.npy,            *_tokens.npy
        """
        esm2_dir = self.features_dir / "esm2"
        gearnet_dir = self.features_dir / "gearnet"
        chemberta_dir = self.features_dir / "chemberta"

        self.seq_embeddings = {}
        self.struct_embeddings = {}
        self.drug_embeddings = {}
        self.drug_pooled = {}

        if esm2_dir.exists():
            # Convention 1 (EGFR): wild_type_per_residue.npy → key "wild_type"
            for f in esm2_dir.glob("*_per_residue.npy"):
                seq_id = f.stem.replace("_per_residue", "")
                self.seq_embeddings[seq_id] = np.load(f)
            # Convention 2 (K562/HeLa): abl1_esm2.npy → key "abl1"
            for f in esm2_dir.glob("*_esm2.npy"):
                seq_id = f.stem.replace("_esm2", "")
                if seq_id not in self.seq_embeddings:
                    self.seq_embeddings[seq_id] = np.load(f)

        if gearnet_dir.exists():
            # Convention 1 (EGFR): 2GS6_residue_embeddings.npy → key "2GS6"
            for f in gearnet_dir.glob("*_residue_embeddings.npy"):
                pdb_id = f.stem.replace("_residue_embeddings", "")
                self.struct_embeddings[pdb_id] = np.load(f)
            # Convention 2 (K562/HeLa): 1IEP_gearnet.npy → key "1IEP"
            for f in gearnet_dir.glob("*_gearnet.npy"):
                pdb_id = f.stem.replace("_gearnet", "")
                if pdb_id not in self.struct_embeddings:
                    self.struct_embeddings[pdb_id] = np.load(f)

        if chemberta_dir.exists():
            # Convention 1 (EGFR): afatinib_per_token.npy → key "afatinib"
            for f in chemberta_dir.glob("*_per_token.npy"):
                drug_key = f.stem.replace("_per_token", "")
                self.drug_embeddings[drug_key] = np.load(f)
                pooled_path = chemberta_dir / f"{drug_key}_pooled.npy"
                if pooled_path.exists():
                    self.drug_pooled[drug_key] = np.load(pooled_path)
            # Convention 2 (K562/HeLa): dasatinib_tokens.npy → key "dasatinib"
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

        # ── Primary PTM channel (auto-discovered ptm_* columns) ──────────
        ptm_values = []
        for col in self._ptm_cols:
            val = row.get(col, 1.0)
            ptm_values.append(float(val) if pd.notna(val) else 1.0)
        ptm_vector = np.array(ptm_values, dtype=np.float32) if ptm_values else np.ones(1, dtype=np.float32)

        delta_ptm_values = []
        for col in self._delta_ptm_cols:
            val = row.get(col, 0.0)
            delta_ptm_values.append(float(val) if pd.notna(val) else 0.0)
        delta_ptm_vector = np.array(delta_ptm_values, dtype=np.float32) if delta_ptm_values else np.zeros(len(ptm_vector), dtype=np.float32)

        # ── Secondary channel (*_slot* columns, if present) ───────────────
        secondary_values = []
        for col in self._secondary_cols:
            val = row.get(col, 1.0)
            secondary_values.append(float(val) if pd.notna(val) else 1.0)
        secondary_vector = np.array(secondary_values, dtype=np.float32) if secondary_values else np.zeros(0, dtype=np.float32)

        delta_secondary_values = []
        for col in self._delta_secondary_cols:
            val = row.get(col, 0.0)
            delta_secondary_values.append(float(val) if pd.notna(val) else 0.0)
        delta_secondary_vector = np.array(delta_secondary_values, dtype=np.float32) if delta_secondary_values else np.zeros(len(secondary_vector), dtype=np.float32)

        # ── target_protein → integer ID (generic) ─────────────────────────
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
            secondary_vector = np.ones_like(secondary_vector) if len(secondary_vector) else secondary_vector
            delta_secondary_vector = np.zeros_like(delta_secondary_vector)
        elif self.ablation_mode == "no_secondary":
            secondary_vector = np.ones_like(secondary_vector) if len(secondary_vector) else secondary_vector
            delta_secondary_vector = np.zeros_like(delta_secondary_vector)
        elif self.ablation_mode == "secondary_only":
            ptm_vector = np.ones_like(ptm_vector)
            delta_ptm_vector = np.zeros_like(delta_ptm_vector)
        elif self.ablation_mode == "no_drug":
            drug_emb = np.zeros_like(drug_emb)
            drug_pool = np.zeros_like(drug_pool)
        elif self.ablation_mode == "no_structure":
            struct_emb = np.zeros_like(struct_emb)
        elif self.ablation_mode == "baseline_only":
            # Keep PTM levels (baseline state), zero all drug-induced changes
            # Token input becomes [level, 0, 0] — "prospective DRP" scenario
            delta_ptm_vector = np.zeros_like(delta_ptm_vector)
            delta_secondary_vector = np.zeros_like(delta_secondary_vector)
        elif self.ablation_mode == "delta_only":
            # Set levels to WT baseline (1.0), keep drug-induced changes
            # Token input becomes [1.0, delta, delta] — purely dynamic signal
            ptm_vector = np.ones_like(ptm_vector)
            if len(secondary_vector):
                secondary_vector = np.ones_like(secondary_vector)
        elif self.ablation_mode == "measured_only":
            # Q7: Keep PTM values ONLY for directly measured samples
            # (propagation_confidence >= 0.90). For propagated samples,
            # reset to WT baseline (1.0 level, 0.0 delta) — tests whether
            # mutation-class propagation priors contribute to prediction.
            if prop_conf < 0.90:
                ptm_vector = np.ones_like(ptm_vector)
                delta_ptm_vector = np.zeros_like(delta_ptm_vector)
                if len(secondary_vector):
                    secondary_vector = np.ones_like(secondary_vector)
                delta_secondary_vector = np.zeros_like(delta_secondary_vector)

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
            "secondary_vector": torch.from_numpy(secondary_vector),
            "delta_secondary_vector": torch.from_numpy(delta_secondary_vector),
            "target_protein": torch.tensor(target_protein, dtype=torch.long),
            "propagation_confidence": torch.tensor([prop_conf], dtype=torch.float32),
            "ln_ic50": torch.tensor([ln_ic50], dtype=torch.float32),
            "resistance_label": torch.tensor([resistance], dtype=torch.float32),
        }
