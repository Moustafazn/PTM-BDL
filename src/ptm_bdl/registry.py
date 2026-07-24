"""
PTM Type Registry — Dynamic subtype system built from config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PTMSiteConfig:
    """Configuration for a single PTM site."""
    position: int
    residue: str
    amino_acid: str
    function: str = ""
    classic_name: str = ""
    homolog_ref: Optional[str] = None


@dataclass
class ProteinPTMConfig:
    """Per-protein PTM site configuration."""
    protein_name: str
    protein_id: int
    uniprot: str
    sites: dict[str, list[PTMSiteConfig]] = field(default_factory=dict)
    max_slots: dict[str, int] = field(default_factory=dict)


class PTMTypeRegistry:
    """
    Dynamic PTM type/subtype system built from config.

    Responsibilities:
      1. Assign contiguous subtype IDs automatically
      2. Build per-protein type_id_table and is_real_table (buffer tensors)
      3. Provide name↔id mappings for XAI reporting
      4. Compute n_subtypes (= N_PTM_TYPES) and n_tokens_per_protein
      5. Provide site labels for each protein

    Usage:
        registry = PTMTypeRegistry.from_config(cfg)
        encoder = PTMBDLEncoder(registry=registry, d_model=64, ...)
    """

    def __init__(
            self,
            ptm_types: dict[str, dict[str, int]],
            ptm_type_descriptions: dict[str, str],
            subtype_descriptions: dict[int, str],
            protein_configs: list[ProteinPTMConfig],
            ptm_type_order: list[str],
    ):
        self._ptm_types = ptm_types  # {ptm_type_name: {amino_acid: subtype_id}}
        self._ptm_type_descriptions = ptm_type_descriptions
        self._subtype_descriptions = subtype_descriptions
        self._protein_configs = {pc.protein_name: pc for pc in protein_configs}
        self._protein_id_map = {pc.protein_name: pc.protein_id for pc in protein_configs}
        self._ptm_type_order = ptm_type_order

        # Compute derived quantities
        self._n_subtypes = max(
            sid for subtypes in ptm_types.values() for sid in subtypes.values()
        ) + 1 if ptm_types else 0

        # Subtype name map: subtype_id → "ptm_type_amino_acid"
        self._subtype_names: dict[int, str] = {}
        self._parent_type_map: dict[int, str] = {}
        for ptm_type, subtypes in ptm_types.items():
            for aa, sid in subtypes.items():
                self._subtype_names[sid] = f"{ptm_type}_{aa}"
                self._parent_type_map[sid] = ptm_type

        # Compute n_tokens (must be same for all proteins in current design)
        self._n_tokens = 0
        if protein_configs:
            tokens_per_protein = {}
            for pc in protein_configs:
                total = sum(pc.max_slots.get(pt, 0) for pt in ptm_type_order)
                tokens_per_protein[pc.protein_name] = total
            # All proteins must have same n_tokens (padded)
            self._n_tokens = max(tokens_per_protein.values()) if tokens_per_protein else 0

        # Build buffer tensors
        self._type_id_table, self._is_real_table = self._build_buffers()

        # Build site labels
        self._site_labels = self._build_site_labels()

        # Build column name mappings
        self._column_names = self._build_column_names()

    @classmethod
    def from_config(cls, cfg: dict) -> "PTMTypeRegistry":
        """
        Build registry from config.yaml.

        Reads from:
          - cfg["ptm_type_registry"] (new format — fully generic)
          - cfg["ptm"] (legacy format — auto-discovers proteins and site keys)

        Assigns subtype IDs in order of (ptm_type, amino_acid) as they appear
        in the config.
        """
        # ── Parse PTM type registry ──────────────────────────────────────
        ptm_type_registry_cfg = cfg.get("ptm_type_registry")
        if ptm_type_registry_cfg:
            return cls._from_new_config(cfg, ptm_type_registry_cfg)
        else:
            return cls._from_legacy_config(cfg)

    @classmethod
    def _from_new_config(cls, cfg: dict, ptr_cfg: dict) -> "PTMTypeRegistry":
        """Build from the new ptm_type_registry config format."""
        ptm_types: dict[str, dict[str, int]] = {}
        ptm_type_descriptions: dict[str, str] = {}
        subtype_descriptions: dict[int, str] = {}
        ptm_type_order: list[str] = []
        sid = 0

        for ptm_type_name, ptm_info in ptr_cfg.items():
            ptm_type_order.append(ptm_type_name)
            ptm_type_descriptions[ptm_type_name] = ptm_info.get("description", "")
            subtypes = {}
            for aa, aa_info in ptm_info.get("subtypes", {}).items():
                subtypes[aa] = sid
                desc = aa_info.get("description", "") if isinstance(aa_info, dict) else str(aa_info)
                subtype_descriptions[sid] = desc
                sid += 1
            ptm_types[ptm_type_name] = subtypes

        # ── Parse protein registry ───────────────────────────────────────
        protein_configs = []
        prot_reg = cfg.get("protein_registry", {})
        for prot_name, prot_info in prot_reg.items():
            sites = {}
            max_slots = {}
            for ptm_type in ptm_type_order:
                ptm_section = prot_info.get("sites", {}).get(ptm_type, {})
                max_s = ptm_section.get("max_slots", 0)
                max_slots[ptm_type] = max_s
                entries = ptm_section.get("entries", [])
                site_list = []
                for entry in entries:
                    site_list.append(PTMSiteConfig(
                        position=entry.get("position", 0),
                        residue=entry.get("residue", ""),
                        amino_acid=entry.get("amino_acid", ""),
                        function=entry.get("function", ""),
                        classic_name=entry.get("classic_name", ""),
                        homolog_ref=entry.get("homolog_ref"),
                    ))
                sites[ptm_type] = site_list

            protein_configs.append(ProteinPTMConfig(
                protein_name=prot_name,
                protein_id=prot_info.get("id", len(protein_configs)),
                uniprot=prot_info.get("uniprot", ""),
                sites=sites,
                max_slots=max_slots,
            ))

        return cls(ptm_types, ptm_type_descriptions, subtype_descriptions,
                   protein_configs, ptm_type_order)

    @classmethod
    def _from_legacy_config(cls, cfg: dict) -> "PTMTypeRegistry":
        """
        Build from the config.yaml format.

        Auto-discovers proteins, site keys, and amino-acid subtypes
        entirely from the config — no hardcoded PTM types.

        Convention (legacy format):
          • The first ``*_sites`` key group → primary channel
            (``ptm_vector``, CSV columns ``ptm_{residue}``)
          • Any additional ``*_sites`` key group → secondary channel
            (``secondary_vector``, CSV columns ``{type}_slot*``)
        """
        ptm_cfg = cfg.get("ptm", {})
        ptm_dim = ptm_cfg.get("ptm_dim", 12)
        # Support both "secondary_dim" and legacy "glyco_dim" config keys
        secondary_dim = ptm_cfg.get("secondary_dim",
                                     ptm_cfg.get("glyco_dim", 0))

        _RESERVED_KEYS = {"ptm_dim", "secondary_dim", "glyco_dim"}

        # ── 1. Discover proteins (any dict-valued key in ptm section) ─────
        protein_names = [
            k for k in ptm_cfg
            if k not in _RESERVED_KEYS and isinstance(ptm_cfg[k], dict)
        ]

        # ── 2. Discover site keys used across all proteins ────────────────
        #   First *_sites group(s) → primary; any remaining → secondary.
        #   Legacy convention: second and subsequent *_sites keys → secondary.
        has_secondary = False
        primary_keys: list[str] = []
        secondary_keys: list[str] = []

        # Discover all unique *_sites keys across proteins in config order.
        # Convention: the FIRST *_sites key → primary channel.
        #             ALL subsequent *_sites keys → secondary channel.
        # This makes the framework truly generic — any PTM type (phospho,
        # glyco, acetyl, ubiquitin, etc.) can be primary or secondary
        # depending on config order.
        all_site_keys_ordered: list[str] = []
        for prot_name in protein_names:
            prot_section = ptm_cfg[prot_name]
            for key in prot_section:
                if key.endswith("_sites") and key not in all_site_keys_ordered:
                    all_site_keys_ordered.append(key)

        for key in all_site_keys_ordered:
            if not primary_keys:
                # First *_sites key → primary channel
                primary_keys.append(key)
            else:
                # Any subsequent *_sites key → secondary channel
                has_secondary = True
                secondary_keys.append(key)

        # ── 3. Discover amino-acid subtypes from actual site entries ──────
        # Scan all proteins to find every unique amino acid used.
        primary_aa_seen: dict[str, str] = {}   # {aa: description}
        secondary_aa_seen: dict[str, str] = {}
        for prot_name in protein_names:
            prot_section = ptm_cfg[prot_name]
            for site_key in primary_keys:
                type_prefix = site_key.replace("_sites", "")  # e.g. "phospho", "acetyl"
                for site in prot_section.get(site_key, []):
                    aa = site.get("amino_acid", "")
                    if aa and aa not in primary_aa_seen:
                        primary_aa_seen[aa] = f"{type_prefix}-{aa}"
            for sec_key in secondary_keys:
                type_prefix = sec_key.replace("_sites", "")
                for site in prot_section.get(sec_key, []):
                    aa = site.get("amino_acid", "N")
                    if aa and aa not in secondary_aa_seen:
                        secondary_aa_seen[aa] = f"{type_prefix}-{aa}"

        # Assign contiguous subtype IDs
        ptm_types: dict[str, dict[str, int]] = {}
        ptm_type_descriptions: dict[str, str] = {}
        subtype_descriptions: dict[int, str] = {}
        sid = 0

        primary_subtypes: dict[str, int] = {}
        for aa, desc in primary_aa_seen.items():
            primary_subtypes[aa] = sid
            subtype_descriptions[sid] = desc
            sid += 1
        ptm_types["primary"] = primary_subtypes
        ptm_type_descriptions["primary"] = "Primary PTM channel"
        ptm_type_order: list[str] = ["primary"]

        if has_secondary:
            sec_subtypes: dict[str, int] = {}
            for aa, desc in secondary_aa_seen.items():
                sec_subtypes[aa] = sid
                subtype_descriptions[sid] = desc
                sid += 1
            ptm_types["secondary"] = sec_subtypes
            ptm_type_descriptions["secondary"] = "Secondary PTM channel"
            ptm_type_order.append("secondary")

        # ── 4. Build per-protein configs ──────────────────────────────────
        protein_configs: list[ProteinPTMConfig] = []
        for prot_idx, prot_name in enumerate(protein_names):
            prot_section = ptm_cfg[prot_name]

            # Primary block: concatenate all primary site keys
            primary_sites: list[PTMSiteConfig] = []
            for site_key in primary_keys:
                for site in prot_section.get(site_key, []):
                    primary_sites.append(PTMSiteConfig(
                        position=site.get("position", 0),
                        residue=site.get("residue", ""),
                        amino_acid=site.get("amino_acid", ""),
                        function=site.get("function", ""),
                        classic_name=site.get("classic_name", ""),
                        homolog_ref=site.get("homolog_ref"),
                    ))

            sites: dict[str, list[PTMSiteConfig]] = {"primary": primary_sites}
            max_slots: dict[str, int] = {"primary": ptm_dim}

            # Secondary block (if present)
            if has_secondary:
                secondary_site_list: list[PTMSiteConfig] = []
                for sec_key in secondary_keys:
                    for site in prot_section.get(sec_key, []):
                        secondary_site_list.append(PTMSiteConfig(
                            position=site.get("position", 0),
                            residue=site.get("residue", ""),
                            amino_acid=site.get("amino_acid", "N"),
                            function=site.get("function", ""),
                            classic_name=site.get("classic_name", ""),
                            homolog_ref=site.get("homolog_ref"),
                        ))
                sites["secondary"] = secondary_site_list
                max_slots["secondary"] = secondary_dim if secondary_dim else ptm_dim

            uniprot_cfg = cfg.get("uniprot", {}).get(prot_name, {})
            protein_configs.append(ProteinPTMConfig(
                protein_name=prot_name,
                protein_id=prot_idx,
                uniprot=uniprot_cfg.get("accession", ""),
                sites=sites,
                max_slots=max_slots,
            ))

        return cls(ptm_types, ptm_type_descriptions, subtype_descriptions,
                   protein_configs, ptm_type_order)

    def _build_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build type_id_table and is_real_table for all proteins.

        Returns:
          type_id_table: (n_proteins, n_tokens) — subtype IDs per slot
          is_real_table: (n_proteins, n_tokens) — True for real sites, False for padding
        """
        n_proteins = self.n_proteins
        n_tokens = self._n_tokens

        if n_proteins == 0 or n_tokens == 0:
            return torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.bool)

        type_ids = torch.zeros(n_proteins, n_tokens, dtype=torch.long)
        is_real = torch.zeros(n_proteins, n_tokens, dtype=torch.bool)

        for pc in self._protein_configs.values():
            pid = pc.protein_id
            slot = 0
            for ptm_type in self._ptm_type_order:
                max_s = pc.max_slots.get(ptm_type, 0)
                sites = pc.sites.get(ptm_type, [])
                subtypes_map = self._ptm_types.get(ptm_type, {})

                for i in range(max_s):
                    if i < len(sites):
                        site = sites[i]
                        aa = site.amino_acid
                        subtype_id = subtypes_map.get(aa, 0)
                        type_ids[pid, slot] = subtype_id
                        is_real[pid, slot] = True
                    else:
                        # Padding slot — use first subtype of this PTM type as default
                        default_id = next(iter(subtypes_map.values()), 0) if subtypes_map else 0
                        type_ids[pid, slot] = default_id
                        is_real[pid, slot] = False
                    slot += 1

        return type_ids, is_real

    def _build_site_labels(self) -> dict[str, dict[str, list[str]]]:
        """Build site labels per protein per PTM type for XAI reporting."""
        labels: dict[str, dict[str, list[str]]] = {}
        for pc in self._protein_configs.values():
            prot_labels: dict[str, list[str]] = {}
            for ptm_type in self._ptm_type_order:
                max_s = pc.max_slots.get(ptm_type, 0)
                sites = pc.sites.get(ptm_type, [])
                slot_labels = []
                for i in range(max_s):
                    if i < len(sites):
                        site = sites[i]
                        label = site.residue
                        if site.classic_name and site.classic_name != site.residue:
                            label = f"{site.residue}({site.classic_name})"
                        slot_labels.append(label)
                    else:
                        slot_labels.append(f"pad_{i + 1:02d}")
                prot_labels[ptm_type] = slot_labels
            labels[pc.protein_name] = prot_labels
        return labels

    def _build_column_names(self) -> dict[str, dict[str, list[str]]]:
        """Build dataset column names per protein per PTM type."""
        columns: dict[str, dict[str, list[str]]] = {}
        for pc in self._protein_configs.values():
            prot_cols: dict[str, list[str]] = {}
            for ptm_type in self._ptm_type_order:
                sites = pc.sites.get(ptm_type, [])
                max_s = pc.max_slots.get(ptm_type, 0)
                # First PTM type uses ptm_{residue} naming; others use {type}_slot{i}
                if ptm_type == self._ptm_type_order[0]:
                    cols = [f"ptm_{s.residue}" for s in sites]
                    while len(cols) < max_s:
                        cols.append(f"ptm_pad_{len(cols):02d}")
                else:
                    cols = [f"{ptm_type}_slot{i:02d}" for i in range(max_s)]
                prot_cols[ptm_type] = cols
            columns[pc.protein_name] = prot_cols
        return columns

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def n_subtypes(self) -> int:
        """Total number of distinct subtypes (= size of type embedding)."""
        return self._n_subtypes

    @property
    def n_proteins(self) -> int:
        """Number of registered proteins."""
        return len(self._protein_configs)

    @property
    def n_tokens(self) -> int:
        """Number of PTM tokens per sample (sum of max_slots across PTM types)."""
        return self._n_tokens

    @property
    def type_id_table(self) -> torch.Tensor:
        """(n_proteins, n_tokens) — subtype IDs per slot."""
        return self._type_id_table

    @property
    def is_real_table(self) -> torch.Tensor:
        """(n_proteins, n_tokens) — True for real sites, False for padding."""
        return self._is_real_table

    @property
    def subtype_names(self) -> dict[int, str]:
        """Map subtype_id → human-readable name (e.g., 0 → 'phospho_Y')."""
        return dict(self._subtype_names)

    @property
    def parent_type(self) -> dict[int, str]:
        """Map subtype_id → parent PTM type (e.g., 0 → 'phospho')."""
        return dict(self._parent_type_map)

    @property
    def ptm_type_order(self) -> list[str]:
        """Ordered list of PTM type names."""
        return list(self._ptm_type_order)

    @property
    def protein_names(self) -> list[str]:
        """Ordered list of protein names."""
        return sorted(self._protein_configs.keys(),
                      key=lambda n: self._protein_configs[n].protein_id)

    @property
    def protein_name_to_id(self) -> dict[str, int]:
        """Map protein_name → protein_id."""
        return dict(self._protein_id_map)

    @property
    def protein_id_to_name(self) -> dict[int, str]:
        """Map protein_id → protein_name."""
        return {v: k for k, v in self._protein_id_map.items()}

    @property
    def site_labels(self) -> dict[str, dict[str, list[str]]]:
        """Site labels per protein per PTM type."""
        return self._site_labels

    # ── Query Methods ────────────────────────────────────────────────────

    def get_ptm_type_slot_range(self, ptm_type: str) -> tuple[int, int]:
        """Return (start, end) slot indices for a given PTM type."""
        start = 0
        for pt in self._ptm_type_order:
            if pt == ptm_type:
                # Use max across all proteins
                max_s = max(
                    pc.max_slots.get(pt, 0)
                    for pc in self._protein_configs.values()
                ) if self._protein_configs else 0
                return start, start + max_s
            max_s = max(
                pc.max_slots.get(pt, 0)
                for pc in self._protein_configs.values()
            ) if self._protein_configs else 0
            start += max_s
        raise ValueError(f"Unknown PTM type: {ptm_type}")

    def get_n_sites_per_type(self, ptm_type: str) -> int:
        """Return max_slots for a given PTM type (across all proteins)."""
        return max(
            pc.max_slots.get(ptm_type, 0)
            for pc in self._protein_configs.values()
        ) if self._protein_configs else 0

    def get_column_names(self, protein_name: str, ptm_type: str) -> list[str]:
        """Return dataset column names for a protein + PTM type."""
        return self._column_names.get(protein_name, {}).get(ptm_type, [])

    def get_flat_site_labels(self, protein_name: str) -> list[str]:
        """Return flattened site labels for all PTM types (in token order)."""
        labels = []
        for ptm_type in self._ptm_type_order:
            labels.extend(self._site_labels.get(protein_name, {}).get(ptm_type, []))
        return labels

    def get_protein_config(self, protein_name: str) -> ProteinPTMConfig:
        """Return the ProteinPTMConfig for a given protein."""
        return self._protein_configs[protein_name]

    def get_subtype_id(self, ptm_type: str, amino_acid: str) -> int:
        """Return the subtype_id for a (ptm_type, amino_acid) pair."""
        return self._ptm_types[ptm_type][amino_acid]

    def get_real_mask_for_protein(self, protein_name: str, ptm_type: str) -> list[bool]:
        """Return real/pad mask for a specific protein and PTM type."""
        pc = self._protein_configs[protein_name]
        max_s = pc.max_slots.get(ptm_type, 0)
        n_real = len(pc.sites.get(ptm_type, []))
        return [True] * min(n_real, max_s) + [False] * max(0, max_s - n_real)

    def get_type_map_for_protein(self, protein_name: str, ptm_type: str) -> list[int]:
        """Return per-slot subtype IDs for a specific protein and PTM type."""
        pc = self._protein_configs[protein_name]
        max_s = pc.max_slots.get(ptm_type, 0)
        sites = pc.sites.get(ptm_type, [])
        subtypes_map = self._ptm_types.get(ptm_type, {})
        result = []
        for i in range(max_s):
            if i < len(sites):
                aa = sites[i].amino_acid
                result.append(subtypes_map.get(aa, 0))
            else:
                result.append(next(iter(subtypes_map.values()), 0) if subtypes_map else 0)
        return result
