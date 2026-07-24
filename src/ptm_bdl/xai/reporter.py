"""
XAI report assembly — combines IG attributions, attention analysis,
and homology validation into a unified JSON report.

The reporter is PTM-type-agnostic — it uses the registry to resolve
site labels and organize results by protein and PTM type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.ptm_bdl.registry import PTMTypeRegistry


def build_xai_report(
        ig_results: dict,
        attention_results: Optional[dict] = None,
        homology_results: Optional[dict] = None,
        registry: Optional[PTMTypeRegistry] = None,
        extra: Optional[dict] = None,
) -> dict:
    """
    Assemble a comprehensive XAI report from IG, attention, and homology results.

    Args:
        ig_results: Output from compute_ig_batch() — per-protein channel attributions.
        attention_results: Output from compute_cross_type_attention() — per-protein quadrants.
        homology_results: Output from check_homology() — concordance flags.
        registry: PTMTypeRegistry for resolving site labels.
        extra: Any additional data to include in the report.

    Returns:
        Dict structured for JSON serialization.
    """
    report: dict = {}

    # IG attributions with site labels
    if ig_results:
        ig_section = {}
        for pid, attrs in ig_results.items():
            if pid == "n_samples":
                continue
            protein_name = (registry.protein_id_to_name.get(pid, f"protein_{pid}")
                            if registry else f"protein_{pid}")
            protein_ig = {"n_samples": attrs.get("n_samples", 0)}

            for channel_key, values in attrs.items():
                if channel_key == "n_samples":
                    continue
                if isinstance(values, np.ndarray):
                    # Rank sites by importance
                    ranked = sorted(enumerate(values), key=lambda x: -x[1])
                    site_ranking = []
                    for rank, (slot, importance) in enumerate(ranked, start=1):
                        entry = {
                            "rank": rank,
                            "slot": slot,
                            "importance": float(importance),
                        }
                        # Add site label from registry if available
                        if registry:
                            flat_labels = registry.get_flat_site_labels(protein_name)
                            if slot < len(flat_labels):
                                entry["site"] = flat_labels[slot]
                        site_ranking.append(entry)
                    protein_ig[channel_key] = {
                        "site_ranking": site_ranking,
                        "raw_values": values.tolist(),
                    }
            ig_section[protein_name] = protein_ig
        report["integrated_gradients"] = ig_section

    # Cross-type attention
    if attention_results:
        report["cross_type_attention"] = attention_results

    # Homology validation
    if homology_results:
        report["homology_validation"] = homology_results

    # Extra data
    if extra:
        report.update(extra)

    return report


def save_xai_report(report: dict, output_path: Path | str) -> None:
    """Save the XAI report to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
