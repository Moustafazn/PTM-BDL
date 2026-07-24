"""
Cross-type attention analysis for PTM-BDL models.

Extracts post-softmax attention weights from the final PTM-BDL transformer layer
and decomposes them into quadrants based on PTM type boundaries. This reveals
cross-type crosstalk (e.g., primary↔secondary) learned by the model.

The module is PTM-type-agnostic — it uses the registry to determine type boundaries.
"""

from __future__ import annotations

import numpy as np
import torch

from src.ptm_bdl.registry import PTMTypeRegistry


def compute_cross_type_attention(
        model,
        dataset,
        indices: list[int],
        registry: PTMTypeRegistry,
) -> dict:
    """
    Average post-softmax attention from the FINAL PTM-BDL transformer layer,
    decomposed by PTM type quadrants per protein.

    Args:
        model: Trained PTM-BDL model (eval mode).
        dataset: PyTorch Dataset.
        indices: Sample indices to process.
        registry: PTMTypeRegistry for protein name lookup.

    Returns:
        Dict keyed by protein_name, each containing:
          "n_samples": int
          "mean_attention_matrix": list[list[float]] — (n_tokens × n_tokens) mean attention
          "quadrants": dict mapping "typeA_to_typeB" → mean attention value
    """
    n_tokens = registry.n_tokens
    protein_id_to_name = registry.protein_id_to_name

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    # Detect model device
    device = next(model.parameters()).device

    for idx in indices:
        sample = dataset[int(idx)]
        tp = sample["target_protein"].view(1).long().to(device)
        pid = int(tp.item())
        protein_name = protein_id_to_name.get(pid, f"protein_{pid}")

        if protein_name not in sums:
            sums[protein_name] = np.zeros((n_tokens, n_tokens))
            counts[protein_name] = 0

        # Move all tensors to model device before calling encoder
        ptm_v = sample["ptm_vector"].unsqueeze(0).to(device)
        dptm_v = sample["delta_ptm_vector"].unsqueeze(0).to(device)
        sec_v = sample["secondary_vector"].unsqueeze(0).to(device) if (
            "secondary_vector" in sample and sample["secondary_vector"].numel() > 0
        ) else torch.zeros(1, 0, device=device)
        dsec_v = sample["delta_secondary_vector"].unsqueeze(0).to(device) if (
            "delta_secondary_vector" in sample and sample["delta_secondary_vector"].numel() > 0
        ) else torch.zeros(1, 0, device=device)

        attn = model.ptm_bdl.compute_attn_weights(
            ptm_v, dptm_v, sec_v, dsec_v, tp,
        )
        sums[protein_name] += attn.squeeze(0).cpu().numpy()
        counts[protein_name] += 1

    # Build quadrant analysis using registry type boundaries
    out = {}
    for protein_name in sums:
        if counts[protein_name] == 0:
            continue
        mean_attn = sums[protein_name] / counts[protein_name]

        # Compute quadrant means for each PTM type pair
        quadrants = {}
        for type_a in registry.ptm_type_order:
            start_a, end_a = registry.get_ptm_type_slot_range(type_a)
            for type_b in registry.ptm_type_order:
                start_b, end_b = registry.get_ptm_type_slot_range(type_b)
                if end_a <= n_tokens and end_b <= n_tokens:
                    quadrants[f"{type_a}_to_{type_b}"] = float(
                        mean_attn[start_a:end_a, start_b:end_b].mean()
                    )

        out[protein_name] = {
            "n_samples": counts[protein_name],
            "mean_attention_matrix": mean_attn.tolist(),
            "quadrants": quadrants,
        }
    return out
