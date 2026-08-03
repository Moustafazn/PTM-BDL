"""
Generic per-token Integrated Gradients for PTM-BDL models.

Reference: Sundararajan, Taly & Yan, "Axiomatic Attribution for Deep Networks", ICML 2017.

Integrates along ALL PTM input channels simultaneously (level + delta for each
PTM type), then returns per-site attributions bucketed by channel.

The module is PTM-type-agnostic — it operates on named channels provided by
the caller. The default is a single flat channel:
  channels = [("ptm_vector", "delta_ptm_vector")]

Baselines:
  level channels: 1.0 (wild-type occupancy — no modulation)
  delta channels: 0.0 (no drug effect)

Per-site importance = |grad_level × Δlevel| + |grad_delta × Δdelta|
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Default PTM channel definition for the model interface.
# All PTM types are concatenated into a single flat vector by the dataset.
DEFAULT_PTM_CHANNELS = [
    ("ptm_vector", "delta_ptm_vector"),
]


def compute_ig_for_sample(
        model,
        sample: dict,
        ptm_channels: Optional[list[tuple[str, str]]] = None,
        n_steps: int = 30,
        target: str = "resistance",
) -> dict:
    """
    Compute Integrated Gradients for a single sample on ALL PTM input channels.

    Args:
        model: The PTM-BDL multimodal model (must be in train mode for grads).
        sample: Dict with keys for each modality (seq_emb, struct_emb, etc.)
                and PTM channels (ptm_vector, delta_ptm_vector, etc.).
        ptm_channels: List of (level_key, delta_key) pairs defining PTM channels.
                      Defaults to DEFAULT_PTM_CHANNELS.
        n_steps: Number of interpolation steps (higher = more accurate).
        target: "resistance" or "ic50" — which output to attribute.

    Returns:
        Dict with:
          "channel_attrs": dict mapping level_key → np.array of per-site importance
          "target_protein": int — protein ID for this sample
    """
    if ptm_channels is None:
        ptm_channels = DEFAULT_PTM_CHANNELS

    # Detect model device and move all inputs there
    device = next(model.parameters()).device

    tp = sample["target_protein"].view(1).long().to(device)
    seq_e = sample["seq_emb"].unsqueeze(0).to(device)
    str_e = sample["struct_emb"].unsqueeze(0).to(device)
    drg_e = sample["drug_emb"].unsqueeze(0).to(device)
    drg_p = sample["drug_pooled"].unsqueeze(0).to(device)

    # Build baselines and actuals for each PTM channel
    channels_info = []
    for level_key, delta_key in ptm_channels:
        actual_level = sample[level_key].to(device)
        actual_delta = sample[delta_key].to(device)
        n_sites = actual_level.shape[-1]
        channels_info.append({
            "level_key": level_key,
            "delta_key": delta_key,
            "actual_level": actual_level,
            "actual_delta": actual_delta,
            "baseline_level": torch.ones(n_sites, device=device),
            "baseline_delta": torch.zeros(n_sites, device=device),
            "grad_level": torch.zeros(n_sites, device=device),
            "grad_delta": torch.zeros(n_sites, device=device),
        })

    # Integrate along all channels simultaneously
    for step in range(n_steps + 1):
        a = step / n_steps

        # Build interpolated inputs for all channels
        interp_inputs = {}
        for ch in channels_info:
            interp_level = (ch["baseline_level"] + a * (ch["actual_level"] - ch["baseline_level"])
                            ).unsqueeze(0).requires_grad_(True)
            interp_delta = (ch["baseline_delta"] + a * (ch["actual_delta"] - ch["baseline_delta"])
                            ).unsqueeze(0).requires_grad_(True)
            interp_inputs[ch["level_key"]] = interp_level
            interp_inputs[ch["delta_key"]] = interp_delta

        # Forward pass with interpolated PTM channels
        ic50_pred, resist_pred = model(
            seq_embeddings=seq_e,
            struct_embeddings=str_e,
            drug_pooled=drg_p,
            drug_embeddings=drg_e,
            target_protein=tp,
            **interp_inputs,
        )

        model.zero_grad()
        output = resist_pred if target == "resistance" else ic50_pred
        output.backward()

        # Accumulate gradients for each channel
        for ch in channels_info:
            level_tensor = interp_inputs[ch["level_key"]]
            delta_tensor = interp_inputs[ch["delta_key"]]
            if level_tensor.grad is not None:
                ch["grad_level"] += level_tensor.grad.squeeze(0).detach()
            if delta_tensor.grad is not None:
                ch["grad_delta"] += delta_tensor.grad.squeeze(0).detach()

    # Compute attributions: |grad_level × Δlevel| + |grad_delta × Δdelta|
    n_s = n_steps + 1
    channel_attrs = {}
    for ch in channels_info:
        d_level = ch["actual_level"] - ch["baseline_level"]
        d_delta = ch["actual_delta"] - ch["baseline_delta"]
        attr = (np.abs(((ch["grad_level"] / n_s) * d_level).detach().cpu().numpy())
                + np.abs(((ch["grad_delta"] / n_s) * d_delta).detach().cpu().numpy()))
        channel_attrs[ch["level_key"]] = attr

    return {
        "channel_attrs": channel_attrs,
        "target_protein": int(tp.item()),
    }


def compute_ig_batch(
        model,
        dataset,
        indices: list[int],
        ptm_channels: Optional[list[tuple[str, str]]] = None,
        n_steps: int = 20,
        target: str = "resistance",
) -> dict:
    """
    Compute IG over multiple samples, returning per-protein aggregated attributions.

    Args:
        model: PTM-BDL model (will be set to train mode for gradients).
        dataset: PyTorch Dataset returning sample dicts.
        indices: List of sample indices to process.
        ptm_channels: List of (level_key, delta_key) pairs. Defaults to DEFAULT_PTM_CHANNELS.
        n_steps: IG interpolation steps.
        target: "resistance" or "ic50".

    Returns:
        Dict keyed by protein_id, each containing:
          dict mapping channel_name → mean attribution array
          "n_samples": int
    """
    if ptm_channels is None:
        ptm_channels = DEFAULT_PTM_CHANNELS

    model.train()

    sums: dict[int, dict] = {}
    counts: dict[int, int] = {}

    for idx in indices:
        sample = dataset[int(idx)]
        result = compute_ig_for_sample(
            model, sample, ptm_channels=ptm_channels,
            n_steps=n_steps, target=target,
        )
        pid = result["target_protein"]

        if pid not in sums:
            sums[pid] = {k: np.zeros_like(v) for k, v in result["channel_attrs"].items()}
            counts[pid] = 0

        for k, v in result["channel_attrs"].items():
            sums[pid][k] += v
        counts[pid] += 1

    model.eval()

    out = {}
    for pid in sums:
        n = max(counts[pid], 1)
        out[pid] = {k: v / n for k, v in sums[pid].items()}
        out[pid]["n_samples"] = counts[pid]
    return out
