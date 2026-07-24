"""Collation function for variable-length sequences in PTM-BDL batches."""

from __future__ import annotations

import torch


def collate_fn(batch):
    """Custom collation: pad variable-length sequences to max batch length."""
    max_L = max(item["seq_emb"].size(0) for item in batch)
    max_M = max(item["struct_emb"].size(0) for item in batch)
    max_N = max(item["drug_emb"].size(0) for item in batch)

    seq_embs = torch.zeros(len(batch), max_L, batch[0]["seq_emb"].size(1))
    struct_embs = torch.zeros(len(batch), max_M, batch[0]["struct_emb"].size(1))
    drug_embs = torch.zeros(len(batch), max_N, batch[0]["drug_emb"].size(1))

    for i, item in enumerate(batch):
        seq_embs[i, :item["seq_emb"].size(0)] = item["seq_emb"]
        struct_embs[i, :item["struct_emb"].size(0)] = item["struct_emb"]
        drug_embs[i, :item["drug_emb"].size(0)] = item["drug_emb"]

    return {
        "seq_emb": seq_embs,
        "struct_emb": struct_embs,
        "drug_emb": drug_embs,
        "drug_pooled": torch.stack([b["drug_pooled"] for b in batch]),
        "ptm_vector": torch.stack([b["ptm_vector"] for b in batch]),
        "delta_ptm_vector": torch.stack([b["delta_ptm_vector"] for b in batch]),
        "secondary_vector": torch.stack([b["secondary_vector"] for b in batch]),
        "delta_secondary_vector": torch.stack([b["delta_secondary_vector"] for b in batch]),
        "target_protein": torch.stack([b["target_protein"] for b in batch]),
        "propagation_confidence": torch.stack([b["propagation_confidence"] for b in batch]),
        "ln_ic50": torch.stack([b["ln_ic50"] for b in batch]),
        "resistance_label": torch.stack([b["resistance_label"] for b in batch]),
    }
