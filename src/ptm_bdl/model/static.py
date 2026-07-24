"""
Static (early-fusion) components — cross-modal self-attention for seq + struct + drug.

Components:
    JointSelfAttentionBlock — Single transformer block with pre-norm
    StaticJointTransformer  — Stack of JointSelfAttentionBlocks
    AttentionPooling        — Ilse et al. ICML 2018 gated attention pooling
    ModalityProjection      — Per-modality linear projection to shared dim
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ModalityProjection(nn.Module):
    """Project a modality's embeddings to a shared dimension."""

    def __init__(self, input_dim: int, shared_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )

    def forward(self, x):
        return self.projection(x)


class JointSelfAttentionBlock(nn.Module):
    """Pre-norm transformer block for joint self-attention."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attention=False):
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return (x, attn_weights) if return_attention else x


class StaticJointTransformer(nn.Module):
    """Stack of JointSelfAttentionBlocks for cross-modal fusion."""

    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            JointSelfAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, return_all_attention=False):
        all_attn = []
        for layer in self.layers:
            if return_all_attention:
                x, attn = layer(x, return_attention=True)
                all_attn.append(attn)
            else:
                x = layer(x)
        x = self.final_norm(x)
        return (x, all_attn) if return_all_attention else x


class AttentionPooling(nn.Module):
    """
    Gated attention pooling (Ilse et al., ICML 2018).

    Learns attention weights over token positions and computes
    a weighted sum to produce a fixed-size representation.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model // 4, 1),
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled, weights.squeeze(-1)
