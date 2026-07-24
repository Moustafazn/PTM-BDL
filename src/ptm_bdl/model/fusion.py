"""
Late Bilinear Fusion — S_rep ⊙ P_rep.

Two-way late fusion where:
  - S_rep (static representation) already contains drug context from
    early joint attention (seq + struct + drug).
  - P_rep (PTM-BDL representation) carries the dynamic biological state,
    drug-conditioned through delta_ptm features in its inputs.

The model asks: "Given the drug CAN bind (S_rep knows),
does the PTM state say it actually WORKS?"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BilinearLateFusion(nn.Module):
    """
    Two-way late fusion: static_rep ⊙ ptm_bdl_rep.

    Drug is NOT a separate branch here — it is already encoded inside
    static_rep via early joint attention. PTM-BDL rep is drug-conditioned
    through its delta_ptm input features, not through a fusion shortcut.
    """

    def __init__(self, static_dim: int, ptm_bdl_dim: int,
                 output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, output_dim)
        self.ptm_proj = nn.Linear(ptm_bdl_dim, output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, static_rep, ptm_rep):
        s = torch.tanh(self.static_proj(static_rep))
        p = torch.tanh(self.ptm_proj(ptm_rep))
        return self.output_proj(s * p)
