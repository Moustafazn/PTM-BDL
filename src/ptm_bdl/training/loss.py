"""
Focal Loss — Class-conditional alpha for handling class imbalance.

FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with class-conditional alpha.

    With α=0.25:
      majority class (label=1) gets weight 0.25
      minority class (label=0) gets weight 0.75 → 3× up-weight on minority
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
