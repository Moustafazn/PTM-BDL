"""
Shared fixtures for all test modules.

Provides model instances, sample batches, and project paths
used across test_model.py, test_ptm_bdl.py, and test_training.py.
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ptm_bdl.config import load_config
from src.ptm_bdl.training.factory import build_model_from_cfg


@pytest.fixture
def cfg():
    """Load EGFR case study config (used for all tests)."""
    return load_config(case_study="egfr_erbb2_tki")


@pytest.fixture
def model(cfg):
    """Small full model with typed attention (for fast testing)."""
    return build_model_from_cfg(cfg, use_typed_attention=True)


@pytest.fixture
def model_mlp(cfg):
    """Model with MLP ablation (no typed attention)."""
    return build_model_from_cfg(cfg, use_typed_attention=False)


@pytest.fixture
def batch(model):
    """Minimal 4-sample batch with mixed proteins.

    PTM vector size matches the registry's n_tokens (all types flat).
    """
    B = 4
    n_tokens = model.registry.n_tokens  # e.g., 24 for EGFR (12 phospho + 12 glyco)
    torch.manual_seed(42)
    return {
        "seq_emb": torch.randn(B, 10, 1280),
        "struct_emb": torch.randn(B, 8, 512),
        "drug_emb": torch.randn(B, 5, 384),
        "drug_pooled": torch.randn(B, 384),
        "ptm_vector": torch.ones(B, n_tokens),
        "delta_ptm_vector": torch.randn(B, n_tokens) * 0.1,
        "target_protein": torch.tensor([0, 1, 0, 1]),
        "ln_ic50": torch.randn(B, 1),
        "resistance_label": torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        "propagation_confidence": torch.tensor([[0.8], [0.4], [0.65], [0.7]]),
    }
