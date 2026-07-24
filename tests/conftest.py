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

from src.ptm_bdl.model.predictor import MultimodalResistancePredictor


@pytest.fixture
def model():
    """Small full model with typed attention (for fast testing)."""
    return MultimodalResistancePredictor(
        seq_dim=1280, struct_dim=512, drug_dim=384,
        ptm_dim=12, glyco_dim=12,
        shared_dim=64, num_heads=4, num_layers=2, dropout=0.0,
        ptm_bdl_d_model=32, ptm_bdl_n_heads=4, ptm_bdl_n_layers=1,
        use_typed_attention=True,
    )


@pytest.fixture
def model_mlp():
    """Model with MLP ablation (no typed attention)."""
    return MultimodalResistancePredictor(
        seq_dim=1280, struct_dim=512, drug_dim=384,
        ptm_dim=12, glyco_dim=12,
        shared_dim=64, num_heads=4, num_layers=2, dropout=0.0,
        ptm_bdl_d_model=32, ptm_bdl_n_heads=4, ptm_bdl_n_layers=1,
        use_typed_attention=False,
    )


@pytest.fixture
def batch():
    """Minimal 4-sample batch with mixed proteins."""
    B = 4
    torch.manual_seed(42)
    return {
        "seq_emb": torch.randn(B, 10, 1280),
        "struct_emb": torch.randn(B, 8, 512),
        "drug_emb": torch.randn(B, 5, 384),
        "drug_pooled": torch.randn(B, 384),
        "ptm_vector": torch.ones(B, 12),
        "delta_ptm_vector": torch.randn(B, 12) * 0.1,
        "secondary_vector": torch.ones(B, 12),
        "delta_secondary_vector": torch.randn(B, 12) * 0.1,
        "target_protein": torch.tensor([0, 1, 0, 1]),
        "ln_ic50": torch.randn(B, 1),
        "resistance_label": torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        "propagation_confidence": torch.tensor([[0.8], [0.4], [0.65], [0.7]]),
    }
