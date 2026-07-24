"""
PTM-BDL Training Package — Loss functions, training loops, metrics, and model factory.

Components:
    FocalLoss                  — Class-conditional focal loss for imbalanced classification
    train_epoch                — Single training epoch with multi-task loss
    validate                   — Validation with comprehensive metrics
    compute_optimal_threshold  — Youden's J threshold calibration on validation set
    compute_metrics            — Classification + regression metric computation
    build_model_from_cfg       — Model factory using PTMTypeRegistry
"""

from src.ptm_bdl.training.factory import build_model_from_cfg
from src.ptm_bdl.training.loss import FocalLoss
from src.ptm_bdl.training.metrics import compute_metrics
from src.ptm_bdl.training.trainer import train_epoch, validate, compute_optimal_threshold
