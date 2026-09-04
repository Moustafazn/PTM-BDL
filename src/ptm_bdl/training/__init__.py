"""
PTM-BDL Training Package — Loss functions, training loops, metrics, checkpoint
I/O, device resolution, class-balanced sampling, and model factory.

Components:
    FocalLoss                  — Class-conditional focal loss for imbalanced classification
    train_epoch                — Single training epoch with multi-task loss
    validate                   — Validation with comprehensive metrics
    compute_optimal_threshold  — Youden's J threshold calibration on validation set
    compute_metrics            — Classification + regression metric computation
    build_model_from_cfg       — Model factory using PTMTypeRegistry
    save_checkpoint            — Save model state_dict to disk
    load_checkpoint            — Load model state_dict (weights_only=True, eval mode)
    resolve_device             — Auto-detect CUDA / MPS / CPU from config
    create_balanced_sampler    — WeightedRandomSampler for class-imbalanced datasets
"""

from src.ptm_bdl.training.factory import build_model_from_cfg
from src.ptm_bdl.training.loss import FocalLoss
from src.ptm_bdl.training.metrics import compute_metrics
from src.ptm_bdl.training.trainer import (
    train_epoch,
    validate,
    compute_optimal_threshold,
    save_checkpoint,
    load_checkpoint,
    resolve_device,
    create_balanced_sampler,
)
