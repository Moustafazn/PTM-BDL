"""
PTM-BDL Data Package — Config-driven dataset and data utilities.

Components:
    ResistanceDataset       — PyTorch Dataset reading PTM columns from registry
    collate_fn              — Custom collation for variable-length sequences
    create_stratified_splits — Stratified train/val/test splitting
"""

from src.ptm_bdl.data.collate import collate_fn
from src.ptm_bdl.data.dataset import ResistanceDataset
from src.ptm_bdl.data.splits import create_stratified_splits
