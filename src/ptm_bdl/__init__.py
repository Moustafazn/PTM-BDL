"""
PTM-BDL Framework — Config-driven extensible framework for PTM-based drug response prediction.

Public API:
    PTMTypeRegistry     — Dynamic PTM type/subtype system built from config
    PTMBDLEncoder       — Typed self-attention encoder for PTM tokens
    MultimodalResistancePredictor — Full two-stage fusion model
"""

from src.ptm_bdl.registry import PTMTypeRegistry
