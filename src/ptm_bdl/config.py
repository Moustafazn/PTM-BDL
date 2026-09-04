"""
Config loader — merges base tool config with case-study-specific config.

The base config (model architecture, training settings, PTM-BDL hyperparameters)
lives at src/ptm_bdl/default_config.yaml (shipped with the package).

Case-study-specific config (proteins, drugs, PTM sites, tissue filters) lives
at src/case_studies/<name>/config.yaml.

Usage:
    from src.ptm_bdl.config import load_config

    # Load merged config (base + case study)
    cfg = load_config(case_study="egfr_erbb2_tki")

    # Load just the base config
    cfg = load_config()
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Package directory (where this file lives)
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent  # src/ptm_bdl -> src -> project_root

# Default config bundled with the tool package
DEFAULT_CONFIG_PATH = _PACKAGE_DIR / "default_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(case_study: str | None = "egfr_erbb2_tki",
                project_root: Path | str | None = None) -> dict:
    """
    Load and merge configuration files.

    Args:
        case_study: Name of the case study (e.g., "egfr_erbb2_tki").
                    If None, only the base tool config is loaded.
        project_root: Override project root path. If None, auto-detected
                      from package location.

    Returns:
        Merged config dict with tool settings + case study settings.
    """
    root = Path(project_root) if project_root else _PROJECT_ROOT

    # Load base tool config (bundled with the package)
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Merge case study config if specified
    if case_study:
        cs_path = root / "src" / "case_studies" / case_study / "config.yaml"
        if cs_path.exists():
            with open(cs_path) as f:
                cs_cfg = yaml.safe_load(f)
            cfg = _deep_merge(cfg, cs_cfg)

    return cfg
