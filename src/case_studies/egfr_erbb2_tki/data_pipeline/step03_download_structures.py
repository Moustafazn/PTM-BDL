#!/usr/bin/env python3
"""Step 03 wrapper — Download PDB structures for EGFR/ERBB2 TKI case study."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from src.case_studies.common.step03_download_structures import run

if __name__ == "__main__":
    run(case_study="egfr_erbb2_tki")
