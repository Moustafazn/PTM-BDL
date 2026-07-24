#!/usr/bin/env python3
"""Step 02 wrapper — Download mutation profiles for EGFR/ERBB2 TKI case study."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from src.case_studies.egfr_erbb2_tki.data_pipeline.step02_download_mutations import run

if __name__ == "__main__":
    run(case_study="egfr_erbb2_tki")
