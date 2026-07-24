#!/usr/bin/env python3
"""Step 01 wrapper — Download GDSC data for HeLa/HDAC case study."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from src.case_studies.common.step01_download_gdsc import run

if __name__ == "__main__":
    run(case_study="hela_hdac")
