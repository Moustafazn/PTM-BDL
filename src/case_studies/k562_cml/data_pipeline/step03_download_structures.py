#!/usr/bin/env python3
"""
Step 03 — Download PDB structures for K562/CML target proteins.

Downloads PDB structures for ABL1 kinase for GearNet embedding
extraction (step08).

PDB structures:
  1IEP — ABL1 kinase + Imatinib (STI-571, Gleevec)
  2GQG — ABL1 kinase + Dasatinib (BMS-354825)
  2HYY — ABL1 kinase apo (DFG-in, active conformation)

Ref: Nagar et al., Cancer Res 2002 — ABL1+Imatinib structure
Ref: Tokarski et al., Cancer Res 2006 — ABL1+Dasatinib structure
"""
import sys
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="k562_cml")
OUT_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "pdb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_STRUCTURES = {
    "1IEP": {"protein": "ABL1", "description": "ABL1 kinase + Imatinib"},
    "2GQG": {"protein": "ABL1", "description": "ABL1 kinase + Dasatinib"},
    "2HYY": {"protein": "ABL1", "description": "ABL1 kinase apo (DFG-in)"},
}


def download_pdb(pdb_id: str) -> bool:
    pdb_path = OUT_DIR / f"{pdb_id}.pdb"
    if pdb_path.exists():
        print(f"  ✓ {pdb_id}: already downloaded")
        return True

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, pdb_path)
        size_kb = pdb_path.stat().st_size / 1024
        print(f"  ✓ Saved: {pdb_path} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 03 — K562/CML: Download PDB Structures                 ║")
    print("║  Structures: ABL1+Imatinib (1IEP), ABL1+Dasatinib (2GQG)    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    for pdb_id, info in PDB_STRUCTURES.items():
        ok = download_pdb(pdb_id)
        if ok:
            print(f"    {pdb_id}: {info['protein']} — {info['description']}")

    print("\n✓ Step 03 complete!")
