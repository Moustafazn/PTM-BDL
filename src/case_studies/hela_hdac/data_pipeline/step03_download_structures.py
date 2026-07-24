#!/usr/bin/env python3
"""
Step 03 — Download PDB structures for HeLa/HDAC target proteins.

Downloads PDB structures for HDAC1 and EP300 for GearNet embedding
extraction (step08).

PDB structures:
  4BKX — HDAC1 catalytic domain (human, 2.0 Å)
  5EDU — HDAC1 + Romidepsin-like peptide inhibitor
  4LXZ — HDAC8 + Vorinostat (SAHA) — closest HDAC+SAHA co-crystal
  4BHW — p300 HAT domain (3.5 Å, human)

Ref: Watson et al., Nature 2012 — HDAC1 crystal structure
Ref: Lasko et al., Nature 2017 — p300 HAT domain
"""
import sys
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="hela_hdac")
OUT_DIR = PROJECT_ROOT / cfg["paths"]["raw_data"] / "pdb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_STRUCTURES = {
    "4BKX": {"protein": "HDAC1", "description": "HDAC1 catalytic domain (human)"},
    "5EDU": {"protein": "HDAC1", "description": "HDAC1 + peptide inhibitor"},
    "4LXZ": {"protein": "HDAC8", "description": "HDAC8 + Vorinostat (SAHA)"},
    "4BHW": {"protein": "EP300", "description": "p300 HAT domain (human)"},
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
    print("║  STEP 03 — HeLa/HDAC: Download PDB Structures                ║")
    print("║  Structures: HDAC1 (4BKX, 5EDU), HDAC8+SAHA (4LXZ), p300    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    for pdb_id, info in PDB_STRUCTURES.items():
        ok = download_pdb(pdb_id)
        if ok:
            print(f"    {pdb_id}: {info['protein']} — {info['description']}")

    print("\n✓ Step 03 complete!")
