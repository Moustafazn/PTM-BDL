#!/usr/bin/env python3
"""
Step 02 — Download protein sequences for HeLa/HDAC target proteins.

Downloads UniProt sequences for HDAC1 (Q13547) and EP300 (Q09472)
for ESM-2 embedding extraction (step07).

Unlike EGFR (which has mutation variants), HDAC inhibitor targets use
wild-type sequences — HDAC inhibition is mechanism-based, not mutation-driven.
"""
import sys
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="hela_hdac")
OUT_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"] / "sequences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROTEINS = {
    "HDAC1": {"uniprot": "Q13547", "description": "Histone deacetylase 1 — primary drug target"},
    "EP300": {"uniprot": "Q09472", "description": "Histone acetyltransferase p300 — A485 target"},
}


def download_sequence(protein_name: str, uniprot_id: str) -> str:
    """Download FASTA from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    fasta_path = OUT_DIR / f"{protein_name.lower()}_{uniprot_id}.fasta"

    if fasta_path.exists():
        print(f"  ✓ {protein_name} ({uniprot_id}): already downloaded")
        with open(fasta_path) as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        return seq

    print(f"  Downloading {protein_name} ({uniprot_id}) from UniProt...")
    try:
        urllib.request.urlretrieve(url, fasta_path)
        print(f"  ✓ Saved: {fasta_path}")
        with open(fasta_path) as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        print(f"    Length: {len(seq)} amino acids")
        return seq
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"    Manual: curl -o {fasta_path} {url}")
        return ""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 02 — HeLa/HDAC: Download Protein Sequences              ║")
    print("║  Proteins: HDAC1 (Q13547), EP300 (Q09472)                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    for name, info in PROTEINS.items():
        seq = download_sequence(name, info["uniprot"])
        if seq:
            print(f"    {name}: {len(seq)} aa — {info['description']}")

    print("\n✓ Step 02 complete!")
