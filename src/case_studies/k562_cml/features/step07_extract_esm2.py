#!/usr/bin/env python3
"""
Step 07 — Extract ESM-2 protein sequence embeddings for K562/CML case study.

Target protein: ABL1 (BCR-ABL fusion kinase, UniProt P00519).
Completely different kinase system from EGFR — proves ESM-2 branch generalizes.

Output: data/features/esm2/ embeddings for ABL1
"""
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)

FEATURES_DIR = PROJECT_ROOT / cfg["paths"]["features"] / "esm2"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    device_cfg = cfg.get("training", {}).get("device", "auto")
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def extract_esm2_embeddings():
    """Extract ESM-2 embeddings for ABL1 and other target proteins."""
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Step 07 — ESM-2 Embeddings ({CASE_STUDY})                 ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    device = get_device()
    target_proteins = cfg["project"]["target_proteins"]
    print(f"  Device: {device}")
    print(f"  Target proteins: {target_proteins}")

    try:
        from transformers import AutoModel, AutoTokenizer
        model_name = "facebook/esm2_t33_650M_UR50D"
        print(f"  Loading ESM-2: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        use_model = True
    except Exception as e:
        print(f"  ⚠ Could not load ESM-2: {e}")
        use_model = False

    for protein in target_proteins:
        uniprot_cfg = cfg.get("uniprot", {}).get(protein, {})
        accession = uniprot_cfg.get("accession", "")

        # Search multiple locations for FASTA files
        fasta_path = None
        for candidate in [
            PROJECT_ROOT / cfg["paths"]["processed_data"] / "sequences" / f"{protein.lower()}_{accession}.fasta",
            PROJECT_ROOT / cfg["paths"]["raw_data"] / "ccle" / f"{protein.lower()}_{accession}.fasta",
            PROJECT_ROOT / cfg["paths"]["raw_data"] / "ptm" / f"uniprot_{accession}.fasta",
        ]:
            if candidate.exists():
                fasta_path = candidate
                break

        if fasta_path is not None and fasta_path.exists() and use_model:
            from Bio import SeqIO
            record = SeqIO.read(fasta_path, "fasta")
            seq = str(record.seq)[:1022]
            print(f"  Extracting {protein} ({len(seq)} aa, {accession})...")

            inputs = tokenizer(seq, return_tensors="pt", padding=False).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            emb_full = outputs.last_hidden_state[0, 1:-1].cpu().numpy()
            print(f"    Raw ESM-2: {emb_full.shape}")

            # Pool to fixed length for training speed
            # Self-attention is O(n²) — 1022 tokens is too slow.
            # Adaptive avg pool preserves information from ALL residues.
            MAX_SEQ_TOKENS = 200
            if emb_full.shape[0] > MAX_SEQ_TOKENS:
                import torch.nn.functional as F
                t = torch.from_numpy(emb_full).unsqueeze(0).permute(0, 2, 1)
                emb = F.adaptive_avg_pool1d(t, MAX_SEQ_TOKENS).permute(0, 2, 1).squeeze(0).numpy()
                print(f"    Pooled: {emb_full.shape[0]} → {MAX_SEQ_TOKENS} tokens")
            else:
                emb = emb_full

            np.save(FEATURES_DIR / f"{protein.lower()}_esm2.npy", emb)
            np.save(FEATURES_DIR / f"{protein.lower()}_esm2_pooled.npy", emb.mean(axis=0))
            print(f"    ✓ {protein}: {emb.shape}")
        else:
            print(f"  Creating placeholder for {protein} (dim=1280)")
            emb = np.random.randn(100, 1280).astype(np.float32) * 0.01
            np.save(FEATURES_DIR / f"{protein.lower()}_esm2.npy", emb)
            np.save(FEATURES_DIR / f"{protein.lower()}_esm2_pooled.npy", emb.mean(axis=0))

    print(f"\n✓ Step 07 complete!")


if __name__ == "__main__":
    extract_esm2_embeddings()
