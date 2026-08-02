#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 07 — Extract ESM-2 Protein Sequence Embeddings                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Use the pre-trained ESM-2 protein language model (Meta AI) to convert     ║
║    each mutant EGFR sequence into a dense embedding that captures            ║
║    evolutionary, structural, and functional context.                         ║
║                                                                              ║
║  WHY ESM-2?                                                                  ║
║    ESM-2 was trained on ~250M protein sequences using masked language        ║
║    modeling. It has learned:                                                 ║
║    • Which amino acids are tolerated at each position (conservation)         ║
║    • How mutations affect protein function (variant effects)                 ║
║    • Implicit structural information (contact prediction emerges)            ║
║    • Evolutionary relationships between protein families                     ║
║                                                                              ║
║    When we feed a MUTANT sequence (e.g., L858R), ESM-2 generates different  ║
║    embeddings than for wild-type because it "knows" that arginine at         ║
║    position 858 is unusual and disrupts normal EGFR function.                ║
║                                                                              ║
║  MODEL: facebook/esm2_t33_650M_UR50D (650M parameters, 33 layers)           ║
║                                                                              ║
║  INPUT:  Mutant EGFR amino acid sequences (from Step 02)                    ║
║  OUTPUT: Per-residue embeddings (L × 1280) and pooled (1 × 1280)            ║
║          Saved to data/features/esm2/                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="egfr_erbb2_tki")

FEATURES_DIR = PROJECT_ROOT / cfg["paths"]["features"] / "esm2"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Select the best available compute device."""
    device_cfg = cfg["training"]["device"]
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def load_mutant_sequences():
    """
    Load mutant EGFR sequences generated in Step 02.
    
    Returns dict: {sequence_id: amino_acid_sequence}
    """
    fasta_path = PROJECT_ROOT / cfg["paths"]["processed_data"] / "ccle" / "egfr_mutant_sequences.fasta"

    sequences = {}

    if fasta_path.exists():
        from Bio import SeqIO
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = record.id.replace("EGFR_", "")
            sequences[seq_id] = str(record.seq)
        print(f"  ✓ Loaded {len(sequences)} mutant sequences from FASTA")
    else:
        print("  ⚠ FASTA file not found. Generating sequences from reference...")
        # Fallback: generate minimal set from UniProt reference
        sequences = generate_minimal_sequences()

    return sequences


def generate_minimal_sequences():
    """Generate minimal set of mutant sequences for development."""
    # Use a short representative kinase domain sequence for testing
    # In production, use full sequences from Step 02

    # EGFR kinase domain (positions 696-1022 of P00533)
    # This is a truncated version for faster development
    kinase_seq = (
        "FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDN"
        "PHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDR"
        "RLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIY"
        "THQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCW"
        "MIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVV"
        "DADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNST"
    )

    sequences = {"wild_type": kinase_seq}

    # L858R: position 858 maps to ~index 162 in kinase domain (858-696=162)
    seq_l858r = list(kinase_seq)
    if len(seq_l858r) > 162:
        seq_l858r[162] = "R"  # L→R
    sequences["L858R"] = "".join(seq_l858r)

    # T790M: position 790 maps to ~index 94 (790-696=94)
    seq_t790m = list(kinase_seq)
    if len(seq_t790m) > 94:
        seq_t790m[94] = "M"  # T→M
    sequences["T790M"] = "".join(seq_t790m)

    # C797S: position 797 maps to ~index 101 (797-696=101)
    seq_c797s = list(kinase_seq)
    if len(seq_c797s) > 101:
        seq_c797s[101] = "S"  # C→S
    sequences["C797S"] = "".join(seq_c797s)

    # Double mutant L858R+T790M
    seq_double = list(kinase_seq)
    if len(seq_double) > 162:
        seq_double[162] = "R"
        seq_double[94] = "M"
    sequences["L858R_T790M"] = "".join(seq_double)

    # Triple mutant
    seq_triple = list(kinase_seq)
    if len(seq_triple) > 162:
        seq_triple[162] = "R"
        seq_triple[94] = "M"
        seq_triple[101] = "S"
    sequences["L858R_T790M_C797S"] = "".join(seq_triple)

    print(f"  Generated {len(sequences)} minimal mutant sequences")
    return sequences


def extract_esm2_embeddings(sequences: dict):
    """
    Extract ESM-2 embeddings for all mutant sequences.
    
    ESM-2 ARCHITECTURE:
    ───────────────────
    • Model: facebook/esm2_t33_650M_UR50D
    • Parameters: 650 million
    • Layers: 33 Transformer layers
    • Hidden dim: 1280
    • Trained on: UniRef50 (250M protein sequences)
    • Training objective: Masked language modeling (15% residues masked)
    
    EMBEDDING EXTRACTION:
    ─────────────────────
    For each sequence, we extract:
    
    1. Per-residue embeddings: (L × 1280) matrix
       - L = sequence length
       - Each row = 1280-dimensional vector for one amino acid
       - Captures the LOCAL context of each residue
       - Used for residue-level attention in the joint Transformer
    
    2. Sequence-level (pooled) embedding: (1 × 1280) vector
       - Mean of all per-residue embeddings
       - Captures the GLOBAL sequence context
       - Used as a summary representation
    
    WHY PER-RESIDUE EMBEDDINGS MATTER:
    ──────────────────────────────────
    The joint self-attention mechanism needs to know WHERE in the sequence
    each mutation is and how it affects local context. Per-residue embeddings
    let the model attend to:
    - Position 858 (L858R mutation site)
    - Position 790 (T790M gatekeeper)
    - Position 797 (C797S covalent binding site)
    - The ATP-binding pocket residues
    
    This allows cross-modal attention between specific residues and
    drug chemical tokens.
    """
    print("\n" + "=" * 70)
    print("STEP 7.1: Extracting ESM-2 Embeddings")
    print("=" * 70)

    device = get_device()
    print(f"  Device: {device}")

    model_name = cfg["model"]["esm2_model"]
    print(f"  Loading ESM-2 model: {model_name}")

    try:
        # ── Method 1: Using HuggingFace Transformers ─────────────────────────
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")

        embeddings = {}

        for seq_id, sequence in tqdm(sequences.items(), desc="  Extracting embeddings"):
            # Tokenize the amino acid sequence
            # ESM-2 uses single-character amino acid codes as tokens
            # Special tokens: <cls> at start, <eos> at end
            inputs = tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024  # ESM-2 supports up to 1024 tokens
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

                # Last hidden state: (1, L+2, 1280) — includes <cls> and <eos>
                last_hidden = outputs.last_hidden_state

                # Remove special tokens (first and last)
                per_residue_full = last_hidden[0, 1:-1, :].cpu().numpy()  # (L, 1280)

            # Pool to fixed length for training speed
            # Self-attention is O(n²) — 1022 tokens is too slow for MPS/GPU.
            # Adaptive avg pool preserves information from ALL residues.
            # (Same approach as K562 step07)
            import torch.nn.functional as F
            MAX_SEQ_TOKENS = 200
            if per_residue_full.shape[0] > MAX_SEQ_TOKENS:
                t = torch.from_numpy(per_residue_full).unsqueeze(0).permute(0, 2, 1)
                per_residue = F.adaptive_avg_pool1d(
                    t, MAX_SEQ_TOKENS).permute(0, 2, 1).squeeze(0).numpy()
                print(f"    {seq_id}: {per_residue_full.shape} → pooled to {per_residue.shape}")
            else:
                per_residue = per_residue_full

            # Mean pooling for sequence-level embedding
            pooled = per_residue.mean(axis=0)  # (1280,)

            embeddings[seq_id] = {
                "per_residue": per_residue,  # (MAX_SEQ_TOKENS, 1280)
                "pooled": pooled,  # (1280,)
                "sequence_length": len(sequence),
            }

        return embeddings

    except ImportError:
        print("  ⚠ Transformers library not available. Trying fair-esm...")
        return extract_esm2_fairesm(sequences, device)
    except Exception as e:
        print(f"  ⚠ ESM-2 extraction failed: {e}")
        print("  → Creating placeholder embeddings for development...")
        return create_placeholder_embeddings(sequences)


def extract_esm2_fairesm(sequences: dict, device):
    """Alternative extraction using the fair-esm library directly."""
    try:
        import esm

        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.to(device)
        model.eval()

        embeddings = {}

        for seq_id, sequence in tqdm(sequences.items(), desc="  Extracting (fair-esm)"):
            data = [(seq_id, sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
                per_residue = results["representations"][33][0, 1:-1, :].cpu().numpy()
                pooled = per_residue.mean(axis=0)

            embeddings[seq_id] = {
                "per_residue": per_residue,
                "pooled": pooled,
                "sequence_length": len(sequence),
            }

        return embeddings

    except Exception as e:
        print(f"  ⚠ fair-esm extraction failed: {e}")
        return create_placeholder_embeddings(sequences)


def create_placeholder_embeddings(sequences: dict):
    """Create random placeholder embeddings for development."""
    np.random.seed(42)
    embeddings = {}

    for seq_id, sequence in sequences.items():
        L = len(sequence)
        embeddings[seq_id] = {
            "per_residue": np.random.randn(L, 1280).astype(np.float32),
            "pooled": np.random.randn(1280).astype(np.float32),
            "sequence_length": L,
        }

    print(f"  ⚠ Created {len(embeddings)} PLACEHOLDER embeddings (dim=1280)")
    print("    These are RANDOM — install ESM-2 for real embeddings!")
    return embeddings


def save_embeddings(embeddings: dict):
    """Save extracted embeddings to disk."""
    print("\n  Saving embeddings...")

    for seq_id, emb in embeddings.items():
        # Save per-residue embeddings as numpy array
        np.save(
            FEATURES_DIR / f"{seq_id}_per_residue.npy",
            emb["per_residue"]
        )
        # Save pooled embedding
        np.save(
            FEATURES_DIR / f"{seq_id}_pooled.npy",
            emb["pooled"]
        )

    # Save metadata
    metadata = {
        seq_id: {
            "sequence_length": emb["sequence_length"],
            "per_residue_shape": list(emb["per_residue"].shape),
            "pooled_shape": list(emb["pooled"].shape),
        }
        for seq_id, emb in embeddings.items()
    }

    import json
    with open(FEATURES_DIR / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved {len(embeddings)} embeddings to {FEATURES_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 07: Extract ESM-2 Protein Sequence Embeddings        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Model: facebook/esm2_t33_650M_UR50D (650M params)         ║")
    print("║  Output: Per-residue (L×1280) + pooled (1280) embeddings   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Load EGFR sequences
    sequences = load_mutant_sequences()

    # Load ERBB2/HER2 sequence (for ERBB family expansion)
    erbb2_fasta = PROJECT_ROOT / cfg["paths"]["processed_data"] / "ccle" / "erbb2_mutant_sequences.fasta"
    if erbb2_fasta.exists():
        from Bio import SeqIO

        for record in SeqIO.parse(erbb2_fasta, "fasta"):
            seq_id = record.id.replace("ERBB2_", "ERBB2_")  # Keep ERBB2_ prefix
            if seq_id not in sequences:
                sequences[seq_id] = str(record.seq)
        print(f"  ✓ Added ERBB2 sequences (total: {len(sequences)})")
    else:
        # Try loading HER2 WT from raw FASTA
        erbb2_raw = PROJECT_ROOT / cfg["paths"]["raw_data"] / "ccle" / cfg["uniprot"]["ERBB2"]["fasta_file"]
        if erbb2_raw.exists():
            from Bio import SeqIO

            for record in SeqIO.parse(erbb2_raw, "fasta"):
                sequences["ERBB2_wild_type"] = str(record.seq)
            print(f"  ✓ Added ERBB2 WT sequence from {erbb2_raw.name} (1255 AA)")
        else:
            print(f"  ⚠ No ERBB2 FASTA found — ERBB2 samples will use placeholder")

    # Extract embeddings
    embeddings = extract_esm2_embeddings(sequences)

    # Save
    save_embeddings(embeddings)

    print("\n✓ Step 07 complete! ESM-2 embeddings ready for fusion model.")
    print(f"  EGFR sequences: {sum(1 for k in embeddings if not k.startswith('ERBB2'))}")
    print(f"  ERBB2 sequences: {sum(1 for k in embeddings if k.startswith('ERBB2'))}")
