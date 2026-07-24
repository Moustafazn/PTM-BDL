#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP 04 — PTM Baseline State Vectors: HDAC1 + EP300                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE:                                                                    ║
║    Create per-site baseline PTM state vectors for the target proteins        ║
║    (HDAC1, EP300). These vectors represent the UNTREATED PTM state of       ║
║    each protein and are used by step06 to build per-sample PTM input.       ║
║                                                                              ║
║  BIOLOGICAL CONTEXT:                                                         ║
║    HDAC1 phospho at S393/S421/S423 by CK2 regulates enzymatic activity     ║
║    and complex formation (Pflum et al., JBC 2001, PMID 11929873).           ║
║    EP300 autoacetylation at K1499 activates HAT function                    ║
║    (Thompson et al., NSMB 2004, PMID 15558049).                             ║
║                                                                              ║
║  OUTPUT:                                                                     ║
║    data/processed/ptm/hdac1_ptm_state_vectors.json                          ║
║    data/processed/ptm/ep300_ptm_state_vectors.json                          ║
║    data/processed/ptm/hdac1_phosphorylation_sites.csv                       ║
║    data/processed/ptm/ep300_phosphorylation_sites.csv                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from src.ptm_bdl.config import load_config

cfg = load_config(case_study="hela_hdac")
OUT_DIR = PROJECT_ROOT / cfg["paths"]["processed_data"] / "ptm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_hdac1_ptm_vectors():
    """
    Build baseline PTM state vectors for HDAC1 (Q13547).

    HDAC1 phosphorylation at S393/S421/S423 by CK2 is constitutive in
    most cancer cells and regulates HDAC1 enzymatic activity and NuRD/Sin3
    complex assembly. Acetylation at K218/K220 regulates catalytic activity.

    Biological contexts:
      - baseline: normal cancer cell HDAC1 phospho/acetyl state
      - hdac_inhibited: under Vorinostat/Romidepsin — acetylation increases
      - hat_inhibited: under A485 — acetylation decreases at EP300 targets

    Ref: Pflum et al., JBC 2001 (PMID 11929873)
    Ref: Galasinski et al., JBC 2002 (PMID 12032141)
    """
    print("\n  Building HDAC1 PTM state vectors...")

    sites = cfg["ptm"]["HDAC1"]
    phospho_sites = sites.get("phospho_sites", [])
    acetyl_sites = sites.get("acetyl_sites", [])

    # Baseline: normal cancer cell state
    # CK2-phosphorylated S393/S421/S423 = high (constitutive)
    # Other phospho sites = moderate
    # Acetyl sites = moderate (dynamic equilibrium)
    baseline = {}
    for s in phospho_sites:
        pos = s["position"]
        if pos in (393, 421, 423):
            baseline[str(pos)] = 2.5  # CK2 constitutive phospho — high
        else:
            baseline[str(pos)] = 1.0  # other phospho — baseline
    for s in acetyl_sites:
        pos = s["position"]
        if pos in (218, 220):
            baseline[str(pos)] = 1.5  # autoacetylation — moderate
        else:
            baseline[str(pos)] = 1.0  # other acetyl — baseline

    vectors = {
        "baseline_level": baseline,
    }

    # Save
    path = OUT_DIR / "hdac1_ptm_state_vectors.json"
    with open(path, "w") as f:
        json.dump(vectors, f, indent=2)
    print(f"    ✓ Saved: {path}")
    print(f"      Backgrounds: {list(vectors.keys())}")
    print(f"      Sites: {len(baseline)} ({len(phospho_sites)} phospho + {len(acetyl_sites)} acetyl)")

    # Save sites CSV
    all_sites = []
    for s in phospho_sites:
        all_sites.append({**s, "ptm_type": "phosphorylation", "protein": "HDAC1",
                          "uniprot": "Q13547", "baseline_level": baseline.get(str(s["position"]), 1.0)})
    for s in acetyl_sites:
        all_sites.append({**s, "ptm_type": "acetylation", "protein": "HDAC1",
                          "uniprot": "Q13547", "baseline_level": baseline.get(str(s["position"]), 1.0)})

    df = pd.DataFrame(all_sites)
    csv_path = OUT_DIR / "hdac1_phosphorylation_sites.csv"
    df.to_csv(csv_path, index=False)
    print(f"    ✓ Saved: {csv_path}")

    return vectors


def build_ep300_ptm_vectors():
    """
    Build baseline PTM state vectors for EP300/p300 (Q09472).

    EP300 autoacetylation at K1499 in the activation loop activates HAT
    enzymatic function. A485 (HAT inhibitor) blocks this autoacetylation.
    S1834 is phosphorylated by AKT, which also activates HAT activity.

    Ref: Thompson et al., NSMB 2004 (PMID 15558049)
    Ref: Liu et al., Mol Cell 2006 (PMID 17189186)
    """
    print("\n  Building EP300 PTM state vectors...")

    sites = cfg["ptm"]["EP300"]
    phospho_sites = sites.get("phospho_sites", [])
    acetyl_sites = sites.get("acetyl_sites", [])

    baseline = {}
    for s in phospho_sites:
        pos = s["position"]
        if pos == 1834:
            baseline[str(pos)] = 2.0  # AKT-phosphorylated — active
        else:
            baseline[str(pos)] = 1.0
    for s in acetyl_sites:
        pos = s["position"]
        if pos == 1499:
            baseline[str(pos)] = 3.0  # autoacetylation — HAT activation
        elif pos in (1549, 1558, 1560, 1568):
            baseline[str(pos)] = 2.0  # HAT domain autoacetylation cluster
        else:
            baseline[str(pos)] = 1.5

    vectors = {
        "baseline_level": baseline,
    }

    path = OUT_DIR / "ep300_ptm_state_vectors.json"
    with open(path, "w") as f:
        json.dump(vectors, f, indent=2)
    print(f"    ✓ Saved: {path}")

    all_sites = []
    for s in phospho_sites:
        all_sites.append({**s, "ptm_type": "phosphorylation", "protein": "EP300",
                          "uniprot": "Q09472", "baseline_level": baseline.get(str(s["position"]), 1.0)})
    for s in acetyl_sites:
        all_sites.append({**s, "ptm_type": "acetylation", "protein": "EP300",
                          "uniprot": "Q09472", "baseline_level": baseline.get(str(s["position"]), 1.0)})

    df = pd.DataFrame(all_sites)
    csv_path = OUT_DIR / "ep300_phosphorylation_sites.csv"
    df.to_csv(csv_path, index=False)
    print(f"    ✓ Saved: {csv_path}")

    return vectors


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  STEP 04 — HeLa/HDAC: PTM Baseline State Vectors              ║")
    print("║  Proteins: HDAC1 (Q13547) + EP300 (Q09472)                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    build_hdac1_ptm_vectors()
    build_ep300_ptm_vectors()

    print("\n✓ Step 04 complete!")
