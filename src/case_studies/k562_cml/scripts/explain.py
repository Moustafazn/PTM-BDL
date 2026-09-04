#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  K562/CML — PTM-BDL Explainability & Biological Validation                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PART 1  Per-sample predictions + group analysis (TKI vs chemo,             ║
║          per-drug, per-target_protein).                                      ║
║                                                                              ║
║  PART 2  PER-PROTEIN INTEGRATED GRADIENTS                                    ║
║          Phospho IG bucketed per protein (ABL1 / CRKL / STAT5A).            ║
║          Site-level rankings against UniProt-annotated site labels.          ║
║          Ref: Sundararajan, Taly & Yan, ICML 2017                           ║
║                                                                              ║
║  PART 3  CROSS-TYPE ATTENTION                                                ║
║          Post-softmax attention weights from the PTM-BDL transformer.       ║
║          For K562 (phospho-only), this shows phospho intra-type attention.  ║
║                                                                              ║
║  PART 4  PER-DRUG IG COMPARISON (TKI vs Chemo)                              ║
║          Dasatinib (multi-kinase TKI) vs Imatinib (BCR-ABL TKI) vs         ║
║          Cytarabine/Paclitaxel/Methotrexat (chemo) — tests whether the     ║
║          model discriminates kinase-targeted from non-kinase mechanisms.     ║
║          Ref: Shah et al., Science 2004 (PMID 15256107)                     ║
║                                                                              ║
║  PART 5  BCR-ABL SUBSTRATE VALIDATION                                        ║
║          Top-ranked IG sites should include:                                 ║
║            CRKL Y207 — canonical BCR-ABL biomarker                          ║
║            STAT5A Y694 — JAK2-BCR-ABL activation                            ║
║            ABL1 Y245/Y412 — activation loop autophosphorylation             ║
║          Ref: ten Hoeve et al., Blood 1994 (PMID 7517861)                   ║
║          Ref: Hantschel, Genes Dev 2012 (PMID 22855830)                     ║
║                                                                              ║
║  PART 6  MODEL VALIDATION SUMMARY                                            ║
║                                                                              ║
║  INPUTS:                                                                     ║
║    data/models/k562_cml/best_model.pt                                        ║
║    data/models/k562_cml/split_indices.json                                   ║
║    data/processed/k562_cml/multimodal_dataset.csv + data/features/*          ║
║                                                                              ║
║  OUTPUTS:                                                                    ║
║    results/k562_cml/xai_report.json                                          ║
║    results/k562_cml/figures/ptm_attribution.png                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

from src.ptm_bdl.data.dataset import ResistanceDataset
from src.ptm_bdl.training.factory import build_model_from_cfg
from src.ptm_bdl.training import load_checkpoint, resolve_device
from src.ptm_bdl.xai.integrated_gradients import compute_ig_batch
from src.ptm_bdl.xai.attention import compute_cross_type_attention
from src.ptm_bdl.config import load_config

CASE_STUDY = "k562_cml"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load optimal threshold (Youden's J from train.py, fallback 0.5)
_threshold_path = MODEL_DIR / "optimal_threshold.json"
if _threshold_path.exists():
    with open(_threshold_path) as _f:
        RESIST_THRESHOLD = float(json.load(_f).get("optimal_threshold", 0.5))
else:
    RESIST_THRESHOLD = 0.5

# Published BCR-ABL substrates that should be top-ranked by IG
PUBLISHED_ABL_SUBSTRATES = {
    "CRKL_Y207": {
        "gene": "CRKL", "site": "Y207",
        "function": "Canonical BCR-ABL biomarker — adaptor protein",
        "ref": "ten Hoeve et al., Blood 1994 (PMID 7517861)",
    },
    "STAT5A_Y694": {
        "gene": "STAT5A", "site": "Y694",
        "function": "JAK2-BCR-ABL transcription factor activation",
        "ref": "Nieborowska-Skorska et al., JEM 1999 (PMID 10364531)",
    },
    "ABL1_Y245": {
        "gene": "ABL1", "site": "Y245",
        "function": "ABL1 kinase autophosphorylation (activation marker)",
        "ref": "Hantschel, Genes Dev 2012 (PMID 22855830)",
    },
    "ABL1_Y412": {
        "gene": "ABL1", "site": "Y412",
        "function": "Activation loop — full kinase activation",
        "ref": "Hantschel, Genes Dev 2012 (PMID 22855830)",
    },
}

# Drug classification from biology.py
TKI_DRUGS = ["Dasatinib", "Imatinib"]
CHEMO_DRUGS = ["Cytarabine", "Paclitaxel", "Methotrexat"]


def _get_device():
    device_str = cfg["training"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _predict_single(model, sample, device=None):
    """Run model on a single sample, return (ic50_pred, p_resist)."""
    batch = {k: (v.view(1) if v.ndim == 0 else v.unsqueeze(0))
             for k, v in sample.items() if isinstance(v, torch.Tensor)}
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        ic50_pred, resist_logits = model(
            seq_embeddings=batch["seq_emb"],
            struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"],
            drug_embeddings=batch["drug_emb"],
            ptm_vector=batch["ptm_vector"],
            delta_ptm_vector=batch["delta_ptm_vector"],
            target_protein=batch["target_protein"],
        )
    return float(ic50_pred.item()), float(torch.sigmoid(resist_logits).item())


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Sample-level predictions + group analysis
# ══════════════════════════════════════════════════════════════════════════════

def collect_predictions(model, dataset, indices, device=None):
    """Collect per-sample predictions with metadata."""
    df = dataset.df
    samples = []
    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        ic50_pred, p_resist = _predict_single(model, sample, device=device)
        row = df.iloc[int(idx)]
        samples.append({
            "idx": int(idx),
            "drug_name": str(row.get("drug_name", "unknown")),
            "target_protein": str(row.get("target_protein", "unknown")),
            "resistance_label": int(row.get("resistance_label", 0)),
            "ic50_true": float(row.get("ln_ic50", 0)),
            "ic50_pred": ic50_pred,
            "resist_prob": p_resist,
            "resist_pred": int(p_resist > RESIST_THRESHOLD),
        })
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(indices)} samples predicted")
    return samples


def group_analysis(samples):
    """Compute per-group stats from predictions."""
    groups = {}

    # By drug
    by_drug = defaultdict(list)
    for s in samples:
        by_drug[s["drug_name"]].append(s)
    groups["by_drug"] = {}
    for drug, ss in sorted(by_drug.items()):
        groups["by_drug"][drug] = {
            "n": len(ss),
            "mean_resist_prob": float(np.mean([s["resist_prob"] for s in ss])),
            "mean_ic50_pred": float(np.mean([s["ic50_pred"] for s in ss])),
            "mean_ic50_true": float(np.mean([s["ic50_true"] for s in ss])),
            "accuracy": float(np.mean([s["resist_pred"] == s["resistance_label"] for s in ss])),
        }

    # By protein
    by_protein = defaultdict(list)
    for s in samples:
        by_protein[s["target_protein"]].append(s)
    groups["by_protein"] = {}
    for protein, ss in sorted(by_protein.items()):
        groups["by_protein"][protein] = {
            "n": len(ss),
            "mean_resist_prob": float(np.mean([s["resist_prob"] for s in ss])),
            "accuracy": float(np.mean([s["resist_pred"] == s["resistance_label"] for s in ss])),
        }

    # TKI vs Chemo
    tki_samples = [s for s in samples if s["drug_name"] in TKI_DRUGS]
    chemo_samples = [s for s in samples if s["drug_name"] in CHEMO_DRUGS]
    groups["tki_vs_chemo"] = {
        "tki": {
            "n": len(tki_samples),
            "mean_resist_prob": float(np.mean([s["resist_prob"] for s in tki_samples])) if tki_samples else 0,
        },
        "chemo": {
            "n": len(chemo_samples),
            "mean_resist_prob": float(np.mean([s["resist_prob"] for s in chemo_samples])) if chemo_samples else 0,
        },
    }
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Per-protein Integrated Gradients with site-level ranking
# ══════════════════════════════════════════════════════════════════════════════

def _get_protein_site_labels(protein_name: str) -> list:
    """Get per-protein site labels from config (resolves the display bug
    where CRKL/STAT5A slots were labeled with ABL1 site names)."""
    ptm_cfg = cfg.get("ptm", {})
    protein_cfg = ptm_cfg.get(protein_name, {})
    ptm_dim = ptm_cfg.get("ptm_dim", 12)

    labels = []
    for site in protein_cfg.get("phospho_sites", []):
        labels.append(site.get("residue", f"slot_{len(labels)}"))
    for site in protein_cfg.get("acetyl_sites", []):
        labels.append(site.get("residue", f"slot_{len(labels)}"))
    while len(labels) < ptm_dim:
        labels.append(f"PAD_{len(labels)}")
    return labels[:ptm_dim]


def compute_per_protein_ig(model, dataset, indices, n_steps=30):
    """
    Compute IG via the tool module, then produce per-protein site rankings
    with PTM site labels resolved PER PROTEIN from config.
    """
    print(f"\n  Computing per-protein IG on {len(indices)} samples ({n_steps} steps)...")
    ig_results = compute_ig_batch(model, dataset, indices, n_steps=n_steps)

    # Resolve protein names
    protein_names = {v: k for k, v in dataset._protein_map.items()}
    ptm_cols = dataset._ptm_cols

    per_protein = {}
    for pid, data in ig_results.items():
        protein = protein_names.get(pid, f"protein_{pid}")
        n_samples = data.pop("n_samples", 0)

        # Get per-protein site labels (not ABL1's column names)
        site_labels = _get_protein_site_labels(protein)

        # Get phospho attributions
        phospho_attr = data.get("ptm_vector", np.zeros(len(ptm_cols)))
        if isinstance(phospho_attr, np.ndarray):
            phospho_attr = phospho_attr.tolist()

        # Build site ranking with CORRECT per-protein labels
        sites = []
        for i, attr_val in enumerate(phospho_attr):
            label = site_labels[i] if i < len(site_labels) else f"slot_{i}"
            if label.startswith("PAD"):
                continue  # skip padded slots
            sites.append({
                "slot": i,
                "site": label,
                "mean_abs_attribution": float(abs(attr_val)),
            })
        sites.sort(key=lambda e: -e["mean_abs_attribution"])
        for rank, e in enumerate(sites, 1):
            e["rank"] = rank

        per_protein[protein] = {
            "n_samples": n_samples,
            "phospho_site_ranking": sites,
            "top_site": sites[0]["site"] if sites else "none",
            "top_attribution": sites[0]["mean_abs_attribution"] if sites else 0,
        }
        print(f"    {protein} (n={n_samples}): top={sites[0]['site'] if sites else '?'} "
              f"({sites[0]['mean_abs_attribution']:.6f})")

    return per_protein


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Per-drug IG comparison (TKI vs Chemo discrimination)
# ══════════════════════════════════════════════════════════════════════════════

def compute_per_drug_ig(model, dataset, indices, n_steps=20):
    """Compute IG stratified by drug to test TKI vs chemo discrimination."""
    print("\n  Per-drug Integrated Gradients (TKI vs chemo)...")
    df = dataset.df
    ptm_cols = dataset._ptm_cols
    per_drug = {}

    for drug in sorted(df.iloc[indices]["drug_name"].unique()):
        drug_mask = df.iloc[indices]["drug_name"] == drug
        drug_idx = [indices[i] for i, m in enumerate(drug_mask.values) if m]
        if len(drug_idx) < 2:
            continue

        drug_ig = compute_ig_batch(model, dataset, drug_idx, n_steps=n_steps)

        # Aggregate across proteins for this drug
        total_attr = np.zeros(len(ptm_cols))
        total_n = 0
        for pid, data in drug_ig.items():
            n = data.get("n_samples", 0)
            attr = data.get("ptm_vector", np.zeros(len(ptm_cols)))
            total_attr += np.abs(attr) * n
            total_n += n
        if total_n > 0:
            total_attr /= total_n

        # Top-3 sites
        top_indices = np.argsort(-total_attr)[:3]
        top_sites = [(ptm_cols[i].replace("ptm_", ""), float(total_attr[i]))
                     for i in top_indices]

        drug_class = "TKI" if drug in TKI_DRUGS else "Chemo"
        per_drug[drug] = {
            "n_samples": total_n,
            "drug_class": drug_class,
            "mean_phospho_attribution": total_attr.tolist(),
            "top_3_sites": [{"site": s, "attr": a} for s, a in top_sites],
            "total_ig_magnitude": float(np.sum(total_attr)),
        }
        print(f"    {drug} ({drug_class}, n={total_n}): "
              f"top={top_sites[0][0]} ({top_sites[0][1]:.6f})")

    return per_drug


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: BCR-ABL substrate validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_bcr_abl_substrates(per_protein_ig, ptm_cols):
    """Check if published BCR-ABL substrates appear in top-ranked IG sites."""
    print("\n  BCR-ABL substrate validation (published benchmarks)...")
    results = {}

    for substrate_id, info in PUBLISHED_ABL_SUBSTRATES.items():
        gene = info["gene"]
        site = info["site"]

        # Check if this protein is in our IG results
        if gene in per_protein_ig:
            ranking = per_protein_ig[gene].get("phospho_site_ranking", [])
            # Search for the site in the ranking
            found_rank = None
            for entry in ranking:
                if site.lower() in entry["site"].lower():
                    found_rank = entry["rank"]
                    break

            results[substrate_id] = {
                "gene": gene,
                "site": site,
                "function": info["function"],
                "reference": info["ref"],
                "rank_in_ig": found_rank,
                "in_top_5": found_rank is not None and found_rank <= 5,
                "in_top_10": found_rank is not None and found_rank <= 10,
            }
            status = f"rank={found_rank}" if found_rank else "not found"
            print(f"    {substrate_id}: {status} — {info['function']}")
        else:
            results[substrate_id] = {
                "gene": gene, "site": site,
                "function": info["function"],
                "reference": info["ref"],
                "rank_in_ig": None,
                "note": f"Protein {gene} not in test set",
            }
            print(f"    {substrate_id}: protein {gene} not in test set")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def explain():
    """Run comprehensive XAI analysis for K562/CML case study."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  K562/CML — XAI Analysis                                   ║")
    print("║  Per-protein IG, TKI vs chemo, BCR-ABL validation          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # XAI runs on CPU (same as EGFR) — IG needs requires_grad_ which
    # doesn't work reliably on MPS
    device = _get_device()
    print(f"  Device: {device}")

    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    dataset = ResistanceDataset(dataset_path, features_dir)
    print(f"  Dataset: {len(dataset)} samples")

    with open(MODEL_DIR / "split_indices.json") as f:
        split = json.load(f)
    test_idx = split["test_idx"]
    print(f"  Test samples: {len(test_idx)}")

    model = build_model_from_cfg(cfg).to(device)
    model_path = MODEL_DIR / "best_model.pt"
    if model_path.exists():
        load_checkpoint(model, model_path, device)
        print(f"  ✓ Loaded: {model_path.name}")
    else:
        print(f"  ⚠ No trained model — using random weights (demo)")
        model.eval()

    # ── PART 1: Predictions + group analysis ─────────────────────────────
    print("\n  PART 1: Per-sample predictions + group analysis")
    predictions = collect_predictions(model, dataset, test_idx, device=device)
    groups = group_analysis(predictions)

    # ── PART 2: Per-protein IG with site ranking ─────────────────────────
    print("\n  PART 2: Per-protein Integrated Gradients")
    per_protein_ig = compute_per_protein_ig(model, dataset, test_idx, n_steps=30)

    # ── PART 3: Cross-type attention ─────────────────────────────────────
    print("\n  PART 3: Cross-type attention analysis")
    try:
        attn_results = compute_cross_type_attention(
            model, dataset, test_idx, model.registry
        )
        print(f"    ✓ Computed for {len(attn_results)} proteins")
    except Exception as e:
        attn_results = {"error": str(e)}
        print(f"    ⚠ Attention extraction failed: {e}")

    # ── PART 4: Per-drug IG comparison ───────────────────────────────────
    print("\n  PART 4: Per-drug IG comparison (TKI vs chemo)")
    per_drug_ig = compute_per_drug_ig(model, dataset, test_idx, n_steps=20)

    # ── PART 5: BCR-ABL substrate validation ─────────────────────────────
    print("\n  PART 5: BCR-ABL substrate validation")
    substrate_validation = validate_bcr_abl_substrates(
        per_protein_ig, dataset._ptm_cols)

    # ── PART 6: Compile XAI report ───────────────────────────────────────
    print("\n  PART 6: Compiling report...")
    xai_report = {
        "case_study": CASE_STUDY,
        "threshold": RESIST_THRESHOLD,
        "n_test_samples": len(test_idx),
        "group_analysis": groups,
        "per_protein_ig": per_protein_ig,
        "cross_type_attention": attn_results,
        "per_drug_ig": per_drug_ig,
        "bcr_abl_substrate_validation": substrate_validation,
        "biological_validation": {
            "bcr_abl_substrates": PUBLISHED_ABL_SUBSTRATES,
            "dasatinib_vs_imatinib": {
                "description": "Dasatinib IG should be stronger than Imatinib "
                               "(325× more potent kinase inhibition)",
                "reference": "O'Hare et al., Blood 2005 (PMID 15256422)",
            },
            "tki_vs_chemo_discrimination": {
                "description": "TKI attention → kinase substrates; "
                               "chemo attention → DNA damage/cell cycle sites",
                "reference": "Hochhaus et al., Leukemia 2020 (PMID 31988391)",
            },
        },
        "references": [
            "Sundararajan et al., ICML 2017 — Integrated Gradients",
            "Shah et al., Science 2004 (PMID 15256107) — Dasatinib BCR-ABL",
            "O'Hare et al., Blood 2005 (PMID 15256422) — Dasatinib vs Imatinib",
            "ten Hoeve et al., Blood 1994 (PMID 7517861) — CRKL biomarker",
            "Hantschel, Genes Dev 2012 (PMID 22855830) — ABL1 autophosphorylation",
            "Badkul et al., DrugPTM-Bench 2024 — primary data source",
        ],
    }

    out_path = RESULTS_DIR / "xai_report.json"
    with open(out_path, "w") as f:
        json.dump(xai_report, f, indent=2, default=str)
    print(f"\n  ✓ XAI report saved: {out_path}")

    # ── Figure: per-protein IG attribution ───────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        proteins = sorted(per_protein_ig.keys())
        if proteins:
            n_proteins = len(proteins)
            fig, axes = plt.subplots(1, n_proteins, figsize=(6 * n_proteins, 5))
            if n_proteins == 1:
                axes = [axes]
            for ax, protein in zip(axes, proteins):
                data = per_protein_ig[protein]
                ranking = data.get("phospho_site_ranking", [])[:10]
                if ranking:
                    sites = [e["site"] for e in ranking]
                    attrs = [e["mean_abs_attribution"] for e in ranking]
                    ax.barh(range(len(sites)), attrs, color="#1f77b4", alpha=0.85)
                    ax.set_yticks(range(len(sites)))
                    ax.set_yticklabels(sites, fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xlabel("Mean |IG Attribution|")
                    ax.set_title(f"{protein} (n={data['n_samples']})")
            plt.suptitle("K562/CML — Per-Protein PTM Attribution (top 10 sites)",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "ptm_attribution.png", dpi=150,
                        bbox_inches="tight")
            plt.close()
            print(f"  ✓ Figure: {FIGURES_DIR / 'ptm_attribution.png'}")
    except Exception as e:
        print(f"  ⚠ Could not generate figure: {e}")

    print("✓ XAI analysis complete!")


if __name__ == "__main__":
    explain()
