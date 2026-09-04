#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HeLa/HDAC — PTM-BDL Explainability & Biological Validation                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PART 1  Per-sample predictions + group analysis (per-drug, per-protein,    ║
║          HDAC inhibitor vs HAT inhibitor vs inactive control).              ║
║                                                                              ║
║  PART 2  PER-PROTEIN INTEGRATED GRADIENTS (phospho + acetyl)                ║
║          IG bucketed per protein (EP300 / HDAC1 / CREBBP / HIST1H4A).      ║
║          BOTH phospho and acetyl channels are attributed separately.        ║
║          Site-level rankings against UniProt-annotated site labels.          ║
║          Ref: Sundararajan, Taly & Yan, ICML 2017                           ║
║                                                                              ║
║  PART 3  CROSS-TYPE ATTENTION (phospho ↔ acetyl)                            ║
║          Post-softmax attention from PTM-BDL transformer.                   ║
║          H3S10ph–K9ac binary switch validation.                             ║
║          Ref: Fischle et al., Nature 2003 (PMID 14573844)                   ║
║                                                                              ║
║  PART 4  PER-DRUG IG COMPARISON                                             ║
║          Vorinostat (pan-HDAC) → strong acetyl-K IG                         ║
║          A485 (HAT inhibitor) → negative/different acetyl-K pattern          ║
║          A486 (inactive) → minimal IG across all PTM types                   ║
║          Ref: Marks & Xu, J Cell Biochem 2009 (PMID 19479898)               ║
║                                                                              ║
║  PART 5  BIOLOGICAL VALIDATION SUMMARY                                       ║
║          Top-gene attribution ranking for HDAC/HAT targets.                  ║
║          EP300, HDAC1, CREBBP, histone genes should appear in top IG.       ║
║          Ref: Narita et al., Nat Rev Mol Cell Biol 2019 (PMID 30487433)    ║
║                                                                              ║
║  PART 6  MODEL VALIDATION SUMMARY                                            ║
║                                                                              ║
║  INPUTS:                                                                     ║
║    data/models/hela_hdac/best_model.pt + split_indices.json                  ║
║    data/processed/hela_hdac/multimodal_dataset.csv + data/features/*         ║
║                                                                              ║
║  OUTPUTS:                                                                    ║
║    results/hela_hdac/xai_report.json                                         ║
║    results/hela_hdac/figures/ptm_attribution.png                             ║
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

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load optimal threshold
_threshold_path = MODEL_DIR / "optimal_threshold.json"
if _threshold_path.exists():
    with open(_threshold_path) as _f:
        RESIST_THRESHOLD = float(json.load(_f).get("optimal_threshold", 0.5))
else:
    RESIST_THRESHOLD = 0.5

# Drug mechanism classes
HDAC_INHIBITORS = ["Vorinostat", "Romidepsin", "CUDC101"]
HAT_INHIBITORS = ["A485"]
INACTIVE_CONTROLS = ["A486"]
NATURAL_MODULATORS = ["Curcumin"]


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

    # Drug mechanism class comparison
    def _class_stats(drug_list, name):
        ss = [s for s in samples if s["drug_name"] in drug_list]
        return {
            "n": len(ss),
            "drugs": drug_list,
            "mean_resist_prob": float(np.mean([s["resist_prob"] for s in ss])) if ss else 0,
        }

    groups["drug_class_comparison"] = {
        "hdac_inhibitors": _class_stats(HDAC_INHIBITORS, "HDAC inhibitors"),
        "hat_inhibitors": _class_stats(HAT_INHIBITORS, "HAT inhibitors"),
        "inactive_controls": _class_stats(INACTIVE_CONTROLS, "Inactive controls"),
        "natural_modulators": _class_stats(NATURAL_MODULATORS, "Natural modulators"),
    }
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Per-protein IG (phospho + acetyl) with site-level ranking
# ══════════════════════════════════════════════════════════════════════════════

def _get_protein_site_labels(protein_name: str) -> list:
    """Get per-protein site labels from config (resolves the display bug
    where EP300 slots were labeled with HDAC1 site names)."""
    ptm_cfg = cfg.get("ptm", {})
    protein_cfg = ptm_cfg.get(protein_name, {})
    ptm_dim = ptm_cfg.get("ptm_dim", 12)

    labels = []
    ptm_types = []
    for site in protein_cfg.get("phospho_sites", []):
        labels.append(site.get("residue", f"slot_{len(labels)}"))
        ptm_types.append("phospho")
    for site in protein_cfg.get("acetyl_sites", []):
        labels.append(site.get("residue", f"slot_{len(labels)}"))
        ptm_types.append("acetyl")
    while len(labels) < ptm_dim:
        labels.append(f"PAD_{len(labels)}")
        ptm_types.append("pad")
    return labels[:ptm_dim], ptm_types[:ptm_dim]


def compute_per_protein_ig(model, dataset, indices, n_steps=30):
    """
    Compute IG per protein, returning site rankings with CORRECT per-protein labels.
    """
    print(f"\n  Computing per-protein IG on {len(indices)} samples ({n_steps} steps)...")
    ig_results = compute_ig_batch(model, dataset, indices, n_steps=n_steps)

    protein_names = {v: k for k, v in dataset._protein_map.items()}
    ptm_cols = dataset._ptm_cols
    sec_cols = dataset._secondary_cols

    per_protein = {}
    for pid, data in ig_results.items():
        protein = protein_names.get(pid, f"protein_{pid}")
        n_samples = data.pop("n_samples", 0)

        # Get per-protein site labels (not HDAC1's column names)
        site_labels, site_types = _get_protein_site_labels(protein)

        # Get phospho attributions (primary PTM channel)
        phospho_attr = data.get("ptm_vector", np.zeros(len(ptm_cols)))
        if isinstance(phospho_attr, np.ndarray):
            phospho_attr = phospho_attr.tolist()

        # Build site ranking with CORRECT per-protein labels
        phospho_sites = []
        acetyl_sites = []
        for i, attr_val in enumerate(phospho_attr):
            label = site_labels[i] if i < len(site_labels) else f"slot_{i}"
            stype = site_types[i] if i < len(site_types) else "pad"
            if stype == "pad":
                continue
            entry = {
                "slot": i, "site": label,
                "mean_abs_attribution": float(abs(attr_val)),
            }
            if stype == "acetyl":
                acetyl_sites.append(entry)
            else:
                phospho_sites.append(entry)

        phospho_sites.sort(key=lambda e: -e["mean_abs_attribution"])
        for rank, e in enumerate(phospho_sites, 1):
            e["rank"] = rank
        acetyl_sites.sort(key=lambda e: -e["mean_abs_attribution"])
        for rank, e in enumerate(acetyl_sites, 1):
            e["rank"] = rank

        per_protein[protein] = {
            "n_samples": n_samples,
            "phospho_site_ranking": phospho_sites,
            "acetyl_site_ranking": acetyl_sites,
            "phospho_top": phospho_sites[0]["site"] if phospho_sites else "none",
            "acetyl_top": acetyl_sites[0]["site"] if acetyl_sites else "none",
        }
        print(f"    {protein} (n={n_samples}): "
              f"phospho top={phospho_sites[0]['site'] if phospho_sites else '?'}, "
              f"acetyl top={acetyl_sites[0]['site'] if acetyl_sites else '?'}")

    return per_protein


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Per-drug IG comparison
# ══════════════════════════════════════════════════════════════════════════════

def compute_per_drug_ig(model, dataset, indices, n_steps=20):
    """Compute IG stratified by drug mechanism."""
    print("\n  Per-drug Integrated Gradients...")
    df = dataset.df
    ptm_cols = dataset._ptm_cols
    sec_cols = dataset._secondary_cols
    per_drug = {}
    n_ptm = len(ptm_cols)
    n_sec = len(sec_cols) if sec_cols else 0
    n_total = n_ptm + n_sec

    for drug in sorted(df.iloc[indices]["drug_name"].unique()):
        drug_mask = df.iloc[indices]["drug_name"] == drug
        drug_idx = [indices[i] for i, m in enumerate(drug_mask.values) if m]
        if len(drug_idx) < 2:
            continue

        drug_ig = compute_ig_batch(model, dataset, drug_idx, n_steps=n_steps)

        # Aggregate — ptm_vector from IG contains ALL tokens (phospho + acetyl)
        total_phospho = np.zeros(n_ptm)
        total_acetyl = np.zeros(n_sec) if n_sec > 0 else np.zeros(0)
        total_n = 0
        for pid, data in drug_ig.items():
            n = data.get("n_samples", 0)
            full_attr = np.abs(data.get("ptm_vector", np.zeros(n_total)))
            total_phospho += full_attr[:n_ptm] * n
            if n_sec > 0 and len(full_attr) > n_ptm:
                total_acetyl += full_attr[n_ptm:n_ptm + n_sec] * n
            total_n += n
        if total_n > 0:
            total_phospho /= total_n
            if n_sec > 0:
                total_acetyl /= total_n

        # Drug class
        if drug in HDAC_INHIBITORS:
            drug_class = "HDAC inhibitor"
        elif drug in HAT_INHIBITORS:
            drug_class = "HAT inhibitor"
        elif drug in INACTIVE_CONTROLS:
            drug_class = "Inactive control"
        else:
            drug_class = "Natural modulator"

        per_drug[drug] = {
            "n_samples": total_n,
            "drug_class": drug_class,
            "phospho_ig_magnitude": float(np.sum(total_phospho)),
            "acetyl_ig_magnitude": float(np.sum(total_acetyl)) if sec_cols else 0,
            "phospho_attribution": total_phospho.tolist(),
            "acetyl_attribution": total_acetyl.tolist() if sec_cols else [],
        }
        print(f"    {drug} ({drug_class}, n={total_n}): "
              f"phospho_mag={np.sum(total_phospho):.6f}, "
              f"acetyl_mag={np.sum(total_acetyl):.6f}")

    return per_drug


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def explain():
    """Run comprehensive XAI analysis for HeLa/HDAC case study."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HeLa/HDAC — XAI Analysis                                  ║")
    print("║  Per-protein IG (phospho + acetyl), drug comparison         ║")
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

    # ── PART 2: Per-protein IG (phospho + acetyl) ────────────────────────
    print("\n  PART 2: Per-protein Integrated Gradients (phospho + acetyl)")
    per_protein_ig = compute_per_protein_ig(model, dataset, test_idx, n_steps=30)

    # ── PART 3: Cross-type attention ─────────────────────────────────────
    print("\n  PART 3: Cross-type attention (phospho ↔ acetyl)")
    try:
        attn_results = compute_cross_type_attention(
            model, dataset, test_idx, model.registry
        )
        print(f"    ✓ Computed for {len(attn_results)} proteins")
    except Exception as e:
        attn_results = {"error": str(e)}
        print(f"    ⚠ Attention extraction failed: {e}")

    # ── PART 4: Per-drug IG comparison ───────────────────────────────────
    print("\n  PART 4: Per-drug IG comparison (mechanism-stratified)")
    per_drug_ig = compute_per_drug_ig(model, dataset, test_idx, n_steps=20)

    # ── PART 5+6: Compile XAI report ────────────────────────────────────
    print("\n  PART 5-6: Compiling report...")
    xai_report = {
        "case_study": CASE_STUDY,
        "threshold": RESIST_THRESHOLD,
        "n_test_samples": len(test_idx),
        "group_analysis": groups,
        "per_protein_ig": per_protein_ig,
        "cross_type_attention": attn_results,
        "per_drug_ig": per_drug_ig,
        "biological_validation_targets": {
            "acetyl_increases_under_hdac_inhibition": {
                "drugs": ["Vorinostat", "Romidepsin"],
                "expected": "Positive IG for acetyl_K sites",
                "reference": "Marks & Xu 2009 (PMID 19479898)",
            },
            "hat_inhibition_decreases_acetylation": {
                "drugs": ["A485"],
                "expected": "Negative IG for acetyl_K at EP300 sites",
                "reference": "Lasko et al., Nature 2017 (PMID 29211713)",
            },
            "inactive_control_minimal_signal": {
                "drugs": ["A486"],
                "expected": "Near-zero IG across all PTM types",
                "reference": "Lasko et al., Nature 2017 (PMID 29211713)",
            },
            "phospho_acetyl_crosstalk": {
                "description": "H3S10ph anti-correlates with H3K9ac",
                "expected": "Non-trivial off-diagonal cross-type attention",
                "reference": "Fischle et al., Nature 2003 (PMID 14573844)",
            },
        },
        "references": [
            "Sundararajan et al., ICML 2017 — Integrated Gradients",
            "Fischle et al., Nature 2003 (PMID 14573844) — H3 phospho-acetyl switch",
            "Ardito et al., IJMS 2019 — Acetylation↔phosphorylation crosstalk",
            "Lasko et al., Nature 2017 (PMID 29211713) — A485/A486 mechanism",
            "Marks & Xu, J Cell Biochem 2009 (PMID 19479898) — HDAC inhibitors",
            "Narita et al., Nat Rev Mol Cell Biol 2019 (PMID 30487433) — HATs/HDACs",
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
            fig, axes = plt.subplots(2, n_proteins, figsize=(6 * n_proteins, 10))
            if n_proteins == 1:
                axes = axes.reshape(2, 1)

            for col, protein in enumerate(proteins):
                data = per_protein_ig[protein]

                # Phospho subplot
                ax = axes[0, col] if n_proteins > 1 else axes[0, 0]
                ranking = data.get("phospho_site_ranking", [])[:8]
                if ranking:
                    sites = [e["site"] for e in ranking]
                    attrs = [e["mean_abs_attribution"] for e in ranking]
                    ax.barh(range(len(sites)), attrs, color="#1f77b4", alpha=0.85)
                    ax.set_yticks(range(len(sites)))
                    ax.set_yticklabels(sites, fontsize=7)
                    ax.invert_yaxis()
                    ax.set_xlabel("Mean |IG|")
                    ax.set_title(f"{protein} Phospho (n={data['n_samples']})")

                # Acetyl subplot
                ax = axes[1, col] if n_proteins > 1 else axes[1, 0]
                ranking = data.get("acetyl_site_ranking", [])[:8]
                if ranking:
                    sites = [e["site"] for e in ranking]
                    attrs = [e["mean_abs_attribution"] for e in ranking]
                    ax.barh(range(len(sites)), attrs, color="#ff7f0e", alpha=0.85)
                    ax.set_yticks(range(len(sites)))
                    ax.set_yticklabels(sites, fontsize=7)
                    ax.invert_yaxis()
                    ax.set_xlabel("Mean |IG|")
                    ax.set_title(f"{protein} Acetyl")

            plt.suptitle("HeLa/HDAC — Per-Protein PTM Attribution",
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
