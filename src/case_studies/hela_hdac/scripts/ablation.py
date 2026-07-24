#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HeLa/HDAC — PTM-BDL Ablation Study                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Three families of tests for the PTM-BDL architecture:                      ║
║                                                                              ║
║  PART 1 — FEATURE & ARCHITECTURE ABLATIONS                                  ║
║    Symmetric across the two PTM channels (phospho ⊕ acetyl):               ║
║      no_ptm              — ALL PTM features zeroed (static baseline)        ║
║      no_secondary        — phospho channel only (drops acetylation)         ║
║      secondary_only      — acetyl channel only (drops phosphorylation)      ║
║      no_drug             — drug embeddings zeroed                            ║
║      no_structure        — GearNet structural embeddings zeroed              ║
║      no_typed_attention  — PTM-BDL with MLP in place of typed self-attn    ║
║      full                — full PTM-BDL (phospho + acetyl)                  ║
║                                                                              ║
║  PART 2 — MULTI-SEED STABILITY (3 seeds × per-protein IG)                  ║
║    Per-protein phospho + acetyl IG at the PTM-BDL token boundary.           ║
║    Ref: Fischle et al., Nature 2003 (PMID 14573844) — H3 phospho/acetyl    ║
║                                                                              ║
║  PART 3 — RANDOMISED PTM CONTROL (inference-only permutation)               ║
║    Three arms: phospho_shuffled / acetyl_shuffled / both_shuffled           ║
║    PASS = real PTM beats shuffled on AUROC ≥ +0.005.                        ║
║    Ref: Breiman 2001; Fisher et al., JMLR 2019                             ║
║                                                                              ║
║  All sub-tests reuse the SAME train/val/test split from train.py.           ║
║  All models are built with `build_model_from_cfg` (single source of truth). ║
║                                                                              ║
║  OUTPUTS:                                                                    ║
║    results/ablation_study.json            — Part 1 metrics + votes          ║
║    results/stability_analysis.json        — Part 2 IG + consistency         ║
║    results/randomized_ptm_control.json    — Part 3 permutation test         ║
║    results/figures/ablation_comparison.png                                   ║
║                                                                              ║
║  Biological context:                                                         ║
║    HeLa (cervical carcinoma, HPV18+) — 6 drugs: Vorinostat, Romidepsin,    ║
║    CUDC-101, A485, A486, Curcumin                                           ║
║    PTM types: phosphorylation (S/T/Y) + acetylation (K) — NEW PTM type     ║
║    Ref: Narita et al., Nat Rev Mol Cell Biol 2019 (PMID 30487433)          ║
║    Ref: Badkul et al., DrugPTM-Bench 2024                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# ── Import from framework packages ──────────────────────────────────────────
from src.ptm_bdl.data.dataset import ResistanceDataset
from src.ptm_bdl.data.collate import collate_fn
from src.ptm_bdl.training.loss import FocalLoss
from src.ptm_bdl.training.trainer import train_epoch, validate
from src.ptm_bdl.training.factory import build_model_from_cfg
from src.ptm_bdl.config import load_config

CASE_STUDY = "hela_hdac"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Common training/data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_device():
    device_str = cfg["training"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_split():
    split_path = MODEL_DIR / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"split_indices.json not found at {split_path}. "
            "Run train.py first to create the train/val/test split."
        )
    with open(split_path) as f:
        split = json.load(f)
    return (
        np.array(split["train_idx"]),
        np.array(split["val_idx"]),
        np.array(split["test_idx"]),
    )


def _make_loaders(dataset, train_idx, val_idx, test_idx):
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # Weighted sampling to handle class imbalance
    train_labels = dataset.df["resistance_label"].values[train_idx].astype(int)
    class_counts = np.bincount(train_labels)
    if class_counts.min() == 0:
        class_weights = np.ones(len(class_counts), dtype=np.float32)
    else:
        class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels].astype(np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_set), replacement=True,
    )
    bs = cfg["model"]["batch_size"]
    return (
        DataLoader(train_set, batch_size=bs, sampler=sampler, collate_fn=collate_fn),
        DataLoader(val_set, batch_size=bs, shuffle=False, collate_fn=collate_fn),
        DataLoader(test_set, batch_size=bs, shuffle=False, collate_fn=collate_fn),
    )


def _train_loop(model, train_loader, val_loader, focal_loss, device, save_path):
    """Standard cosine-LR loop with early stopping on max(AUROC, BAcc)."""
    lr = cfg["model"]["learning_rate"]
    wd = cfg["model"]["weight_decay"]
    n_epochs = cfg["model"]["num_epochs"]
    patience = cfg["model"]["early_stopping_patience"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01,
    )
    best_score = 0.0
    counter = 0
    epoch_done = 0

    print(f"    Training: max {n_epochs} epochs, patience={patience}, lr={lr}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 focal_loss, 1.0, 2.0, device)
        vm = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
        elapsed = time.time() - t0
        score = max(vm.get("auroc", 0), vm.get("balanced_acc", 0))
        improved = score > best_score

        if epoch <= 2 or epoch % 5 == 0 or improved:
            marker = " ★" if improved else ""
            print(f"    Ep {epoch:3d}/{n_epochs} | "
                  f"loss={train_loss:.4f} | "
                  f"AUROC={vm.get('auroc', 0):.3f} | "
                  f"BAcc={vm.get('balanced_acc', 0):.3f} | "
                  f"RMSE={vm.get('rmse', 0):.3f} | "
                  f"score={score:.3f} ({elapsed:.1f}s){marker}")

        if improved:
            best_score = score
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"    Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs, "
                      f"best={best_score:.3f})")
                epoch_done = epoch
                break
        epoch_done = epoch

    return best_score, epoch_done


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Feature & Architecture Ablations
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "no_ptm": dict(
        label="Model A: No PTM",
        description="All PTM features (phospho + acetyl) zeroed — static baseline.",
        color="#d62728", use_typed_attention=True, data_mode="no_ptm",
    ),
    "no_secondary": dict(
        label="Model B: No Acetyl",
        description="Phospho channel active, acetyl channel zeroed.",
        color="#9467bd", use_typed_attention=True, data_mode="no_secondary",
    ),
    "secondary_only": dict(
        label="Model C: Acetyl Only",
        description="Acetyl channel active, phospho channel zeroed.",
        color="#8c564b", use_typed_attention=True, data_mode="secondary_only",
    ),
    "no_drug": dict(
        label="Model D: No Drug",
        description="Drug embeddings (ChemBERTa) zeroed.",
        color="#ff7f0e", use_typed_attention=True, data_mode="no_drug",
    ),
    "no_structure": dict(
        label="Model E: No Structure",
        description="Structural embeddings (GearNet) zeroed.",
        color="#2ca02c", use_typed_attention=True, data_mode="no_structure",
    ),
    "no_typed_attention": dict(
        label="Model F: No Typed Attention",
        description="PTM-BDL typed self-attention replaced with MLP.",
        color="#e377c2", use_typed_attention=False, data_mode="full",
    ),
    "full": dict(
        label="Model G: Full PTM-BDL",
        description="All features + typed self-attention (production model).",
        color="#1f77b4", use_typed_attention=True, data_mode="full",
    ),
}

ABLATION_ORDER = [
    "no_ptm", "no_secondary", "secondary_only",
    "no_drug", "no_structure", "no_typed_attention", "full",
]


def train_ablation_model(mode_key, dataset_path, features_dir,
                         train_idx, val_idx, test_idx, device):
    spec = ABLATION_CONFIGS[mode_key]
    print(f"\n  {'─' * 60}")
    print(f"  {spec['label']}")
    print(f"  {spec['description']}")
    print(f"  {'─' * 60}")

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # KEY FIX: pass ablation_mode to ResistanceDataset
    dataset = ResistanceDataset(dataset_path, features_dir,
                                ablation_mode=spec["data_mode"])
    train_loader, val_loader, test_loader = _make_loaders(
        dataset, train_idx, val_idx, test_idx)

    model = build_model_from_cfg(
        cfg, use_typed_attention=spec["use_typed_attention"]).to(device)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    save_path = MODEL_DIR / f"ablation_{mode_key}.pt"

    t0 = time.time()
    best_score, n_epochs = _train_loop(
        model, train_loader, val_loader, focal_loss, device, save_path)
    elapsed = time.time() - t0

    # Reload best model and evaluate on both val and test sets
    model.load_state_dict(torch.load(save_path, map_location=device,
                                     weights_only=True))
    val_m = validate(model, val_loader, focal_loss, 1.0, 2.0, device)
    test_m = validate(model, test_loader, focal_loss, 1.0, 2.0, device)
    print(f"    Test: BAcc={test_m['balanced_acc']:.3f}, "
          f"AUROC={test_m.get('auroc', 0):.3f}, "
          f"RMSE={test_m.get('rmse', 0):.3f}, "
          f"R={test_m.get('pearson_r', 0):.3f}  ({n_epochs} epochs / {elapsed:.0f}s)")

    return {
        "label": spec["label"],
        "description": spec["description"],
        "val_metrics": val_m,
        "test_metrics": test_m,
        "training_epochs": n_epochs,
        "training_time_seconds": round(elapsed, 1),
        "use_typed_attention": spec["use_typed_attention"],
        "data_mode": spec["data_mode"],
    }


def run_ablation_study(device):
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 1: PTM-BDL Feature & Architecture Ablations")
    print("══════════════════════════════════════════════════════════════")

    train_idx, val_idx, test_idx = _load_split()
    print(f"  Using shared split: train={len(train_idx)}, val={len(val_idx)}, "
          f"test={len(test_idx)}")
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    results = {}
    for mode in ABLATION_ORDER:
        results[mode] = train_ablation_model(
            mode, dataset_path, features_dir,
            train_idx, val_idx, test_idx, device,
        )

    # ── Comparison table ────────────────────────────────────────────────
    print("\n  " + "=" * 96)
    print(f"  {'Model':<30s} | {'AUROC':>6s} | {'PR-S':>5s} | {'BAcc':>5s} | "
          f"{'F1mac':>5s} | {'RMSE':>5s} | {'R':>5s} | {'Δ AUROC vs no_ptm':>17s}")
    print("  " + "-" * 96)
    base_auroc = results["no_ptm"]["test_metrics"].get("auroc", 0)
    for mode in ABLATION_ORDER:
        t = results[mode]["test_metrics"]
        d = t.get("auroc", 0) - base_auroc
        print(f"  {results[mode]['label']:<30s} | {t.get('auroc', 0):6.3f} | "
              f"{t.get('auprc_sensitive', 0):5.3f} | {t['balanced_acc']:5.3f} | "
              f"{t.get('f1_macro', 0):5.3f} | {t.get('rmse', 0):5.3f} | "
              f"{t.get('pearson_r', 0):5.3f} | {d:+17.3f}")
    print("  " + "=" * 96)

    # ── Interpretation: did PTM-BDL help? (votes) ───────────────────────
    full_m = results["full"]["test_metrics"]
    noptm_m = results["no_ptm"]["test_metrics"]
    gains = {
        "auroc": full_m.get("auroc", 0) - noptm_m.get("auroc", 0),
        "auprc_sensitive": (full_m.get("auprc_sensitive", 0)
                            - noptm_m.get("auprc_sensitive", 0)),
        "bacc": full_m["balanced_acc"] - noptm_m["balanced_acc"],
        "f1_macro": full_m.get("f1_macro", 0) - noptm_m.get("f1_macro", 0),
    }
    votes_help = sum(1 for v in gains.values() if v > 0.01)
    votes_total = len(gains)

    # Channel-level and modality-level marginal gains
    acetyl_marginal = (full_m.get("auroc", 0)
                       - results["no_secondary"]["test_metrics"].get("auroc", 0))
    phospho_marginal = (full_m.get("auroc", 0)
                        - results["secondary_only"]["test_metrics"].get("auroc", 0))
    drug_marginal = (full_m.get("auroc", 0)
                     - results["no_drug"]["test_metrics"].get("auroc", 0))
    struct_marginal = (full_m.get("auroc", 0)
                       - results["no_structure"]["test_metrics"].get("auroc", 0))
    typed_attn_marginal = (full_m.get("auroc", 0)
                           - results["no_typed_attention"]["test_metrics"].get("auroc", 0))

    print(f"\n  Modality-level marginal AUROC gains (vs full):")
    print(f"    PTM (all) marginal          : {gains['auroc']:+.3f} "
          f"(full vs no_ptm)")
    print(f"    Phospho marginal            : {phospho_marginal:+.3f} "
          f"(full vs acetyl-only)")
    print(f"    Acetyl marginal             : {acetyl_marginal:+.3f} "
          f"(full vs phospho-only)")
    print(f"    Drug marginal               : {drug_marginal:+.3f} "
          f"(full vs no_drug)")
    print(f"    Structure marginal          : {struct_marginal:+.3f} "
          f"(full vs no_structure)")
    print(f"    Typed-attention marginal     : {typed_attn_marginal:+.3f} "
          f"(typed-attn vs MLP)")
    print(f"\n  PTM-BDL vs No-PTM votes      : {votes_help}/{votes_total} "
          f"metrics positive")

    summary = {
        "ptm_gain_auroc": round(gains["auroc"], 4),
        "ptm_gain_auprc_sensitive": round(gains["auprc_sensitive"], 4),
        "ptm_gain_bacc": round(gains["bacc"], 4),
        "ptm_gain_f1_macro": round(gains["f1_macro"], 4),
        "phospho_marginal_auroc": round(phospho_marginal, 4),
        "acetyl_marginal_auroc": round(acetyl_marginal, 4),
        "drug_marginal_auroc": round(drug_marginal, 4),
        "structure_marginal_auroc": round(struct_marginal, 4),
        "typed_attention_marginal_auroc": round(typed_attn_marginal, 4),
        "votes_ptm_helps": votes_help,
        "votes_total": votes_total,
        "conclusion": ("PTM_BDL_HELPS" if votes_help >= 3
                       else "MIXED" if votes_help >= 1 else "NO_HELP"),
    }

    save = {mode: {
        "label": results[mode]["label"],
        "description": results[mode]["description"],
        "use_typed_attention": results[mode]["use_typed_attention"],
        "data_mode": results[mode]["data_mode"],
        "val_metrics": results[mode]["val_metrics"],
        "test_metrics": results[mode]["test_metrics"],
        "training_epochs": results[mode]["training_epochs"],
        "training_time_seconds": results[mode]["training_time_seconds"],
    } for mode in ABLATION_ORDER}
    save["_summary"] = summary

    out_path = RESULTS_DIR / "ablation_study.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")

    # ── Figure ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [results[m]["label"].split(":")[0] for m in ABLATION_ORDER]
        colors = [ABLATION_CONFIGS[m]["color"] for m in ABLATION_ORDER]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, key, title in [
            (axes[0], "auroc", "AUROC (test)"),
            (axes[1], "balanced_acc", "Balanced Accuracy (test)"),
            (axes[2], "rmse", "RMSE on ln(IC50) (test)"),
        ]:
            vals = [results[m]["test_metrics"].get(key, 0) for m in ABLATION_ORDER]
            ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
        plt.suptitle("HeLa/HDAC — PTM-BDL Ablation Study", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure saved: {FIGURES_DIR / 'ablation_comparison.png'}")
    except Exception as e:
        print(f"  ⚠ Could not generate figure: {e}")

    return save


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Multi-seed stability + per-protein IG
# ══════════════════════════════════════════════════════════════════════════════

def _run_ptm_bdl_ig(model, dataset, indices, n_steps: int = 20):
    """
    Integrated Gradients on the PTM-BDL phospho + acetyl input plane.

    HeLa has both phospho and acetyl (secondary) channels, so integrates along:
      - ptm_vector / delta_ptm_vector (phospho)
      - secondary_vector / delta_secondary_vector (acetyl)

    Per-site importance = |grad_level × Δlevel| + |grad_delta × Δdelta|

    Returns dict with arrays per protein (EP300, HDAC1, etc.).
    """
    model.train()  # need grads
    dev = next(model.parameters()).device
    ptm_dim = dataset._ptm_dim
    sec_dim = dataset._secondary_dim
    baseline_phospho = torch.ones(ptm_dim, device=dev)
    baseline_dphospho = torch.zeros(ptm_dim, device=dev)
    baseline_acetyl = torch.ones(sec_dim, device=dev) if sec_dim > 0 else torch.zeros(0, device=dev)
    baseline_dacetyl = torch.zeros(sec_dim, device=dev) if sec_dim > 0 else torch.zeros(0, device=dev)

    sums_ph = {}
    sums_ac = {}
    counts = {}
    protein_names = {v: k for k, v in dataset._protein_map.items()}

    for idx in indices:
        sample = dataset[int(idx)]
        actual_phospho = sample["ptm_vector"].to(dev)
        actual_dphospho = sample["delta_ptm_vector"].to(dev)
        actual_acetyl = sample["secondary_vector"].to(dev)
        actual_dacetyl = sample["delta_secondary_vector"].to(dev)
        tp = sample["target_protein"].view(1).long().to(dev)
        pid = int(tp.item())
        protein = protein_names.get(pid, f"protein_{pid}")

        if protein not in sums_ph:
            sums_ph[protein] = np.zeros(ptm_dim)
            sums_ac[protein] = np.zeros(sec_dim) if sec_dim > 0 else np.zeros(0)
            counts[protein] = 0
        counts[protein] += 1

        seq_e = sample["seq_emb"].unsqueeze(0).to(dev)
        str_e = sample["struct_emb"].unsqueeze(0).to(dev)
        drg_e = sample["drug_emb"].unsqueeze(0).to(dev)
        drg_p = sample["drug_pooled"].unsqueeze(0).to(dev)

        grads_phospho = torch.zeros(ptm_dim, device=dev)
        grads_dphospho = torch.zeros(ptm_dim, device=dev)
        grads_acetyl = torch.zeros(sec_dim, device=dev) if sec_dim > 0 else torch.zeros(0, device=dev)
        grads_dacetyl = torch.zeros(sec_dim, device=dev) if sec_dim > 0 else torch.zeros(0, device=dev)

        for step in range(n_steps + 1):
            a = step / n_steps
            iph = (baseline_phospho + a * (actual_phospho - baseline_phospho)
                   ).unsqueeze(0).requires_grad_(True)
            idph = (baseline_dphospho + a * (actual_dphospho - baseline_dphospho)
                    ).unsqueeze(0).requires_grad_(True)

            if sec_dim > 0:
                iac = (baseline_acetyl + a * (actual_acetyl - baseline_acetyl)
                       ).unsqueeze(0).requires_grad_(True)
                idac = (baseline_dacetyl + a * (actual_dacetyl - baseline_dacetyl)
                        ).unsqueeze(0).requires_grad_(True)
            else:
                iac = actual_acetyl.unsqueeze(0)
                idac = actual_dacetyl.unsqueeze(0)

            _, resist_pred = model(
                seq_embeddings=seq_e,
                struct_embeddings=str_e,
                drug_pooled=drg_p,
                drug_embeddings=drg_e,
                ptm_vector=iph,
                delta_ptm_vector=idph,
                secondary_vector=iac,
                delta_secondary_vector=idac,
                target_protein=tp,
            )
            model.zero_grad()
            resist_pred.backward()
            if iph.grad is not None:
                grads_phospho += iph.grad.squeeze(0).detach()
            if idph.grad is not None:
                grads_dphospho += idph.grad.squeeze(0).detach()
            if sec_dim > 0:
                if iac.grad is not None:
                    grads_acetyl += iac.grad.squeeze(0).detach()
                if idac.grad is not None:
                    grads_dacetyl += idac.grad.squeeze(0).detach()

        n_s = n_steps + 1
        delta_ph = actual_phospho - baseline_phospho
        delta_dph = actual_dphospho - baseline_dphospho
        attr_ph = (np.abs(((grads_phospho / n_s) * delta_ph).cpu().numpy())
                   + np.abs(((grads_dphospho / n_s) * delta_dph).cpu().numpy()))
        sums_ph[protein] += attr_ph

        if sec_dim > 0:
            delta_ac = actual_acetyl - baseline_acetyl
            delta_dac = actual_dacetyl - baseline_dacetyl
            attr_ac = (np.abs(((grads_acetyl / n_s) * delta_ac).cpu().numpy())
                       + np.abs(((grads_dacetyl / n_s) * delta_dac).cpu().numpy()))
            sums_ac[protein] += attr_ac

    model.eval()
    out = {}
    for protein in sums_ph:
        n = max(counts[protein], 1)
        entry = {
            "phospho": (sums_ph[protein] / n).tolist(),
            "n_samples": counts[protein],
        }
        if sec_dim > 0:
            entry["acetyl"] = (sums_ac[protein] / n).tolist()
        out[protein] = entry
    return out


def run_stability_analysis(device, n_seeds: int = 3):
    print("\n══════════════════════════════════════════════════════════════")
    print(f"PART 2: Multi-Seed Stability ({n_seeds} seeds) + per-protein IG")
    print("══════════════════════════════════════════════════════════════")
    train_idx, val_idx, test_idx = _load_split()
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    seeds = [42, 123, 456][:n_seeds]

    per_seed = []
    for s_i, seed in enumerate(seeds):
        print(f"\n  Seed {s_i + 1}/{n_seeds} (seed={seed}):")
        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset = ResistanceDataset(dataset_path, features_dir)
        train_loader, val_loader, _ = _make_loaders(
            dataset, train_idx, val_idx, test_idx)
        model = build_model_from_cfg(cfg).to(device)
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        save_path = MODEL_DIR / f"stability_seed{seed}.pt"
        best_score, _ = _train_loop(model, train_loader, val_loader,
                                    focal_loss, device, save_path)
        model.load_state_dict(torch.load(save_path, map_location=device,
                                         weights_only=True))
        print(f"    val max(AUROC, BAcc) = {best_score:.3f}")

        # IG on first 20 test samples
        ig = _run_ptm_bdl_ig(model, dataset, test_idx.tolist()[:20])
        per_seed.append(ig)
        for protein, data in ig.items():
            if data["n_samples"] > 0:
                ph_top = int(np.argmax(data["phospho"]))
                ac_top = int(np.argmax(data["acetyl"])) if "acetyl" in data else -1
                print(f"    {protein} (n={data['n_samples']}): "
                      f"phospho top={ph_top}, acetyl top={ac_top}")

    # ── Aggregate across seeds ──────────────────────────────────────────
    all_proteins = set()
    for ig in per_seed:
        all_proteins.update(ig.keys())

    ptm_cols = None
    sec_cols = None
    try:
        tmp_ds = ResistanceDataset(dataset_path, features_dir)
        ptm_cols = tmp_ds._ptm_cols
        sec_cols = tmp_ds._secondary_cols
    except Exception:
        pass

    # ── Resolve per-protein site labels from config ──────────────────────
    # The dataset columns (ptm_cols / sec_cols) are shared across proteins
    # (e.g. "secondary_slot00"..."secondary_slot06"), but each protein has
    # its own biological sites defined in config.yaml → ptm → PROTEIN.
    def _get_protein_labels(protein_name, channel="phospho"):
        """Get per-protein site labels from config (resolves generic IDs)."""
        ptm_cfg = cfg.get("ptm", {})
        protein_cfg = ptm_cfg.get(protein_name, {})
        key = "phospho_sites" if channel == "phospho" else "acetyl_sites"
        return [site.get("residue", f"slot_{i}")
                for i, site in enumerate(protein_cfg.get(key, []))]

    aggregated = {}
    for protein in sorted(all_proteins):
        ph_arrs = [np.array(s[protein]["phospho"])
                   for s in per_seed if protein in s and s[protein]["n_samples"] > 0]
        ac_arrs = [np.array(s[protein]["acetyl"])
                   for s in per_seed
                   if protein in s and "acetyl" in s[protein] and s[protein]["n_samples"] > 0]

        # Get per-protein resolved labels (biological names, not slot IDs)
        ph_labels = _get_protein_labels(protein, "phospho")
        ac_labels = _get_protein_labels(protein, "acetyl")

        entry = {}
        if ph_arrs:
            mean_ph = np.mean(ph_arrs, axis=0)
            top_ph = int(np.argmax(mean_ph))
            entry["phospho_mean_importance"] = mean_ph.tolist()
            entry["phospho_top_slot"] = top_ph
            # Use per-protein labels if available, fall back to dataset columns
            if ph_labels and top_ph < len(ph_labels):
                entry["phospho_top_site"] = ph_labels[top_ph]
                entry["phospho_site_labels"] = ph_labels
            else:
                entry["phospho_top_site"] = (ptm_cols[top_ph] if ptm_cols and top_ph < len(ptm_cols)
                                             else f"slot_{top_ph}")
                if ptm_cols:
                    entry["phospho_site_labels"] = ptm_cols
        if ac_arrs:
            mean_ac = np.mean(ac_arrs, axis=0)
            top_ac = int(np.argmax(mean_ac))
            entry["acetyl_mean_importance"] = mean_ac.tolist()
            entry["acetyl_top_slot"] = top_ac
            # Use per-protein labels if available, fall back to dataset columns
            if ac_labels and top_ac < len(ac_labels):
                entry["acetyl_top_site"] = ac_labels[top_ac]
                entry["acetyl_site_labels"] = ac_labels
            else:
                entry["acetyl_top_site"] = (sec_cols[top_ac] if sec_cols and top_ac < len(sec_cols)
                                            else f"slot_{top_ac}")
                if sec_cols:
                    entry["acetyl_site_labels"] = sec_cols
        entry["n_seeds_contributing"] = len(ph_arrs)
        aggregated[protein] = entry

    # Cross-seed consistency
    top_sites_per_protein = {}
    for protein in sorted(all_proteins):
        ph_tops = []
        for s in per_seed:
            if protein in s and s[protein]["n_samples"] > 0:
                ph_tops.append(int(np.argmax(s[protein]["phospho"])))
        top_sites_per_protein[protein] = {"phospho": ph_tops}
        concordant = len(set(ph_tops)) == 1 if ph_tops else False
        aggregated[protein]["cross_seed_phospho_concordant"] = concordant
        print(f"  {protein}: phospho top across seeds = {ph_tops} → "
              f"{'✓ concordant' if concordant else '✗ discordant'}")

    out = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "per_protein": aggregated,
        "top_sites_per_seed": top_sites_per_protein,
    }
    out_path = RESULTS_DIR / "stability_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Saved: {out_path}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Randomised PTM control (inference-only permutation)
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_inference_shuffled_arm(arm_name, columns_to_shuffle,
                                    model, dataset, test_idx, device):
    """Shuffle PTM columns in test set, evaluate same model."""
    import copy
    print(f"\n  ── Arm: {arm_name} (inference-only) ──")

    df_shuffled = dataset.df.copy()
    np.random.seed(cfg["training"]["seed"])
    perm = np.random.permutation(len(df_shuffled))
    n_shuffled = 0
    for col in columns_to_shuffle:
        if col in df_shuffled.columns:
            df_shuffled[col] = df_shuffled[col].values[perm]
            n_shuffled += 1
    print(f"    Shuffled {n_shuffled} columns (inference-only, no retraining)")

    shuffled_dataset = copy.copy(dataset)
    shuffled_dataset.df = df_shuffled

    test_set = Subset(shuffled_dataset, test_idx)
    bs = cfg["model"]["batch_size"]
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             collate_fn=collate_fn)

    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    test_m = validate(model, test_loader, focal_loss, 1.0, 2.0, device)
    print(f"    Shuffled-test AUROC={test_m.get('auroc', 0):.3f}, "
          f"BAcc={test_m['balanced_acc']:.3f}, "
          f"RMSE={test_m.get('rmse', 0):.3f}")
    return test_m


def run_randomized_ptm_control(device):
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 3: Randomised PTM control — inference-only permutation")
    print("  Method: Permutation Feature Importance (Breiman 2001)")
    print("══════════════════════════════════════════════════════════════")

    train_idx, val_idx, test_idx = _load_split()
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / CASE_STUDY / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    # Discover PTM columns from dataset
    dataset = ResistanceDataset(dataset_path, features_dir)
    phospho_cols = dataset._ptm_cols + dataset._delta_ptm_cols
    acetyl_cols = dataset._secondary_cols + dataset._delta_secondary_cols
    print(f"  Phospho columns: {len(phospho_cols)}, Acetyl columns: {len(acetyl_cols)}")

    # Load trained full model
    full_model_path = MODEL_DIR / "ablation_full.pt"
    if not full_model_path.exists():
        full_model_path = MODEL_DIR / "best_model.pt"
    if not full_model_path.exists():
        print("  ✗ No trained model found — run Part 1 first.")
        return {}

    model = build_model_from_cfg(cfg).to(device)
    model.load_state_dict(torch.load(full_model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"  ✓ Loaded trained model: {full_model_path.name}")

    # Reference: evaluate on unshuffled test set
    test_set = Subset(dataset, test_idx)
    bs = cfg["model"]["batch_size"]
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             collate_fn=collate_fn)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    full_metrics_ref = validate(model, test_loader, focal_loss, 1.0, 2.0, device)
    print(f"\n  Reference (real PTM): AUROC={full_metrics_ref.get('auroc', 0):.3f}, "
          f"BAcc={full_metrics_ref['balanced_acc']:.3f}")

    # Run each shuffled arm
    arms = {}
    arms["phospho_shuffled"] = _run_one_inference_shuffled_arm(
        "phospho_shuffled", phospho_cols, model, dataset, test_idx, device)
    if acetyl_cols:
        arms["acetyl_shuffled"] = _run_one_inference_shuffled_arm(
            "acetyl_shuffled", acetyl_cols, model, dataset, test_idx, device)
        arms["both_shuffled"] = _run_one_inference_shuffled_arm(
            "both_shuffled", phospho_cols + acetyl_cols, model, dataset, test_idx, device)

    # Drop summary
    def _drops(arm_m):
        return {
            "drop_auroc": round(full_metrics_ref.get("auroc", 0)
                                - arm_m.get("auroc", 0), 4),
            "drop_bacc": round(full_metrics_ref.get("balanced_acc", 0)
                               - arm_m.get("balanced_acc", 0), 4),
            "drop_auprc_sensitive": round(full_metrics_ref.get("auprc_sensitive", 0)
                                          - arm_m.get("auprc_sensitive", 0), 4),
        }

    print("\n  ── Drop summary (positive ⇒ real PTM beats shuffled) ──")
    print(f"  {'Arm':<22s} | {'ΔAUROC':>7s} | {'ΔBAcc':>6s} | {'ΔAUPRCsens':>11s}")
    print("  " + "-" * 55)
    drops_summary = {}
    for arm, m in arms.items():
        d = _drops(m)
        drops_summary[arm] = d
        print(f"  {arm:<22s} | {d['drop_auroc']:+7.3f} | "
              f"{d['drop_bacc']:+6.3f} | {d['drop_auprc_sensitive']:+11.3f}")

    primary_arm = "both_shuffled" if "both_shuffled" in drops_summary else "phospho_shuffled"
    primary = drops_summary[primary_arm]
    primary_pass = (primary["drop_auroc"] >= 0.005
                    and primary["drop_auprc_sensitive"] >= 0.0)
    print(f"\n  PRIMARY ({primary_arm}) pass criterion "
          f"(ΔAUROC ≥ +0.005 AND ΔAUPRC-sens ≥ 0.0): "
          f"{'✓ PASS' if primary_pass else '✗ FAIL'}")

    out = {
        "method": "Inference-only Permutation Feature Importance (Breiman 2001)",
        "model_used": str(full_model_path.name),
        "reference_full_metrics": full_metrics_ref,
        "arms": {arm: {"shuffled_metrics": m, "drops": drops_summary[arm]}
                 for arm, m in arms.items()},
        "primary_arm": primary_arm,
        "primary_pass": bool(primary_pass),
    }
    out_path = RESULTS_DIR / "randomized_ptm_control.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Saved: {out_path}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HeLa/HDAC — PTM-BDL Ablation + Stability + Randomised Ctrl║")
    print("╚══════════════════════════════════════════════════════════════╝")
    device = _get_device()
    print(f"  Device: {device}")

    run_ablation_study(device)
    run_stability_analysis(device, n_seeds=3)
    run_randomized_ptm_control(device)
    print("\n✓ HeLa/HDAC ablation study complete!")
