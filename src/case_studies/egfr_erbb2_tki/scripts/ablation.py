#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EGFR/ERBB2 TKI — PTM-BDL Ablation Study                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Three families of tests for the PTM-BDL architecture                       ║
║  (PTM_Biological_Dynamics_Layer.md §9):                                      ║
║                                                                              ║
║  PART 1 — FEATURE & ARCHITECTURE ABLATIONS                                  ║
║    Symmetric across the two PTM channels (phospho ⊕ glyco):                 ║
║      no_ptm              — all PTM features zeroed (baseline)               ║
║      no_glyco        — primary only (phospho); zeros glyco slot range   ║
║      glyco_only      — secondary only (glyco); zeros phospho slot range ║
║      no_typed_attention  — PTM-BDL with MLP in place of typed self-attn     ║
║      full                — full PTM-BDL (= proposal Model C, phospho+glyco) ║
║    Per-PTM-type ablation uses the registry's zero_slot_range to zero        ║
║    specific PTM type slots in the flat ptm_vector.                          ║
║                                                                              ║
║  PART 2 — MULTI-SEED STABILITY (3 seeds × per-protein × per-mod-type IG)    ║
║    Per-mod-type IG buckets at the PTM-BDL token boundary:                   ║
║       phospho (slots 0..11) and glyco_N (slots 12..23).                     ║
║    Cross-receptor homology check at the GRB2-docking slot (index 7):        ║
║       EGFR Y1092 ≡ ERBB2 Y1221, both phospho_Y.                              ║
║                                                                              ║
║  PART 3 — RANDOMISED PTM CONTROL                                            ║
║    Three independent shuffles writing to one report:                        ║
║       (a) phospho shuffled, glyco unchanged                                 ║
║       (b) glyco shuffled, phospho unchanged                                 ║
║       (c) both shuffled (the legacy "all PTM shuffled" control)             ║
║    PASS = real PTM beats shuffled on AUROC and BAcc by ≥ +0.005 / +0.02.    ║
║                                                                              ║
║  All sub-tests reuse the SAME train/val/test split saved by step11.         ║
║  All models are built with `build_model_from_cfg` (single source of truth). ║
║                                                                              ║
║  OUTPUTS:                                                                    ║
║    results/ablation_study.json            — Part 1 metrics + votes          ║
║    results/stability_analysis.json        — Part 2 IG buckets + homology    ║
║    results/randomized_ptm_control.json    — Part 3 phospho/glyco/both       ║
║    results/figures/ablation_comparison.png                                   ║
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

# ── Import from tool packages ──────────────────────────────────────────
from src.ptm_bdl.data.dataset import ResistanceDataset
from src.ptm_bdl.data.collate import collate_fn
from src.ptm_bdl.training.loss import FocalLoss
from src.ptm_bdl.training.trainer import train_epoch, validate
from src.ptm_bdl.training.factory import build_model_from_cfg

# ── Import from case study biology ──────────────────────────────────────────
from src.case_studies.egfr_erbb2_tki.biology import (
    PHOSPHO_LABELS_EGFR, PHOSPHO_LABELS_ERBB2,
    GLYCO_LABELS_EGFR, GLYCO_LABELS_ERBB2,
    GRB2_PHOSPHO_INDEX, EGFR_N528_INDEX, ERBB2_N530_INDEX,
)

# Protein ID constants from registry
PROTEIN_ID_EGFR = 0
PROTEIN_ID_ERBB2 = 1

from src.ptm_bdl.config import load_config

CASE_STUDY = "egfr_erbb2_tki"
cfg = load_config(case_study=CASE_STUDY)

MODEL_DIR = PROJECT_ROOT / cfg["paths"]["models"] / CASE_STUDY
RESULTS_DIR = PROJECT_ROOT / cfg["paths"]["results"] / CASE_STUDY
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Site labels, homology indices are imported from case study biology above


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
            "Run step11 first to create the train/val/test split."
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
    """Standard cosine-LR Stage-1 loop with early stopping on max(AUROC, BAcc)."""
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
        description="All PTM features (phospho + glyco + L2) zeroed.",
        color="#d62728", use_typed_attention=True, data_mode="no_ptm",
    ),
    "baseline_only": dict(
        label="Model B0: Baseline PTM only",
        description="PTM levels (baseline state) active, all drug-induced deltas zeroed. "
                    "Tests prospective DRP scenario (no drug-PTM interaction data).",
        color="#ff7f0e", use_typed_attention=True, data_mode="baseline_only",
    ),
    "delta_only": dict(
        label="Model B1: Delta PTM only",
        description="PTM levels set to WT (1.0), drug-induced deltas active. "
                    "Isolates purely dynamic pharmacodynamic signal.",
        color="#2ca02c", use_typed_attention=True, data_mode="delta_only",
    ),
    "no_glyco": dict(
        label="Model E: No secondary (glyco)",
        description="Primary (phospho) channel active, secondary (glyco) channel zeroed "
                    "via zero_slot_range on the flat ptm_vector.",
        color="#9467bd", use_typed_attention=True, data_mode="full",
        zero_slot_range=(12, 24),  # zero glyco slots in flat vector
    ),
    "glyco_only": dict(
        label="Model F: Secondary only (glyco)",
        description="Secondary (glyco) channel active, primary (phospho) channel zeroed "
                    "via zero_slot_range on the flat ptm_vector.",
        color="#8c564b", use_typed_attention=True, data_mode="full",
        zero_slot_range=(0, 12),  # zero phospho slots in flat vector
    ),
    "no_typed_attention": dict(
        label="Model G: No typed attention",
        description="PTM-BDL typed self-attention replaced with MLP "
                    "(input richness same, inter-token dependencies removed).",
        color="#e377c2", use_typed_attention=False, data_mode="full",
    ),
    "full": dict(
        label="Model D: Full PTM-BDL",
        description="All features + typed self-attention (production model).",
        color="#1f77b4", use_typed_attention=True, data_mode="full",
    ),
}

# Order: no_ptm → baseline_only → delta_only decomposes the PTM signal
# into static vs dynamic contributions (addresses reviewer Q1/Q3).
ABLATION_CONFIGS["measured_only"] = dict(
    label="Model M: Measured PTM only",
    description="PTM values kept only for directly measured samples (conf≥0.90). "
                "Propagated samples reset to WT baseline. Tests Q7: contribution "
                "of mutation-class propagation priors vs direct measurements.",
    color="#17becf", use_typed_attention=True, data_mode="measured_only",
)

ABLATION_ORDER = [
    "no_ptm", "baseline_only", "delta_only", "measured_only",
    "no_glyco", "glyco_only", "no_typed_attention", "full",
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

    zero_slot_range = spec.get("zero_slot_range", None)
    dataset = ResistanceDataset(dataset_path, features_dir,
                                ablation_mode=spec["data_mode"],
                                zero_slot_range=zero_slot_range)
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
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    results = {}
    for mode in ABLATION_ORDER:
        results[mode] = train_ablation_model(
            mode, dataset_path, features_dir,
            train_idx, val_idx, test_idx, device,
        )

    # ── Comparison table ────────────────────────────────────────────────
    print("\n  " + "=" * 92)
    print(f"  {'Model':<28s} | {'AUROC':>6s} | {'PR-S':>5s} | {'BAcc':>5s} | "
          f"{'F1mac':>5s} | {'RMSE':>5s} | {'R':>5s} | {'Δ AUROC vs no_ptm':>17s}")
    print("  " + "-" * 92)
    base_auroc = results["no_ptm"]["test_metrics"].get("auroc", 0)
    for mode in ABLATION_ORDER:
        t = results[mode]["test_metrics"]
        d = t.get("auroc", 0) - base_auroc
        print(f"  {results[mode]['label']:<28s} | {t.get('auroc', 0):6.3f} | "
              f"{t.get('auprc_sensitive', 0):5.3f} | {t['balanced_acc']:5.3f} | "
              f"{t.get('f1_macro', 0):5.3f} | {t.get('rmse', 0):5.3f} | "
              f"{t.get('pearson_r', 0):5.3f} | {d:+17.3f}")
    print("  " + "=" * 92)

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

    # Channel-level gains
    secondary_marginal = (full_m.get("auroc", 0)
                          - results["no_glyco"]["test_metrics"].get("auroc", 0))
    primary_marginal = (full_m.get("auroc", 0)
                        - results["glyco_only"]["test_metrics"].get("auroc", 0))
    typed_attn_marginal = (full_m.get("auroc", 0)
                           - results["no_typed_attention"]["test_metrics"].get("auroc", 0))

    print(f"\n  Channel-level marginal AUROC gains (vs full):")
    print(f"    Primary (phospho) marginal   : {primary_marginal:+.3f} "
          f"(full vs secondary-only)")
    print(f"    Secondary (glyco) marginal   : {secondary_marginal:+.3f} "
          f"(full vs primary-only)")
    print(f"    Typed-attention marginal     : {typed_attn_marginal:+.3f} "
          f"(typed-attn vs MLP)")
    print(f"\n  PTM-BDL vs No-PTM votes      : {votes_help}/{votes_total} "
          f"metrics positive")

    summary = {
        "ptm_gain_auroc": round(gains["auroc"], 4),
        "ptm_gain_auprc_sensitive": round(gains["auprc_sensitive"], 4),
        "ptm_gain_bacc": round(gains["bacc"], 4),
        "ptm_gain_f1_macro": round(gains["f1_macro"], 4),
        "primary_marginal_auroc": round(primary_marginal, 4),
        "secondary_marginal_auroc": round(secondary_marginal, 4),
        "typed_attention_marginal_auroc": round(typed_attn_marginal, 4),
        "votes_ptm_helps": votes_help,
        "votes_total": votes_total,
        "votes_ptm_bdl_helps": votes_help,  # mirror key required by §3 pass-criteria
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

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [results[m]["label"].split(":")[0] for m in ABLATION_ORDER]
        colors = [ABLATION_CONFIGS[m]["color"] for m in ABLATION_ORDER]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, key, title in [
            (axes[0], "auroc", "AUROC (test)"),
            (axes[1], "balanced_acc", "Balanced Accuracy (test)"),
            (axes[2], "rmse", "RMSE on ln(IC50) (test)"),
        ]:
            vals = [results[m]["test_metrics"].get(key, 0) for m in ABLATION_ORDER]
            ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
        plt.suptitle("PTM-BDL Ablation Study", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure saved: {FIGURES_DIR / 'ablation_comparison.png'}")
    except Exception as e:
        print(f"  ⚠ Could not generate figure: {e}")

    return save


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Multi-seed stability + per-mod-type IG
# ══════════════════════════════════════════════════════════════════════════════

def _run_ptm_bdl_ig(model, dataset, indices, df, n_steps: int = 20):
    """
    Integrated Gradients on the PTM-BDL flat 24-token input plane.

    Uses the flat ptm_vector (24 tokens: 12 phospho + 12 glyco) and
    delta_ptm_vector (24 tokens).  Integrates along both level and delta
    channels simultaneously.

    Per-site importance = |grad_level × Δlevel| + |grad_delta × Δdelta|

    After integration, slices results into phospho (0:12) and glyco (12:24)
    for per-protein, per-mod-type reporting.

    Returns dict with arrays per mod-type bucket, per protein.
    """
    model.train()  # need grads
    n_tokens = 24  # 12 phospho + 12 glyco
    baseline_level = torch.ones(n_tokens)   # WT = no modulation
    baseline_delta = torch.zeros(n_tokens)  # no drug effect

    sums = {
        "EGFR_phospho": np.zeros(12), "EGFR_glyco": np.zeros(12),
        "ERBB2_phospho": np.zeros(12), "ERBB2_glyco": np.zeros(12),
    }
    counts = {"EGFR": 0, "ERBB2": 0}

    for idx in indices:
        sample = dataset[int(idx)]
        actual_level = sample["ptm_vector"]        # (24,) flat
        actual_delta = sample["delta_ptm_vector"]  # (24,) flat
        tp = sample["target_protein"].view(1).long()
        protein = "ERBB2" if tp.item() == PROTEIN_ID_ERBB2 else "EGFR"
        counts[protein] += 1

        seq_e = sample["seq_emb"].unsqueeze(0)
        str_e = sample["struct_emb"].unsqueeze(0)
        drg_e = sample["drug_emb"].unsqueeze(0)
        drg_p = sample["drug_pooled"].unsqueeze(0)

        # Gradient accumulators for level and delta channels
        grads_level = torch.zeros(n_tokens)
        grads_delta = torch.zeros(n_tokens)

        for step in range(n_steps + 1):
            a = step / n_steps
            interp_level = (baseline_level + a * (actual_level - baseline_level)
                           ).unsqueeze(0).requires_grad_(True)
            interp_delta = (baseline_delta + a * (actual_delta - baseline_delta)
                           ).unsqueeze(0).requires_grad_(True)

            _, resist_pred = model(
                seq_embeddings=seq_e,
                struct_embeddings=str_e,
                drug_pooled=drg_p,
                drug_embeddings=drg_e,
                ptm_vector=interp_level,
                delta_ptm_vector=interp_delta,
                target_protein=tp,
            )
            model.zero_grad()
            resist_pred.backward()
            if interp_level.grad is not None:
                grads_level += interp_level.grad.squeeze(0).detach()
            if interp_delta.grad is not None:
                grads_delta += interp_delta.grad.squeeze(0).detach()

        # IG formula: |avg_grad_level × Δlevel| + |avg_grad_delta × Δdelta|
        d_level = actual_level - baseline_level
        d_delta = actual_delta - baseline_delta
        n_s = n_steps + 1
        attr = (np.abs(((grads_level / n_s) * d_level).numpy())
                + np.abs(((grads_delta / n_s) * d_delta).numpy()))

        # Slice into phospho (0:12) and glyco (12:24)
        sums[f"{protein}_phospho"] += attr[:12]
        sums[f"{protein}_glyco"] += attr[12:24]

    model.eval()
    out = {}
    for protein in ["EGFR", "ERBB2"]:
        n = max(counts[protein], 1)
        out[protein] = {
            "phospho": sums[f"{protein}_phospho"] / n,
            "glyco": sums[f"{protein}_glyco"] / n,
            "n_samples": counts[protein],
        }
    return out


def run_stability_analysis(device, n_seeds: int = 3):
    print("\n══════════════════════════════════════════════════════════════")
    print(f"PART 2: Multi-Seed Stability ({n_seeds} seeds) + per-mod-type IG")
    print("══════════════════════════════════════════════════════════════")
    train_idx, val_idx, test_idx = _load_split()
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]
    seeds = [42, 123, 456][:n_seeds]

    # Per-seed IG accumulators
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
        ig = _run_ptm_bdl_ig(model, dataset, test_idx.tolist()[:20], dataset.df)
        per_seed.append(ig)
        for protein in ["EGFR", "ERBB2"]:
            if ig[protein]["n_samples"] > 0:
                ph_top = int(np.argmax(ig[protein]["phospho"]))
                gl_top = int(np.argmax(ig[protein]["glyco"]))
                labels_ph = (PHOSPHO_LABELS_EGFR if protein == "EGFR"
                             else PHOSPHO_LABELS_ERBB2)
                labels_gl = (GLYCO_LABELS_EGFR if protein == "EGFR"
                             else GLYCO_LABELS_ERBB2)
                print(f"    {protein} (n={ig[protein]['n_samples']}): "
                      f"phospho top={labels_ph[ph_top]}, "
                      f"glyco top={labels_gl[gl_top]}")

    # ── Aggregate across seeds + IG rank stability (Q5) ─────────────────
    from src.ptm_bdl.evaluation.statistical import ig_rank_stability

    def _mean_axis(seed_list, protein, mod):
        arr = [s[protein][mod] for s in seed_list if s[protein]["n_samples"] > 0]
        return np.mean(arr, axis=0) if arr else np.zeros(12)

    def _per_seed_arrays(seed_list, protein, mod):
        return [s[protein][mod] for s in seed_list if s[protein]["n_samples"] > 0]

    egfr_phospho_mean = _mean_axis(per_seed, "EGFR", "phospho")
    egfr_glyco_mean = _mean_axis(per_seed, "EGFR", "glyco")
    erbb2_phospho_mean = _mean_axis(per_seed, "ERBB2", "phospho")
    erbb2_glyco_mean = _mean_axis(per_seed, "ERBB2", "glyco")

    # Compute IG rank stability (Spearman ρ across seeds — Reviewer Q5)
    stability_metrics = {}
    for protein, mod, labels in [
        ("EGFR", "phospho", PHOSPHO_LABELS_EGFR),
        ("EGFR", "glyco", GLYCO_LABELS_EGFR),
        ("ERBB2", "phospho", PHOSPHO_LABELS_ERBB2),
        ("ERBB2", "glyco", GLYCO_LABELS_ERBB2),
    ]:
        arrays = _per_seed_arrays(per_seed, protein, mod)
        if len(arrays) >= 2:
            stab = ig_rank_stability(arrays, site_labels=labels)
            stability_metrics[f"{protein}_{mod}"] = stab
            print(f"    {protein} {mod}: Spearman ρ (mean)={stab['mean_spearman_rho']:.3f}, "
                  f"top1_consistent={stab['top1_consistent']}, "
                  f"top3_Jaccard={stab['top3_jaccard_overlap']:.3f}")

    egfr_phospho_top_idx = int(np.argmax(egfr_phospho_mean))
    erbb2_phospho_top_idx = int(np.argmax(erbb2_phospho_mean))
    egfr_glyco_top_idx = int(np.argmax(egfr_glyco_mean))
    # ERBB2 glyco: enforce real-slot search (mask pads 7-11)
    erbb2_glyco_real = erbb2_glyco_mean.copy()
    erbb2_glyco_real[7:] = -np.inf
    erbb2_glyco_top_idx = int(np.argmax(erbb2_glyco_real)) if np.isfinite(
        erbb2_glyco_real.max()) else int(np.argmax(erbb2_glyco_mean))

    homology_phospho_concordant = (
            egfr_phospho_top_idx == GRB2_PHOSPHO_INDEX
            and erbb2_phospho_top_idx == GRB2_PHOSPHO_INDEX
    )
    homology_glyco_concordant = (
            egfr_glyco_top_idx == EGFR_N528_INDEX
            and erbb2_glyco_top_idx == ERBB2_N530_INDEX
    )

    print("\n  Cross-receptor homology checks:")
    print(f"    Phospho (Y1068 ≡ Y1221, slot {GRB2_PHOSPHO_INDEX}): "
          f"EGFR top idx={egfr_phospho_top_idx}, "
          f"ERBB2 top idx={erbb2_phospho_top_idx} → "
          f"{'✓ concordant' if homology_phospho_concordant else '✗ discordant'}")
    print(f"    Glyco (EGFR-N528 slot {EGFR_N528_INDEX} ↔ ERBB2-N530 slot "
          f"{ERBB2_N530_INDEX}): "
          f"EGFR top idx={egfr_glyco_top_idx}, "
          f"ERBB2 top idx={erbb2_glyco_top_idx} → "
          f"{'✓ concordant' if homology_glyco_concordant else '✗ discordant'}")

    out = {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "egfr": {
            "phospho_sites": PHOSPHO_LABELS_EGFR,
            "glyco_sites": GLYCO_LABELS_EGFR,
            "phospho_mean_importance": egfr_phospho_mean.tolist(),
            "glyco_mean_importance": egfr_glyco_mean.tolist(),
            "phospho_top_site": PHOSPHO_LABELS_EGFR[egfr_phospho_top_idx],
            "glyco_top_site": GLYCO_LABELS_EGFR[egfr_glyco_top_idx],
            # Per-seed arrays for Q5 rank stability analysis
            "phospho_per_seed": [a.tolist() for a in _per_seed_arrays(per_seed, "EGFR", "phospho")],
            "glyco_per_seed": [a.tolist() for a in _per_seed_arrays(per_seed, "EGFR", "glyco")],
        },
        "erbb2": {
            "phospho_sites": PHOSPHO_LABELS_ERBB2,
            "glyco_sites": GLYCO_LABELS_ERBB2,
            "phospho_mean_importance": erbb2_phospho_mean.tolist(),
            "glyco_mean_importance": erbb2_glyco_mean.tolist(),
            "phospho_top_site": PHOSPHO_LABELS_ERBB2[erbb2_phospho_top_idx],
            "glyco_top_site": GLYCO_LABELS_ERBB2[erbb2_glyco_top_idx],
            # Per-seed arrays for Q5 rank stability analysis
            "phospho_per_seed": [a.tolist() for a in _per_seed_arrays(per_seed, "ERBB2", "phospho")],
            "glyco_per_seed": [a.tolist() for a in _per_seed_arrays(per_seed, "ERBB2", "glyco")],
        },
        "homology_phospho_concordant": bool(homology_phospho_concordant),
        "homology_glyco_concordant": bool(homology_glyco_concordant),
        # Q5: IG rank stability metrics (Spearman ρ, top-k Jaccard across seeds)
        "ig_rank_stability": stability_metrics,
    }
    out_path = RESULTS_DIR / "stability_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Saved: {out_path}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Randomised PTM control (inference-only permutation)
# ──────────────────────────────────────────────────────────────────────────────
# REVISED 2026-07-03: Uses INFERENCE-ONLY permutation (Permutation Feature
# Importance) instead of retrain-from-scratch.
#
# Method: Load the TRAINED "full" model → shuffle PTM columns in TEST SET
# only → run inference → compare metrics.  No retraining needed.
#
# Scientific basis: Permutation Feature Importance (Breiman 2001, Machine
# Learning 45:5-32; Fisher, Rudin & Dominici 2019, JMLR 20:1-81).  This is
# the standard method used by SHAP, SAGE, and SAGE-net (Nat Methods 2026).
#
# Advantages over retrain:
#   1. Eliminates training stochasticity (same model, same weights)
#   2. Directly tests "has the model learned to USE PTM features?"
#   3. Complements Part 1 (ablation) which tests "does PTM carry learnable signal?"
#   4. Saves ~9 days of compute (no retraining per arm)
#
# Three arms: phospho_shuffled / glyco_shuffled / both_shuffled
# PASS = real PTM beats shuffled on AUROC ≥ +0.005 AND AUPRC-sens ≥ 0.0
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_inference_shuffled_arm(arm_name: str, columns_to_shuffle,
                                    model, dataset, test_idx, device):
    """
    Inference-only permutation test: shuffle PTM columns in the TEST SET,
    then evaluate the SAME trained model.  No retraining.

    Ref: Breiman (2001) Random Forests, §10 Variable Importance;
         Fisher et al. (2019) JMLR 20:1-81, Model Reliance.
    """
    import copy
    print(f"\n  ── Arm: {arm_name} (inference-only) ──")

    # Deep copy the dataset's DataFrame to avoid mutating the original
    df_shuffled = dataset.df.copy()

    # Shuffle within test indices only — preserves train/val integrity
    # But we use a GLOBAL permutation of the PTM columns to break the
    # mutation→PTM correspondence (same as the original retrain approach)
    np.random.seed(cfg["training"]["seed"])
    perm = np.random.permutation(len(df_shuffled))
    n_shuffled = 0
    for col in columns_to_shuffle:
        if col in df_shuffled.columns:
            df_shuffled[col] = df_shuffled[col].values[perm]
            n_shuffled += 1
    print(f"    Shuffled {n_shuffled} columns (inference-only, no retraining)")

    # Create a temporary dataset with shuffled PTM
    shuffled_dataset = copy.copy(dataset)
    shuffled_dataset.df = df_shuffled

    # Build test loader from shuffled dataset
    test_set = Subset(shuffled_dataset, test_idx)
    bs = cfg["model"]["batch_size"]
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             collate_fn=collate_fn)

    # Evaluate the SAME model (no retraining)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    test_m = validate(model, test_loader, focal_loss, 1.0, 2.0, device)
    print(f"    Shuffled-test AUROC={test_m.get('auroc', 0):.3f}, "
          f"BAcc={test_m['balanced_acc']:.3f}, "
          f"RMSE={test_m.get('rmse', 0):.3f}, "
          f"AUPRC_sens={test_m.get('auprc_sensitive', 0):.3f}")
    return test_m


def run_randomized_ptm_control(device):
    print("\n══════════════════════════════════════════════════════════════")
    print("PART 3: Randomised PTM control — inference-only permutation")
    print("  Method: Permutation Feature Importance (Breiman 2001,")
    print("          Fisher et al. JMLR 2019)")
    print("  Same trained model, shuffled PTM at test time")
    print("══════════════════════════════════════════════════════════════")

    train_idx, val_idx, test_idx = _load_split()
    dataset_path = (PROJECT_ROOT / cfg["paths"]["processed_data"]
                    / "multimodal_dataset.csv")
    features_dir = PROJECT_ROOT / cfg["paths"]["features"]

    # Column lists (only model-consumed columns)
    phospho_cols = ([f"ptm_{s}" for s in [
        "Y869", "S991", "Y998", "Y1016", "S1039", "T1041",
        "Y1069", "Y1092", "Y1110", "Y1125", "Y1172", "Y1197"]]
                    + [f"delta_ptm_{s}" for s in [
                "Y869", "S991", "Y998", "Y1016", "S1039", "T1041",
                "Y1069", "Y1092", "Y1110", "Y1125", "Y1172", "Y1197"]])
    glyco_cols = ([f"glyco_slot{i:02d}" for i in range(12)]
                  + [f"delta_glyco_slot{i:02d}" for i in range(12)])

    # ── Load the trained full model ─────────────────────────────────────
    full_model_path = MODEL_DIR / "ablation_full.pt"
    if not full_model_path.exists():
        full_model_path = MODEL_DIR / "best_model.pt"
    if not full_model_path.exists():
        print("  ✗ No trained model found — cannot run inference-only control.")
        print("    Run Part 1 (ablation) first to produce ablation_full.pt.")
        return {}

    model = build_model_from_cfg(cfg).to(device)
    model.load_state_dict(torch.load(full_model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"  ✓ Loaded trained model: {full_model_path.name}")

    # ── Reference: evaluate full model on UNSHUFFLED test set ──────────
    dataset = ResistanceDataset(dataset_path, features_dir)
    test_set = Subset(dataset, test_idx)
    bs = cfg["model"]["batch_size"]
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             collate_fn=collate_fn)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    full_metrics_ref = validate(model, test_loader, focal_loss, 1.0, 2.0, device)
    print(f"\n  Reference (real PTM): AUROC={full_metrics_ref.get('auroc', 0):.3f}, "
          f"BAcc={full_metrics_ref['balanced_acc']:.3f}, "
          f"AUPRC_sens={full_metrics_ref.get('auprc_sensitive', 0):.3f}")

    # ── Run each shuffled arm (inference-only) ─────────────────────────
    arms = {}
    arms["phospho_shuffled"] = _run_one_inference_shuffled_arm(
        "phospho_shuffled", phospho_cols, model, dataset, test_idx, device)
    arms["glyco_shuffled"] = _run_one_inference_shuffled_arm(
        "glyco_shuffled", glyco_cols, model, dataset, test_idx, device)
    arms["both_shuffled"] = _run_one_inference_shuffled_arm(
        "both_shuffled", phospho_cols + glyco_cols, model, dataset, test_idx, device)

    # ── Drop summary (real PTM minus shuffled — positive = real PTM wins) ──
    def _drops(arm_m):
        return {
            "drop_auroc": round(full_metrics_ref.get("auroc", 0)
                                - arm_m.get("auroc", 0), 4),
            "drop_bacc": round(full_metrics_ref.get("balanced_acc", 0)
                               - arm_m.get("balanced_acc", 0), 4),
            "drop_auprc_sensitive": round(full_metrics_ref.get("auprc_sensitive", 0)
                                          - arm_m.get("auprc_sensitive", 0), 4),
            "drop_rmse": round(arm_m.get("rmse", 0)
                               - full_metrics_ref.get("rmse", 0), 4),
        }

    print("\n  ── Drop summary (positive ⇒ real PTM beats shuffled) ──")
    print(f"  {'Arm':<22s} | {'ΔAUROC':>7s} | {'ΔBAcc':>6s} | "
          f"{'ΔAUPRCsens':>11s} | {'ΔRMSE':>6s}")
    print("  " + "-" * 64)
    drops_summary = {}
    for arm, m in arms.items():
        d = _drops(m)
        drops_summary[arm] = d
        print(f"  {arm:<22s} | {d['drop_auroc']:+7.3f} | "
              f"{d['drop_bacc']:+6.3f} | "
              f"{d['drop_auprc_sensitive']:+11.3f} | "
              f"{d['drop_rmse']:+6.3f}")

    # Pass criterion (revised 2026-07-03):
    #   AUROC + AUPRC-sensitive (threshold-independent, appropriate for 92:8 imbalance).
    #   BAcc removed: with 12 sensitive test samples, all models produce identical
    #   confusion matrices → BAcc cannot differentiate (see BENCHMARKING_PLAN.md §2).
    #   Ref: Saito & Rehmsmeier, PLOS ONE 2015; Fisher et al., JMLR 2019.
    primary = drops_summary["both_shuffled"]
    primary_pass = (primary["drop_auroc"] >= 0.005
                    and primary["drop_auprc_sensitive"] >= 0.0)
    print(f"\n  PRIMARY (both_shuffled) pass criterion "
          f"(ΔAUROC ≥ +0.005 AND ΔAUPRC-sens ≥ 0.0): "
          f"{'✓ PASS' if primary_pass else '✗ FAIL'}")

    out = {
        "method": "Inference-only Permutation Feature Importance (Breiman 2001, Fisher et al. 2019)",
        "note": "Same trained model, PTM columns shuffled at test time only. "
                "No retraining — directly tests model reliance on PTM features.",
        "model_used": str(full_model_path.name),
        "reference_full_metrics": full_metrics_ref,
        "arms": {arm: {"shuffled_metrics": m, "drops": drops_summary[arm]}
                 for arm, m in arms.items()},
        "primary_arm": "both_shuffled",
        "primary_pass": bool(primary_pass),
    }
    out_path = RESULTS_DIR / "randomized_ptm_control.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ Saved: {out_path}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Run all analyses
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  STEP 11b: PTM-BDL Ablation + Stability + Randomised Control║")
    print("╚══════════════════════════════════════════════════════════════╝")
    device = _get_device()
    print(f"  Device: {device}")

    run_ablation_study(device)
    run_stability_analysis(device, n_seeds=3)
    run_randomized_ptm_control(device)
    print("\n✓ Step 11b complete!")
