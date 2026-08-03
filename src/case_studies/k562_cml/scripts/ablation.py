#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  K562/CML — PTM-BDL Ablation Study                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Three families of tests for the PTM-BDL architecture:                      ║
║                                                                              ║
║  PART 1 — FEATURE & ARCHITECTURE ABLATIONS                                  ║
║    no_ptm              — all PTM features zeroed (static baseline)           ║
║    no_drug             — drug embeddings zeroed                              ║
║    no_structure        — GearNet structural embeddings zeroed                ║
║    no_typed_attention  — PTM-BDL with MLP in place of typed self-attn       ║
║    full                — full PTM-BDL (phospho only for K562)               ║
║                                                                              ║
║  PART 2 — MULTI-SEED STABILITY (3 seeds × per-protein IG)                  ║
║    Per-protein phospho IG at the PTM-BDL token boundary.                    ║
║    Cross-protein check: ABL1 Y245/Y412 activation loop sites should        ║
║    rank consistently across seeds and proteins.                              ║
║    Ref: Hantschel, Genes Dev 2012 (PMID 22855830)                          ║
║                                                                              ║
║  PART 3 — RANDOMISED PTM CONTROL (inference-only permutation)               ║
║    Shuffle phospho columns at test time, same trained model.                ║
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
║    K562 (CML, BCR-ABL+) — 5 drugs: Dasatinib, Imatinib (TKIs) +           ║
║    Cytarabine, Paclitaxel, Methotrexat (chemo)                              ║
║    Ref: Shah et al., Science 2004 (PMID 15256107)                           ║
║    Ref: Badkul et al., DrugPTM-Bench 2024                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gc
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

CASE_STUDY = "k562_cml"
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
        description="All PTM features (phospho) zeroed — static baseline.",
        color="#d62728", use_typed_attention=True, data_mode="no_ptm",
    ),
    "baseline_only": dict(
        label="Model A1: Baseline PTM only",
        description="PTM levels (baseline state) active, all drug-induced deltas zeroed. "
                    "Tests prospective DRP scenario (no drug-PTM interaction data).",
        color="#bcbd22", use_typed_attention=True, data_mode="baseline_only",
    ),
    "delta_only": dict(
        label="Model A2: Delta PTM only",
        description="PTM levels set to WT (1.0), drug-induced deltas active. "
                    "Isolates purely dynamic pharmacodynamic signal.",
        color="#17becf", use_typed_attention=True, data_mode="delta_only",
    ),
    "no_drug": dict(
        label="Model B: No Drug",
        description="Drug embeddings (ChemBERTa) zeroed.",
        color="#ff7f0e", use_typed_attention=True, data_mode="no_drug",
    ),
    "no_structure": dict(
        label="Model C: No Structure",
        description="Structural embeddings (GearNet) zeroed.",
        color="#2ca02c", use_typed_attention=True, data_mode="no_structure",
    ),
    "no_typed_attention": dict(
        label="Model D: No Typed Attention",
        description="PTM-BDL typed self-attention replaced with MLP.",
        color="#e377c2", use_typed_attention=False, data_mode="full",
    ),
    "full": dict(
        label="Model E: Full PTM-BDL",
        description="All features + typed self-attention (production model).",
        color="#1f77b4", use_typed_attention=True, data_mode="full",
    ),
}

ABLATION_CONFIGS["measured_only"] = dict(
    label="Model M: Measured PTM only",
    description="PTM values kept only for directly measured samples (conf≥0.90). "
                "Propagated samples reset to WT baseline. Tests Q7: contribution "
                "of mutation-class propagation priors vs direct measurements.",
    color="#aec7e8", use_typed_attention=True, data_mode="measured_only",
)

ABLATION_ORDER = [
    "no_ptm", "baseline_only", "delta_only", "measured_only",
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

    # Modality-level marginal gains
    drug_marginal = (full_m.get("auroc", 0)
                     - results["no_drug"]["test_metrics"].get("auroc", 0))
    struct_marginal = (full_m.get("auroc", 0)
                       - results["no_structure"]["test_metrics"].get("auroc", 0))
    typed_attn_marginal = (full_m.get("auroc", 0)
                           - results["no_typed_attention"]["test_metrics"].get("auroc", 0))

    print(f"\n  Modality-level marginal AUROC gains (vs full):")
    print(f"    PTM marginal                : {gains['auroc']:+.3f} "
          f"(full vs no_ptm)")
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
        plt.suptitle("K562/CML — PTM-BDL Ablation Study", fontsize=13, fontweight="bold")
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
    Integrated Gradients on the PTM-BDL phospho input plane.

    K562 has phospho only (no secondary channel), so integrates along:
      - ptm_vector       (phospho baseline level, baseline=1.0 = WT)
      - delta_ptm_vector (drug-induced phospho change, baseline=0.0)

    Per-site importance = |grad_level × Δlevel| + |grad_delta × Δdelta|

    Returns dict with arrays per protein (ABL1, CRKL, STAT5A).
    """
    model.train()  # need grads
    dev = next(model.parameters()).device
    ptm_dim = dataset._ptm_dim
    baseline_phospho = torch.ones(ptm_dim, device=dev)
    baseline_dphospho = torch.zeros(ptm_dim, device=dev)

    sums = {}
    counts = {}
    protein_names = {v: k for k, v in dataset._protein_map.items()}

    for idx in indices:
        sample = dataset[int(idx)]
        actual_phospho = sample["ptm_vector"].to(dev)
        actual_dphospho = sample["delta_ptm_vector"].to(dev)
        tp = sample["target_protein"].view(1).long().to(dev)
        pid = int(tp.item())
        protein = protein_names.get(pid, f"protein_{pid}")

        if protein not in sums:
            sums[protein] = np.zeros(ptm_dim)
            counts[protein] = 0
        counts[protein] += 1

        seq_e = sample["seq_emb"].unsqueeze(0).to(dev)
        str_e = sample["struct_emb"].unsqueeze(0).to(dev)
        drg_e = sample["drug_emb"].unsqueeze(0).to(dev)
        drg_p = sample["drug_pooled"].unsqueeze(0).to(dev)

        grads_phospho = torch.zeros(ptm_dim, device=dev)
        grads_dphospho = torch.zeros(ptm_dim, device=dev)

        for step in range(n_steps + 1):
            a = step / n_steps
            iph = (baseline_phospho + a * (actual_phospho - baseline_phospho)
                   ).unsqueeze(0).requires_grad_(True)
            idph = (baseline_dphospho + a * (actual_dphospho - baseline_dphospho)
                    ).unsqueeze(0).requires_grad_(True)

            _, resist_pred = model(
                seq_embeddings=seq_e,
                struct_embeddings=str_e,
                drug_pooled=drg_p,
                drug_embeddings=drg_e,
                ptm_vector=iph,
                delta_ptm_vector=idph,
                target_protein=tp,
            )
            model.zero_grad()
            resist_pred.backward()
            if iph.grad is not None:
                grads_phospho += iph.grad.squeeze(0).detach()
            if idph.grad is not None:
                grads_dphospho += idph.grad.squeeze(0).detach()

        delta_ph = actual_phospho - baseline_phospho
        delta_dph = actual_dphospho - baseline_dphospho
        n_s = n_steps + 1

        attr_ph_level = np.abs(((grads_phospho / n_s) * delta_ph).cpu().numpy())
        attr_ph_delta = np.abs(((grads_dphospho / n_s) * delta_dph).cpu().numpy())
        attr_ph = attr_ph_level + attr_ph_delta

        sums[protein] += attr_ph

    model.eval()
    out = {}
    for protein in sums:
        n = max(counts[protein], 1)
        out[protein] = {
            "phospho": (sums[protein] / n).tolist(),
            "n_samples": counts[protein],
        }
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
                top_idx = int(np.argmax(data["phospho"]))
                print(f"    {protein} (n={data['n_samples']}): "
                      f"phospho top slot={top_idx}")

    # ── Aggregate across seeds ──────────────────────────────────────────
    all_proteins = set()
    for ig in per_seed:
        all_proteins.update(ig.keys())

    ptm_cols = None
    # Try to get PTM column names for reporting
    try:
        tmp_ds = ResistanceDataset(dataset_path, features_dir)
        ptm_cols = tmp_ds._ptm_cols
    except Exception:
        pass

    # ── Resolve per-protein site labels from config ──────────────────────
    # The dataset columns (ptm_cols) are shared across all proteins and
    # reflect ABL1's sites (the first protein alphabetically). But CRKL
    # has Y132/Y198/Y207/Y251, STAT5A has Y694/Y699/S725/S779 — these
    # must be read from config.yaml → ptm → PROTEIN → phospho_sites.
    def _get_protein_phospho_labels(protein_name):
        """Get per-protein phospho site labels from config."""
        ptm_cfg = cfg.get("ptm", {})
        protein_cfg = ptm_cfg.get(protein_name, {})
        return [site.get("residue", f"slot_{i}")
                for i, site in enumerate(protein_cfg.get("phospho_sites", []))]

    aggregated = {}
    for protein in sorted(all_proteins):
        arrs = [np.array(s[protein]["phospho"])
                for s in per_seed if protein in s and s[protein]["n_samples"] > 0]
        if arrs:
            mean_ig = np.mean(arrs, axis=0)
            top_idx = int(np.argmax(mean_ig))

            # Use per-protein labels from config (not shared ptm_cols)
            protein_labels = _get_protein_phospho_labels(protein)
            if protein_labels and top_idx < len(protein_labels):
                top_label = protein_labels[top_idx]
            else:
                top_label = ptm_cols[top_idx] if ptm_cols and top_idx < len(ptm_cols) else f"slot_{top_idx}"

            aggregated[protein] = {
                "phospho_mean_importance": mean_ig.tolist(),
                "phospho_top_slot": top_idx,
                "phospho_top_site": top_label,
                "n_seeds_contributing": len(arrs),
            }
            # Use per-protein labels if available, fall back to shared ptm_cols
            if protein_labels:
                aggregated[protein]["phospho_site_labels"] = protein_labels
            elif ptm_cols:
                aggregated[protein]["phospho_site_labels"] = ptm_cols

    # Cross-seed consistency: do seeds agree on top site?
    top_sites_per_protein = {}
    for protein in sorted(all_proteins):
        tops = []
        for s in per_seed:
            if protein in s and s[protein]["n_samples"] > 0:
                tops.append(int(np.argmax(s[protein]["phospho"])))
        top_sites_per_protein[protein] = tops
        concordant = len(set(tops)) == 1 if tops else False
        aggregated.setdefault(protein, {})["cross_seed_top_concordant"] = concordant
        print(f"  {protein}: top sites across seeds = {tops} → "
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
# ──────────────────────────────────────────────────────────────────────────────
# Method: Load the TRAINED "full" model → shuffle PTM columns in TEST SET
# only → run inference → compare metrics.  No retraining needed.
#
# Scientific basis: Permutation Feature Importance (Breiman 2001,
# Fisher, Rudin & Dominici 2019, JMLR 20:1-81).
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
    print(f"  Phospho columns to shuffle: {len(phospho_cols)}")

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

    # Run shuffled arm
    arms = {}
    arms["phospho_shuffled"] = _run_one_inference_shuffled_arm(
        "phospho_shuffled", phospho_cols, model, dataset, test_idx, device)

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
    drops_summary = {}
    for arm, m in arms.items():
        d = _drops(m)
        drops_summary[arm] = d
        print(f"  {arm}: ΔAUROC={d['drop_auroc']:+.3f}, "
              f"ΔBAcc={d['drop_bacc']:+.3f}, "
              f"ΔAUPRC-sens={d['drop_auprc_sensitive']:+.3f}")

    primary = drops_summary["phospho_shuffled"]
    primary_pass = (primary["drop_auroc"] >= 0.005
                    and primary["drop_auprc_sensitive"] >= 0.0)
    print(f"\n  PRIMARY pass criterion (ΔAUROC ≥ +0.005 AND ΔAUPRC-sens ≥ 0.0): "
          f"{'✓ PASS' if primary_pass else '✗ FAIL'}")

    out = {
        "method": "Inference-only Permutation Feature Importance (Breiman 2001)",
        "model_used": str(full_model_path.name),
        "reference_full_metrics": full_metrics_ref,
        "arms": {arm: {"shuffled_metrics": m, "drops": drops_summary[arm]}
                 for arm, m in arms.items()},
        "primary_arm": "phospho_shuffled",
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
    print("║  K562/CML — PTM-BDL Ablation + Stability + Randomised Ctrl ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    device = _get_device()
    print(f"  Device: {device}")

    run_ablation_study(device)
    run_stability_analysis(device, n_seeds=3)
    run_randomized_ptm_control(device)
    print("\n✓ K562/CML ablation study complete!")
