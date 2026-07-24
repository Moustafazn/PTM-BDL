"""
Tests for training utilities, loss functions, metrics, and config.

Covers: FocalLoss, compute_metrics, build_model_from_cfg,
config loading, and one-step training sanity check.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ptm_bdl.config import load_config
from src.ptm_bdl.training.loss import FocalLoss
from src.ptm_bdl.training.metrics import compute_metrics as compute_metrics_fn
from src.ptm_bdl.training.factory import build_model_from_cfg as build_model_from_cfg_fn
from src.ptm_bdl.model.encoder import PTMBDLEncoder
from src.ptm_bdl.model.ablation import PTMBDLMlpAblation


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:

    @staticmethod
    @pytest.fixture
    def cfg():
        return load_config(case_study="egfr_erbb2_tki")

    @staticmethod
    def test_config_loads(cfg):
        assert "model" in cfg
        assert "ptm" in cfg
        assert "training" in cfg
        assert "drugs" in cfg

    @staticmethod
    def test_model_hyperparams(cfg):
        assert cfg["model"]["shared_dim"] > 0
        assert cfg["model"]["num_attention_heads"] > 0
        assert cfg["model"]["num_joint_attention_layers"] > 0
        assert cfg["model"]["dropout"] >= 0

    @staticmethod
    def test_ptm_dimensions(cfg):
        assert cfg["ptm"]["ptm_dim"] == 12
        assert cfg["ptm"]["glyco_dim"] == 12

    @staticmethod
    def test_both_proteins_configured(cfg):
        assert "EGFR" in cfg["ptm"]
        assert "ERBB2" in cfg["ptm"]
        assert len(cfg["ptm"]["EGFR"]["phospho_sites"]) == 12

    @staticmethod
    def test_six_drugs_configured(cfg):
        assert len(cfg["drugs"]) >= 6
        drug_names = list(cfg["drugs"].keys())
        for expected in ["osimertinib", "gefitinib", "afatinib", "erlotinib"]:
            assert expected in drug_names, f"Missing drug: {expected}"

    @staticmethod
    def test_drug_smiles_present(cfg):
        for name, drug in cfg["drugs"].items():
            assert "smiles" in drug, f"Drug {name} missing SMILES"
            assert len(drug["smiles"]) > 10, f"Drug {name} SMILES too short"

    @staticmethod
    def test_training_seed_set(cfg):
        assert "seed" in cfg["training"]
        assert isinstance(cfg["training"]["seed"], int)

    @staticmethod
    def test_ptm_bdl_config(cfg):
        bdl = cfg.get("ptm_bdl", {})
        assert bdl.get("d_model", 64) > 0
        assert bdl.get("n_heads", 4) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. FocalLoss Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFocalLoss:

    @staticmethod
    @pytest.fixture
    def focal():
        return FocalLoss(alpha=0.25, gamma=2.0)

    @staticmethod
    def test_output_is_scalar(focal):
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = focal(logits, targets)
        assert loss.dim() == 0  # scalar

    @staticmethod
    def test_loss_positive(focal):
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = focal(logits, targets)
        assert loss.item() > 0

    @staticmethod
    def test_perfect_prediction_low_loss(focal):
        logits = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        loss = focal(logits, targets)
        assert loss.item() < 0.01, f"Perfect prediction should have near-zero loss, got {loss.item()}"

    @staticmethod
    def test_wrong_prediction_high_loss(focal):
        logits = torch.tensor([[10.0], [10.0], [10.0], [10.0]])
        targets = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        loss = focal(logits, targets)
        assert loss.item() > 0.1

    @staticmethod
    def test_class_conditional_alpha(focal):
        """Minority class (label=0) should get higher weight (1-alpha=0.75)."""
        logits = torch.zeros(2, 1)  # neutral
        loss_pos = focal(logits, torch.ones(2, 1))  # majority (alpha=0.25)
        loss_neg = focal(logits, torch.zeros(2, 1))  # minority (1-alpha=0.75)
        assert loss_neg.item() > loss_pos.item(), \
            "Minority class should have higher loss weight"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Metrics Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    @staticmethod
    def test_perfect_classification():
        preds = [1, 1, 0, 0, 1]
        probs = [0.9, 0.8, 0.1, 0.2, 0.95]
        labels = [1, 1, 0, 0, 1]
        m = compute_metrics_fn(preds, probs, labels)
        assert m["accuracy"] == 1.0
        assert m["balanced_acc"] == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    @staticmethod
    def test_all_wrong_classification():
        preds = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.9, 0.8]
        labels = [1, 1, 0, 0]
        m = compute_metrics_fn(preds, probs, labels)
        assert m["accuracy"] == 0.0
        assert m["sensitivity"] == 0.0
        assert m["specificity"] == 0.0

    @staticmethod
    def test_regression_metrics():
        preds = [1, 1, 0, 0]
        probs = [0.9, 0.8, 0.1, 0.2]
        labels = [1, 1, 0, 0]
        ic50_pred = [1.0, 2.0, 3.0, 4.0]
        ic50_true = [1.1, 2.1, 3.1, 4.1]
        m = compute_metrics_fn(preds, probs, labels, ic50_pred, ic50_true)
        assert "mse" in m
        assert "rmse" in m
        assert "pearson_r" in m
        assert m["rmse"] < 0.2

    @staticmethod
    def test_auroc_present():
        preds = [1, 0, 1, 0]
        probs = [0.8, 0.3, 0.7, 0.2]
        labels = [1, 0, 1, 0]
        m = compute_metrics_fn(preds, probs, labels)
        assert "auroc" in m
        assert m["auroc"] > 0.5

    @staticmethod
    def test_confusion_matrix():
        preds = [1, 0, 1, 0]
        probs = [0.8, 0.3, 0.7, 0.2]
        labels = [1, 0, 1, 0]
        m = compute_metrics_fn(preds, probs, labels)
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. build_model_from_cfg Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildModel:

    @staticmethod
    @pytest.fixture
    def cfg():
        return load_config(case_study="egfr_erbb2_tki")

    @staticmethod
    def test_builds_with_typed_attention(cfg):
        model = build_model_from_cfg_fn(cfg, use_typed_attention=True)
        assert isinstance(model.ptm_bdl, PTMBDLEncoder)

    @staticmethod
    def test_builds_with_mlp_ablation(cfg):
        model = build_model_from_cfg_fn(cfg, use_typed_attention=False)
        assert isinstance(model.ptm_bdl, PTMBDLMlpAblation)

    @staticmethod
    def test_model_forward_works(cfg):
        model = build_model_from_cfg_fn(cfg)
        model.eval()
        with torch.no_grad():
            ic50, _resist = model(
                seq_embeddings=torch.randn(1, 10, 1280),
                struct_embeddings=torch.randn(1, 8, 512),
                drug_pooled=torch.randn(1, 384),
                ptm_vector=torch.ones(1, 12),
                delta_ptm_vector=torch.zeros(1, 12),
            )
        assert ic50.shape == (1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Training Sanity Check (one step, no data needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingSanity:

    @staticmethod
    def test_one_gradient_step(model, batch):
        """Model should be able to do one forward + backward + step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        ic50, resist = model(
            seq_embeddings=batch["seq_emb"],
            struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"],
            drug_embeddings=batch["drug_emb"],
            ptm_vector=batch["ptm_vector"],
            delta_ptm_vector=batch["delta_ptm_vector"],
            secondary_vector=batch["secondary_vector"],
            delta_secondary_vector=batch["delta_secondary_vector"],
            target_protein=batch["target_protein"],
        )

        loss = ((ic50 - batch["ln_ic50"]) ** 2).mean() + \
               F.binary_cross_entropy_with_logits(resist, batch["resistance_label"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    @staticmethod
    def test_loss_decreases_over_steps(model, batch):
        """Loss should decrease after multiple gradient steps on same batch."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        for _ in range(5):
            ic50, _resist = model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch["drug_emb"],
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                secondary_vector=batch["secondary_vector"],
                delta_secondary_vector=batch["delta_secondary_vector"],
                target_protein=batch["target_protein"],
            )
            loss = ((ic50 - batch["ln_ic50"]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
