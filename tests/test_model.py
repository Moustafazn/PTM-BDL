"""
Tests for the full MultimodalResistancePredictor model.

Covers: architecture, forward pass shapes, optional inputs,
attention/PTM-BDL output returns, gradient flow, and PTM sensitivity.
"""

import torch

# noinspection PyProtectedMember
from src.models.multimodal_predictor import (
    PTMBDLEncoder, PTMBDLMlpAblation, ModalityProjection, BilinearLateFusion,
)


# ── Architecture ─────────────────────────────────────────────────────────────

class TestArchitecture:

    @staticmethod
    def test_typed_attention_backend(model):
        assert isinstance(model.ptm_bdl, PTMBDLEncoder)

    @staticmethod
    def test_mlp_ablation_backend(model_mlp):
        assert isinstance(model_mlp.ptm_bdl, PTMBDLMlpAblation)

    @staticmethod
    def test_has_all_components(model):
        for attr in ["seq_projection", "struct_projection", "drug_projection",
                      "static_transformer", "static_pool", "ptm_bdl",
                      "fusion", "regression_head", "classification_head"]:
            assert hasattr(model, attr), f"Missing: {attr}"

    @staticmethod
    def test_all_params_trainable(model):
        for name, p in model.named_parameters():
            assert p.requires_grad, f"{name} is frozen"

    @staticmethod
    def test_param_count_positive(model):
        assert sum(p.numel() for p in model.parameters()) > 0


# ── Forward Pass ─────────────────────────────────────────────────────────────

class TestForwardPass:

    @staticmethod
    def _run(model, batch):
        model.eval()
        with torch.no_grad():
            return model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch["drug_emb"],
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                glyco_vector=batch["glyco_vector"],
                delta_glyco_vector=batch["delta_glyco_vector"],
                target_protein=batch["target_protein"],
            )

    def test_output_shapes(self, model, batch):
        ic50, resist = self._run(model, batch)
        assert ic50.shape == (4, 1)
        assert resist.shape == (4, 1)

    def test_mlp_ablation_shapes(self, model_mlp, batch):
        ic50, resist = self._run(model_mlp, batch)
        assert ic50.shape == (4, 1)
        assert resist.shape == (4, 1)

    def test_no_nans(self, model, batch):
        ic50, resist = self._run(model, batch)
        assert not torch.isnan(ic50).any()
        assert not torch.isnan(resist).any()

    @staticmethod
    def test_batch_size_one(model):
        model.eval()
        with torch.no_grad():
            ic50, _r = model(
                seq_embeddings=torch.randn(1, 10, 1280),
                struct_embeddings=torch.randn(1, 8, 512),
                drug_pooled=torch.randn(1, 384),
                ptm_vector=torch.ones(1, 12),
                delta_ptm_vector=torch.zeros(1, 12),
            )
        assert ic50.shape == (1, 1)

    @staticmethod
    def test_without_optional_inputs(model):
        model.eval()
        with torch.no_grad():
            ic50, _r = model(
                seq_embeddings=torch.randn(2, 10, 1280),
                struct_embeddings=torch.randn(2, 8, 512),
                drug_pooled=torch.randn(2, 384),
                ptm_vector=torch.ones(2, 12),
                delta_ptm_vector=torch.zeros(2, 12),
            )
        assert ic50.shape == (2, 1)

    @staticmethod
    def test_returns_attention(model, batch):
        model.eval()
        with torch.no_grad():
            _, _, extras = model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch["drug_emb"],
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                glyco_vector=batch["glyco_vector"],
                delta_glyco_vector=batch["delta_glyco_vector"],
                target_protein=batch["target_protein"],
                return_attention=True,
            )
        assert "static_attention_maps" in extras

    @staticmethod
    def test_returns_ptm_bdl(model, batch):
        model.eval()
        with torch.no_grad():
            _, _, extras = model(
                seq_embeddings=batch["seq_emb"],
                struct_embeddings=batch["struct_emb"],
                drug_pooled=batch["drug_pooled"],
                drug_embeddings=batch["drug_emb"],
                ptm_vector=batch["ptm_vector"],
                delta_ptm_vector=batch["delta_ptm_vector"],
                glyco_vector=batch["glyco_vector"],
                delta_glyco_vector=batch["delta_glyco_vector"],
                target_protein=batch["target_protein"],
                return_ptm_bdl=True,
            )
        for key in ["pooled", "tokens", "mask", "type_ids"]:
            assert key in extras["ptm_bdl"]


# ── Gradient Flow ────────────────────────────────────────────────────────────

class TestGradientFlow:

    @staticmethod
    def test_ptm_grads(model, batch):
        ptm = batch["ptm_vector"].clone().requires_grad_(True)
        _, resist = model(
            seq_embeddings=batch["seq_emb"], struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"], drug_embeddings=batch["drug_emb"],
            ptm_vector=ptm, delta_ptm_vector=batch["delta_ptm_vector"],
            glyco_vector=batch["glyco_vector"],
            delta_glyco_vector=batch["delta_glyco_vector"],
            target_protein=batch["target_protein"],
        )
        resist.sum().backward()
        assert ptm.grad is not None and ptm.grad.abs().sum() > 0

    @staticmethod
    def test_glyco_grads(model, batch):
        glyco = batch["glyco_vector"].clone().requires_grad_(True)
        ic50, _ = model(
            seq_embeddings=batch["seq_emb"], struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"], drug_embeddings=batch["drug_emb"],
            ptm_vector=batch["ptm_vector"], delta_ptm_vector=batch["delta_ptm_vector"],
            glyco_vector=glyco, delta_glyco_vector=batch["delta_glyco_vector"],
            target_protein=batch["target_protein"],
        )
        ic50.sum().backward()
        assert glyco.grad is not None and glyco.grad.abs().sum() > 0

    @staticmethod
    def test_seq_grads(model, batch):
        seq = batch["seq_emb"].clone().requires_grad_(True)
        _, resist = model(
            seq_embeddings=seq, struct_embeddings=batch["struct_emb"],
            drug_pooled=batch["drug_pooled"],
            ptm_vector=batch["ptm_vector"],
            delta_ptm_vector=batch["delta_ptm_vector"],
        )
        resist.sum().backward()
        assert seq.grad.abs().sum() > 0


# ── Component Units ──────────────────────────────────────────────────────────

class TestComponents:

    @staticmethod
    def test_modality_projection():
        out = ModalityProjection(1280, 64)(torch.randn(2, 10, 1280))
        assert out.shape == (2, 10, 64)

    @staticmethod
    def test_bilinear_fusion():
        out = BilinearLateFusion(64, 32, 64)(torch.randn(2, 64), torch.randn(2, 32))
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()


# ── Sensitivity ──────────────────────────────────────────────────────────────

class TestSensitivity:

    @staticmethod
    def test_ptm_change_changes_output(model, batch):
        model.eval()
        with torch.no_grad():
            _, r1 = model(
                seq_embeddings=batch["seq_emb"][:1],
                struct_embeddings=batch["struct_emb"][:1],
                drug_pooled=batch["drug_pooled"][:1],
                ptm_vector=torch.ones(1, 12),
                delta_ptm_vector=torch.zeros(1, 12),
                target_protein=torch.tensor([0]),
            )
            ptm_mod = torch.ones(1, 12)
            ptm_mod[0, 7] = 5.0  # Y1092 up 5x
            _, r2 = model(
                seq_embeddings=batch["seq_emb"][:1],
                struct_embeddings=batch["struct_emb"][:1],
                drug_pooled=batch["drug_pooled"][:1],
                ptm_vector=ptm_mod,
                delta_ptm_vector=torch.zeros(1, 12),
                target_protein=torch.tensor([0]),
            )
        assert not torch.allclose(r1, r2, atol=1e-6)
