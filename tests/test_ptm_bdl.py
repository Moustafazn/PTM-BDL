"""
Tests for the PTM Biological Dynamics Layer (PTM-BDL).

Covers: PTMBDLEncoder, PTMBDLMlpAblation, pad masking, type embeddings,
attention weights, cross-token independence (MLP), and registry integration.
"""

import torch

from src.ptm_bdl.model.encoder import PTMBDLEncoder
from src.ptm_bdl.model.ablation import PTMBDLMlpAblation
from src.ptm_bdl.registry import PTMTypeRegistry
from src.ptm_bdl.config import load_config


# ── Helper: build encoder from EGFR config ───────────────────────────────────

def _make_registry():
    cfg = load_config(case_study="egfr_erbb2_tki")
    return PTMTypeRegistry.from_config(cfg)


def _make_encoder(d_model=32):
    registry = _make_registry()
    return PTMBDLEncoder(registry=registry, d_model=d_model, n_heads=4, n_layers=1)


def _make_mlp(d_model=32):
    registry = _make_registry()
    return PTMBDLMlpAblation(registry=registry, d_model=d_model, n_layers=1)


# ── PTM-BDL Encoder (Typed Self-Attention) ───────────────────────────────────

class TestPTMBDLEncoder:

    @staticmethod
    def _run(enc, protein_id=0):
        n = enc.n_tokens
        return enc(
            ptm_vector=torch.ones(1, n),
            delta_ptm_vector=torch.zeros(1, n),
            target_protein=torch.tensor([protein_id]),
        )

    def test_output_keys(self):
        out = self._run(_make_encoder())
        for key in ["pooled", "tokens", "mask", "type_ids"]:
            assert key in out

    def test_output_shapes(self):
        enc = _make_encoder()
        n = enc.n_tokens
        out = self._run(enc)
        assert out["pooled"].shape == (1, 32)
        assert out["tokens"].shape == (1, n, 32)
        assert out["mask"].shape == (1, n)
        assert out["type_ids"].shape == (1, n)

    def test_protein_0_has_real_tokens(self):
        out = self._run(_make_encoder(), protein_id=0)
        n_real = out["mask"][0].sum().item()
        assert n_real > 0, "Protein 0 should have at least some real tokens"

    def test_protein_1_may_have_pads(self):
        """Protein 1 (ERBB2) may have fewer real tokens due to padding."""
        enc = _make_encoder()
        out0 = self._run(enc, protein_id=0)
        out1 = self._run(enc, protein_id=1)
        # ERBB2 should have <= EGFR real tokens (some are padded)
        assert out1["mask"][0].sum().item() <= out0["mask"][0].sum().item()

    def test_type_ids_have_multiple_subtypes(self):
        """Type IDs should contain multiple subtypes (e.g., Y, S, T, N)."""
        out = self._run(_make_encoder(), protein_id=0)
        unique_types = out["type_ids"][0].unique().tolist()
        assert len(unique_types) >= 2, \
            f"Expected multiple subtypes, got {unique_types}"

    def test_attention_weights_shape(self):
        enc = _make_encoder()
        n = enc.n_tokens
        attn = enc.compute_attn_weights(
            ptm_vector=torch.ones(2, n),
            delta_ptm_vector=torch.zeros(2, n),
            target_protein=torch.tensor([0, 1]),
        )
        assert attn.shape == (2, n, n)

    def test_attention_rows_sum_to_one(self):
        enc = _make_encoder()
        n = enc.n_tokens
        attn = enc.compute_attn_weights(
            ptm_vector=torch.ones(1, n),
            delta_ptm_vector=torch.zeros(1, n),
            target_protein=torch.tensor([0]),
        )
        row_sums = attn[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(n), atol=0.1), \
            f"Attention rows should sum to ~1.0, got range [{row_sums.min():.3f}, {row_sums.max():.3f}]"

    def test_batch_mixed_proteins(self):
        enc = _make_encoder()
        n = enc.n_tokens
        out = enc(
            ptm_vector=torch.ones(3, n),
            delta_ptm_vector=torch.zeros(3, n),
            target_protein=torch.tensor([0, 1, 0]),
        )
        # Protein 0 and 2 (same) should have same mask
        assert out["mask"][0].sum().item() == out["mask"][2].sum().item()

    def test_different_inputs_different_pooled(self):
        enc = _make_encoder()
        n = enc.n_tokens
        enc.eval()
        with torch.no_grad():
            out1 = enc(torch.ones(1, n), torch.zeros(1, n), torch.tensor([0]))
            ptm2 = torch.ones(1, n)
            ptm2[0, 7] = 3.0
            out2 = enc(ptm2, torch.zeros(1, n), torch.tensor([0]))
        assert not torch.allclose(out1["pooled"], out2["pooled"], atol=1e-5)


# ── PTM-BDL MLP Ablation ────────────────────────────────────────────────────

class TestPTMBDLMlpAblation:

    def test_same_shapes_as_encoder(self):
        mlp = _make_mlp()
        n = mlp.n_tokens
        out = mlp(torch.ones(2, n), torch.zeros(2, n), torch.tensor([0, 1]))
        assert out["pooled"].shape == (2, 32)
        assert out["tokens"].shape == (2, n, 32)

    def test_no_cross_token_dependency(self):
        """Changing one token should NOT affect other tokens in MLP."""
        mlp = _make_mlp()
        n = mlp.n_tokens
        mlp.eval()

        ptm1 = torch.ones(1, n)
        ptm2 = ptm1.clone()
        ptm2[0, 0] = 5.0  # Only change first site

        with torch.no_grad():
            out1 = mlp(ptm1, torch.zeros(1, n), torch.tensor([0]))
            out2 = mlp(ptm2, torch.zeros(1, n), torch.tensor([0]))

        # Token 1 (second site) must be identical
        assert torch.allclose(out1["tokens"][0, 1], out2["tokens"][0, 1], atol=1e-5)
        # Token 0 should be different
        assert not torch.allclose(out1["tokens"][0, 0], out2["tokens"][0, 0], atol=1e-5)

    def test_has_cross_token_in_encoder(self):
        """The Encoder (attention) SHOULD have cross-token dependencies."""
        enc = _make_encoder()
        n = enc.n_tokens
        enc.eval()

        ptm1 = torch.ones(1, n)
        ptm2 = ptm1.clone()
        ptm2[0, 0] = 5.0

        with torch.no_grad():
            out1 = enc(ptm1, torch.zeros(1, n), torch.tensor([0]))
            out2 = enc(ptm2, torch.zeros(1, n), torch.tensor([0]))

        # In attention, changing token 0 SHOULD affect token 1
        assert not torch.allclose(out1["tokens"][0, 1], out2["tokens"][0, 1], atol=1e-4), \
            "Self-attention encoder should have cross-token dependencies"


# ── Registry Integration ─────────────────────────────────────────────────────

class TestRegistryIntegration:

    def test_registry_builds_from_config(self):
        registry = _make_registry()
        assert registry.n_subtypes > 0
        assert registry.n_proteins > 0
        assert registry.n_tokens > 0

    def test_registry_has_type_tables(self):
        registry = _make_registry()
        assert registry.type_id_table.shape == (registry.n_proteins, registry.n_tokens)
        assert registry.is_real_table.shape == (registry.n_proteins, registry.n_tokens)

    def test_registry_subtype_names(self):
        registry = _make_registry()
        names = registry.subtype_names
        assert len(names) == registry.n_subtypes
        # Should have at least Y, S, T subtypes
        name_values = list(names.values())
        assert any("Y" in n for n in name_values), f"No Y subtype in {name_values}"

    def test_registry_slot_ranges(self):
        registry = _make_registry()
        for ptm_type in registry.ptm_type_order:
            start, end = registry.get_ptm_type_slot_range(ptm_type)
            assert 0 <= start < end <= registry.n_tokens
