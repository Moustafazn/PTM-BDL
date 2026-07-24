"""
Tests for the PTM Biological Dynamics Layer (PTM-BDL).

Covers: PTMBDLEncoder, PTMBDLMlpAblation, pad masking, type embeddings,
attention weights, cross-token independence (MLP), and biological constraints.
"""

import torch

from src.ptm_bdl.model import (
    PTMBDLEncoder, PTMBDLMlpAblation,
    N_PTM_TYPES,
)
# noinspection PyProtectedMember
from src.ptm_bdl.model import (
    _TYPE_Y, _TYPE_S, _TYPE_T, _TYPE_N,
    _PAD_EGFR, _PAD_ERBB2,
)


# ── PTM-BDL Encoder (Typed Self-Attention) ───────────────────────────────────

class TestPTMBDLEncoder:

    @staticmethod
    def _make(**kw):
        return PTMBDLEncoder(d_model=kw.get("d_model", 32),
                             n_heads=4, n_layers=1)

    @staticmethod
    def _run(enc, protein_id=0):
        return enc(
            ptm_vector=torch.ones(1, 12),
            delta_ptm_vector=torch.zeros(1, 12),
            secondary_vector=torch.ones(1, 12),
            delta_secondary_vector=torch.zeros(1, 12),
            target_protein=torch.tensor([protein_id]),
        )

    def test_output_keys(self):
        out = self._run(self._make())
        for key in ["pooled", "tokens", "mask", "type_ids"]:
            assert key in out

    def test_output_shapes(self):
        out = self._run(self._make())
        assert out["pooled"].shape == (1, 32)
        assert out["tokens"].shape == (1, 24, 32)
        assert out["mask"].shape == (1, 24)
        assert out["type_ids"].shape == (1, 24)

    def test_egfr_24_real_tokens(self):
        out = self._run(self._make(), protein_id=0)
        assert out["mask"][0].sum().item() == 24

    def test_erbb2_17_real_tokens(self):
        out = self._run(self._make(), protein_id=1)
        n_real = out["mask"][0].sum().item()
        assert n_real == 17, f"Expected 17 real, got {n_real}"

    def test_erbb2_phospho_pads_at_10_11(self):
        out = self._run(self._make(), protein_id=1)
        mask = out["mask"][0]
        assert not mask[10].item()  # pad
        assert not mask[11].item()  # pad

    def test_erbb2_secondary_pads_at_19_to_23(self):
        out = self._run(self._make(), protein_id=1)
        mask = out["mask"][0]
        for i in range(19, 24):
            assert not mask[i].item(), f"Slot {i} should be padded"

    def test_secondary_type_ids_all_N(self):
        out = self._run(self._make(), protein_id=0)
        secondary_types = out["type_ids"][0, 12:].unique().tolist()
        assert secondary_types == [_TYPE_N]

    def test_phospho_types_include_Y_S_T(self):
        out = self._run(self._make(), protein_id=0)
        phospho_types = set(out["type_ids"][0, :12].tolist())
        assert _TYPE_Y in phospho_types  # At least Y
        assert len(phospho_types) >= 2  # Should have Y + S or T

    def test_attention_weights_shape(self):
        enc = self._make()
        attn = enc.compute_attn_weights(
            ptm_vector=torch.ones(2, 12),
            delta_ptm_vector=torch.zeros(2, 12),
            secondary_vector=torch.ones(2, 12),
            delta_secondary_vector=torch.zeros(2, 12),
            target_protein=torch.tensor([0, 1]),
        )
        assert attn.shape == (2, 24, 24)

    def test_attention_rows_sum_to_one(self):
        enc = self._make()
        attn = enc.compute_attn_weights(
            ptm_vector=torch.ones(1, 12),
            delta_ptm_vector=torch.zeros(1, 12),
            secondary_vector=torch.ones(1, 12),
            delta_secondary_vector=torch.zeros(1, 12),
            target_protein=torch.tensor([0]),
        )
        row_sums = attn[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(24), atol=0.1), \
            f"Attention rows should sum to ~1.0, got range [{row_sums.min():.3f}, {row_sums.max():.3f}]"

    def test_batch_mixed_proteins(self):
        enc = self._make()
        out = enc(
            ptm_vector=torch.ones(3, 12),
            delta_ptm_vector=torch.zeros(3, 12),
            secondary_vector=torch.ones(3, 12),
            delta_secondary_vector=torch.zeros(3, 12),
            target_protein=torch.tensor([0, 1, 0]),
        )
        assert out["mask"][0].sum().item() == 24  # EGFR
        assert out["mask"][1].sum().item() == 17  # ERBB2
        assert out["mask"][2].sum().item() == 24  # EGFR

    def test_different_inputs_different_pooled(self):
        enc = self._make()
        enc.eval()
        with torch.no_grad():
            out1 = enc(torch.ones(1, 12), torch.zeros(1, 12),
                       torch.ones(1, 12), torch.zeros(1, 12), torch.tensor([0]))
            ptm2 = torch.ones(1, 12)
            ptm2[0, 7] = 3.0
            out2 = enc(ptm2, torch.zeros(1, 12),
                       torch.ones(1, 12), torch.zeros(1, 12), torch.tensor([0]))
        assert not torch.allclose(out1["pooled"], out2["pooled"], atol=1e-5)


# ── PTM-BDL MLP Ablation ────────────────────────────────────────────────────

class TestPTMBDLMlpAblation:

    def test_same_shapes_as_encoder(self):
        mlp = PTMBDLMlpAblation(d_model=32, n_layers=1)
        out = mlp(torch.ones(2, 12), torch.zeros(2, 12),
                  torch.ones(2, 12), torch.zeros(2, 12), torch.tensor([0, 1]))
        assert out["pooled"].shape == (2, 32)
        assert out["tokens"].shape == (2, 24, 32)

    def test_no_cross_token_dependency(self):
        """Changing one token should NOT affect other tokens in MLP."""
        mlp = PTMBDLMlpAblation(d_model=32, n_layers=1)
        mlp.eval()

        ptm1 = torch.ones(1, 12)
        ptm2 = ptm1.clone()
        ptm2[0, 0] = 5.0  # Only change first phospho site

        with torch.no_grad():
            out1 = mlp(ptm1, torch.zeros(1, 12), torch.ones(1, 12),
                       torch.zeros(1, 12), torch.tensor([0]))
            out2 = mlp(ptm2, torch.zeros(1, 12), torch.ones(1, 12),
                       torch.zeros(1, 12), torch.tensor([0]))

        # Token 1 (second phospho) must be identical
        assert torch.allclose(out1["tokens"][0, 1], out2["tokens"][0, 1], atol=1e-5)
        # Token 0 should be different
        assert not torch.allclose(out1["tokens"][0, 0], out2["tokens"][0, 0], atol=1e-5)

    def test_has_cross_token_in_encoder(self):
        """The Encoder (attention) SHOULD have cross-token dependencies."""
        enc = PTMBDLEncoder(d_model=32, n_heads=4, n_layers=1)
        enc.eval()

        ptm1 = torch.ones(1, 12)
        ptm2 = ptm1.clone()
        ptm2[0, 0] = 5.0

        with torch.no_grad():
            out1 = enc(ptm1, torch.zeros(1, 12), torch.ones(1, 12),
                       torch.zeros(1, 12), torch.tensor([0]))
            out2 = enc(ptm2, torch.zeros(1, 12), torch.ones(1, 12),
                       torch.zeros(1, 12), torch.tensor([0]))

        # In attention, changing token 0 SHOULD affect token 1
        assert not torch.allclose(out1["tokens"][0, 1], out2["tokens"][0, 1], atol=1e-4), \
            "Self-attention encoder should have cross-token dependencies"


# ── Biological Constants ─────────────────────────────────────────────────────

class TestBiologicalConstants:

    def test_four_ptm_types(self):
        assert N_PTM_TYPES == 4

    def test_type_ids_values(self):
        assert _TYPE_Y == 0  # phospho-tyrosine
        assert _TYPE_S == 1  # phospho-serine
        assert _TYPE_T == 2  # phospho-threonine
        assert _TYPE_N == 3  # N-linked secondary PTM

    def test_egfr_no_pads(self):
        assert _PAD_EGFR == [False] * 24

    def test_erbb2_pad_count(self):
        n_pad = sum(1 for x in _PAD_ERBB2 if x)
        assert n_pad == 7  # 2 phospho + 5 secondary pads
