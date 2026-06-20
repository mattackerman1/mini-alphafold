"""Shape contract tests for src/models/evoformer.py."""

import torch
import pytest

from src.models.evoformer import (
    EvoformerStack,
    EvoformerBlock,
    TriangleMultiplication,
    build_evoformer,
    EvoformerOutput,
)
from src.data.msa_encoder import FEATURE_DIM as MSA_FEAT_DIM


# Small dims for fast tests
C_M, C_Z, NHEAD, C_OUTER, C_T = 32, 16, 4, 8, 8


def _msa_input(B: int, N: int, L: int) -> torch.Tensor:
    return torch.randn(B, N, L, MSA_FEAT_DIM)


# ---------------------------------------------------------------------------
# TriangleMultiplication
# ---------------------------------------------------------------------------

def test_triangle_mult_outgoing_shape():
    """TriangleMultiplication (outgoing) preserves pair shape."""
    tri = TriangleMultiplication(C_Z, c_t=C_T, outgoing=True)
    pair = torch.randn(2, 10, 10, C_Z)
    out = tri(pair)
    assert out.shape == (2, 10, 10, C_Z)


def test_triangle_mult_incoming_shape():
    """TriangleMultiplication (incoming) preserves pair shape."""
    tri = TriangleMultiplication(C_Z, c_t=C_T, outgoing=False)
    pair = torch.randn(2, 10, 10, C_Z)
    out = tri(pair)
    assert out.shape == (2, 10, 10, C_Z)


def test_triangle_mult_modifies_pair():
    """Triangle update should change the pair representation (non-trivial op)."""
    tri = TriangleMultiplication(C_Z, c_t=C_T, outgoing=True)
    pair = torch.randn(1, 8, 8, C_Z)
    out = tri(pair)
    assert not torch.allclose(out, pair)


# ---------------------------------------------------------------------------
# EvoformerBlock
# ---------------------------------------------------------------------------

def test_evoformer_block_shapes():
    """EvoformerBlock preserves MSA and pair shapes."""
    block = EvoformerBlock(c_m=C_M, c_z=C_Z, nhead=NHEAD, c_outer=C_OUTER,
                           c_t=C_T, use_triangle=True)
    msa = torch.randn(2, 4, 12, C_M)
    pair = torch.randn(2, 12, 12, C_Z)
    msa_out, pair_out = block(msa, pair)
    assert msa_out.shape == (2, 4, 12, C_M)
    assert pair_out.shape == (2, 12, 12, C_Z)


def test_evoformer_block_no_triangle():
    """EvoformerBlock without triangle updates still preserves shapes."""
    block = EvoformerBlock(c_m=C_M, c_z=C_Z, nhead=NHEAD, c_outer=C_OUTER,
                           use_triangle=False)
    msa = torch.randn(1, 3, 8, C_M)
    pair = torch.randn(1, 8, 8, C_Z)
    msa_out, pair_out = block(msa, pair)
    assert msa_out.shape == (1, 3, 8, C_M)
    assert pair_out.shape == (1, 8, 8, C_Z)


# ---------------------------------------------------------------------------
# EvoformerStack
# ---------------------------------------------------------------------------

def test_evoformer_stack_output_shapes():
    """EvoformerStack returns correct MSA and pair shapes."""
    evo = build_evoformer(n_blocks=1, c_m=C_M, c_z=C_Z, nhead=NHEAD,
                          c_outer=C_OUTER, c_t=C_T)
    msa_feat = _msa_input(B=2, N=4, L=16)
    out = evo(msa_feat)
    assert isinstance(out, EvoformerOutput)
    assert out.msa_repr.shape == (2, 4, 16, C_M)
    assert out.pair_repr.shape == (2, 16, 16, C_Z)


def test_evoformer_query_repr_shape():
    """EvoformerOutput.query_repr is row-0 of MSA: (B, L, c_m)."""
    evo = build_evoformer(n_blocks=1, c_m=C_M, c_z=C_Z, nhead=NHEAD,
                          c_outer=C_OUTER, c_t=C_T)
    msa_feat = _msa_input(B=2, N=4, L=16)
    out = evo(msa_feat)
    assert out.query_repr.shape == (2, 16, C_M)


def test_evoformer_single_sequence():
    """EvoformerStack works with N=1 (single-sequence pseudo-MSA)."""
    evo = build_evoformer(n_blocks=1, c_m=C_M, c_z=C_Z, nhead=NHEAD,
                          c_outer=C_OUTER, c_t=C_T)
    msa_feat = _msa_input(B=1, N=1, L=10)
    out = evo(msa_feat)
    assert out.msa_repr.shape == (1, 1, 10, C_M)
    assert out.pair_repr.shape == (1, 10, 10, C_Z)


def test_evoformer_gradient_flows():
    """Gradients flow through all Evoformer parameters."""
    evo = build_evoformer(n_blocks=1, c_m=C_M, c_z=C_Z, nhead=NHEAD,
                          c_outer=C_OUTER, c_t=C_T)
    msa_feat = _msa_input(B=2, N=3, L=8)
    out = evo(msa_feat)
    loss = out.msa_repr.sum() + out.pair_repr.sum()
    loss.backward()
    grads = [p.grad for p in evo.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(g.isfinite().all() for g in grads), "NaN/Inf gradients detected"
