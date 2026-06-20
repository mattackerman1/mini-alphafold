"""Shape contract tests for src/models/structure_module.py."""

import torch
import pytest

from src.models.structure_module import (
    StructureModule,
    StructureOutput,
    CoordHead,
    FrameHead,
    gram_schmidt_frame,
    build_structure_model,
    build_msa_structure_model,
)
from src.data.protein_dataset import VOCAB_SIZE
from src.data.msa_encoder import FEATURE_DIM as MSA_FEAT_DIM


def _make_tokens(B: int, L: int) -> torch.Tensor:
    return torch.randint(2, VOCAB_SIZE, (B, L))


def _make_msa(B: int, N: int, L: int) -> torch.Tensor:
    return torch.randn(B, N, L, MSA_FEAT_DIM)


# ---------------------------------------------------------------------------
# Gram-Schmidt frame utility
# ---------------------------------------------------------------------------

def test_gram_schmidt_frame_shape():
    """gram_schmidt_frame returns (B, L, 3, 3) rotation matrices."""
    a = torch.randn(2, 10, 3)
    b = torch.randn(2, 10, 3)
    R = gram_schmidt_frame(a, b)
    assert R.shape == (2, 10, 3, 3)


def test_gram_schmidt_frame_orthogonal():
    """Rotation matrices must satisfy R^T R ≈ I."""
    a = torch.randn(2, 5, 3)
    b = torch.randn(2, 5, 3)
    R = gram_schmidt_frame(a, b)
    I_approx = torch.bmm(
        R.reshape(-1, 3, 3).transpose(-1, -2),
        R.reshape(-1, 3, 3),
    )
    I = torch.eye(3).unsqueeze(0).expand_as(I_approx)
    assert torch.allclose(I_approx, I, atol=1e-5)


# ---------------------------------------------------------------------------
# CoordHead
# ---------------------------------------------------------------------------

def test_coord_head_shape():
    """CoordHead produces (B, L, 3) Cα coordinates."""
    head = CoordHead(hidden_dim=32, inner_dim=16)
    h = torch.randn(2, 10, 32)
    ca = head(h)
    assert ca.shape == (2, 10, 3)


# ---------------------------------------------------------------------------
# FrameHead
# ---------------------------------------------------------------------------

def test_frame_head_shape():
    """FrameHead produces (B, L, 3, 3) rotation matrices."""
    head = FrameHead(hidden_dim=32)
    h = torch.randn(2, 10, 32)
    R = head(h)
    assert R.shape == (2, 10, 3, 3)


# ---------------------------------------------------------------------------
# StructureModule — single-sequence path
# ---------------------------------------------------------------------------

def test_structure_module_single_seq_shape():
    """StructureModule (encoder path) returns (B, L, 3) coords and (B, L, 3, 3) frames."""
    model = build_structure_model(kind="bilstm", hidden_dim=16, num_layers=1, use_frames=True)
    tokens = _make_tokens(2, 15)
    out = model(tokens)
    assert isinstance(out, StructureOutput)
    assert out.ca_coords.shape == (2, 15, 3)
    assert out.rotations is not None
    assert out.rotations.shape == (2, 15, 3, 3)


def test_structure_module_no_frames():
    """StructureModule with use_frames=False has no rotation output."""
    model = build_structure_model(kind="bilstm", hidden_dim=16, num_layers=1, use_frames=False)
    tokens = _make_tokens(2, 10)
    out = model(tokens)
    assert out.ca_coords.shape == (2, 10, 3)
    assert not out.has_frames


# ---------------------------------------------------------------------------
# StructureModule — MSA path
# ---------------------------------------------------------------------------

def test_structure_module_msa_path_shape():
    """StructureModule (Evoformer path) returns correct shapes for MSA input."""
    model = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, nhead=4, c_outer=8, c_t=8, use_frames=True
    )
    tokens = _make_tokens(2, 12)
    msa_feat = _make_msa(B=2, N=4, L=12)
    out = model(tokens, msa_features=msa_feat)
    assert out.ca_coords.shape == (2, 12, 3)
    assert out.rotations is not None
    assert out.rotations.shape == (2, 12, 3, 3)


def test_structure_module_gradient_flows():
    """Gradients flow through the full structure module."""
    model = build_structure_model(kind="bilstm", hidden_dim=16, num_layers=1)
    tokens = _make_tokens(2, 10)
    out = model(tokens)
    out.ca_coords.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
