"""Shape contract and symmetry tests for src/models/contact_predictor.py."""

import torch
import pytest

from src.models.contact_predictor import ContactHead, ContactPredictor, build_contact_model
from src.data.protein_dataset import VOCAB_SIZE


def _make_tokens(B: int, L: int) -> torch.Tensor:
    return torch.randint(2, VOCAB_SIZE, (B, L))


# ---------------------------------------------------------------------------
# ContactHead
# ---------------------------------------------------------------------------

def test_contact_head_output_shape():
    """ContactHead produces (B, L, L) logits from (B, L, H) hidden states."""
    head = ContactHead(hidden_dim=32, pair_dim=16)
    h = torch.randn(2, 20, 32)
    logits = head(h)
    assert logits.shape == (2, 20, 20)


def test_contact_head_output_is_symmetric():
    """ContactHead output is symmetric: logits[b,i,j] == logits[b,j,i]."""
    head = ContactHead(hidden_dim=32, pair_dim=16)
    head.eval()
    h = torch.randn(1, 15, 32)
    with torch.no_grad():
        logits = head(h)
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-5), \
        "Contact logits should be symmetric"


def test_contact_head_single_residue():
    """ContactHead works on L=1 sequences."""
    head = ContactHead(hidden_dim=32, pair_dim=16)
    h = torch.randn(1, 1, 32)
    logits = head(h)
    assert logits.shape == (1, 1, 1)


def test_contact_head_gradient_flows():
    """Gradients flow back through ContactHead."""
    head = ContactHead(hidden_dim=32, pair_dim=16)
    h = torch.randn(2, 10, 32, requires_grad=True)
    loss = head(h).sum()
    loss.backward()
    assert h.grad is not None


# ---------------------------------------------------------------------------
# ContactPredictor (full model)
# ---------------------------------------------------------------------------

def test_contact_predictor_bilstm_shape():
    """ContactPredictor(bilstm) returns (B, L, L) logits."""
    model = build_contact_model(kind="bilstm", hidden_dim=16, num_layers=1)
    tokens = _make_tokens(2, 15)
    logits = model(tokens)
    assert logits.shape == (2, 15, 15)


def test_contact_predictor_transformer_shape():
    """ContactPredictor(transformer) returns (B, L, L) logits."""
    model = build_contact_model(kind="transformer", d_model=16, nhead=2, num_layers=1)
    tokens = _make_tokens(2, 12)
    logits = model(tokens)
    assert logits.shape == (2, 12, 12)


def test_contact_predictor_symmetry():
    """Full ContactPredictor output is symmetric."""
    model = build_contact_model(kind="bilstm", hidden_dim=16, num_layers=1)
    model.eval()
    tokens = _make_tokens(1, 10)
    with torch.no_grad():
        logits = model(tokens)
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-5)
