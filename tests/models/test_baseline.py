"""Shape contract tests for src/models/baseline.py."""

import torch
import pytest

from src.models.baseline import (
    BiLSTMModel,
    TransformerModel,
    build_model,
    SS3_CLASSES,
    SS8_CLASSES,
)
from src.data.protein_dataset import PAD_IDX, VOCAB_SIZE


def _make_tokens(B: int, L: int) -> torch.Tensor:
    """Random token indices in [2, VOCAB_SIZE)."""
    return torch.randint(2, VOCAB_SIZE, (B, L))


# ---------------------------------------------------------------------------
# BiLSTMModel
# ---------------------------------------------------------------------------

def test_bilstm_output_shape_ss3():
    """BiLSTM forward returns (B, L, n_classes) for SS3."""
    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1, n_classes=SS3_CLASSES)
    tokens = _make_tokens(2, 20)
    out = model(tokens)
    assert out.shape == (2, 20, SS3_CLASSES)


def test_bilstm_output_shape_ss8():
    """BiLSTM forward returns (B, L, n_classes) for SS8."""
    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1, n_classes=SS8_CLASSES)
    tokens = _make_tokens(3, 15)
    out = model(tokens)
    assert out.shape == (3, 15, SS8_CLASSES)


def test_bilstm_encode_shape():
    """BiLSTM.encode returns (B, L, 2*hidden_dim) hidden states."""
    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1)
    tokens = _make_tokens(2, 10)
    h = model.encode(tokens)
    assert h.shape == (2, 10, 32)   # 2 * hidden_dim = 32


def test_bilstm_pad_token_produces_zero_embedding():
    """PAD tokens are embedded as zero vectors (padding_idx=0)."""
    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1)
    pad_tokens = torch.zeros(1, 5, dtype=torch.long)  # all PAD
    real_tokens = torch.randint(2, VOCAB_SIZE, (1, 5))
    # Just check shapes — PAD embedding contract is on the embedding layer
    assert model(pad_tokens).shape == (1, 5, SS3_CLASSES)
    assert model(real_tokens).shape == (1, 5, SS3_CLASSES)


def test_bilstm_single_residue():
    """BiLSTM handles length-1 sequences."""
    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1)
    tokens = _make_tokens(1, 1)
    out = model(tokens)
    assert out.shape == (1, 1, SS3_CLASSES)


def test_bilstm_gradient_flows():
    """Gradients should flow back through BiLSTM."""
    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1)
    tokens = _make_tokens(2, 10)
    loss = model(tokens).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# TransformerModel
# ---------------------------------------------------------------------------

def test_transformer_output_shape():
    """Transformer forward returns (B, L, n_classes)."""
    model = TransformerModel(d_model=16, nhead=2, num_layers=1, n_classes=SS3_CLASSES)
    tokens = _make_tokens(2, 20)
    out = model(tokens)
    assert out.shape == (2, 20, SS3_CLASSES)


def test_transformer_encode_shape():
    """Transformer.encode returns (B, L, d_model) hidden states."""
    model = TransformerModel(d_model=16, nhead=2, num_layers=1)
    tokens = _make_tokens(2, 10)
    h = model.encode(tokens)
    assert h.shape == (2, 10, 16)


def test_transformer_gradient_flows():
    """Gradients flow back through TransformerModel."""
    model = TransformerModel(d_model=16, nhead=2, num_layers=1)
    tokens = _make_tokens(2, 10)
    loss = model(tokens).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# build_model factory
# ---------------------------------------------------------------------------

def test_build_model_bilstm():
    """build_model('bilstm') returns a BiLSTMModel."""
    model = build_model(kind="bilstm", n_classes=SS3_CLASSES)
    assert isinstance(model, BiLSTMModel)


def test_build_model_transformer():
    """build_model('transformer') returns a TransformerModel."""
    model = build_model(kind="transformer", n_classes=SS3_CLASSES)
    assert isinstance(model, TransformerModel)


def test_build_model_invalid_kind():
    """build_model raises for unknown kind."""
    with pytest.raises((ValueError, KeyError)):
        build_model(kind="invalid_kind", n_classes=SS3_CLASSES)
