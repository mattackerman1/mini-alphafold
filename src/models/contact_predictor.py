"""
Residue-residue contact map predictor for mini-alphafold.

Architecture:  sequence encoder  →  ContactHead  →  (B, L, L) logits

The ContactHead takes per-residue hidden states (B, L, H) and produces a
symmetric matrix of raw logits.  Apply sigmoid to get contact probabilities.

Symmetry is guaranteed by construction:
    pair_feat[b, i, j] = 0.5 * (q[b,i]*k[b,j] + k[b,i]*q[b,j])
which is invariant under swapping i and j.

Memory note: the ContactHead materialises an (B, L, L, pair_dim) tensor,
so it is O(L²).  For sequences ≤ 500 residues this is fine on CPU/GPU.

Public API
----------
  ContactPredictor          — full model (encoder + head)
  ContactHead               — pairwise head (reusable standalone)
  build_contact_model(kind) — factory
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.data.protein_dataset import PAD_IDX, VOCAB_SIZE
from src.models.baseline import (
    BiLSTMModel,
    ModelKind,
    SS3_CLASSES,
    TransformerModel,
    build_model,
)


# ---------------------------------------------------------------------------
# Pairwise contact head
# ---------------------------------------------------------------------------

class ContactHead(nn.Module):
    """Projects per-residue hidden states to a symmetric (L, L) contact map.

    Args:
        hidden_dim: Dimensionality of encoder output (input to this head).
        pair_dim:   Internal dimension of the pairwise feature vector.
        dropout:    Dropout before the output linear layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, pair_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, pair_dim, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: (batch, seq_len, hidden_dim) — encoder output.
        Returns:
            logits: (batch, seq_len, seq_len) — raw (pre-sigmoid) contact scores,
                    symmetric along the last two dimensions.
        """
        q = self.proj_q(h)   # (B, L, d)
        k = self.proj_k(h)   # (B, L, d)

        # Symmetric pairwise features: 0.5*(q_i*k_j + k_i*q_j)
        # Broadcasting: (B, L, 1, d) * (B, 1, L, d) → (B, L, L, d)
        pair = 0.5 * (
            q.unsqueeze(2) * k.unsqueeze(1)   # q_i * k_j
            + k.unsqueeze(2) * q.unsqueeze(1) # k_i * q_j
        )                                      # (B, L, L, pair_dim)

        return self.out(pair).squeeze(-1)      # (B, L, L)


# ---------------------------------------------------------------------------
# Full contact predictor
# ---------------------------------------------------------------------------

class ContactPredictor(nn.Module):
    """Sequence encoder + ContactHead for residue-residue contact prediction.

    Args:
        encoder:    A BiLSTMModel or TransformerModel.  Its encode() method
                    is used to obtain per-residue hidden states.
        pair_dim:   Internal dimension of the ContactHead.
        dropout:    Dropout in the ContactHead.
    """

    def __init__(
        self,
        encoder: BiLSTMModel | TransformerModel,
        pair_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_dim = _encoder_hidden_dim(encoder)
        self.contact_head = ContactHead(hidden_dim, pair_dim=pair_dim, dropout=dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (batch, seq_len) — integer token indices from ProteinDataset.
        Returns:
            logits: (batch, seq_len, seq_len) — raw contact logits (apply sigmoid
                    for probabilities).  Symmetric: logits[b,i,j] == logits[b,j,i].
        """
        h = self.encoder.encode(tokens)          # (B, L, H)
        return self.contact_head(h)              # (B, L, L)

    def predict_proba(self, tokens: Tensor) -> Tensor:
        """Convenience wrapper: forward → sigmoid → (B, L, L) contact probabilities."""
        return torch.sigmoid(self(tokens))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        enc = type(self.encoder).__name__
        return (
            f"ContactPredictor(encoder={enc}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _encoder_hidden_dim(encoder: BiLSTMModel | TransformerModel) -> int:
    """Return the output dimensionality of an encoder's encode() method."""
    if isinstance(encoder, BiLSTMModel):
        return encoder.lstm.hidden_size * 2   # bidirectional
    if isinstance(encoder, TransformerModel):
        return encoder.embedding.embedding_dim
    raise TypeError(f"Unsupported encoder type: {type(encoder)}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_contact_model(
    kind: ModelKind = "bilstm",
    pair_dim: int = 64,
    dropout: float = 0.1,
    **encoder_kwargs,
) -> ContactPredictor:
    """Build a ContactPredictor with the specified encoder backbone.

    Args:
        kind:            "bilstm" or "transformer".
        pair_dim:        Internal dimension of the ContactHead.
        dropout:         Dropout in ContactHead.
        **encoder_kwargs: Forwarded to the encoder constructor
                          (e.g. hidden_dim=256, num_layers=3).

    Returns:
        ContactPredictor ready for training.

    Example::

        model = build_contact_model("bilstm", hidden_dim=128)
        logits = model(tokens)   # (B, L, L)
    """
    # n_classes is irrelevant for the encoder when used in ContactPredictor
    # (the SS head is never called), but the encoder needs a value to build.
    encoder = build_model(kind=kind, n_classes=SS3_CLASSES, **encoder_kwargs)
    return ContactPredictor(encoder, pair_dim=pair_dim, dropout=dropout)
