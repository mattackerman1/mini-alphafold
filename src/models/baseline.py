"""
Baseline sequence model for per-residue secondary structure prediction.

Architecture: bidirectional LSTM encoder → linear projection head.
A Transformer encoder variant is also provided as a drop-in replacement.

Input:  (batch, seq_len)          — token indices from ProteinDataset
Output: (batch, seq_len, n_classes) — per-residue class logits

Both models ignore PAD positions during the forward pass; loss masking
is the responsibility of the training loop (use ignore_index=PAD_IDX in
CrossEntropyLoss).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.data.protein_dataset import PAD_IDX, VOCAB_SIZE

# Secondary structure label sets
SS3_CLASSES = 3   # coil (C), helix (H), strand (E)
SS8_CLASSES = 8   # DSSP 8-state encoding


# ---------------------------------------------------------------------------
# Shared projection head
# ---------------------------------------------------------------------------

class ResidueHead(nn.Module):
    """Linear projection from hidden dim to per-residue class logits."""

    def __init__(self, hidden_dim: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, seq_len, n_classes)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# BiLSTM model
# ---------------------------------------------------------------------------

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM encoder for per-residue sequence labelling.

    Args:
        vocab_size:  Number of token types (default: VOCAB_SIZE from dataset).
        embed_dim:   Amino acid embedding dimension.
        hidden_dim:  LSTM hidden size (each direction); output is 2×hidden_dim.
        num_layers:  Number of stacked LSTM layers.
        n_classes:   Number of output classes (3 for SS3, 8 for SS8).
        dropout:     Dropout applied between LSTM layers and before the head.
        padding_idx: Token index to embed as all-zeros (PAD_IDX).
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_classes: int = SS3_CLASSES,
        dropout: float = 0.2,
        padding_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional doubles the output dim
        self.head = ResidueHead(hidden_dim * 2, n_classes, dropout=dropout)

    def encode(self, tokens: Tensor) -> Tensor:
        """Return per-residue hidden states before the classification head.

        Args:
            tokens: (batch, seq_len)
        Returns:
            h: (batch, seq_len, 2*hidden_dim)
        """
        x = self.embedding(tokens)
        x, _ = self.lstm(x)
        return x

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (batch, seq_len) — integer token indices.
        Returns:
            logits: (batch, seq_len, n_classes)
        """
        return self.head(self.encode(tokens))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Transformer encoder model
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)                   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)                           # (1, max_len, d)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder for per-residue sequence labelling.

    Args:
        vocab_size:  Number of token types.
        d_model:     Embedding / model dimension (must be divisible by nhead).
        nhead:       Number of attention heads.
        num_layers:  Number of TransformerEncoderLayer blocks.
        dim_ff:      Feed-forward inner dimension.
        n_classes:   Number of output classes.
        dropout:     Dropout rate throughout.
        padding_idx: Token index to embed as all-zeros.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        n_classes: int = SS3_CLASSES,
        dropout: float = 0.1,
        padding_idx: int = PAD_IDX,
    ) -> None:
        super().__init__()

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = ResidueHead(d_model, n_classes, dropout=dropout)

    def encode(self, tokens: Tensor) -> Tensor:
        """Return per-residue hidden states before the classification head.

        Args:
            tokens: (batch, seq_len)
        Returns:
            h: (batch, seq_len, d_model)
        """
        pad_mask = tokens == self.padding_idx
        x = self.embedding(tokens)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (batch, seq_len) — integer token indices.
        Returns:
            logits: (batch, seq_len, n_classes)
        """
        return self.head(self.encode(tokens))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ModelKind = Literal["bilstm", "transformer"]


def build_model(
    kind: ModelKind = "bilstm",
    n_classes: int = SS3_CLASSES,
    **kwargs,
) -> BiLSTMModel | TransformerModel:
    """Instantiate a baseline model by name.

    Args:
        kind:      "bilstm" or "transformer".
        n_classes: Number of output classes (SS3_CLASSES=3 or SS8_CLASSES=8).
        **kwargs:  Forwarded to the model constructor for hyperparameter overrides.

    Returns:
        Initialised model.
    """
    if kind == "bilstm":
        return BiLSTMModel(n_classes=n_classes, **kwargs)
    if kind == "transformer":
        return TransformerModel(n_classes=n_classes, **kwargs)
    raise ValueError(f"Unknown model kind {kind!r}. Choose 'bilstm' or 'transformer'.")
