"""
Coarse 3D structure prediction module (Phase 2).

Architecture
------------
  sequence encoder  →  CoordHead    →  (B, L, 3)    Cα coordinates
                    →  FrameHead    →  (B, L, 3, 3) per-residue rotation matrices

The CoordHead predicts absolute Cα positions in an arbitrary global frame.
The FrameHead predicts a local rotation matrix per residue via Gram-Schmidt
orthonormalisation of two learned 3-vectors.  Together, (R_i, CA_i) defines the
backbone frame used by FAPE loss.

FAPE is invariant to the global rotation/translation applied to all predicted
coordinates, so the model is free to "choose" any global frame during training.

Public API
----------
  StructureModule          — full model
  StructureOutput          — output dataclass
  build_structure_model()  — factory
  gram_schmidt_frame()     — standalone utility (also used in losses.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.protein_dataset import PAD_IDX, VOCAB_SIZE
from src.models.baseline import BiLSTMModel, ModelKind, SS3_CLASSES, TransformerModel, build_model


# ---------------------------------------------------------------------------
# Gram-Schmidt orthonormalisation
# ---------------------------------------------------------------------------

def gram_schmidt_frame(a: Tensor, b: Tensor) -> Tensor:
    """Construct a rotation matrix from two (possibly non-orthogonal) 3-vectors.

    Applies Gram-Schmidt to produce an orthonormal right-handed frame:
      e1 = normalize(a)
      e2 = normalize(b - (b·e1) * e1)
      e3 = cross(e1, e2)

    Args:
        a: (..., 3) — first direction vector.
        b: (..., 3) — second direction vector (need not be perpendicular to a).

    Returns:
        R: (..., 3, 3) rotation matrix whose *columns* are [e1, e2, e3].
           Satisfies R^T R = I and det(R) = +1 (proper rotation).
    """
    e1 = F.normalize(a, dim=-1)                              # (..., 3)
    e2 = b - (b * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(e2, dim=-1)                             # (..., 3)
    e3 = torch.cross(e1, e2, dim=-1)                         # (..., 3)
    return torch.stack([e1, e2, e3], dim=-1)                 # (..., 3, 3)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructureOutput:
    """Output of StructureModule.forward().

    Attributes:
        ca_coords:  (B, L, 3) float32 — predicted Cα positions.
        rotations:  (B, L, 3, 3) float32 — per-residue rotation matrices
                    (None when use_frames=False).
    """
    ca_coords: Tensor
    rotations: Optional[Tensor] = None   # (B, L, 3, 3) or None

    @property
    def has_frames(self) -> bool:
        return self.rotations is not None


# ---------------------------------------------------------------------------
# Coordinate head
# ---------------------------------------------------------------------------

class CoordHead(nn.Module):
    """Projects per-residue embeddings to Cα coordinates.

    A deliberately simple MLP: LayerNorm → Linear → ReLU → Linear(3).
    The final layer has no bias so coordinates are initialised near zero,
    which avoids large initial FAPE loss values.

    Args:
        hidden_dim: Dimensionality of encoder output.
        inner_dim:  Hidden size of the intermediate linear layer.
    """

    def __init__(self, hidden_dim: int, inner_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, 3, bias=False),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: (B, L, hidden_dim)
        Returns:
            ca_coords: (B, L, 3)
        """
        return self.net(h)


# ---------------------------------------------------------------------------
# Frame head
# ---------------------------------------------------------------------------

class FrameHead(nn.Module):
    """Predicts a per-residue rotation matrix from embeddings.

    Projects to 6 raw scalars (two 3-vectors), then applies Gram-Schmidt
    to obtain a valid rotation matrix.

    Args:
        hidden_dim: Dimensionality of encoder output.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 6)

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: (B, L, hidden_dim)
        Returns:
            R: (B, L, 3, 3) — per-residue rotation matrices.
        """
        x = self.proj(self.norm(h))          # (B, L, 6)
        a = x[..., :3]                       # (B, L, 3)
        b = x[..., 3:]                       # (B, L, 3)
        return gram_schmidt_frame(a, b)      # (B, L, 3, 3)


# ---------------------------------------------------------------------------
# Full structure module
# ---------------------------------------------------------------------------

class StructureModule(nn.Module):
    """Encoder + coordinate head + optional frame head.

    Args:
        encoder:    BiLSTMModel or TransformerModel (its encode() method is used).
        use_frames: If True, also predict per-residue rotation frames for FAPE.
        inner_dim:  Hidden dim of the CoordHead MLP.
    """

    def __init__(
        self,
        encoder: BiLSTMModel | TransformerModel,
        use_frames: bool = True,
        inner_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_dim = _encoder_hidden_dim(encoder)
        self.coord_head = CoordHead(hidden_dim, inner_dim=inner_dim)
        self.frame_head = FrameHead(hidden_dim) if use_frames else None

    def forward(self, tokens: Tensor) -> StructureOutput:
        """
        Args:
            tokens: (B, L) integer token indices from ProteinDataset.
        Returns:
            StructureOutput with .ca_coords (B, L, 3) and
            .rotations (B, L, 3, 3) if use_frames=True.
        """
        h = self.encoder.encode(tokens)                   # (B, L, H)
        ca_coords = self.coord_head(h)                    # (B, L, 3)
        rotations = self.frame_head(h) if self.frame_head is not None else None
        return StructureOutput(ca_coords=ca_coords, rotations=rotations)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        enc = type(self.encoder).__name__
        frames = self.frame_head is not None
        return (
            f"StructureModule(encoder={enc}, frames={frames}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _encoder_hidden_dim(encoder: BiLSTMModel | TransformerModel) -> int:
    if isinstance(encoder, BiLSTMModel):
        return encoder.lstm.hidden_size * 2
    if isinstance(encoder, TransformerModel):
        return encoder.embedding.embedding_dim
    raise TypeError(f"Unsupported encoder type: {type(encoder)}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_structure_model(
    kind: ModelKind = "bilstm",
    use_frames: bool = True,
    **encoder_kwargs,
) -> StructureModule:
    """Build a StructureModule with the specified encoder backbone.

    Args:
        kind:            "bilstm" or "transformer".
        use_frames:      Whether to predict rotation frames (required for FAPE).
        **encoder_kwargs: Forwarded to the encoder (e.g. hidden_dim=256).

    Returns:
        StructureModule ready for training.

    Example::

        model = build_structure_model("bilstm", hidden_dim=256)
        out = model(tokens)           # StructureOutput
        ca = out.ca_coords            # (B, L, 3)
        R  = out.rotations            # (B, L, 3, 3)
    """
    encoder = build_model(kind=kind, n_classes=SS3_CLASSES, **encoder_kwargs)
    return StructureModule(encoder, use_frames=use_frames)
