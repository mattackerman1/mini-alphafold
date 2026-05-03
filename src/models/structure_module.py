"""
Coarse 3D structure prediction module (Phase 2 / Phase 3).

Architecture
------------
  Single-sequence path (Phase 2, default):
    sequence encoder  →  CoordHead    →  (B, L, 3)    Cα coordinates
                      →  FrameHead    →  (B, L, 3, 3) per-residue rotation matrices

  MSA-aware path (Phase 3):
    EvoformerStack    →  query_repr   →  CoordHead    →  (B, L, 3)
                                      →  FrameHead    →  (B, L, 3, 3)

  The two paths share identical CoordHead and FrameHead; only the source of
  per-residue hidden states differs.  Both are exposed through a unified
  forward(tokens, msa_features=None) signature.

Backward compatibility
----------------------
  build_structure_model()          — Phase 2, sequence encoder only (unchanged)
  build_msa_structure_model()      — Phase 3, EvoformerStack only
  StructureModule(encoder=...)     — single-sequence mode
  StructureModule(evoformer=...)   — MSA mode
  StructureModule(encoder=..., evoformer=...)  — both; MSA path used when
                                                 msa_features is provided

Public API
----------
  StructureOutput          — output dataclass
  StructureModule          — full model
  build_structure_model()  — factory (Phase 2, backward-compat)
  build_msa_structure_model() — factory (Phase 3, Evoformer backbone)
  gram_schmidt_frame()     — standalone utility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.protein_dataset import PAD_IDX, VOCAB_SIZE
from src.models.baseline import BiLSTMModel, ModelKind, SS3_CLASSES, TransformerModel, build_model

if TYPE_CHECKING:
    from src.models.evoformer import EvoformerStack


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
    """Per-residue Cα coordinate and frame predictor.

    Accepts either a sequence encoder (single-sequence, Phase 2) or an
    EvoformerStack (MSA-aware, Phase 3) as the representation backbone.
    Both paths feed identical CoordHead / FrameHead prediction heads.

    Args:
        encoder:    BiLSTMModel or TransformerModel.  Used when msa_features
                    is not provided in forward(), or when no evoformer is set.
        evoformer:  EvoformerStack.  Used when msa_features is provided.
                    Its query_repr (row 0 of msa_repr) is passed to the heads.
        use_frames: If True, also predict per-residue rotation frames for FAPE.
        inner_dim:  Hidden dim of the CoordHead MLP.

    At least one of ``encoder`` or ``evoformer`` must be provided.

    Forward signature::

        out = model(tokens)                        # single-sequence path
        out = model(tokens, msa_features=msa)      # MSA-aware path
    """

    def __init__(
        self,
        encoder:   Optional[BiLSTMModel | TransformerModel] = None,
        evoformer: Optional["EvoformerStack"] = None,
        use_frames: bool = True,
        inner_dim:  int  = 128,
    ) -> None:
        super().__init__()

        if encoder is None and evoformer is None:
            raise ValueError("At least one of encoder or evoformer must be provided.")

        self.encoder   = encoder
        self.evoformer = evoformer

        # Head input dim depends on which backbone is primary
        if evoformer is not None:
            hidden_dim = evoformer.c_m
        else:
            hidden_dim = _encoder_hidden_dim(encoder)  # type: ignore[arg-type]

        self.coord_head = CoordHead(hidden_dim, inner_dim=inner_dim)
        self.frame_head = FrameHead(hidden_dim) if use_frames else None

    def forward(
        self,
        tokens:       Tensor,
        msa_features: Optional[Tensor] = None,
    ) -> StructureOutput:
        """
        Args:
            tokens:       (B, L) integer token indices from ProteinDataset.
            msa_features: (B, N_seq, L, feat_dim) float32 — MSA features from
                          MSAEncoder / MSADataset.  When provided and an
                          evoformer is attached, the MSA path is used.
                          When None, falls back to the sequence encoder.

        Returns:
            StructureOutput with .ca_coords (B, L, 3) and
            .rotations (B, L, 3, 3) if use_frames=True.
        """
        if self.evoformer is not None and msa_features is not None:
            # MSA-aware path (Phase 3)
            evo_out = self.evoformer(msa_features)
            h = evo_out.query_repr                     # (B, L, c_m) — row 0
        elif self.encoder is not None:
            # Single-sequence path (Phase 2, backward compat)
            h = self.encoder.encode(tokens)            # (B, L, H)
        else:
            raise ValueError(
                "No valid forward path: provide msa_features when using an "
                "evoformer-only StructureModule, or attach a sequence encoder."
            )

        ca_coords = self.coord_head(h)                 # (B, L, 3)
        rotations = self.frame_head(h) if self.frame_head is not None else None
        return StructureOutput(ca_coords=ca_coords, rotations=rotations)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        if self.evoformer is not None:
            backbone = (
                f"EvoformerStack(n_blocks={len(self.evoformer.blocks)}, "
                f"c_m={self.evoformer.c_m})"
            )
        else:
            backbone = type(self.encoder).__name__  # type: ignore[union-attr]
        frames = self.frame_head is not None
        return (
            f"StructureModule(backbone={backbone}, frames={frames}, "
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
# Factories
# ---------------------------------------------------------------------------

def build_structure_model(
    kind: ModelKind = "bilstm",
    use_frames: bool = True,
    **encoder_kwargs,
) -> StructureModule:
    """Build a StructureModule with a sequence encoder backbone (Phase 2).

    Fully backward-compatible with the original Phase 2 interface.

    Args:
        kind:             "bilstm" or "transformer".
        use_frames:       Whether to predict rotation frames (required for FAPE).
        **encoder_kwargs: Forwarded to the encoder (e.g. hidden_dim=256).

    Returns:
        StructureModule (encoder-only, no Evoformer).

    Example::

        model = build_structure_model("bilstm", hidden_dim=256)
        out = model(tokens)           # StructureOutput
        ca = out.ca_coords            # (B, L, 3)
        R  = out.rotations            # (B, L, 3, 3)
    """
    encoder = build_model(kind=kind, n_classes=SS3_CLASSES, **encoder_kwargs)
    return StructureModule(encoder=encoder, use_frames=use_frames)


def build_msa_structure_model(
    n_blocks:     int  = 2,
    c_m:          int  = 64,
    c_z:          int  = 32,
    use_frames:   bool = True,
    inner_dim:    int  = 128,
    use_triangle: bool = True,
    **evoformer_kwargs,
) -> StructureModule:
    """Build a StructureModule with an EvoformerStack backbone (Phase 3).

    The resulting model expects msa_features=(B, N_seq, L, 45) in forward().
    Use MSADataset / collate_msa_fn to build batches.

    Args:
        n_blocks:          Number of Evoformer blocks.
        c_m:               MSA channel width (also the head input dim).
        c_z:               Pair channel width.
        use_frames:        Whether to predict rotation frames.
        inner_dim:         CoordHead MLP hidden size.
        use_triangle:      Enable triangle multiplicative updates.
        **evoformer_kwargs: Forwarded to build_evoformer
                           (nhead, c_outer, c_t, expand, dropout).

    Returns:
        StructureModule (Evoformer-only, no sequence encoder).

    Example::

        model = build_msa_structure_model(n_blocks=4, c_m=128, c_z=64)
        out = model(tokens, msa_features=msa)   # StructureOutput
        ca  = out.ca_coords                      # (B, L, 3)
    """
    from src.models.evoformer import build_evoformer

    evoformer = build_evoformer(
        n_blocks=n_blocks, c_m=c_m, c_z=c_z,
        use_triangle=use_triangle, **evoformer_kwargs,
    )
    return StructureModule(evoformer=evoformer, use_frames=use_frames, inner_dim=inner_dim)
