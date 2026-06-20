"""
Recycling mechanism for iterative structure refinement (Phase 3).

In AlphaFold2, recycling runs the Evoformer + Structure module N times.
Each pass receives embeddings derived from the previous pass's outputs:
  - Previous Cα-Cα pairwise distances (binned and embedded) → added to pair rep
  - Previous MSA first row (via LayerNorm) → added to MSA first row

Gradients flow only through the final recycle pass; all earlier passes
run under torch.no_grad() to save memory during training.

Public API
----------
  RecyclingEmbedder     — computes (msa_delta, pair_delta) from prev pass outputs
  RecycledEvoformer     — wraps EvoformerStack + StructureModule for recycled inference
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.evoformer import EvoformerOutput, EvoformerStack
from src.models.structure_module import StructureModule, StructureOutput


# ---------------------------------------------------------------------------
# RecyclingEmbedder
# ---------------------------------------------------------------------------

class RecyclingEmbedder(nn.Module):
    """Embeds previous-pass outputs into additive recycling corrections.

    Produces two delta tensors injected into the next Evoformer pass:
      - ``pair_delta``  (B, L, L, c_z): from binned Cα-Cα distances.
      - ``msa_delta``   (B, L, c_m):    LayerNorm of the previous MSA first row.

    Args:
        c_m:           MSA channel dimension.
        c_z:           Pair channel dimension.
        num_dist_bins: Number of distance bins spanning 0–20 Å (default 15).
    """

    def __init__(self, c_m: int, c_z: int, num_dist_bins: int = 15) -> None:
        super().__init__()
        self.num_dist_bins = num_dist_bins

        # Fixed bin edges: 0–20 Å linearly spaced
        self.register_buffer(
            "bin_edges",
            torch.linspace(0.0, 20.0, num_dist_bins + 1),
        )

        # Pair delta: embed bin index → pair channel
        self.dist_embedding = nn.Embedding(num_dist_bins, c_z)

        # MSA delta: normalise previous first row
        self.msa_norm = nn.LayerNorm(c_m)

    def forward(
        self,
        prev_ca: Tensor,        # (B, L, 3)   — prev Cα coordinates
        prev_pair: Tensor,      # (B, L, L, c_z) — prev pair representation
        prev_msa_row: Tensor,   # (B, L, c_m)  — prev MSA first row
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            prev_ca:       (B, L, 3) predicted Cα positions from the previous pass.
            prev_pair:     (B, L, L, c_z) pair representation from the previous pass.
                           Currently unused directly; retained for API extensibility.
            prev_msa_row:  (B, L, c_m) MSA first-row representation from the previous pass.

        Returns:
            msa_delta:  (B, L, c_m)    — add to MSA first row after input projection.
            pair_delta: (B, L, L, c_z) — add to initial (zero) pair representation.
        """
        # --- pair delta: pairwise Cα distances → bins → embeddings ---
        diff = prev_ca.unsqueeze(2) - prev_ca.unsqueeze(1)         # (B, L, L, 3)
        dists = diff.norm(dim=-1)                                    # (B, L, L)

        # bucketize returns index in [0, num_dist_bins]; clamp to valid range
        bin_edges: Tensor = self.bin_edges  # type: ignore[assignment]
        bin_idx = torch.bucketize(dists.clamp(0.0, 20.0), bin_edges[1:])
        bin_idx = bin_idx.clamp(0, self.num_dist_bins - 1)          # (B, L, L) long

        pair_delta = self.dist_embedding(bin_idx)                    # (B, L, L, c_z)

        # --- MSA delta: LayerNorm of previous first-row representation ---
        msa_delta = self.msa_norm(prev_msa_row)                      # (B, L, c_m)

        return msa_delta, pair_delta


# ---------------------------------------------------------------------------
# RecycledEvoformer
# ---------------------------------------------------------------------------

class RecycledEvoformer(nn.Module):
    """Iteratively refines structure predictions by recycling Evoformer outputs.

    Runs the full Evoformer + Structure-prediction heads ``num_recycles`` times.
    The first pass starts from zero-initialised recycle inputs.  Each subsequent
    pass receives the Cα coordinates, pair representation, and MSA first row from
    the previous pass via ``RecyclingEmbedder``.

    Only the **last** recycle pass participates in gradient computation.  All
    earlier passes run under ``torch.no_grad()`` to save memory.

    Args:
        evoformer:        A pre-built ``EvoformerStack``.
        structure_module: A ``StructureModule`` whose ``coord_head`` and
                          ``frame_head`` will be called with the Evoformer's
                          query representation.  Its internal evoformer (if any)
                          is bypassed; only the prediction heads are used.
        num_recycles:     Total number of forward passes (default 3).
    """

    def __init__(
        self,
        evoformer: EvoformerStack,
        structure_module: StructureModule,
        num_recycles: int = 3,
    ) -> None:
        super().__init__()
        self.evoformer = evoformer
        self.structure_module = structure_module
        self.recycling_embedder = RecyclingEmbedder(
            c_m=evoformer.c_m,
            c_z=evoformer.c_z,
        )
        self.num_recycles = num_recycles

    def _single_pass(
        self,
        msa_features: Tensor,
        prev_ca: Tensor,
        prev_pair: Tensor,
        prev_msa_row: Tensor,
    ) -> tuple[EvoformerOutput, StructureOutput]:
        """One recycled Evoformer pass followed by structure head prediction."""
        B, N, L, _ = msa_features.shape

        # 1. Project raw MSA features to model dimension
        msa = self.evoformer.input_proj(msa_features)           # (B, N, L, c_m)
        pair = msa_features.new_zeros(B, L, L, self.evoformer.c_z)

        # 2. Inject recycling corrections
        msa_delta, pair_delta = self.recycling_embedder(prev_ca, prev_pair, prev_msa_row)
        msa = msa.clone()
        msa[:, 0, :, :] = msa[:, 0, :, :] + msa_delta          # first row only
        pair = pair + pair_delta

        # 3. Run Evoformer blocks
        for block in self.evoformer.blocks:
            msa, pair = block(msa, pair)

        evo_out = EvoformerOutput(msa_repr=msa, pair_repr=pair)

        # 4. Run structure heads directly on the query representation
        h = evo_out.query_repr                                   # (B, L, c_m)
        ca_coords = self.structure_module.coord_head(h)          # (B, L, 3)
        rotations: Optional[Tensor] = (
            self.structure_module.frame_head(h)
            if self.structure_module.frame_head is not None
            else None
        )
        struct_out = StructureOutput(ca_coords=ca_coords, rotations=rotations)

        return evo_out, struct_out

    def forward(
        self,
        msa_features: Tensor,
    ) -> tuple[EvoformerOutput, StructureOutput]:
        """
        Args:
            msa_features: (B, N, L, feat_dim) raw MSA feature tensor.

        Returns:
            evo_out:    ``EvoformerOutput`` from the final recycle pass.
            struct_out: ``StructureOutput`` from the final recycle pass.
        """
        B, N, L, _ = msa_features.shape
        c_m = self.evoformer.c_m
        c_z = self.evoformer.c_z
        device = msa_features.device
        dtype = msa_features.dtype

        # Zero-initialised recycle inputs for the first pass
        prev_ca      = torch.zeros(B, L, 3,       device=device, dtype=dtype)
        prev_pair    = torch.zeros(B, L, L, c_z,  device=device, dtype=dtype)
        prev_msa_row = torch.zeros(B, L, c_m,     device=device, dtype=dtype)

        evo_out: Optional[EvoformerOutput] = None
        struct_out: Optional[StructureOutput] = None

        for recycle_idx in range(self.num_recycles):
            is_last = recycle_idx == self.num_recycles - 1

            if is_last:
                evo_out, struct_out = self._single_pass(
                    msa_features, prev_ca, prev_pair, prev_msa_row
                )
            else:
                with torch.no_grad():
                    evo_out, struct_out = self._single_pass(
                        msa_features, prev_ca, prev_pair, prev_msa_row
                    )

            # Detach to prevent gradient flow through intermediate passes
            prev_ca      = struct_out.ca_coords.detach()
            prev_pair    = evo_out.pair_repr.detach()
            prev_msa_row = evo_out.query_repr.detach()

        assert evo_out is not None and struct_out is not None
        return evo_out, struct_out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    from src.models.evoformer import build_evoformer
    from src.models.structure_module import build_msa_structure_model

    print("=" * 60)
    print("RecyclingEmbedder + RecycledEvoformer — smoke test")
    print("=" * 60)

    torch.manual_seed(42)
    B, N, L = 2, 4, 20
    C_M, C_Z = 32, 16
    from src.data.msa_encoder import FEATURE_DIM as FEAT_DIM

    # Build components
    evoformer = build_evoformer(n_blocks=1, c_m=C_M, c_z=C_Z, nhead=4, c_outer=8, c_t=8)
    struct_mod = build_msa_structure_model(n_blocks=1, c_m=C_M, c_z=C_Z,
                                           nhead=4, c_outer=8, c_t=8)

    msa_feat = torch.randn(B, N, L, FEAT_DIM)

    # 1. RecyclingEmbedder standalone
    embedder = RecyclingEmbedder(c_m=C_M, c_z=C_Z)
    prev_ca      = torch.zeros(B, L, 3)
    prev_pair    = torch.zeros(B, L, L, C_Z)
    prev_msa_row = torch.randn(B, L, C_M)
    msa_delta, pair_delta = embedder(prev_ca, prev_pair, prev_msa_row)
    assert msa_delta.shape  == (B, L, C_M),      f"msa_delta shape: {msa_delta.shape}"
    assert pair_delta.shape == (B, L, L, C_Z),   f"pair_delta shape: {pair_delta.shape}"
    print(f"[1] RecyclingEmbedder: msa_delta {msa_delta.shape}, pair_delta {pair_delta.shape}  (PASS)")

    # 2. RecycledEvoformer forward — 3 recycles
    recycled = RecycledEvoformer(evoformer=evoformer, structure_module=struct_mod, num_recycles=3)
    evo_out, struct_out = recycled(msa_feat)
    assert evo_out.msa_repr.shape  == (B, N, L, C_M)
    assert evo_out.pair_repr.shape == (B, L, L, C_Z)
    assert struct_out.ca_coords.shape == (B, L, 3)
    assert struct_out.rotations is not None and struct_out.rotations.shape == (B, L, 3, 3)
    print(f"[2] RecycledEvoformer(num_recycles=3): "
          f"msa {evo_out.msa_repr.shape}, pair {evo_out.pair_repr.shape}, "
          f"ca {struct_out.ca_coords.shape}  (PASS)")

    # 3. Gradient flows through final pass only
    loss = struct_out.ca_coords.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in recycled.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients found"
    assert all(math.isfinite(g) for g in grad_norms), "NaN/Inf gradients"
    print(f"[3] Gradients: {len(grad_norms)} params, max={max(grad_norms):.4f}  (PASS)")

    # 4. num_recycles=1 (no intermediate passes)
    recycled1 = RecycledEvoformer(evoformer=evoformer, structure_module=struct_mod, num_recycles=1)
    evo1, s1 = recycled1(msa_feat)
    assert s1.ca_coords.shape == (B, L, 3)
    print(f"[4] num_recycles=1: ca_coords {s1.ca_coords.shape}  (PASS)")

    print("\nAll tests passed.")
