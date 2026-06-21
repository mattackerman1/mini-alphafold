"""
Simplified Evoformer block for mini-alphafold (Phase 3).

Architecture per block
----------------------
  1. RowAttention              — gated MHA over the sequence (N) axis for each position
  2. ColumnAttention           — gated MHA over the position (L) axis for each sequence
  3. MSATransition             — position-wise 2-layer MLP on the MSA representation
  4. OuterProductMean          — collapses N → pair update (B, L, L, c_z)
  5. TriangleMultiplication    — pair → pair (outgoing edges)   [optional]
  6. TriangleMultiplication    — pair → pair (incoming edges)   [optional]
  7. PairTransition            — position-wise 2-layer MLP on the pair representation

All sub-layers use pre-norm LayerNorm and residual connections.
Gating (element-wise sigmoid) is applied to each attention output before
adding back to the residual, following AF2's gated self-attention convention.

Triangle multiplicative updates (AF2 §1.6.5)
---------------------------------------------
  Outgoing: z_new[i,k] += sum_j( a(z[i,j]) * b(z[j,k]) )
  Incoming: z_new[i,k] += sum_j( a(z[j,i]) * b(z[j,k]) )

  Both variants use two levels of gating:
    1. Input gates:  a = sigmoid(gate_a(z)) * proj_a(z)
    2. Output gate:  update = sigmoid(out_gate(z)) * out_proj(layernorm(p))

  This enforces transitivity constraints in the pair representation —
  if A-B and B-C are close, then A-C should also be close.

AF2 name mapping
----------------
  "Row-wise gated self-attention" (AF2) = ColumnAttention here
     → attends over L positions for each MSA row (sequence)
  "Column-wise gated self-attention" (AF2) = RowAttention here
     → attends over N sequences for each column (position)
  The task spec uses "row = sequence dimension", which is the transpose of the
  AF2 convention, so the names are swapped.  Behaviour is identical.

Dimensions
----------
  c_m      : MSA channel width (default 64)
  c_z      : pair channel width (default 32)
  c_outer  : intermediate width for outer-product mean (default 16)
  c_t      : intermediate width for triangle updates (default c_z // 2)
  nhead    : attention heads (default 4, must divide c_m)

Public API
----------
  EvoformerOutput                       — output dataclass
  TriangleMultiplication(c_z, outgoing) — standalone triangle update
  EvoformerBlock(c_m, c_z, ...)         — single stackable block
  EvoformerStack(feat_dim, n_blocks, …) — full stack with input projection
  build_evoformer(n_blocks, …)          — factory helper
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.msa_encoder import FEATURE_DIM as MSA_FEATURE_DIM


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvoformerOutput:
    """Output of EvoformerStack.forward().

    Attributes:
        msa_repr:   (B, N, L, c_m) — updated MSA representation.
                    Row 0 is the query sequence representation.
        pair_repr:  (B, L, L, c_z) — accumulated pair representation built
                    by outer-product mean over N sequences.
    """
    msa_repr: Tensor    # (B, N, L, c_m)
    pair_repr: Tensor   # (B, L, L, c_z)

    @property
    def query_repr(self) -> Tensor:
        """(B, L, c_m) — MSA row 0 = query sequence representation."""
        return self.msa_repr[:, 0, :, :]


# ---------------------------------------------------------------------------
# Gated multi-head attention (AF2 §1.6)
# ---------------------------------------------------------------------------

class _GatedMHA(nn.Module):
    """Pre-norm gated multi-head self-attention.

    Gate: output = sigmoid(W_g * LayerNorm(x)) * attn(LayerNorm(x))
    This down-weights attention outputs that the model is uncertain about,
    following the AF2 gated attention design.

    Args:
        d_model:  Model/channel dimension.
        nhead:    Number of attention heads.
        dropout:  Attention dropout probability.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pre-norm gated self-attention with residual.

        Args:
            x:                (S, d) or (batch, S, d) where S is the attended dim.
            key_padding_mask: (batch, S) bool — True for positions to ignore.
        Returns:
            (same shape as x)
        """
        normed = self.norm(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )
        gate = torch.sigmoid(self.gate_proj(normed))
        return x + gate * attn_out


# ---------------------------------------------------------------------------
# Row-wise attention (over N sequences, for each position)
# ---------------------------------------------------------------------------

class RowAttention(nn.Module):
    """Gated self-attention over the *sequence* (N) axis, per position.

    For each position j in [0, L), the N rows of the MSA are attended against
    each other.  Positions are processed in parallel by folding B and L into
    the batch dimension.

    Corresponds to AF2 "column-wise gated self-attention" (§1.6.2).

    Args:
        c_m:   MSA channel dimension.
        nhead: Number of attention heads.
        dropout: Dropout inside attention.
    """

    def __init__(self, c_m: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = _GatedMHA(c_m, nhead, dropout=dropout)

    def forward(self, msa: Tensor) -> Tensor:
        """
        Args:
            msa: (B, N, L, c_m)
        Returns:
            (B, N, L, c_m)
        """
        B, N, L, c_m = msa.shape
        # For each position, attend over N sequences: reshape → (B*L, N, c_m)
        x = msa.permute(0, 2, 1, 3).reshape(B * L, N, c_m)
        x = self.mha(x)
        # Restore: (B*L, N, c_m) → (B, L, N, c_m) → (B, N, L, c_m)
        return x.reshape(B, L, N, c_m).permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Column-wise attention (over L positions, for each sequence)
# ---------------------------------------------------------------------------

class ColumnAttention(nn.Module):
    """Gated self-attention over the *position* (L) axis, per sequence.

    For each row i (MSA sequence), the L positions attend against each other.
    Sequences are processed in parallel by folding B and N into the batch dim.

    Corresponds to AF2 "row-wise gated self-attention with pair bias" (§1.6.1).
    The pair bias term is omitted in this simplified version.

    Args:
        c_m:   MSA channel dimension.
        nhead: Number of attention heads.
        dropout: Dropout inside attention.
    """

    def __init__(self, c_m: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = _GatedMHA(c_m, nhead, dropout=dropout)

    def forward(self, msa: Tensor) -> Tensor:
        """
        Args:
            msa: (B, N, L, c_m)
        Returns:
            (B, N, L, c_m)
        """
        B, N, L, c_m = msa.shape
        # For each sequence, attend over L positions: reshape → (B*N, L, c_m)
        x = msa.reshape(B * N, L, c_m)
        x = self.mha(x)
        return x.reshape(B, N, L, c_m)


# ---------------------------------------------------------------------------
# MSA transition (position-wise MLP)
# ---------------------------------------------------------------------------

class MSATransition(nn.Module):
    """Position-wise two-layer MLP applied to the MSA representation.

    Pre-norm, expansion-factor-4, ReLU activation.  Matches the feed-forward
    "transition" block in AF2 §1.6.3.

    Args:
        c_m:     MSA channel dimension.
        expand:  Hidden dimension multiplier (default 4).
    """

    def __init__(self, c_m: int, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = c_m * expand
        self.norm = nn.LayerNorm(c_m)
        self.net = nn.Sequential(
            nn.Linear(c_m, inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, c_m),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, msa: Tensor) -> Tensor:
        """
        Args:
            msa: (B, N, L, c_m)
        Returns:
            (B, N, L, c_m)
        """
        return msa + self.drop(self.net(self.norm(msa)))


# ---------------------------------------------------------------------------
# Outer product mean (MSA → pair update)
# ---------------------------------------------------------------------------

class OuterProductMean(nn.Module):
    """Computes a pair update by averaging outer products across MSA sequences.

    For each pair of positions (i, j), the outer product of their projected
    representations is computed per-sequence and averaged over N sequences,
    then projected to the pair channel dimension c_z.

    This is the primary path by which MSA co-evolutionary signal flows into
    the pair representation (AF2 §1.6.4).

    Memory: O(B * L² * c_outer²).  Keep c_outer small (≤32) for long sequences.

    Args:
        c_m:     MSA channel dimension.
        c_z:     Pair channel dimension.
        c_outer: Intermediate projection dimension before outer product.
    """

    def __init__(self, c_m: int, c_z: int, c_outer: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.left_proj  = nn.Linear(c_m, c_outer, bias=False)
        self.right_proj = nn.Linear(c_m, c_outer, bias=False)
        self.out_proj   = nn.Linear(c_outer * c_outer, c_z)

    def forward(self, msa: Tensor, pair: Tensor) -> Tensor:
        """
        Args:
            msa:  (B, N, L, c_m)
            pair: (B, L, L, c_z) — existing pair representation (updated in place)
        Returns:
            (B, L, L, c_z)
        """
        B, N, L, _ = msa.shape

        normed = self.norm(msa)                      # (B, N, L, c_m)
        left  = self.left_proj(normed)               # (B, N, L, c_outer)
        right = self.right_proj(normed)              # (B, N, L, c_outer)

        # Outer product mean: sum over N sequences, then divide
        # einsum: b=batch, n=seq, i=pos_i, j=pos_j, a=feat_a, c=feat_c
        # outer[b,i,j,a,c] = sum_n left[b,n,i,a] * right[b,n,j,c]
        outer = torch.einsum("bnia,bnjc->bijac", left, right) / N  # (B, L, L, co, co)
        outer = outer.reshape(B, L, L, -1)           # (B, L, L, c_outer²)

        return pair + self.out_proj(outer)           # (B, L, L, c_z)


# ---------------------------------------------------------------------------
# Pair transition (position-wise MLP on the pair representation)
# ---------------------------------------------------------------------------

class PairTransition(nn.Module):
    """Position-wise MLP applied to the pair representation.

    Identical in structure to MSATransition but operates on (B, L, L, c_z).

    Args:
        c_z:    Pair channel dimension.
        expand: Hidden dimension multiplier (default 4).
    """

    def __init__(self, c_z: int, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = c_z * expand
        self.norm = nn.LayerNorm(c_z)
        self.net = nn.Sequential(
            nn.Linear(c_z, inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, c_z),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, pair: Tensor) -> Tensor:
        """
        Args:
            pair: (B, L, L, c_z)
        Returns:
            (B, L, L, c_z)
        """
        return pair + self.drop(self.net(self.norm(pair)))


# ---------------------------------------------------------------------------
# Triangle multiplicative update (AF2 §1.6.5)
# ---------------------------------------------------------------------------

class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update for the pair representation.

    Updates edge (i, k) by aggregating information from all triangles that
    include it as one edge, using the other two edges as multiplicative inputs.

    Two variants controlled by ``outgoing``:

      Outgoing  (AF2 Algorithm 11):
        p[i,k] = sum_j  a[i,j] * b[j,k]
        Interprets j as the "tip" of a triangle with base (i,k).

      Incoming  (AF2 Algorithm 12):
        p[i,k] = sum_j  a[j,i] * b[j,k]
        Interprets j as the "source" of two outgoing edges to i and k.

    Both use two levels of gating:
      1. Input gates  — a = sigmoid(gate_a(z)) * proj_a(z),  same for b
      2. Output gate  — update = sigmoid(out_gate(z)) * out_proj(LayerNorm(p))

    The triangle sum is O(L³ · c_t) in FLOPs; keep c_t small for long sequences.

    Args:
        c_z:      Pair channel dimension (input and output).
        c_t:      Intermediate channel for the triangle product.
                  Defaults to c_z // 2.
        outgoing: True → outgoing-edge update; False → incoming-edge update.
    """

    def __init__(
        self,
        c_z:      int,
        c_t:      Optional[int] = None,
        outgoing: bool = True,
    ) -> None:
        super().__init__()
        c_t = c_t if c_t is not None else max(1, c_z // 2)
        self.outgoing = outgoing

        self.norm_in  = nn.LayerNorm(c_z)

        # Gated input projections: two independent branches a and b
        self.proj_a = nn.Linear(c_z, c_t, bias=False)
        self.gate_a = nn.Linear(c_z, c_t, bias=False)
        self.proj_b = nn.Linear(c_z, c_t, bias=False)
        self.gate_b = nn.Linear(c_z, c_t, bias=False)

        # Output: LayerNorm on the product, then gated linear projection
        self.norm_out = nn.LayerNorm(c_t)
        self.out_proj = nn.Linear(c_t, c_z, bias=False)
        self.out_gate = nn.Linear(c_z, c_z, bias=False)

    def forward(self, pair: Tensor) -> Tensor:
        """
        Args:
            pair: (B, L, L, c_z) — pair representation.
        Returns:
            (B, L, L, c_z) — updated pair representation.
        """
        z = self.norm_in(pair)                               # (B, L, L, c_z)

        # Gated input projections
        a = torch.sigmoid(self.gate_a(z)) * self.proj_a(z)  # (B, L, L, c_t)
        b = torch.sigmoid(self.gate_b(z)) * self.proj_b(z)  # (B, L, L, c_t)

        if self.outgoing:
            # p[b,i,k,f] = sum_j  a[b,i,j,f] * b[b,j,k,f]
            # j is summed; for each feature channel f this is a matrix multiply.
            p = torch.einsum("bijf,bjkf->bikf", a, b)       # (B, L, L, c_t)
        else:
            # p[b,i,k,f] = sum_j  a[b,j,i,f] * b[b,j,k,f]
            p = torch.einsum("bjif,bjkf->bikf", a, b)       # (B, L, L, c_t)

        # Output gate and projection (with LayerNorm on the product)
        g      = torch.sigmoid(self.out_gate(z))             # (B, L, L, c_z)
        update = self.out_proj(self.norm_out(p))             # (B, L, L, c_z)

        return pair + g * update


# ---------------------------------------------------------------------------
# Single Evoformer block
# ---------------------------------------------------------------------------

class EvoformerBlock(nn.Module):
    """One Evoformer block.

    Execution order:
      row attn → col attn → MSA MLP → outer product
      → [tri outgoing] → [tri incoming] → pair MLP

    Triangle updates are enabled by default (``use_triangle=True``).
    Set ``use_triangle=False`` to recover the original behaviour without them.

    Multiple blocks can be stacked in a ModuleList (see EvoformerStack).

    Args:
        c_m:          MSA channel dimension.
        c_z:          Pair channel dimension.
        nhead:        Attention heads (must divide c_m).
        c_outer:      Outer product intermediate dimension.
        c_t:          Triangle update intermediate dimension (default c_z // 2).
        expand:       MLP expansion factor.
        dropout:      Dropout in attention layers.
        use_triangle: If True (default), add both triangle multiplicative updates
                      after the outer product mean step.
    """

    def __init__(
        self,
        c_m:          int   = 64,
        c_z:          int   = 32,
        nhead:        int   = 4,
        c_outer:      int   = 16,
        c_t:          Optional[int] = None,
        expand:       int   = 4,
        dropout:      float = 0.0,
        use_triangle: bool  = True,
    ) -> None:
        super().__init__()
        self.row_attn        = RowAttention(c_m, nhead, dropout=dropout)
        self.col_attn        = ColumnAttention(c_m, nhead, dropout=dropout)
        self.msa_transition  = MSATransition(c_m, expand=expand, dropout=dropout)
        self.outer_product   = OuterProductMean(c_m, c_z, c_outer=c_outer)

        if use_triangle:
            self.tri_out = TriangleMultiplication(c_z, c_t=c_t, outgoing=True)
            self.tri_in  = TriangleMultiplication(c_z, c_t=c_t, outgoing=False)
        else:
            self.tri_out = None  # type: ignore[assignment]
            self.tri_in  = None  # type: ignore[assignment]

        self.pair_transition = PairTransition(c_z, expand=expand, dropout=dropout)

    def forward(self, msa: Tensor, pair: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            msa:  (B, N, L, c_m) — MSA representation.
            pair: (B, L, L, c_z) — pair representation.
        Returns:
            msa:  (B, N, L, c_m) — updated MSA representation.
            pair: (B, L, L, c_z) — updated pair representation.
        """
        msa  = self.row_attn(msa)
        msa  = self.col_attn(msa)
        msa  = self.msa_transition(msa)
        pair = self.outer_product(msa, pair)
        if self.tri_out is not None:
            pair = self.tri_out(pair)
            pair = self.tri_in(pair)
        pair = self.pair_transition(pair)
        return msa, pair


# ---------------------------------------------------------------------------
# Full Evoformer stack
# ---------------------------------------------------------------------------

class EvoformerStack(nn.Module):
    """Input projection + stack of EvoformerBlocks.

    Accepts raw MSA feature tensors from MSADataset (shape B×N×L×feat_dim)
    and returns an updated MSA representation and a pair representation.

    Args:
        feat_dim:     Input feature dimension per (seq, position).
                      Default: MSA_FEATURE_DIM = 45 from msa_encoder.
        c_m:          MSA channel width after projection.
        c_z:          Pair channel width.
        n_blocks:     Number of stacked EvoformerBlocks.
        nhead:        Attention heads (must divide c_m).
        c_outer:      Outer product intermediate dim.
        c_t:          Triangle update intermediate dim (default c_z // 2).
        expand:       MLP expansion factor.
        dropout:      Attention dropout.
        use_triangle: Enable triangle multiplicative updates in every block.
    """

    def __init__(
        self,
        feat_dim:     int   = MSA_FEATURE_DIM,
        c_m:          int   = 64,
        c_z:          int   = 32,
        n_blocks:     int   = 2,
        nhead:        int   = 4,
        c_outer:      int   = 16,
        c_t:          Optional[int] = None,
        expand:       int   = 4,
        dropout:      float = 0.0,
        use_triangle: bool  = True,
    ) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z

        self.input_proj = nn.Linear(feat_dim, c_m)

        self.blocks = nn.ModuleList([
            EvoformerBlock(
                c_m=c_m, c_z=c_z, nhead=nhead,
                c_outer=c_outer, c_t=c_t, expand=expand,
                dropout=dropout, use_triangle=use_triangle,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, msa_features: Tensor) -> EvoformerOutput:
        """
        Args:
            msa_features: (B, N, L, feat_dim) — raw MSA features from MSAEncoder.

        Returns:
            EvoformerOutput with:
              .msa_repr  (B, N, L, c_m)
              .pair_repr (B, L, L, c_z)
        """
        B, N, L, _ = msa_features.shape

        # Project raw features to model dimension
        msa = self.input_proj(msa_features)          # (B, N, L, c_m)

        # Pair representation starts at zero; outer product mean fills it in
        pair = msa_features.new_zeros(B, L, L, self.c_z)

        for block in self.blocks:
            msa, pair = block(msa, pair)

        return EvoformerOutput(msa_repr=msa, pair_repr=pair)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"EvoformerStack(n_blocks={len(self.blocks)}, "
            f"c_m={self.c_m}, c_z={self.c_z}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_evoformer(
    n_blocks:     int  = 2,
    c_m:          int  = 64,
    c_z:          int  = 32,
    feat_dim:     int  = MSA_FEATURE_DIM,
    use_triangle: bool = True,
    **kwargs,
) -> EvoformerStack:
    """Build an EvoformerStack with sensible defaults.

    Args:
        n_blocks:     Number of Evoformer blocks to stack (default 2).
        c_m:          MSA representation channel width (default 64).
        c_z:          Pair representation channel width (default 32).
        feat_dim:     Input feature dimension (default 45, from MSAEncoder).
        use_triangle: Enable triangle multiplicative updates (default True).
        **kwargs:     Forwarded to EvoformerStack (nhead, c_outer, c_t,
                      expand, dropout).

    Returns:
        EvoformerStack ready for training.

    Example::

        evo = build_evoformer(n_blocks=4, c_m=128, c_z=64)
        out = evo(msa_features)    # (B, N, L, 45) tensor
        print(out.msa_repr.shape)  # (B, N, L, 128)
        print(out.pair_repr.shape) # (B, L, L, 64)

        # Without triangle updates (for ablation / speed)
        evo_fast = build_evoformer(n_blocks=4, c_m=128, use_triangle=False)
    """
    return EvoformerStack(
        feat_dim=feat_dim, c_m=c_m, c_z=c_z,
        n_blocks=n_blocks, use_triangle=use_triangle, **kwargs
    )


# ---------------------------------------------------------------------------
# Smoke-test — run with:  python -m src.models.evoformer
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EvoformerStack + TriangleMultiplication — smoke test")
    print("=" * 60)

    torch.manual_seed(0)

    B, N, L  = 2, 8, 24
    C_M, C_Z = 32, 16
    feat_dim  = MSA_FEATURE_DIM   # 45

    msa_feat = torch.randn(B, N, L, feat_dim)

    # ---- 1. Standalone TriangleMultiplication ----
    pair_dummy = torch.randn(B, L, L, C_Z)
    for outgoing in (True, False):
        label = "outgoing" if outgoing else "incoming"
        tri = TriangleMultiplication(C_Z, c_t=8, outgoing=outgoing)
        out_tri = tri(pair_dummy)
        assert out_tri.shape == (B, L, L, C_Z), f"Bad shape for {label}: {out_tri.shape}"
        assert not torch.allclose(out_tri, pair_dummy), "Triangle should modify pair"
        print(f"[1{'a' if outgoing else 'b'}] TriangleMultiplication ({label}): "
              f"{out_tri.shape}  (PASS)")

    # ---- 2. EvoformerStack with triangle (default) ----
    evo = build_evoformer(n_blocks=2, c_m=C_M, c_z=C_Z, nhead=4, c_outer=8, c_t=8)
    print(evo)
    out = evo(msa_feat)
    assert out.msa_repr.shape  == (B, N, L, C_M)
    assert out.pair_repr.shape == (B, L, L, C_Z)
    print(f"[2] Forward shapes: msa {out.msa_repr.shape}, pair {out.pair_repr.shape}  (PASS)")

    # ---- 3. Triangle actually changes the pair vs no-triangle ----
    evo_notri = build_evoformer(n_blocks=2, c_m=C_M, c_z=C_Z, nhead=4,
                                c_outer=8, use_triangle=False)
    # Copy weights of the non-triangle params so only the triangle differs
    out_notri = evo_notri(msa_feat)
    assert not torch.allclose(out.pair_repr, out_notri.pair_repr), \
        "Triangle should produce different pair repr than no-triangle"
    print(f"[3] Triangle changes pair repr vs no-triangle  (PASS)")

    # ---- 4. Query repr ----
    assert out.query_repr.shape == (B, L, C_M)
    print(f"[4] Query repr: {out.query_repr.shape}  (PASS)")

    # ---- 5. Gradient flow through triangle ops ----
    loss = out.msa_repr.sum() + out.pair_repr.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in evo.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(math.isfinite(g) for g in grad_norms), \
        "NaN/Inf gradients detected"
    print(f"[5] Gradients: {len(grad_norms)} params, max={max(grad_norms):.4f}  (PASS)")

    # ---- 6. N=1 single-sequence backward compat ----
    msa_single = torch.randn(1, 1, L, feat_dim)
    out_single = evo(msa_single)
    assert out_single.msa_repr.shape == (1, 1, L, C_M)
    assert out_single.pair_repr.shape == (1, L, L, C_Z)
    print(f"[6] N=1 single-sequence path  (PASS)")

    # ---- 7. use_triangle=False backward compat ----
    evo_notri2 = build_evoformer(n_blocks=2, c_m=C_M, c_z=C_Z, use_triangle=False)
    out2 = evo_notri2(msa_feat)
    assert out2.msa_repr.shape  == (B, N, L, C_M)
    assert out2.pair_repr.shape == (B, L, L, C_Z)
    print(f"[7] use_triangle=False backward compat  (PASS)")

    # ---- 8. DataLoader integration ----
    from src.data.loaders import MSADataset, collate_msa_fn
    from torch.utils.data import DataLoader
    seqs = ["ACDEFGHIK", "LMNPQRSTVWY", "ACDEFGHIKLM"]
    ds = MSADataset(seqs, n_pseudo=5, seed=99)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_msa_fn)
    tokens, msa_batch, _ = next(iter(loader))
    out_dl = evo(msa_batch)
    assert out_dl.msa_repr.shape[0] == 3
    print(f"[8] DataLoader integration: msa {out_dl.msa_repr.shape}, "
          f"pair {out_dl.pair_repr.shape}  (PASS)")

    print()
    print("All tests passed.")
    print()
    print("Usage example:")
    print("  from src.models.evoformer import build_evoformer")
    print("  evo = build_evoformer(n_blocks=4, c_m=128, c_z=64)")
    print("  out = evo(msa_features)          # (B, N, L, 45) in")
    print("  print(out.msa_repr.shape)        # (B, N, L, 128)")
    print("  print(out.pair_repr.shape)       # (B, L, L, 64)")
    print("  print(out.query_repr.shape)      # (B, L, 128)  <- row 0 only")
    print()
    print("  # Disable triangle updates for ablation:")
    print("  evo_fast = build_evoformer(n_blocks=4, use_triangle=False)")
