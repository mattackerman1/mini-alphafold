"""
Loss functions for contact map prediction.

All losses accept a boolean mask that identifies valid (i, j) pairs to
include in the loss computation.  Pairs involving PAD tokens, the diagonal,
and optionally short-range neighbours (|i-j| < sep_min) should be masked out.

Public API
----------
  contact_bce_loss(logits, targets, mask)          — binary cross-entropy
  focal_loss(logits, targets, mask, gamma, alpha)  — focal loss (class-imbalance)
  contact_loss(logits, targets, mask, kind, ...)   — dispatcher
  make_contact_mask(lengths, L, sep_min, device)   — build the valid-pair mask
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------

def make_contact_mask(
    lengths: Tensor,
    L: int,
    sep_min: int = 6,
    device: torch.device | None = None,
) -> Tensor:
    """Build a boolean mask for valid contact pairs in a padded batch.

    A pair (i, j) is valid when:
      1. Both residues are non-PAD  (i < length_b and j < length_b)
      2. |i - j| >= sep_min        (exclude trivially local contacts)
      3. i != j                    (diagonal — implied by sep_min >= 1)

    Args:
        lengths: (batch,) int tensor — actual sequence length per sample.
        L:       Padded sequence length (second dim of the logit tensor).
        sep_min: Minimum sequence separation to include (default 6,
                 so short-range secondary structure contacts are excluded).
        device:  Target device; defaults to lengths.device.

    Returns:
        mask: (batch, L, L) bool tensor — True where the loss should be computed.
    """
    if device is None:
        device = lengths.device
    B = lengths.size(0)

    # Position indices
    idx = torch.arange(L, device=device)
    # Residue-in-sequence mask: (B, L)
    seq_mask = idx.unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)

    # Pairwise validity: both i and j must be in-sequence
    pair_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)  # (B, L, L)

    # Separation mask: |i - j| >= sep_min
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()           # (L, L)
    sep_mask = sep.unsqueeze(0) >= sep_min                       # (1, L, L)

    return pair_mask & sep_mask   # (B, L, L)


# ---------------------------------------------------------------------------
# Binary cross-entropy loss
# ---------------------------------------------------------------------------

def contact_bce_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """Masked binary cross-entropy loss for contact map prediction.

    Args:
        logits:  (B, L, L) raw logits (pre-sigmoid).
        targets: (B, L, L) float tensor with values in {0.0, 1.0}.
        mask:    (B, L, L) bool tensor — True for valid pairs.

    Returns:
        Scalar mean loss over valid pairs.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0   # differentiable zero

    loss_per_pair = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )                                              # (B, L, L)
    return loss_per_pair[mask].mean()


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Masked focal loss for class-imbalanced contact prediction.

    Contact maps are sparse (~5% positive pairs), so focal loss down-weights
    easy negatives and focuses training on hard examples.

    Args:
        logits:  (B, L, L) raw logits.
        targets: (B, L, L) float tensor in {0.0, 1.0}.
        mask:    (B, L, L) bool — valid pairs.
        gamma:   Focusing parameter (default 2.0).
        alpha:   Weighting for positive class (default 0.25).

    Returns:
        Scalar mean focal loss over valid pairs.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0

    targets_f = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
    p_t = torch.exp(-bce)                        # probability of correct class
    alpha_t = alpha * targets_f + (1 - alpha) * (1 - targets_f)
    loss_per_pair = alpha_t * (1 - p_t) ** gamma * bce
    return loss_per_pair[mask].mean()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

LossKind = Literal["bce", "focal"]


def contact_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    kind: LossKind = "bce",
    **kwargs,
) -> Tensor:
    """Compute contact map loss by name.

    Args:
        logits:   (B, L, L) raw logits.
        targets:  (B, L, L) float tensor in {0.0, 1.0}.
        mask:     (B, L, L) bool — valid pairs.
        kind:     "bce" or "focal".
        **kwargs: Forwarded to the chosen loss (e.g. gamma=2.0 for focal).

    Returns:
        Scalar loss.
    """
    if kind == "bce":
        return contact_bce_loss(logits, targets, mask)
    if kind == "focal":
        return focal_loss(logits, targets, mask, **kwargs)
    raise ValueError(f"Unknown loss kind {kind!r}. Choose 'bce' or 'focal'.")
