"""
Loss functions for structure prediction and contact map prediction.

Contact map losses
------------------
  contact_bce_loss(logits, targets, mask)          — binary cross-entropy
  focal_loss(logits, targets, mask, gamma, alpha)  — focal loss (class-imbalance)
  contact_loss(logits, targets, mask, kind, ...)   — dispatcher
  make_contact_mask(lengths, L, sep_min, device)   — build the valid-pair mask

Structure losses (Phase 2)
--------------------------
  make_backbone_frames(backbone_coords)            — true frames from N/CA/C atoms
  fape_loss(pred_t, true_t, pred_R, true_R, mask)  — Frame-Aligned Point Error
  backbone_rmsd(pred_ca, true_ca, mask)            — per-batch RMSD (no alignment)
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


# ===========================================================================
# Structure losses (Phase 2)
# ===========================================================================

# Atom indices in backbone_coords tensor (matches pdb_parser.BACKBONE_ATOMS)
_N_IDX, _CA_IDX, _C_IDX = 0, 1, 2


def make_backbone_frames(
    backbone_coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Construct local backbone frames from N, Cα, C atom positions.

    The local frame for residue i is defined by:
      e1 = normalize(N_i  − Cα_i)        — points from Cα toward N
      e3 = normalize(e1 × (C_i − Cα_i))  — normal to the backbone plane
      e2 = e3 × e1                        — completes the right-handed frame
      t  = Cα_i                           — origin is Cα

    This convention matches the AF2 backbone frame (Jumper et al. Suppl. §1.8).

    Args:
        backbone_coords: (..., L, 4, 3) float32 — atom order [N, CA, C, O].
                         Leading dimensions may be batch dims.

    Returns:
        rotations:    (..., L, 3, 3) — rotation matrix, columns = [e1, e2, e3].
        translations: (..., L, 3)   — Cα positions.
    """
    N  = backbone_coords[..., _N_IDX,  :]   # (..., L, 3)
    CA = backbone_coords[..., _CA_IDX, :]   # (..., L, 3)
    C  = backbone_coords[..., _C_IDX,  :]   # (..., L, 3)

    v1 = N - CA                                       # N → Cα direction
    v2 = C - CA                                       # Cα → C direction

    e1 = F.normalize(v1, dim=-1)
    e3 = F.normalize(torch.cross(e1, v2, dim=-1), dim=-1)
    e2 = torch.cross(e3, e1, dim=-1)

    rotations = torch.stack([e1, e2, e3], dim=-1)    # (..., L, 3, 3) columns
    return rotations, CA


def fape_loss(
    pred_t: Tensor,
    true_t: Tensor,
    pred_R: Tensor,
    true_R: Tensor,
    seq_mask: Tensor,
    d_clamp: float = 10.0,
    eps: float = 1e-6,
    z: float = 10.0,
) -> Tensor:
    """Frame-Aligned Point Error (FAPE) loss.

    For each residue i (the "frame"), transform all Cα positions j into
    residue i's local coordinate system — in both prediction and ground truth —
    then measure the distance between the two transformed points.

    FAPE_i = (1/L) * Σ_j  clamp(||R_i^T(x_j−t_i) − R_i^{*T}(x_j^*−t_i^*)||, d_clamp)

    Averaged over all valid frames i and points j, then divided by z (the
    AF2 normalisation constant, default 10 Å) to give dimensionless loss values.

    FAPE is invariant to global rotation and translation of the predicted
    structure (as long as pred_R transforms consistently with pred_t).

    Args:
        pred_t:   (B, L, 3) — predicted Cα positions.
        true_t:   (B, L, 3) — true Cα positions.
        pred_R:   (B, L, 3, 3) — predicted rotation matrices.
        true_R:   (B, L, 3, 3) — true rotation matrices (from make_backbone_frames).
        seq_mask: (B, L) bool — True for non-PAD residues.
        d_clamp:  Maximum per-pair error contributing to the loss (Å).
        eps:      Small constant added before sqrt for numerical stability.
        z:        Normalisation constant (default 10 Å, as in AF2).

    Returns:
        Scalar FAPE loss.
    """
    # --- transform all points j into each frame i ---
    # diff[b, i, j, :] = t_j − t_i    (B, L_frame, L_point, 3)
    diff_pred = pred_t.unsqueeze(2) - pred_t.unsqueeze(1)   # (B, L, L, 3)
    diff_true = true_t.unsqueeze(2) - true_t.unsqueeze(1)   # (B, L, L, 3)

    # Apply R_i^T to each diff vector:
    # local[b,i,j,k] = Σ_m R[b,i,m,k] * diff[b,i,j,m]   (R^T = R transposed on last 2 dims)
    local_pred = torch.einsum("bimk,bijm->bijk", pred_R, diff_pred)  # (B, L, L, 3)
    local_true = torch.einsum("bimk,bijm->bijk", true_R, diff_true)  # (B, L, L, 3)

    # Per-pair distance error (with numerical stabilisation)
    sq_err = (local_pred - local_true).pow(2).sum(dim=-1)    # (B, L, L)
    err = torch.sqrt(sq_err + eps)                           # (B, L, L)
    clamped = torch.clamp(err, max=d_clamp)                  # (B, L, L)

    # Valid-pair mask: both frame i and point j must be non-PAD
    pair_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)  # (B, L, L)

    if pair_mask.sum() == 0:
        return pred_t.sum() * 0.0   # differentiable zero

    return clamped[pair_mask].mean() / z


def backbone_rmsd(
    pred_ca: Tensor,
    true_ca: Tensor,
    seq_mask: Tensor,
) -> Tensor:
    """Per-sample Cα RMSD, averaged over the batch.

    Note: this is *unaligned* RMSD — use Kabsch alignment in evaluation code
    for a meaningful structural metric.  During training, FAPE is preferred.

    Args:
        pred_ca:  (B, L, 3)
        true_ca:  (B, L, 3)
        seq_mask: (B, L) bool

    Returns:
        Scalar mean RMSD (Å) over the batch.
    """
    sq = (pred_ca - true_ca).pow(2).sum(dim=-1)   # (B, L)
    rmsds = []
    for b in range(pred_ca.size(0)):
        m = seq_mask[b]
        if m.sum() > 0:
            rmsds.append(sq[b][m].mean().sqrt())
    if not rmsds:
        return pred_ca.sum() * 0.0
    return torch.stack(rmsds).mean()


