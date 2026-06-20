"""Tests for src/training/losses.py — contact map and structure losses."""

import math
import torch
import pytest

from src.training.losses import (
    contact_bce_loss,
    focal_loss,
    contact_loss,
    make_contact_mask,
    fape_loss,
    backbone_rmsd,
    make_backbone_frames,
)


# ---------------------------------------------------------------------------
# make_contact_mask
# ---------------------------------------------------------------------------

def test_make_contact_mask_shape():
    """make_contact_mask returns (B, L, L) bool tensor."""
    lengths = torch.tensor([8, 5])
    mask = make_contact_mask(lengths, L=10, sep_min=6)
    assert mask.shape == (2, 10, 10)
    assert mask.dtype == torch.bool


def test_make_contact_mask_excludes_pad():
    """Positions beyond sequence length must be masked out."""
    lengths = torch.tensor([5])
    mask = make_contact_mask(lengths, L=10, sep_min=1)
    # Positions 5..9 should all be False
    assert not mask[0, 5:, :].any()
    assert not mask[0, :, 5:].any()


def test_make_contact_mask_excludes_short_range():
    """Pairs with |i-j| < sep_min are excluded."""
    lengths = torch.tensor([20])
    mask = make_contact_mask(lengths, L=20, sep_min=6)
    # Diagonal and near-diagonal must be False
    for i in range(20):
        for j in range(max(0, i - 5), min(20, i + 6)):
            assert not mask[0, i, j].item(), f"({i},{j}) should be masked (sep<6)"


# ---------------------------------------------------------------------------
# contact_bce_loss
# ---------------------------------------------------------------------------

def test_contact_bce_loss_scalar():
    """contact_bce_loss returns a scalar tensor."""
    B, L = 2, 10
    logits = torch.randn(B, L, L)
    targets = torch.randint(0, 2, (B, L, L)).float()
    mask = make_contact_mask(torch.full((B,), L), L=L, sep_min=1)
    loss = contact_bce_loss(logits, targets, mask)
    assert loss.shape == ()
    assert loss.item() >= 0.0
    assert math.isfinite(loss.item())


def test_contact_bce_loss_backward():
    """contact_bce_loss supports backward()."""
    B, L = 2, 8
    logits = torch.randn(B, L, L, requires_grad=True)
    targets = torch.randint(0, 2, (B, L, L)).float()
    mask = make_contact_mask(torch.full((B,), L), L=L, sep_min=1)
    contact_bce_loss(logits, targets, mask).backward()
    assert logits.grad is not None


def test_contact_bce_loss_empty_mask():
    """BCE loss with empty mask returns zero (differentiable)."""
    B, L = 2, 5
    logits = torch.randn(B, L, L, requires_grad=True)
    targets = torch.zeros(B, L, L)
    mask = torch.zeros(B, L, L, dtype=torch.bool)
    loss = contact_bce_loss(logits, targets, mask)
    assert loss.item() == pytest.approx(0.0)
    loss.backward()  # should not error


def test_contact_bce_loss_excludes_pad():
    """BCE loss ignores positions beyond sequence length via mask."""
    B, L = 1, 10
    # Only 5 real residues
    lengths = torch.tensor([5])
    mask = make_contact_mask(lengths, L=L, sep_min=1)
    # Set logits to very large value outside mask — should not affect loss
    logits = torch.zeros(B, L, L)
    logits[:, 5:, :] = 1e6
    logits[:, :, 5:] = 1e6
    targets = torch.zeros(B, L, L)
    loss = contact_bce_loss(logits, targets, mask)
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# focal_loss
# ---------------------------------------------------------------------------

def test_focal_loss_scalar():
    """focal_loss returns a scalar, finite, non-negative value."""
    B, L = 2, 10
    logits = torch.randn(B, L, L)
    targets = torch.randint(0, 2, (B, L, L)).float()
    mask = make_contact_mask(torch.full((B,), L), L=L, sep_min=1)
    loss = focal_loss(logits, targets, mask)
    assert loss.shape == ()
    assert loss.item() >= 0.0
    assert math.isfinite(loss.item())


def test_focal_loss_backward():
    """focal_loss supports backward()."""
    B, L = 2, 8
    logits = torch.randn(B, L, L, requires_grad=True)
    targets = torch.randint(0, 2, (B, L, L)).float()
    mask = make_contact_mask(torch.full((B,), L), L=L, sep_min=1)
    focal_loss(logits, targets, mask).backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# contact_loss dispatcher
# ---------------------------------------------------------------------------

def test_contact_loss_dispatcher_bce():
    """contact_loss(kind='bce') dispatches to BCE."""
    B, L = 2, 8
    logits = torch.randn(B, L, L)
    targets = torch.randint(0, 2, (B, L, L)).float()
    mask = make_contact_mask(torch.full((B,), L), L=L, sep_min=1)
    bce_direct = contact_bce_loss(logits, targets, mask)
    bce_dispatch = contact_loss(logits, targets, mask, kind="bce")
    assert torch.allclose(bce_direct, bce_dispatch)


def test_contact_loss_dispatcher_invalid():
    """contact_loss raises ValueError for unknown kind."""
    with pytest.raises(ValueError):
        contact_loss(torch.zeros(1, 5, 5), torch.zeros(1, 5, 5),
                     torch.ones(1, 5, 5, dtype=torch.bool), kind="invalid")


# ---------------------------------------------------------------------------
# make_backbone_frames
# ---------------------------------------------------------------------------

def test_make_backbone_frames_shapes():
    """make_backbone_frames returns (B, L, 3, 3) rotations and (B, L, 3) translations."""
    B, L = 2, 10
    backbone_coords = torch.randn(B, L, 4, 3)
    rotations, translations = make_backbone_frames(backbone_coords)
    assert rotations.shape == (B, L, 3, 3)
    assert translations.shape == (B, L, 3)


def test_make_backbone_frames_orthogonal():
    """Rotation matrices from backbone frames must be orthogonal (R^T R ≈ I)."""
    B, L = 2, 8
    backbone_coords = torch.randn(B, L, 4, 3)
    rotations, _ = make_backbone_frames(backbone_coords)
    R = rotations.reshape(-1, 3, 3)
    I_approx = R.transpose(-1, -2) @ R
    I = torch.eye(3).unsqueeze(0).expand_as(I_approx)
    assert torch.allclose(I_approx, I, atol=1e-5)


# ---------------------------------------------------------------------------
# fape_loss
# ---------------------------------------------------------------------------

def test_fape_loss_scalar():
    """fape_loss returns a scalar, finite, non-negative value."""
    B, L = 2, 10
    pred_t = torch.randn(B, L, 3)
    true_t = torch.randn(B, L, 3)
    pred_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    true_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    loss = fape_loss(pred_t, true_t, pred_R, true_R, seq_mask)
    assert loss.shape == ()
    assert loss.item() >= 0.0
    assert math.isfinite(loss.item())


def test_fape_loss_zero_on_perfect_prediction():
    """fape_loss is zero (or near-zero) when predictions equal targets."""
    B, L = 2, 8
    coords = torch.randn(B, L, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    loss = fape_loss(coords, coords, R, R, seq_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_fape_loss_not_nan_short_sequence():
    """fape_loss must not produce NaN for very short sequences (L=2)."""
    B, L = 1, 2
    pred_t = torch.randn(B, L, 3)
    true_t = torch.randn(B, L, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    loss = fape_loss(pred_t, true_t, R, R, seq_mask)
    assert math.isfinite(loss.item()), "FAPE loss is NaN for short sequences"


def test_fape_loss_backward():
    """fape_loss supports backward()."""
    B, L = 2, 8
    pred_t = torch.randn(B, L, 3, requires_grad=True)
    true_t = torch.randn(B, L, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    fape_loss(pred_t, true_t, R, R, seq_mask).backward()
    assert pred_t.grad is not None


def test_fape_loss_masked_padding():
    """fape_loss ignores PAD positions via seq_mask."""
    B, L = 1, 10
    pred_t = torch.randn(B, L, 3, requires_grad=True)
    true_t = torch.randn(B, L, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    # Only first 5 residues are real
    seq_mask = torch.zeros(B, L, dtype=torch.bool)
    seq_mask[:, :5] = True
    loss = fape_loss(pred_t, true_t, R, R, seq_mask)
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# backbone_rmsd
# ---------------------------------------------------------------------------

def test_backbone_rmsd_scalar():
    """backbone_rmsd returns a scalar, non-negative value."""
    B, L = 2, 10
    pred = torch.randn(B, L, 3)
    true = torch.randn(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    rmsd = backbone_rmsd(pred, true, mask)
    assert rmsd.shape == ()
    assert rmsd.item() >= 0.0


def test_backbone_rmsd_zero_on_perfect():
    """backbone_rmsd is zero when prediction equals target."""
    B, L = 2, 8
    coords = torch.randn(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    rmsd = backbone_rmsd(coords, coords, mask)
    assert rmsd.item() == pytest.approx(0.0, abs=1e-5)
