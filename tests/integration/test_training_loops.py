"""Integration tests: one optimizer step for each training script's core loop.

Each test verifies:
  - Loss is a finite, non-negative scalar
  - backward() completes without error
  - At least one model parameter receives a gradient
"""

import math
import torch
import torch.nn as nn
import pytest

pytestmark = pytest.mark.integration

# Small dims shared across all tests for fast execution
B, L = 2, 20
VOCAB_SIZE = 22
MSA_FEAT_DIM = 45


def _tokens(B=B, L=L):
    return torch.randint(2, VOCAB_SIZE, (B, L))


def _msa(B=B, N=4, L=L):
    return torch.randn(B, N, L, MSA_FEAT_DIM)


def _seq_mask(B=B, L=L):
    return torch.ones(B, L, dtype=torch.bool)


def _assert_valid_loss(loss):
    assert loss.shape == (), "loss must be scalar"
    assert math.isfinite(loss.item()), f"loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"loss is negative: {loss.item()}"


def _assert_gradients(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no parameter received a gradient"
    assert all(g.isfinite().all() for g in grads), "NaN/Inf gradient detected"


# ---------------------------------------------------------------------------
# Baseline (SS3 secondary structure)
# ---------------------------------------------------------------------------

def test_baseline_bilstm_training_step():
    """BiLSTM SS3 model: one forward+backward step produces finite loss and gradients."""
    from src.models.baseline import BiLSTMModel, SS3_CLASSES

    model = BiLSTMModel(embed_dim=16, hidden_dim=16, num_layers=1, n_classes=SS3_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = _tokens()
    labels = torch.randint(0, SS3_CLASSES, (B, L))

    opt.zero_grad()
    logits = model(tokens)                            # (B, L, 3)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, SS3_CLASSES), labels.reshape(-1))
    loss.backward()
    opt.step()

    _assert_valid_loss(loss)
    _assert_gradients(model)


def test_baseline_transformer_training_step():
    """Transformer SS3 model: one forward+backward step produces finite loss and gradients."""
    from src.models.baseline import TransformerModel, SS3_CLASSES

    model = TransformerModel(d_model=16, nhead=2, num_layers=1, n_classes=SS3_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = _tokens()
    labels = torch.randint(0, SS3_CLASSES, (B, L))

    opt.zero_grad()
    logits = model(tokens)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, SS3_CLASSES), labels.reshape(-1))
    loss.backward()
    opt.step()

    _assert_valid_loss(loss)
    _assert_gradients(model)


# ---------------------------------------------------------------------------
# Contact prediction
# ---------------------------------------------------------------------------

def test_contact_training_step():
    """ContactPredictor: one forward+backward step with BCE loss."""
    from src.models.contact_predictor import build_contact_model
    from src.training.losses import contact_bce_loss, make_contact_mask

    model = build_contact_model(kind="bilstm", hidden_dim=16, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = _tokens()
    targets = torch.randint(0, 2, (B, L, L)).float()
    lengths = torch.full((B,), L)
    mask = make_contact_mask(lengths, L=L, sep_min=1)

    opt.zero_grad()
    logits = model(tokens)                           # (B, L, L)
    loss = contact_bce_loss(logits, targets, mask)
    loss.backward()
    opt.step()

    _assert_valid_loss(loss)
    _assert_gradients(model)


# ---------------------------------------------------------------------------
# Structure module (encoder path)
# ---------------------------------------------------------------------------

def test_structure_encoder_training_step():
    """StructureModule (BiLSTM encoder): one step with FAPE loss."""
    from src.models.structure_module import build_structure_model
    from src.training.losses import fape_loss

    model = build_structure_model(kind="bilstm", hidden_dim=16, num_layers=1, use_frames=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = _tokens()
    true_ca = torch.randn(B, L, 3)
    true_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = _seq_mask()

    opt.zero_grad()
    out = model(tokens)
    pred_R = out.rotations if out.rotations is not None else true_R
    loss = fape_loss(out.ca_coords, true_ca, pred_R, true_R, seq_mask)
    loss.backward()
    opt.step()

    _assert_valid_loss(loss)
    _assert_gradients(model)


# ---------------------------------------------------------------------------
# End-to-end (Evoformer + structure module)
# ---------------------------------------------------------------------------

def test_end_to_end_training_step():
    """Full MSA → Evoformer → StructureModule pipeline: one step with FAPE loss."""
    from src.models.structure_module import build_msa_structure_model
    from src.training.losses import fape_loss

    model = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokens = _tokens()
    msa_feat = _msa()
    true_ca = torch.randn(B, L, 3)
    true_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = _seq_mask()

    opt.zero_grad()
    out = model(tokens, msa_features=msa_feat)
    pred_R = out.rotations if out.rotations is not None else true_R
    loss = fape_loss(out.ca_coords, true_ca, pred_R, true_R, seq_mask)
    loss.backward()
    opt.step()

    _assert_valid_loss(loss)
    _assert_gradients(model)


def test_end_to_end_loss_decreases_over_steps():
    """Loss should decrease (or at least not increase) over several optimizer steps."""
    from src.models.structure_module import build_msa_structure_model
    from src.training.losses import fape_loss

    torch.manual_seed(42)
    model = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    tokens = _tokens(B=1, L=10)
    msa_feat = _msa(B=1, N=4, L=10)
    true_ca = torch.randn(1, 10, 3)
    true_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 10, 3, 3)
    seq_mask = torch.ones(1, 10, dtype=torch.bool)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = model(tokens, msa_features=msa_feat)
        pred_R = out.rotations if out.rotations is not None else true_R
        loss = fape_loss(out.ca_coords, true_ca, pred_R, true_R, seq_mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(math.isfinite(v) for v in losses), f"NaN/Inf in losses: {losses}"
    # Loss over 5 steps should decrease at least once (not strictly monotone)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
