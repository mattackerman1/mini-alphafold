"""Integration tests: full data pipeline through model forward pass + recycling."""

import math
import torch
import pytest
from torch.utils.data import DataLoader

pytestmark = pytest.mark.integration

VOCAB_SIZE = 22
MSA_FEAT_DIM = 45

# Synthetic amino acid sequences (all standard AAs)
_AAS = "ACDEFGHIKLMNPQRSTVWY"
_SEQS = [
    "ACDEFGHIKLMNPQRSTVWY",      # L=20
    "MKTAYIAKQRQISFVKSHFSRQ",    # L=22
    "GVPQLQSRLK",                # L=10
    "ACDEFGHIK",                 # L=9 (shortest, for padding test)
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dataset(seqs=None, n_pseudo=4, seed=0):
    from src.data.loaders import MSADataset
    return MSADataset(seqs or _SEQS, n_pseudo=n_pseudo, seed=seed)


# ---------------------------------------------------------------------------
# DataLoader → Evoformer
# ---------------------------------------------------------------------------

def test_dataloader_to_evoformer():
    """MSADataset → DataLoader → collate_msa_fn → EvoformerStack produces correct shapes."""
    from src.data.loaders import collate_msa_fn
    from src.models.evoformer import build_evoformer, EvoformerOutput

    ds = _make_dataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_msa_fn)
    tokens, msa, _ = next(iter(loader))   # (B, L), (B, N, L, 45), None

    B, L = tokens.shape
    N = msa.shape[1]

    evo = build_evoformer(n_blocks=1, c_m=32, c_z=16, nhead=4, c_outer=8, c_t=8)
    out = evo(msa)

    assert isinstance(out, EvoformerOutput)
    assert out.msa_repr.shape == (B, N, L, 32)
    assert out.pair_repr.shape == (B, L, L, 16)
    assert out.query_repr.shape == (B, L, 32)
    assert out.msa_repr.isfinite().all()
    assert out.pair_repr.isfinite().all()


def test_dataloader_to_structure_module():
    """Full pipeline: DataLoader batch → StructureModule → ca_coords finite and correct shape."""
    from src.data.loaders import collate_msa_fn
    from src.models.structure_module import build_msa_structure_model

    ds = _make_dataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_msa_fn)
    tokens, msa, _ = next(iter(loader))

    model = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False
    )
    model.eval()
    with torch.no_grad():
        out = model(tokens, msa_features=msa)

    B, L = tokens.shape
    assert out.ca_coords.shape == (B, L, 3)
    assert out.ca_coords.isfinite().all(), "ca_coords has NaN/Inf"


# ---------------------------------------------------------------------------
# Variable-length batch padding
# ---------------------------------------------------------------------------

def test_variable_length_batch_padding():
    """collate_msa_fn pads variable-length sequences to max length in batch."""
    from src.data.loaders import collate_msa_fn
    from src.data.protein_dataset import PAD_IDX

    # Use sequences of deliberately different lengths
    seqs = ["ACG", "ACDEFGHIKLM", "MNPQ"]
    ds = _make_dataset(seqs=seqs)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_msa_fn)
    tokens, msa, _ = next(iter(loader))

    max_len = max(len(s) for s in seqs)
    assert tokens.shape == (3, max_len), f"Expected (3, {max_len}), got {tokens.shape}"
    assert msa.shape[2] == max_len, f"MSA L dim should be {max_len}, got {msa.shape[2]}"

    # Shortest sequence (len=3) should have PAD tokens beyond position 3
    # Find the batch index corresponding to "ACG"
    for i, s in enumerate(seqs):
        if len(s) < max_len:
            assert (tokens[i, len(s):] == PAD_IDX).all(), \
                f"Padding not applied correctly for seq {i}"


def test_padded_batch_through_evoformer():
    """A padded variable-length batch passes through EvoformerStack without error."""
    from src.data.loaders import collate_msa_fn
    from src.models.evoformer import build_evoformer

    seqs = ["ACG", "ACDEFGHIKLM", "MNPQ"]
    ds = _make_dataset(seqs=seqs)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_msa_fn)
    tokens, msa, _ = next(iter(loader))

    evo = build_evoformer(n_blocks=1, c_m=32, c_z=16, nhead=4, c_outer=8, c_t=8)
    out = evo(msa)
    assert out.msa_repr.isfinite().all()


# ---------------------------------------------------------------------------
# RecycledEvoformer
# ---------------------------------------------------------------------------

def test_recycled_evoformer_output_shape():
    """RecycledEvoformer returns (B, L, 3) ca_coords after all recycles."""
    from src.models.evoformer import build_evoformer
    from src.models.structure_module import build_msa_structure_model
    from src.models.recycling import RecycledEvoformer

    evo = build_evoformer(n_blocks=1, c_m=32, c_z=16, nhead=4, c_outer=8, c_t=8)
    struct = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False
    )
    model = RecycledEvoformer(evo, struct, num_recycles=3)

    msa = torch.randn(1, 4, 20, MSA_FEAT_DIM)
    evo_out, struct_out = model(msa)

    assert struct_out.ca_coords.shape == (1, 20, 3)
    assert struct_out.ca_coords.isfinite().all()


def test_recycling_changes_output():
    """RecycledEvoformer with num_recycles=3 produces different output than num_recycles=1."""
    from src.models.evoformer import build_evoformer
    from src.models.structure_module import build_msa_structure_model
    from src.models.recycling import RecycledEvoformer

    torch.manual_seed(0)
    evo = build_evoformer(n_blocks=1, c_m=32, c_z=16, nhead=4, c_outer=8, c_t=8)
    struct = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False
    )

    msa = torch.randn(1, 4, 15, MSA_FEAT_DIM)

    model1 = RecycledEvoformer(evo, struct, num_recycles=1)
    model3 = RecycledEvoformer(evo, struct, num_recycles=3)

    model1.eval()
    model3.eval()
    with torch.no_grad():
        _, out1 = model1(msa)
        _, out3 = model3(msa)

    assert not torch.allclose(out1.ca_coords, out3.ca_coords), \
        "Recycling had no effect — 1 and 3 recycles produced identical outputs"


# ---------------------------------------------------------------------------
# Gradient flow: full pipeline
# ---------------------------------------------------------------------------

def test_gradient_flow_dataloader_to_loss():
    """Full pipeline: DataLoader → StructureModule → fape_loss → backward succeeds."""
    from src.data.loaders import collate_msa_fn
    from src.models.structure_module import build_msa_structure_model
    from src.training.losses import fape_loss

    ds = _make_dataset(seqs=["ACDEFGHIKL", "MNPQRSTVWY"])
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_msa_fn)
    tokens, msa, _ = next(iter(loader))

    B, L = tokens.shape
    model = build_msa_structure_model(
        n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    true_ca = torch.randn(B, L, 3)
    true_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    seq_mask = torch.ones(B, L, dtype=torch.bool)

    opt.zero_grad()
    out = model(tokens, msa_features=msa)
    pred_R = out.rotations if out.rotations is not None else true_R
    loss = fape_loss(out.ca_coords, true_ca, pred_R, true_R, seq_mask)
    loss.backward()
    opt.step()

    assert math.isfinite(loss.item()), f"loss is not finite: {loss.item()}"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients after backward"
    assert all(g.isfinite().all() for g in grads), "NaN/Inf gradient detected"
