"""Integration tests: checkpoint save → load → inference cycle."""

import math
import torch
import pytest

pytestmark = pytest.mark.integration

VOCAB_SIZE = 22
MSA_FEAT_DIM = 45
B, L = 2, 15


def _tokens(B=B, L=L):
    return torch.randint(2, VOCAB_SIZE, (B, L))


def _msa(B=B, N=4, L=L):
    return torch.randn(B, N, L, MSA_FEAT_DIM)


# ---------------------------------------------------------------------------
# StructureModule (MSA path) checkpoint tests
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_msa_structure_model(tmp_path):
    """Save and reload MSA StructureModule; inference output is finite and correct shape."""
    from src.models.structure_module import build_msa_structure_model

    model = build_msa_structure_model(n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False)
    ckpt_path = tmp_path / "msa_model.pt"
    torch.save({"model_state": model.state_dict(), "args": {"n_blocks": 1, "c_m": 32, "c_z": 16}}, ckpt_path)

    model2 = build_msa_structure_model(n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ckpt["model_state"])
    model2.eval()

    with torch.no_grad():
        out = model2(_tokens(), msa_features=_msa())

    assert out.ca_coords.shape == (B, L, 3)
    assert out.ca_coords.isfinite().all(), "ca_coords contains NaN/Inf after checkpoint reload"


def test_checkpoint_state_dict_keys_match(tmp_path):
    """State dict keys are identical before and after save/load."""
    from src.models.structure_module import build_msa_structure_model

    model = build_msa_structure_model(n_blocks=1, c_m=32, c_z=16, use_triangle=False)
    ckpt_path = tmp_path / "keys_test.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(model.state_dict().keys()) == set(ckpt["model_state"].keys())


def test_checkpoint_args_roundtrip(tmp_path):
    """Training args survive the checkpoint round-trip exactly."""
    from src.models.structure_module import build_msa_structure_model

    args = {"n_blocks": 1, "c_m": 32, "c_z": 16, "lr": 1e-3, "epochs": 10, "use_triangle": False}
    model = build_msa_structure_model(n_blocks=1, c_m=32, c_z=16, use_triangle=False)
    ckpt_path = tmp_path / "args_test.pt"
    torch.save({"model_state": model.state_dict(), "args": args}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["args"] == args


def test_inference_reproducible_after_load(tmp_path):
    """Two models loaded from the same checkpoint produce identical outputs."""
    from src.models.structure_module import build_msa_structure_model

    model = build_msa_structure_model(n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False)
    ckpt_path = tmp_path / "repro_test.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    def _load():
        m = build_msa_structure_model(n_blocks=1, c_m=32, c_z=16, use_frames=True, use_triangle=False)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        return m

    tokens = _tokens(B=1, L=10)
    msa = _msa(B=1, N=4, L=10)

    with torch.no_grad():
        out1 = _load()(tokens, msa_features=msa)
        out2 = _load()(tokens, msa_features=msa)

    assert torch.allclose(out1.ca_coords, out2.ca_coords), \
        "Outputs differ between two models loaded from the same checkpoint"


# ---------------------------------------------------------------------------
# Baseline model checkpoint tests
# ---------------------------------------------------------------------------

def test_baseline_checkpoint_roundtrip(tmp_path):
    """BiLSTM baseline: save, reload, run inference — shape and finite check."""
    from src.models.baseline import build_model, SS3_CLASSES

    model = build_model(kind="bilstm", n_classes=SS3_CLASSES)
    ckpt_path = tmp_path / "baseline.pt"
    torch.save({"model_state": model.state_dict(), "args": {"kind": "bilstm"}}, ckpt_path)

    model2 = build_model(kind="bilstm", n_classes=SS3_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ckpt["model_state"])
    model2.eval()

    with torch.no_grad():
        out = model2(_tokens())

    assert out.shape == (B, L, SS3_CLASSES)
    assert out.isfinite().all(), "baseline logits contain NaN/Inf after checkpoint reload"


def test_checkpoint_weights_are_preserved(tmp_path):
    """Parameter values are numerically identical after save/load."""
    from src.models.baseline import build_model, SS3_CLASSES

    model = build_model(kind="bilstm", n_classes=SS3_CLASSES)
    ckpt_path = tmp_path / "weights_test.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    model2 = build_model(kind="bilstm", n_classes=SS3_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ckpt["model_state"])

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter mismatch after reload: {n1}"
