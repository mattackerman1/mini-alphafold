"""Tests for src/data/msa_encoder.py — MSA feature encoding.

MSA homolog count behavior
--------------------------
MSAEncoder.encode() accepts a list of pre-aligned sequences and returns an
(N_seq, L, FEATURE_DIM) tensor where N_seq equals len(sequences).  The encoder
does NOT enforce a fixed N_seq across calls — that is the caller's responsibility
when constructing batches.  MSADataset uses a fixed n_pseudo per dataset, so all
items in a batch have the same N_seq.
"""

import torch
import pytest

from src.data.msa_encoder import (
    FEATURE_DIM,
    ONE_HOT_DIM,
    MSAEncoder,
    MSAFeatures,
    generate_pseudo_msa,
    encode_single_sequence,
)


def test_feature_dim_constant():
    """FEATURE_DIM must be 45 (22 one-hot + 1 deletion + 22 profile)."""
    assert FEATURE_DIM == 45
    assert ONE_HOT_DIM == 22


def test_encode_single_sequence_shape():
    """encode_single_sequence produces MSAFeatures with shape (1, L, FEATURE_DIM)."""
    result = encode_single_sequence("ACDEF")
    assert isinstance(result, MSAFeatures)
    assert result.features.shape == (1, 5, FEATURE_DIM)
    assert result.features.dtype == torch.float32


def test_msa_encoder_fixed_depth():
    """MSAEncoder.encode() returns (N_seq, L, FEATURE_DIM) with N_seq from input."""
    seqs = ["ACDEF", "ACDEF", "GHIKL"]  # N=3
    enc = MSAEncoder()
    result = enc.encode(seqs)
    assert result.features.shape == (3, 5, FEATURE_DIM)
    assert result.n_seq == 3
    assert result.seq_len == 5


def test_msa_encoder_variable_depth():
    """MSAEncoder.encode() supports variable N_seq across calls — no fixed-depth constraint.

    Design decision: N_seq is caller-controlled.  The encoder does not enforce
    a uniform depth across batches.  MSADataset fixes this via n_pseudo.
    """
    enc = MSAEncoder()
    r3 = enc.encode(["ACDEF", "GHIKL", "MNPQR"])
    r5 = enc.encode(["ACDEF", "GHIKL", "MNPQR", "STVWY", "ACDEG"])
    assert r3.features.shape[0] == 3
    assert r5.features.shape[0] == 5


def test_gap_feature():
    """Gap characters ('-') should produce a 1.0 deletion feature at that position."""
    enc = MSAEncoder()
    feat = enc.encode(["A-C"]).features   # (1, 3, 45)
    deletion_channel = ONE_HOT_DIM        # index 22 is the deletion dim
    assert feat[0, 1, deletion_channel].item() == pytest.approx(1.0)
    assert feat[0, 0, deletion_channel].item() == pytest.approx(0.0)
    assert feat[0, 2, deletion_channel].item() == pytest.approx(0.0)


def test_profile_shape():
    """Profile portion of features should sum to ~1.0 per position (frequency distribution)."""
    enc = MSAEncoder()
    feat = enc.encode(["ACDEF", "ACDEF"]).features   # (2, 5, 45)
    # Profile is the last 22 dims (indices 23..44)
    profile = feat[0, :, ONE_HOT_DIM + 1:]           # (5, 22)
    sums = profile.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5)


def test_generate_pseudo_msa_count():
    """generate_pseudo_msa returns n_sequences+1 sequences (query + n_sequences homologs)."""
    seqs = generate_pseudo_msa("ACDEF", n_sequences=7, mutation_rate=0.1, seed=0)
    assert len(seqs) == 8          # query + 7 homologs
    assert all(len(s) == 5 for s in seqs)
    assert seqs[0] == "ACDEF"     # query is unchanged at index 0


def test_generate_pseudo_msa_reproducible():
    """Same seed produces identical pseudo-MSA."""
    a = generate_pseudo_msa("ACDEF", n_sequences=4, seed=42)
    b = generate_pseudo_msa("ACDEF", n_sequences=4, seed=42)
    assert a == b


def test_encode_single_residue():
    """Single-residue sequence encodes without error."""
    result = encode_single_sequence("A")
    assert result.features.shape == (1, 1, FEATURE_DIM)
