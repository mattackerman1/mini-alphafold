"""Tests for src/data/protein_dataset.py — tokenization and ProteinDataset."""

import torch
import pytest

from src.data.protein_dataset import (
    AA_TO_IDX,
    IDX_TO_AA,
    PAD_IDX,
    UNK_IDX,
    VOCAB_SIZE,
    tokenize,
    ProteinDataset,
)

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def test_vocab_size():
    """VOCAB_SIZE should be 22 (20 AAs + PAD + UNK)."""
    assert VOCAB_SIZE == 22


def test_pad_unk_indices():
    """PAD=0 and UNK=1 are reserved; amino acids start at 2."""
    assert PAD_IDX == 0
    assert UNK_IDX == 1


def test_all_standard_aas_tokenized():
    """All 20 standard amino acids must map to indices in [2, 21]."""
    for aa in _AA_ALPHABET:
        idx = AA_TO_IDX[aa]
        assert 2 <= idx <= 21, f"{aa} → {idx} out of range"


def test_aa_idx_roundtrip():
    """AA_TO_IDX and IDX_TO_AA must be inverses."""
    for aa, idx in AA_TO_IDX.items():
        assert IDX_TO_AA[idx] == aa


def test_tokenize_known_residues():
    """Tokenize a known sequence and check every index is in [2, 21]."""
    tokens = tokenize("ACDEF")
    assert tokens.dtype == torch.long
    assert tokens.shape == (5,)
    assert tokens.min().item() >= 2
    assert tokens.max().item() <= 21


def test_tokenize_unknown_residue():
    """Unknown characters (e.g. 'X') map to UNK_IDX."""
    tokens = tokenize("AXZ")
    assert tokens[1].item() == UNK_IDX
    assert tokens[2].item() == UNK_IDX


def test_tokenize_case_insensitive():
    """Lowercase amino acids should tokenize the same as uppercase."""
    upper = tokenize("ACDEF")
    lower = tokenize("acdef")
    assert torch.equal(upper, lower)


def test_tokenize_single_residue():
    """Single-residue sequence tokenizes to a length-1 tensor."""
    tokens = tokenize("A")
    assert tokens.shape == (1,)


def test_protein_dataset_len():
    """ProteinDataset.__len__ returns number of sequences."""
    ds = ProteinDataset(["ACDEF", "GHIKL", "MNP"])
    assert len(ds) == 3


def test_protein_dataset_getitem_no_labels():
    """Without labels, __getitem__ returns a token tensor."""
    ds = ProteinDataset(["ACDEF"])
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (5,)


def test_protein_dataset_getitem_with_labels():
    """With labels, __getitem__ returns (tokens, labels) tuple."""
    seqs = ["ACDEF", "GHI"]
    labels = [torch.zeros(5, dtype=torch.long), torch.ones(3, dtype=torch.long)]
    ds = ProteinDataset(seqs, labels=labels)
    tokens, lab = ds[0]
    assert tokens.shape == (5,)
    assert lab.shape == (5,)


def test_protein_dataset_length_mismatch_raises():
    """Mismatched sequences and labels should raise ValueError."""
    with pytest.raises(ValueError):
        ProteinDataset(["ACDEF"], labels=[torch.zeros(5), torch.zeros(5)])
