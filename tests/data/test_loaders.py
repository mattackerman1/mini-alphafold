"""Tests for src/data/loaders.py — collation and MSADataset."""

import torch
import pytest
from torch.utils.data import DataLoader

from src.data.loaders import (
    collate_fn,
    MSADataset,
    collate_msa_fn,
    SS3_LABEL_MAP,
    SS8_LABEL_MAP,
)
from src.data.protein_dataset import PAD_IDX, ProteinDataset


# ---------------------------------------------------------------------------
# collate_fn (sequence + label batching)
# ---------------------------------------------------------------------------

def _make_batch(seqs, labels=None):
    """Helper: build a batch list as collate_fn expects."""
    from src.data.protein_dataset import tokenize
    if labels is None:
        return [(tokenize(s),) for s in seqs]
    return [(tokenize(s), lab) for s, lab in zip(seqs, labels)]


def test_collate_fn_pads_tokens():
    """collate_fn pads token tensors to the longest sequence in the batch."""
    from src.data.protein_dataset import tokenize
    batch = [
        (tokenize("ACF"), torch.zeros(3, dtype=torch.long)),
        (tokenize("GHIKL"), torch.zeros(5, dtype=torch.long)),
    ]
    tokens, labels = collate_fn(batch)
    assert tokens.shape == (2, 5)                      # padded to max len 5
    assert tokens[0, 3].item() == PAD_IDX              # short seq padded at end
    assert tokens[0, 4].item() == PAD_IDX


def test_collate_fn_token_dtype():
    """Collated token tensor should be long (int64)."""
    from src.data.protein_dataset import tokenize
    batch = [(tokenize("ACDEF"), torch.zeros(5, dtype=torch.long))]
    tokens, _ = collate_fn(batch)
    assert tokens.dtype == torch.long


def test_collate_fn_label_padding():
    """Labels are padded to max length with -1 (CrossEntropyLoss ignore_index)."""
    from src.data.protein_dataset import tokenize
    batch = [
        (tokenize("AC"), torch.tensor([0, 1])),
        (tokenize("GHIKL"), torch.tensor([2, 0, 1, 2, 0])),
    ]
    _, labels = collate_fn(batch)
    assert labels.shape == (2, 5)
    # Short sequence padded with -1
    assert labels[0, 2].item() == -1
    assert labels[0, 3].item() == -1
    assert labels[0, 4].item() == -1


def test_collate_fn_single_item():
    """Batch of one item should work without errors."""
    from src.data.protein_dataset import tokenize
    batch = [(tokenize("ACDEF"), torch.zeros(5, dtype=torch.long))]
    tokens, labels = collate_fn(batch)
    assert tokens.shape == (1, 5)
    assert labels.shape == (1, 5)


# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

def test_ss3_label_map():
    """SS3 label map has exactly 3 classes: C, H, E."""
    assert set(SS3_LABEL_MAP.keys()) == {"C", "H", "E"}
    assert set(SS3_LABEL_MAP.values()) == {0, 1, 2}


def test_ss8_label_map():
    """SS8 label map has exactly 8 classes."""
    assert len(SS8_LABEL_MAP) == 8
    assert set(SS8_LABEL_MAP.values()) == set(range(8))


# ---------------------------------------------------------------------------
# MSADataset
# ---------------------------------------------------------------------------

def test_msa_dataset_len():
    """MSADataset.__len__ returns the number of sequences."""
    seqs = ["ACDEF", "GHIKL", "MNPQR"]
    ds = MSADataset(seqs, n_pseudo=4)
    assert len(ds) == 3


def test_msa_dataset_getitem_shapes():
    """MSADataset.__getitem__ without labels returns (tokens, msa_features)."""
    seqs = ["ACDEF", "GHIKL"]
    ds = MSADataset(seqs, n_pseudo=5)
    item = ds[0]
    tokens, msa_feat = item   # no labels
    L = 5
    from src.data.msa_encoder import FEATURE_DIM
    assert tokens.shape == (L,)
    # n_pseudo=5 → 5+1=6 sequences (query + 5 homologs)
    assert msa_feat.shape[1] == L
    assert msa_feat.shape[2] == FEATURE_DIM


def test_collate_msa_fn_output_shapes():
    """collate_msa_fn pads tokens and MSA features to batch max length."""
    seqs = ["ACF", "GHIKL"]
    ds = MSADataset(seqs, n_pseudo=3)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_msa_fn)
    batch = next(iter(loader))
    # Without labels, returns (tokens, msa, None) or (tokens, msa)
    tokens, msa_feat = batch[0], batch[1]
    B, max_L = 2, 5
    from src.data.msa_encoder import FEATURE_DIM
    assert tokens.shape == (B, max_L)
    assert msa_feat.shape[0] == B
    assert msa_feat.shape[2] == max_L
    assert msa_feat.shape[3] == FEATURE_DIM
