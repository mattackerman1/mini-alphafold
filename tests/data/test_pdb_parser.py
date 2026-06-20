"""Tests for src/data/pdb_parser.py — tokenization constants and StructureData."""

import torch
import pytest

from src.data.pdb_parser import BACKBONE_ATOMS, ATOM_INDEX, StructureData


def test_backbone_atom_order():
    """Backbone atom order must be [N, CA, C, O] — changing it breaks all consumers."""
    assert BACKBONE_ATOMS == ["N", "CA", "C", "O"]


def test_atom_index_keys():
    """ATOM_INDEX must index all four backbone atoms at 0-3."""
    assert ATOM_INDEX == {"N": 0, "CA": 1, "C": 2, "O": 3}


def test_structure_data_len():
    """StructureData.__len__ returns the sequence length."""
    coords = torch.zeros(5, 4, 3)
    sd = StructureData(sequence="ACDEF", coords=coords)
    assert len(sd) == 5


def test_structure_data_coord_shape():
    """Coords tensor must be (L, 4, 3) float32."""
    L = 10
    coords = torch.randn(L, 4, 3)
    sd = StructureData(sequence="A" * L, coords=coords)
    assert sd.coords.shape == (L, 4, 3)


def test_structure_data_empty_sequence():
    """StructureData can hold an empty sequence (edge case)."""
    coords = torch.zeros(0, 4, 3)
    sd = StructureData(sequence="", coords=coords)
    assert len(sd) == 0
    assert sd.coords.shape == (0, 4, 3)
