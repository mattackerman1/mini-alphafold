"""
PDB / mmCIF structure parser for mini-alphafold.

Loads a structure file with Biopython and extracts:
  - Amino acid sequence (one-letter codes)
  - Backbone coordinates for N, CA, C, O atoms per residue
  - Optional residue identifiers for debugging

Public API
----------
  parse_structure_file(filepath)  -> StructureData
  StructureData                   -> dataclass with .sequence, .coords, .residue_ids

Coordinate tensor shape: (L, 4, 3)
  axis 0: residues (length L)
  axis 1: atoms in order [N, CA, C, O]
  axis 2: x, y, z in Angstroms

Chain selection: chain A is preferred; falls back to the first chain present.
HETATM residues (non-standard / ligand) are silently skipped.
Residues with any missing backbone atom are skipped with a warning.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Atom order used throughout the project — do not change without updating all consumers
BACKBONE_ATOMS: list[str] = ["N", "CA", "C", "O"]
ATOM_INDEX: dict[str, int] = {a: i for i, a in enumerate(BACKBONE_ATOMS)}

# Biopython's three-letter -> one-letter residue map (standard 20 AAs only)
try:
    from Bio.Data.IUPACData import protein_letters_3to1  # type: ignore[import]
    _3TO1: dict[str, str] = {k.upper(): v for k, v in protein_letters_3to1.items()}
except ImportError:
    # Fallback in case Biopython internal layout changes
    _3TO1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructureData:
    """Parsed output for a single protein chain.

    Attributes:
        sequence:    One-letter amino acid string of length L.
        coords:      Float32 tensor of shape (L, 4, 3).
                     Atom order: N=0, CA=1, C=2, O=3.
                     Units: Angstroms.
        residue_ids: List of (chain_id, res_seq, insertion_code) tuples,
                     one per residue, for debugging and alignment.
        pdb_id:      Source PDB/file identifier (optional).
        chain_id:    Which chain was parsed.
    """
    sequence: str
    coords: Tensor                          # (L, 4, 3) float32
    residue_ids: list[tuple] = field(default_factory=list)
    pdb_id: str = ""
    chain_id: str = ""

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        return (
            f"StructureData(pdb={self.pdb_id!r}, chain={self.chain_id!r}, "
            f"L={len(self)}, coords={tuple(self.coords.shape)})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_chain(structure):
    """Return chain A if present, otherwise the first chain in model 0."""
    model = structure[0]
    if "A" in model:
        return model["A"]
    first = next(iter(model))
    logger.warning("Chain A not found; using chain %r instead.", first.id)
    return first


def _residue_to_one_letter(residue) -> Optional[str]:
    """Convert a Biopython Residue to a one-letter code, or None if non-standard."""
    resname = residue.get_resname().strip().upper()
    return _3TO1.get(resname)


def _extract_backbone(residue) -> Optional[list[list[float]]]:
    """Return [[x,y,z] for N, CA, C, O] or None if any atom is missing."""
    coords = []
    for atom_name in BACKBONE_ATOMS:
        if atom_name not in residue:
            return None
        atom = residue[atom_name]
        coords.append(list(atom.get_vector()))
    return coords


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def _parse_chain(chain, pdb_id: str = "") -> StructureData:
    """Extract sequence and backbone coordinates from a Biopython Chain object."""
    sequences: list[str] = []
    all_coords: list[list[list[float]]] = []
    residue_ids: list[tuple] = []
    skipped_missing = 0
    skipped_nonstandard = 0

    for residue in chain.get_residues():
        # Skip HETATM records (water, ligands, modified residues)
        if residue.get_id()[0] != " ":
            skipped_nonstandard += 1
            continue

        one_letter = _residue_to_one_letter(residue)
        if one_letter is None:
            skipped_nonstandard += 1
            continue

        backbone = _extract_backbone(residue)
        if backbone is None:
            skipped_missing += 1
            logger.debug(
                "Residue %s %s missing backbone atom(s); skipped.",
                residue.get_resname(), residue.get_id(),
            )
            continue

        sequences.append(one_letter)
        all_coords.append(backbone)
        het, seq_num, icode = residue.get_id()
        residue_ids.append((chain.id, seq_num, icode.strip()))

    if skipped_missing:
        warnings.warn(
            f"{pdb_id} chain {chain.id}: {skipped_missing} residue(s) skipped "
            f"due to missing backbone atoms.",
            UserWarning,
            stacklevel=3,
        )

    if not sequences:
        raise ValueError(
            f"No standard residues with complete backbone found in "
            f"{pdb_id!r} chain {chain.id!r}."
        )

    coords_tensor = torch.tensor(all_coords, dtype=torch.float32)  # (L, 4, 3)

    return StructureData(
        sequence="".join(sequences),
        coords=coords_tensor,
        residue_ids=residue_ids,
        pdb_id=pdb_id,
        chain_id=chain.id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_structure_file(filepath: str | Path) -> StructureData:
    """Parse a PDB or mmCIF file and return backbone coordinates + sequence.

    Format is auto-detected from the file extension:
      .pdb, .ent          -> PDB format
      .cif, .mmcif        -> mmCIF format

    Chain selection: chain A is used if present; otherwise the first chain.
    Residues with missing backbone atoms (N, CA, C, O) are skipped with a warning.

    Args:
        filepath: Path to a .pdb, .ent, .cif, or .mmcif file.

    Returns:
        StructureData with .sequence, .coords (L, 4, 3), .residue_ids.

    Raises:
        ImportError:  if Biopython is not installed.
        FileNotFoundError: if the file does not exist.
        ValueError:   if no valid residues are found.
    """
    try:
        from Bio import PDB as biopdb  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for structure parsing.\n"
            "Install it with:  pip install biopython"
        ) from exc

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()
    pdb_id = path.stem.upper()

    if suffix in (".pdb", ".ent"):
        parser = biopdb.PDBParser(QUIET=True)
    elif suffix in (".cif", ".mmcif"):
        parser = biopdb.MMCIFParser(QUIET=True)
    else:
        raise ValueError(
            f"Unrecognised file extension {suffix!r}. "
            "Expected one of: .pdb, .ent, .cif, .mmcif"
        )

    structure = parser.get_structure(pdb_id, str(path))
    chain = _select_chain(structure)
    return _parse_chain(chain, pdb_id=pdb_id)


# ---------------------------------------------------------------------------
# Smoke-test — python -m src.data.pdb_parser
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile
    import urllib.request

    PDB_ID = sys.argv[1].upper() if len(sys.argv) > 1 else "1UBQ"

    # Download from RCSB in both formats and round-trip
    for fmt, url_tpl, suffix in [
        ("PDB",   "https://files.rcsb.org/download/{}.pdb",  ".pdb"),
        ("mmCIF", "https://files.rcsb.org/download/{}.cif",  ".cif"),
    ]:
        url = url_tpl.format(PDB_ID)
        print(f"\nFetching {PDB_ID} ({fmt}) from RCSB...")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            urllib.request.urlretrieve(url, tmp_path)
            data = parse_structure_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        print(f"  {data}")
        print(f"  Sequence ({len(data.sequence)} residues): {data.sequence[:40]}...")
        print(f"  Coords shape : {tuple(data.coords.shape)}")
        print(f"  Coords dtype : {data.coords.dtype}")
        print(f"  CA[0] (first residue CA): {data.coords[0, ATOM_INDEX['CA']]}")

        # Sanity checks
        assert len(data.sequence) == data.coords.shape[0], "Length mismatch!"
        assert data.coords.shape[1:] == (4, 3), "Unexpected coord shape!"
        assert not torch.isnan(data.coords).any(), "NaN in coordinates!"

    print(f"\nAll checks passed for {PDB_ID}.")
