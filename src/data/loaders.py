"""
Real protein secondary structure dataset loaders.

Primary source
--------------
  proteinea/secondary_structure_prediction  (Hugging Face)
  - 10,800 amino acid sequences from CATH/SCOPe
  - Columns: input (sequence), dssp3 (3-state SS), dssp8 (8-state SS)
  - Single 'train' split; we carve our own val set in the training script

Label alphabets
---------------
  SS3 (3-state):  C=coil, H=helix, E=strand
  SS8 (8-state):  C=coil, H=α-helix, E=strand, G=3₁₀-helix,
                  I=π-helix, T=turn, S=bend, B=β-bridge

Public API
----------
  load_ss_dataset(ss_type, ...)  → ProteinDataset
  collate_fn(batch)              → (padded_tokens, padded_labels)
  MSADataset                     — dataset returning tokens + MSA features
  collate_msa_fn(batch)          → (padded_tokens, padded_msa, padded_labels)
  SS3_LABEL_MAP, SS8_LABEL_MAP   — char → int index dicts
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.protein_dataset import PAD_IDX, ProteinDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

# Integer indices are stable; DO NOT reorder without updating trained models.
SS3_LABEL_MAP: dict[str, int] = {"C": 0, "H": 1, "E": 2}
SS8_LABEL_MAP: dict[str, int] = {
    "C": 0,
    "H": 1,
    "E": 2,
    "G": 3,   # 3₁₀-helix
    "I": 4,   # π-helix
    "T": 5,   # turn
    "S": 6,   # bend
    "B": 7,   # β-bridge
}

SSType = Literal["ss3", "ss8"]

HF_DATASET_ID = "proteinea/secondary_structure_prediction"
_LABEL_COL = {"ss3": "dssp3", "ss8": "dssp8"}


# ---------------------------------------------------------------------------
# Label conversion helpers
# ---------------------------------------------------------------------------

def _label_str_to_tensor(label_str: str, label_map: dict[str, int]) -> Tensor:
    """Convert a per-residue label string (e.g. 'CCHHEE') to a long tensor.

    Characters not in `label_map` are mapped to -1 so CrossEntropyLoss
    can ignore them via ignore_index=-1.
    """
    return torch.tensor(
        [label_map.get(ch, -1) for ch in label_str],
        dtype=torch.long,
    )


def _is_valid_row(seq: str, label: str) -> bool:
    """Return True if sequence and label are non-empty and the same length."""
    return bool(seq) and bool(label) and len(seq) == len(label)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_ss_dataset(
    ss_type: SSType = "ss3",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> ProteinDataset:
    """Download and return a ProteinDataset of real secondary structure data.

    Uses the ``proteinea/secondary_structure_prediction`` dataset from
    Hugging Face (≈ 10,800 sequences, ~20 MB download, cached after first use).

    Args:
        ss_type:     "ss3" for 3-state (C/H/E) or "ss8" for 8-state labels.
        max_samples: Cap the number of sequences loaded (useful for quick runs).
        cache_dir:   Optional HuggingFace cache directory override.

    Returns:
        ProteinDataset with sequences and per-residue integer label tensors.

    Raises:
        ImportError:  if the ``datasets`` library is not installed.
        RuntimeError: if the download fails and no cached version exists.
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for real data loading.\n"
            "Install it with:  pip install datasets"
        ) from exc

    label_col = _LABEL_COL[ss_type]
    label_map = SS3_LABEL_MAP if ss_type == "ss3" else SS8_LABEL_MAP

    logger.info("Downloading %s (split='train', task=%s)…", HF_DATASET_ID, ss_type)
    print(f"Loading {HF_DATASET_ID} [{ss_type}] from Hugging Face…")

    try:
        hf_ds = hf_load(
            HF_DATASET_ID,
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {HF_DATASET_ID!r} from Hugging Face.\n"
            f"Check your internet connection or pass a local cache_dir.\n"
            f"Original error: {exc}"
        ) from exc

    sequences: list[str] = []
    labels: list[Tensor] = []
    skipped = 0

    iterable = hf_ds if max_samples is None else hf_ds.select(range(min(max_samples * 2, len(hf_ds))))

    for row in iterable:
        seq: str = row["input"].strip()
        lbl_str: str = row[label_col].strip()

        if not _is_valid_row(seq, lbl_str):
            skipped += 1
            continue

        sequences.append(seq)
        labels.append(_label_str_to_tensor(lbl_str, label_map))

        if max_samples is not None and len(sequences) >= max_samples:
            break

    if skipped:
        logger.warning("Skipped %d malformed rows (len mismatch or empty).", skipped)

    print(
        f"  Loaded {len(sequences):,} sequences "
        f"({'ss3' if ss_type == 'ss3' else 'ss8'}, "
        f"{len(label_map)} classes, "
        f"{skipped} rows skipped)"
    )

    return ProteinDataset(sequences, labels=labels)


# ---------------------------------------------------------------------------
# Shared collate function (imported by training scripts)
# ---------------------------------------------------------------------------

def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Pad a variable-length batch of (tokens, labels) to the same length.

    Token padding uses PAD_IDX (0); label padding uses -1 so that
    CrossEntropyLoss(ignore_index=-1) ignores those positions.
    """
    seqs, lbls = zip(*batch)
    return (
        pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(lbls, batch_first=True, padding_value=-1),
    )


# ---------------------------------------------------------------------------
# MSA-aware dataset and collate
# ---------------------------------------------------------------------------

class MSADataset(Dataset):
    """Dataset returning (tokens, msa_features, [label]) tuples.

    Each item's MSA is either:
      - A real pre-aligned MSA supplied as a list of strings, or
      - A pseudo-MSA auto-generated by random mutation of the query.

    The MSA feature tensor has shape (N_seq, L, FEATURE_DIM=45) and is
    computed lazily in __getitem__.

    Args:
        sequences:     Query amino-acid strings.
        msa_sequences: Optional per-sequence MSA, each a list of aligned strings
                       (query must be first, all same length as the query).
                       When None, pseudo-MSAs are generated automatically.
        labels:        Optional per-residue label tensors (e.g. SS labels).
        n_pseudo:      Number of synthetic homologues when msa_sequences is None.
        mutation_rate: Per-residue mutation rate for pseudo-MSA generation.
        seed:          Optional RNG seed for reproducible pseudo-MSAs.
        device:        Device for MSA feature tensors.
    """

    def __init__(
        self,
        sequences: list[str],
        msa_sequences: Optional[list[list[str]]] = None,
        labels: Optional[list[Tensor]] = None,
        n_pseudo: int = 8,
        mutation_rate: float = 0.15,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        from src.data.msa_encoder import MSAEncoder, generate_pseudo_msa

        if msa_sequences is not None and len(msa_sequences) != len(sequences):
            raise ValueError(
                f"msa_sequences must have the same length as sequences "
                f"({len(sequences)} vs {len(msa_sequences)})"
            )
        if labels is not None and len(labels) != len(sequences):
            raise ValueError(
                f"labels must have the same length as sequences "
                f"({len(sequences)} vs {len(labels)})"
            )

        self.sequences = sequences
        self.labels = labels
        self._encoder = MSAEncoder(device=device)
        self._generate_pseudo_msa = generate_pseudo_msa

        # Pre-tokenise query sequences (same as ProteinDataset)
        from src.data.protein_dataset import tokenize
        self._tokens: list[Tensor] = [tokenize(s) for s in sequences]

        # Resolve MSA source per sample
        if msa_sequences is not None:
            self._msa_seqs: list[list[str]] = msa_sequences
        else:
            per_seed = None if seed is None else None  # keep seed stable per item
            self._msa_seqs = [
                generate_pseudo_msa(seq, n_sequences=n_pseudo,
                                    mutation_rate=mutation_rate,
                                    seed=None if seed is None else seed + i)
                for i, seq in enumerate(sequences)
            ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """Return (tokens, msa_tensor, [label]).

        Returns:
            tokens:     (L,) long tensor — query token indices.
            msa_tensor: (N_seq, L, 45) float32 — MSA features.
            label:      (L,) long tensor if labels were supplied, else omitted.
        """
        tokens = self._tokens[idx]
        msa_feat = self._encoder.encode(self._msa_seqs[idx])
        msa_tensor = msa_feat.features   # (N_seq, L, 45)

        if self.labels is not None:
            return tokens, msa_tensor, self.labels[idx]
        return tokens, msa_tensor

    def __repr__(self) -> str:
        return (
            f"MSADataset(n={len(self)}, labelled={self.labels is not None}, "
            f"msa_n_seq={len(self._msa_seqs[0]) if self._msa_seqs else 0})"
        )


def collate_msa_fn(
    batch: list[tuple],
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """Pad a variable-length batch of (tokens, msa_tensor, [label]) tuples.

    Token padding uses PAD_IDX (0); label padding uses -1; MSA features
    are zero-padded along the sequence dimension.

    Returns:
        tokens:  (B, L_max) long
        msa:     (B, N_seq, L_max, 45) float32
        labels:  (B, L_max) long — or None if labels absent in the batch
    """
    has_labels = len(batch[0]) == 3

    if has_labels:
        tokens_list, msa_list, label_list = zip(*batch)
    else:
        tokens_list, msa_list = zip(*batch)
        label_list = None

    # Pad tokens: (B, L_max)
    tokens_pad = pad_sequence(tokens_list, batch_first=True, padding_value=PAD_IDX)

    # Pad MSA: each item is (N, L_i, 45); pad L dimension to L_max
    L_max = tokens_pad.shape[1]
    N_seq = msa_list[0].shape[0]
    feat_dim = msa_list[0].shape[2]
    B = len(msa_list)
    msa_pad = torch.zeros(B, N_seq, L_max, feat_dim)
    for b, msa in enumerate(msa_list):
        L_i = msa.shape[1]
        msa_pad[b, :, :L_i, :] = msa

    # Pad labels if present
    if label_list is not None:
        labels_pad = pad_sequence(label_list, batch_first=True, padding_value=-1)
    else:
        labels_pad = None

    return tokens_pad, msa_pad, labels_pad


# ---------------------------------------------------------------------------
# Quick smoke-test:  python -m src.data.loaders
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ss = sys.argv[1] if len(sys.argv) > 1 else "ss3"
    assert ss in ("ss3", "ss8"), "Usage: python -m src.data.loaders [ss3|ss8]"

    ds = load_ss_dataset(ss_type=ss, max_samples=200)  # type: ignore[arg-type]
    print(ds)

    # Spot-check first sample
    tokens, lbl = ds[0]
    print(f"  Sequence length : {len(tokens)}")
    print(f"  Label length    : {len(lbl)}")
    print(f"  Unique labels   : {sorted(lbl.unique().tolist())}")
    assert len(tokens) == len(lbl), "Token / label length mismatch!"

    # DataLoader round-trip
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    batch_tok, batch_lbl = next(iter(loader))
    print(f"  Batch tokens    : {batch_tok.shape}")
    print(f"  Batch labels    : {batch_lbl.shape}")
    print("Smoke-test passed.")
