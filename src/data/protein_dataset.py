"""
Protein sequence dataset for mini-alphafold.

Converts raw amino acid strings into integer tensors suitable for a DataLoader.
Special tokens:
  <PAD> = 0  — used by collate functions to pad variable-length batches
  <UNK> = 1  — any character outside the 20 standard amino acids
  A–Y   = 2–21
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

# 20 standard amino acids in IUPAC order
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

PAD_IDX = 0
UNK_IDX = 1
_OFFSET = 2  # first real amino acid index

AA_TO_IDX: dict[str, int] = {
    aa: idx + _OFFSET for idx, aa in enumerate(_AA_ALPHABET)
}
IDX_TO_AA: dict[int, str] = {v: k for k, v in AA_TO_IDX.items()}
VOCAB_SIZE = len(_AA_ALPHABET) + _OFFSET  # 22


def tokenize(sequence: str) -> Tensor:
    """Convert an amino acid string to a 1-D integer tensor.

    Unknown characters map to UNK_IDX; sequences are not padded here —
    padding is handled by a DataLoader collate function.
    """
    return torch.tensor(
        [AA_TO_IDX.get(aa.upper(), UNK_IDX) for aa in sequence],
        dtype=torch.long,
    )


class ProteinDataset(Dataset):
    """Dataset of amino acid sequences with optional per-residue labels.

    Args:
        sequences: List of amino acid strings (e.g. ["ACDEF", "GHIKL"]).
        labels:    Optional list of label tensors (one per sequence).
                   Each label tensor should have the same length as its
                   corresponding sequence (e.g. secondary structure tags
                   encoded as integers).  Pass None to work with unlabelled
                   data (returns only the sequence tensor).
    """

    def __init__(
        self,
        sequences: list[str],
        labels: Optional[list[Tensor]] = None,
    ) -> None:
        if labels is not None and len(sequences) != len(labels):
            raise ValueError(
                f"sequences and labels must have the same length "
                f"({len(sequences)} vs {len(labels)})"
            )

        self.sequences = sequences
        self.labels = labels
        # Pre-tokenise everything so __getitem__ is fast
        self._tokens: list[Tensor] = [tokenize(s) for s in sequences]

    # ------------------------------------------------------------------
    # Required Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | Tensor:
        tokens = self._tokens[idx]
        if self.labels is not None:
            return tokens, self.labels[idx]
        return tokens

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def decode(self, token_tensor: Tensor) -> str:
        """Convert a 1-D token tensor back to an amino acid string."""
        return "".join(
            IDX_TO_AA.get(t.item(), "?") for t in token_tensor  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return (
            f"ProteinDataset("
            f"n={len(self)}, "
            f"labelled={self.labels is not None}, "
            f"vocab_size={VOCAB_SIZE})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test — run with: python -m src.data.protein_dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(
        batch: list[tuple[Tensor, Tensor] | Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Pad a batch of variable-length sequences to the same length."""
        if isinstance(batch[0], tuple):
            seqs, lbls = zip(*batch)
            return (
                pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX),
                pad_sequence(lbls, batch_first=True, padding_value=-1),
            )
        return pad_sequence(batch, batch_first=True, padding_value=PAD_IDX), None

    # --- unlabelled example ---
    raw_seqs = ["ACDEFGHIK", "LMNPQRST", "VWYACDE"]
    ds = ProteinDataset(raw_seqs)
    print(ds)

    loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn)
    batch_tokens, _ = next(iter(loader))
    print("Batch shape (unlabelled):", batch_tokens.shape)  # [3, 9]

    # Round-trip decode
    for i, seq in enumerate(raw_seqs):
        decoded = ds.decode(ds._tokens[i])
        assert decoded == seq, f"Round-trip failed: {seq!r} → {decoded!r}"
    print("Round-trip decode: OK")

    # --- labelled example (dummy secondary structure: 0=coil, 1=helix, 2=sheet) ---
    import random
    random.seed(0)
    labels = [
        torch.tensor([random.randint(0, 2) for _ in seq], dtype=torch.long)
        for seq in raw_seqs
    ]
    ds_labelled = ProteinDataset(raw_seqs, labels=labels)
    print(ds_labelled)

    loader2 = DataLoader(ds_labelled, batch_size=3, collate_fn=collate_fn)
    batch_tokens2, batch_labels2 = next(iter(loader2))
    print("Batch shape (labelled):", batch_tokens2.shape)
    print("Labels shape:", batch_labels2.shape)

    print("\nVocab size:", VOCAB_SIZE)
    print("PAD index:", PAD_IDX, "  UNK index:", UNK_IDX)
    print("AA->index sample:", {k: AA_TO_IDX[k] for k in "ACWY"})
    print("\nAll tests passed.")
