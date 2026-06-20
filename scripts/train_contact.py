"""
Contact map prediction training script.

Data sources
------------
  Synthetic (default): random sequences with block-structured contact maps.
                        No download required — good for verifying the pipeline.
  Real (--pdb-ids):    Comma-separated PDB IDs downloaded from RCSB.
                        Requires Biopython.  Example: --pdb-ids 1UBQ,4HHB,1CRN

Contact definition
------------------
  Cα–Cα distance < threshold (default 8 Å).
  Pairs with |i-j| < sep_min (default 6) are excluded from loss and metrics
  to focus on non-trivial contacts.

Metrics
-------
  Precision@L   : precision among the top-L highest-confidence predictions
                  (L = sequence length).  Standard benchmark metric.
  Precision@L/2 : same for top L/2 predictions.
  Precision@L/5 : same for top L/5 predictions.
  Recall@0.5    : recall at sigmoid threshold 0.5.

Usage
-----
  # Synthetic smoke-test
  python scripts/train_contact.py

  # Real PDB data, Transformer, focal loss
  python scripts/train_contact.py --pdb-ids 1UBQ,4HHB,1CRN,2KHO,1L2Y \\
      --model transformer --loss focal --epochs 20

  # More options
  python scripts/train_contact.py --help
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.protein_dataset import PAD_IDX, tokenize
from src.models.contact_predictor import ModelKind, build_contact_model
from src.training.losses import LossKind, contact_loss, make_contact_mask

# ---------------------------------------------------------------------------
# Contact map utilities
# ---------------------------------------------------------------------------

CA_IDX = 1   # index of Cα in BACKBONE_ATOMS = [N, CA, C, O]


def coords_to_contact_map(
    coords: Tensor,
    threshold: float = 8.0,
    sep_min: int = 6,
) -> Tensor:
    """Compute a binary contact map from backbone coordinates.

    Args:
        coords:    (L, 4, 3) float32 — backbone atom positions (N, CA, C, O).
        threshold: Cα–Cα distance cutoff in Å (default 8.0).
        sep_min:   Pairs with |i-j| < sep_min are set to 0 (not contact).

    Returns:
        contact: (L, L) float32 tensor with values in {0.0, 1.0}.
    """
    ca = coords[:, CA_IDX, :]                           # (L, 3)
    diff = ca.unsqueeze(0) - ca.unsqueeze(1)            # (L, L, 3)
    dist = diff.pow(2).sum(-1).sqrt()                   # (L, L)
    contact = (dist < threshold).float()

    # Zero out short-range pairs (local secondary structure)
    L = coords.size(0)
    idx = torch.arange(L, device=coords.device)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    contact[sep < sep_min] = 0.0

    return contact


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContactDataset(Dataset):
    """Dataset of (sequence_tokens, contact_map) pairs.

    Args:
        sequences:    List of amino acid strings.
        contact_maps: List of (L, L) float32 tensors.
    """

    def __init__(self, sequences: list[str], contact_maps: list[Tensor]) -> None:
        assert len(sequences) == len(contact_maps)
        self.tokens: list[Tensor] = [tokenize(s) for s in sequences]
        self.contact_maps = contact_maps

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.tokens[idx], self.contact_maps[idx]


def collate_contact(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a variable-length batch of (tokens, contact_map) pairs.

    Returns:
        tokens:    (B, L_max) long — padded token sequences.
        contacts:  (B, L_max, L_max) float32 — padded contact maps.
        lengths:   (B,) int — original sequence lengths (for mask).
    """
    seqs, maps = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    L = int(lengths.max().item())

    # Pad tokens
    tokens = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)  # (B, L)

    # Pad contact maps to (B, L, L)
    B = len(maps)
    contacts = torch.zeros(B, L, L, dtype=torch.float32)
    for i, cm in enumerate(maps):
        l = cm.size(0)
        contacts[i, :l, :l] = cm

    return tokens, contacts, lengths


# ---------------------------------------------------------------------------
# Synthetic data (no download required)
# ---------------------------------------------------------------------------

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _synthetic_contact_map(length: int, threshold: float = 8.0) -> Tensor:
    """Generate a plausible-looking synthetic contact map.

    Uses a simple block-diagonal structure (mimicking secondary structure)
    plus a few randomly placed long-range contacts.
    """
    cm = torch.zeros(length, length)

    # Local contacts: |i-j| in [6, 12] with decreasing probability
    for i in range(length):
        for offset in range(6, min(13, length - i)):
            j = i + offset
            if random.random() < 0.4:
                cm[i, j] = cm[j, i] = 1.0

    # Long-range contacts: random sparse pairs
    n_lr = max(1, length // 20)
    for _ in range(n_lr):
        i = random.randint(0, length - 1)
        j = random.randint(0, length - 1)
        if abs(i - j) >= 6:
            cm[i, j] = cm[j, i] = 1.0

    return cm


def build_synthetic_dataset(
    n: int = 256,
    min_len: int = 30,
    max_len: int = 100,
    seed: int = 42,
) -> ContactDataset:
    """Random amino acid sequences with structured synthetic contact maps."""
    rng = random.Random(seed)
    sequences, maps = [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        seq = "".join(rng.choice(_AA) for _ in range(length))
        sequences.append(seq)
        maps.append(_synthetic_contact_map(length))
    return ContactDataset(sequences, maps)


# ---------------------------------------------------------------------------
# Real data: download PDB files and parse coordinates
# ---------------------------------------------------------------------------

def build_real_dataset(
    pdb_ids: list[str],
    threshold: float = 8.0,
    sep_min: int = 6,
) -> ContactDataset:
    """Download PDB files from RCSB and build a ContactDataset.

    Args:
        pdb_ids:   List of 4-character PDB identifiers (e.g. ["1UBQ", "4HHB"]).
        threshold: Cα–Cα contact distance cutoff in Å.
        sep_min:   Minimum sequence separation included in contact maps.

    Returns:
        ContactDataset with real sequences and ground-truth contact maps.
    """
    try:
        from src.data.pdb_parser import parse_structure_file
    except ImportError as exc:
        raise ImportError("pdb_parser requires Biopython: pip install biopython") from exc

    sequences, maps = [], []

    for pdb_id in pdb_ids:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"  Downloading {pdb_id.upper()} from RCSB…")
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(url, tmp_path)
            data = parse_structure_file(tmp_path)
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"  WARNING: skipping {pdb_id} — {exc}")
            continue

        cm = coords_to_contact_map(data.coords, threshold=threshold, sep_min=sep_min)
        sequences.append(data.sequence)
        maps.append(cm)
        print(
            f"  {pdb_id.upper()} chain {data.chain_id}: "
            f"L={len(data.sequence)}, "
            f"contacts={int(cm.sum() // 2)} pairs"
        )

    if not sequences:
        raise RuntimeError("No structures loaded successfully.")

    return ContactDataset(sequences, maps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    k_factor: float = 1.0,
) -> float:
    """Precision among the top k*L highest-confidence predictions.

    Args:
        logits:   (B, L, L) raw contact logits.
        targets:  (B, L, L) ground-truth contacts.
        mask:     (B, L, L) valid pairs (excludes PAD and short-range).
        k_factor: Multiplier on L (1.0 → Precision@L, 0.5 → @L/2, etc.)

    Returns:
        Mean precision across batch samples.
    """
    precisions = []
    B = logits.size(0)

    for b in range(B):
        L = int(mask[b].any(dim=1).sum().item())   # effective length
        k = max(1, int(k_factor * L))

        valid_logits = logits[b][mask[b]]
        valid_targets = targets[b][mask[b]]

        if valid_logits.numel() < k:
            continue

        top_k_idx = valid_logits.topk(k).indices
        precisions.append(valid_targets[top_k_idx].float().mean().item())

    return float(sum(precisions) / max(len(precisions), 1))


def recall_at_threshold(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    threshold: float = 0.5,
) -> float:
    """Recall at a fixed sigmoid probability threshold."""
    probs = torch.sigmoid(logits)
    tp, fn = 0, 0
    B = logits.size(0)
    for b in range(B):
        t = targets[b][mask[b]]
        p = (probs[b][mask[b]] >= threshold).float()
        tp += (p * t).sum().item()
        fn += ((1 - p) * t).sum().item()
    return tp / max(tp + fn, 1)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def make_loaders(
    dataset: ContactDataset,
    val_fraction: float = 0.2,
    batch_size: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_contact
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_contact
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_kind: LossKind,
    sep_min: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Run one pass.  Returns (mean_loss, P@L, P@L/5, recall@0.5)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_logits, all_targets, all_masks = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for tokens, contacts, lengths in loader:
            tokens = tokens.to(device)
            contacts = contacts.to(device)
            lengths = lengths.to(device)

            logits = model(tokens)                                    # (B, L, L)
            mask = make_contact_mask(lengths, logits.size(-1),
                                     sep_min=sep_min, device=device)  # (B, L, L)

            loss = contact_loss(logits, contacts, mask, kind=loss_kind)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * tokens.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(contacts.cpu())
            all_masks.append(mask.cpu())

    n = len(loader.dataset)   # type: ignore[arg-type]
    all_l = torch.cat([pad_to_same(x, all_logits) for x in all_logits], dim=0) if len(all_logits) > 1 else all_logits[0]
    all_t = torch.cat([pad_to_same(x, all_targets) for x in all_targets], dim=0) if len(all_targets) > 1 else all_targets[0]
    all_m = torch.cat([pad_to_same(x, all_masks) for x in all_masks], dim=0) if len(all_masks) > 1 else all_masks[0]

    p_at_l    = precision_at_k(all_l, all_t, all_m, k_factor=1.0)
    p_at_l5   = precision_at_k(all_l, all_t, all_m, k_factor=0.2)
    rec       = recall_at_threshold(all_l, all_t, all_m)

    return total_loss / n, p_at_l, p_at_l5, rec


def pad_to_same(tensor: Tensor, group: list[Tensor]) -> Tensor:
    """Pad a (B, L, L) tensor to the maximum L in the group."""
    max_L = max(t.size(-1) for t in group)
    L = tensor.size(-1)
    if L == max_L:
        return tensor
    pad = max_L - L
    return torch.nn.functional.pad(tensor, (0, pad, 0, pad))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    if args.pdb_ids:
        print(f"Loading real PDB structures: {args.pdb_ids}")
        pdb_list = [p.strip().upper() for p in args.pdb_ids.split(",")]
        dataset = build_real_dataset(pdb_list, threshold=args.threshold, sep_min=args.sep_min)
    else:
        print(f"Building synthetic dataset ({args.n_samples} sequences)…")
        dataset = build_synthetic_dataset(n=args.n_samples)

    train_loader, val_loader = make_loaders(
        dataset, batch_size=args.batch_size, val_fraction=0.2
    )
    print(
        f"  Train: {len(train_loader.dataset):,} | "   # type: ignore[arg-type]
        f"Val: {len(val_loader.dataset):,}"             # type: ignore[arg-type]
    )

    # --- model ---
    model = build_contact_model(
        kind=args.model, pair_dim=args.pair_dim
    ).to(device)
    print(f"Model: {model}")

    # --- optimisation ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- loop ---
    best_p_at_l = 0.0
    hdr = (f"\n{'Epoch':>5}  {'Loss':>8}  {'P@L':>7}  {'P@L/5':>7}  "
           f"{'Recall':>7}  {'Time':>6}")
    print(hdr)
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _, _ = run_epoch(
            model, train_loader, optimizer, args.loss, args.sep_min, device
        )
        val_loss, p_at_l, p_at_l5, rec = run_epoch(
            model, val_loader, None, args.loss, args.sep_min, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        marker = " *" if p_at_l > best_p_at_l else ""
        best_p_at_l = max(best_p_at_l, p_at_l)

        print(
            f"{epoch:>5}  {val_loss:>8.4f}  {p_at_l:>6.1%}  "
            f"{p_at_l5:>6.1%}  {rec:>6.1%}  {elapsed:>5.1f}s{marker}"
        )

    print(f"\nBest val Precision@L: {best_p_at_l:.1%}")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, save_path)
        print(f"Checkpoint saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a YAML config and return a flat dict aligned with argparse dest names."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    flat: dict = {}
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                flat[k.replace("-", "_")] = v
    return flat


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_ns, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Train contact map predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML experiment config (CLI args override config values).",
    )

    data = p.add_argument_group("data")
    data.add_argument(
        "--pdb-ids", type=str, default=None,
        help="Comma-separated PDB IDs to download (e.g. 1UBQ,4HHB,1CRN). "
             "Omit to use synthetic data.",
    )
    data.add_argument("--n-samples", type=int, default=256,
                      help="Synthetic sequences to generate")
    data.add_argument("--threshold", type=float, default=8.0,
                      help="Ca-Ca contact distance cutoff in Angstroms")
    data.add_argument("--sep-min", type=int, default=6,
                      help="Minimum sequence separation for contact pairs")

    model_g = p.add_argument_group("model")
    model_g.add_argument("--model", choices=["bilstm", "transformer"], default="bilstm")
    model_g.add_argument("--pair-dim", type=int, default=64,
                         help="Contact head pair dimension")
    model_g.add_argument("--loss", choices=["bce", "focal"], default="bce")

    train_g = p.add_argument_group("training")
    train_g.add_argument("--epochs",     type=int,   default=10)
    train_g.add_argument("--batch-size", type=int,   default=4)
    train_g.add_argument("--lr",         type=float, default=1e-3)
    train_g.add_argument("--save",       type=str,   default=None)

    if pre_ns.config:
        p.set_defaults(**_load_config(pre_ns.config))

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
