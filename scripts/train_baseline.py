"""
Baseline training script for secondary structure prediction.

Uses synthetic data by default so the script runs immediately without any
external dataset.  Replace the `build_synthetic_dataset` call with real data
once the CATH / SCOPe loader exists (Phase 0).

Usage
-----
  python scripts/train_baseline.py                      # BiLSTM, SS3, synthetic
  python scripts/train_baseline.py --model transformer  # Transformer variant
  python scripts/train_baseline.py --epochs 20 --lr 3e-4 --batch-size 32
  python scripts/train_baseline.py --help
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

# Make the repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.protein_dataset import PAD_IDX, ProteinDataset
from src.models.baseline import SS3_CLASSES, SS8_CLASSES, ModelKind, build_model

# ---------------------------------------------------------------------------
# Synthetic dataset (placeholder until real data loader is ready)
# ---------------------------------------------------------------------------

_AA = "ACDEFGHIKLMNPQRSTVWY"


def build_synthetic_dataset(
    n: int = 512,
    min_len: int = 20,
    max_len: int = 100,
    n_classes: int = SS3_CLASSES,
    seed: int = 42,
) -> ProteinDataset:
    """Generate random amino acid sequences with random SS labels.

    This is purely for smoke-testing the training pipeline.
    """
    rng = random.Random(seed)
    sequences, labels = [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        seq = "".join(rng.choice(_AA) for _ in range(length))
        lbl = torch.tensor([rng.randrange(n_classes) for _ in range(length)], dtype=torch.long)
        sequences.append(seq)
        labels.append(lbl)
    return ProteinDataset(sequences, labels=labels)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Pad a batch of (tokens, labels) pairs to the same length."""
    seqs, lbls = zip(*batch)
    return (
        pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX),
        pad_sequence(lbls, batch_first=True, padding_value=-1),   # -1 → ignored by loss
    )


def make_loaders(
    dataset: ProteinDataset,
    val_fraction: float = 0.15,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# One epoch of training / evaluation
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over `loader`.

    Returns:
        (mean_loss, accuracy) — both scalars, accuracy ignores PAD positions.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total_tokens = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for tokens, labels in loader:
            tokens = tokens.to(device)   # (B, L)
            labels = labels.to(device)   # (B, L)

            logits = model(tokens)       # (B, L, C)

            # Flatten for cross-entropy: (B*L, C) vs (B*L,)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * tokens.size(0)

            # Accuracy: only count non-padding positions (label != -1)
            mask = labels != -1
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

    n = len(loader.dataset)  # type: ignore[arg-type]
    return total_loss / n, correct / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    print("Building synthetic dataset...")
    dataset = build_synthetic_dataset(n=args.n_samples, n_classes=args.n_classes)
    train_loader, val_loader = make_loaders(
        dataset, batch_size=args.batch_size, val_fraction=0.15
    )
    print(
        f"  Train: {len(train_loader.dataset)} samples | "  # type: ignore[arg-type]
        f"Val: {len(val_loader.dataset)} samples"           # type: ignore[arg-type]
    )

    # --- model ---
    model = build_model(kind=args.model, n_classes=args.n_classes).to(device)
    print(f"Model: {args.model}  |  Parameters: {model.count_parameters():,}")

    # --- loss: ignore padding (-1) and weight classes equally ---
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- loop ---
    best_val_acc = 0.0
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        elapsed = time.time() - t0
        marker = " *" if val_acc > best_val_acc else ""
        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
            f"{val_loss:>8.4f}  {val_acc:>6.1%}  {elapsed:>5.1f}s{marker}"
        )

    print(f"\nBest val accuracy: {best_val_acc:.1%}")

    # --- optional checkpoint save ---
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, save_path)
        print(f"Checkpoint saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline secondary structure model")
    p.add_argument("--model", choices=["bilstm", "transformer"], default="bilstm",
                   help="Model architecture (default: bilstm)")
    p.add_argument("--n-classes", type=int, choices=[3, 8], default=SS3_CLASSES,
                   help="Number of SS classes: 3 (SS3) or 8 (SS8) (default: 3)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--n-samples", type=int, default=512,
                   help="Number of synthetic sequences (default: 512)")
    p.add_argument("--save", type=str, default=None,
                   help="Path to save model checkpoint, e.g. checkpoints/baseline.pt")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
