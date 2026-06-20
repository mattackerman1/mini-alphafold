"""
Coarse 3D structure prediction training script (Phase 2).

Trains a StructureModule end-to-end to predict Cα backbone coordinates
using FAPE loss, then evaluates with TM-score and GDT-TS.

Data sources
------------
  Synthetic (default):  random sequences with helix/sheet coordinate patterns.
                         No download — validates the training pipeline.
  Real (--pdb-ids):     Comma-separated PDB IDs downloaded from RCSB.
                         Example: --pdb-ids 1UBQ,4HHB,1CRN,2KHO,1VII

Metrics (after Kabsch alignment)
----------------------------------
  RMSD        : Cα root-mean-square deviation (Å) — lower is better.
  TM-score    : Template modelling score [0, 1] — > 0.5 ≈ same fold.
  GDT-TS      : % residues within 1/2/4/8 Å — higher is better.

Usage
-----
  # Synthetic smoke-test
  python scripts/train_structure.py

  # Real structures, Transformer encoder
  python scripts/train_structure.py --pdb-ids 1UBQ,1CRN,4HHB,1L2Y,1VII \\
      --model transformer --epochs 30 --lr 5e-4

  python scripts/train_structure.py --help
"""

from __future__ import annotations

import argparse
import math
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
from src.models.structure_module import ModelKind, StructureOutput, build_structure_model
from src.training.losses import backbone_rmsd, fape_loss, make_backbone_frames

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StructureDataset(Dataset):
    """Pairs of (token sequence, backbone coordinates).

    Args:
        sequences:  List of amino acid strings.
        coords:     List of (L, 4, 3) float32 tensors — atom order [N, CA, C, O].
    """

    def __init__(self, sequences: list[str], coords: list[Tensor]) -> None:
        assert len(sequences) == len(coords)
        self.tokens: list[Tensor] = [tokenize(s) for s in sequences]
        self.coords = coords

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.tokens[idx], self.coords[idx]


def collate_structure(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad tokens and backbone coords to the batch maximum length.

    Returns:
        tokens:  (B, L) long
        coords:  (B, L, 4, 3) float32 — padded with zeros
        lengths: (B,) int
    """
    seqs, coord_list = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    L = int(lengths.max().item())
    B = len(seqs)

    tokens = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)

    coords_padded = torch.zeros(B, L, 4, 3, dtype=torch.float32)
    for i, c in enumerate(coord_list):
        l = c.size(0)
        coords_padded[i, :l] = c

    return tokens, coords_padded, lengths


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _helix_coords(length: int) -> Tensor:
    """Approximate alpha-helix backbone (all 4 atoms placed at Cα-like positions)."""
    coords = torch.zeros(length, 4, 3)
    r, rise, twist = 2.3, 1.5, math.radians(100)
    for i in range(length):
        angle = i * twist
        ca = torch.tensor([r * math.cos(angle), r * math.sin(angle), i * rise])
        # Approximate N and C by small offsets along helix axis
        n  = ca + torch.tensor([-0.5,  0.1, -0.4])
        c  = ca + torch.tensor([ 0.5, -0.1,  0.4])
        o  = c  + torch.tensor([ 0.0,  0.0,  1.2])
        coords[i] = torch.stack([n, ca, c, o])
    return coords


def build_synthetic_dataset(
    n: int = 128,
    min_len: int = 20,
    max_len: int = 80,
    seed: int = 42,
) -> StructureDataset:
    """Random sequences with approximate helix-like coordinates (+ noise).

    The coordinates are geometrically plausible enough to test the training
    pipeline, but carry no real structural signal.
    """
    rng = random.Random(seed)
    sequences, coords_list = [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        seq = "".join(rng.choice(_AA) for _ in range(length))
        c = _helix_coords(length) + torch.randn(length, 4, 3) * 0.5
        sequences.append(seq)
        coords_list.append(c)
    return StructureDataset(sequences, coords_list)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def build_real_dataset(pdb_ids: list[str]) -> StructureDataset:
    """Download PDB files from RCSB and build a StructureDataset."""
    try:
        from src.data.pdb_parser import parse_structure_file
    except ImportError as exc:
        raise ImportError("pdb_parser requires Biopython: pip install biopython") from exc

    sequences, coords_list = [], []
    for pdb_id in pdb_ids:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"  Downloading {pdb_id.upper()}…")
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(url, tmp_path)
            data = parse_structure_file(tmp_path)
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"  WARNING: skipping {pdb_id} — {exc}")
            continue

        sequences.append(data.sequence)
        coords_list.append(data.coords)
        print(f"  {pdb_id.upper()} chain {data.chain_id}: L={len(data.sequence)}")

    if not sequences:
        raise RuntimeError("No structures loaded.")
    return StructureDataset(sequences, coords_list)


# ---------------------------------------------------------------------------
# Evaluation metrics (require Kabsch alignment)
# ---------------------------------------------------------------------------

def kabsch_align(P: Tensor, Q: Tensor) -> Tensor:
    """Align point cloud P onto Q using the Kabsch algorithm.

    Args:
        P: (L, 3) — predicted coordinates (to be rotated).
        Q: (L, 3) — target coordinates (reference, unchanged).

    Returns:
        P_aligned: (L, 3) — P after optimal rotation + translation onto Q.
    """
    P_c = P - P.mean(dim=0, keepdim=True)
    Q_c = Q - Q.mean(dim=0, keepdim=True)

    H = P_c.T @ Q_c                                   # (3, 3) covariance
    U, _, Vt = torch.linalg.svd(H, full_matrices=True)

    # Correct for reflection (det < 0 means improper rotation)
    d = torch.linalg.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=P.device))
    R = Vt.T @ D @ U.T                                # (3, 3) proper rotation

    return P_c @ R.T + Q.mean(dim=0, keepdim=True)


def _d0(L: int) -> float:
    """TM-score length normalisation parameter."""
    if L < 22:
        return 0.5
    return 1.24 * (L - 15) ** (1 / 3) - 1.8


def tm_score(pred_ca: Tensor, true_ca: Tensor, mask: Tensor) -> float:
    """TM-score after Kabsch alignment (per-sample, averaged over batch).

    Args:
        pred_ca: (B, L, 3)
        true_ca: (B, L, 3)
        mask:    (B, L) bool — non-PAD residues

    Returns:
        Mean TM-score across the batch.
    """
    scores = []
    for b in range(pred_ca.size(0)):
        m = mask[b]
        if m.sum() < 3:
            continue
        P = pred_ca[b][m]
        Q = true_ca[b][m]
        L = Q.size(0)
        d0 = _d0(L)
        P_al = kabsch_align(P, Q)
        di = (P_al - Q).pow(2).sum(-1).sqrt()         # (L,)
        score = (1 / (1 + (di / d0) ** 2)).mean().item()
        scores.append(score)
    return sum(scores) / max(len(scores), 1)


def gdt_ts(pred_ca: Tensor, true_ca: Tensor, mask: Tensor) -> float:
    """GDT-TS score after Kabsch alignment.

    GDT-TS = (GDT_P1 + GDT_P2 + GDT_P4 + GDT_P8) / 4
    where GDT_Pk = fraction of residues within k Å.

    Returns:
        Mean GDT-TS across the batch (0–1 scale).
    """
    scores = []
    thresholds = [1.0, 2.0, 4.0, 8.0]
    for b in range(pred_ca.size(0)):
        m = mask[b]
        if m.sum() < 3:
            continue
        P = pred_ca[b][m]
        Q = true_ca[b][m]
        P_al = kabsch_align(P, Q)
        di = (P_al - Q).pow(2).sum(-1).sqrt()
        gdt = sum((di < t).float().mean().item() for t in thresholds) / 4.0
        scores.append(gdt)
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    d_clamp: float,
) -> tuple[float, float, float, float]:
    """Returns (mean_fape, mean_rmsd, mean_tm, mean_gdt)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_fape = 0.0
    all_pred, all_true, all_masks = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for tokens, coords, lengths in loader:
            tokens = tokens.to(device)
            coords = coords.to(device)

            seq_mask = torch.arange(tokens.size(1), device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)

            out: StructureOutput = model(tokens)
            true_R, true_t = make_backbone_frames(coords)   # (B, L, 3, 3), (B, L, 3)

            loss = fape_loss(
                pred_t=out.ca_coords,
                true_t=true_t,
                pred_R=out.rotations if out.has_frames else true_R,
                true_R=true_R,
                seq_mask=seq_mask,
                d_clamp=d_clamp,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_fape += loss.item() * tokens.size(0)
            all_pred.append(out.ca_coords.detach().cpu())
            all_true.append(true_t.cpu())
            all_masks.append(seq_mask.cpu())

    n = len(loader.dataset)   # type: ignore[arg-type]

    # Collect eval tensors (pad to common L for metric computation)
    max_L = max(t.size(1) for t in all_pred)

    def pad2d(t: Tensor) -> Tensor:
        pad = max_L - t.size(1)
        return torch.nn.functional.pad(t, (0, 0, 0, pad)) if pad > 0 else t

    pred_cat = torch.cat([pad2d(t) for t in all_pred], dim=0)
    true_cat = torch.cat([pad2d(t) for t in all_true], dim=0)
    mask_cat = torch.cat([
        torch.nn.functional.pad(m, (0, max_L - m.size(1))) for m in all_masks
    ], dim=0)

    rmsd = backbone_rmsd(pred_cat, true_cat, mask_cat).item()
    tm   = tm_score(pred_cat, true_cat, mask_cat)
    gdt  = gdt_ts(pred_cat, true_cat, mask_cat)

    return total_fape / n, rmsd, tm, gdt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    if args.pdb_ids:
        pdb_list = [p.strip().upper() for p in args.pdb_ids.split(",")]
        print(f"Loading real PDB structures: {pdb_list}")
        dataset = build_real_dataset(pdb_list)
    else:
        print(f"Building synthetic dataset ({args.n_samples} sequences)…")
        dataset = build_synthetic_dataset(n=args.n_samples)

    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_structure)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_structure)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # --- model ---
    model = build_structure_model(
        kind=args.model, use_frames=True
    ).to(device)
    print(f"Model: {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- loop ---
    best_tm = 0.0
    hdr = (f"\n{'Epoch':>5}  {'FAPE':>7}  {'RMSD':>7}  "
           f"{'TM-score':>8}  {'GDT-TS':>7}  {'Time':>6}")
    print(hdr)
    print("-" * 54)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_fape, _, _, _ = run_epoch(
            model, train_loader, optimizer, device, args.d_clamp
        )
        val_fape, rmsd, tm, gdt = run_epoch(
            model, val_loader, None, device, args.d_clamp
        )
        scheduler.step()

        elapsed = time.time() - t0
        marker = " *" if tm > best_tm else ""
        best_tm = max(best_tm, tm)

        print(
            f"{epoch:>5}  {val_fape:>7.4f}  {rmsd:>7.2f}  "
            f"{tm:>8.4f}  {gdt:>7.3f}  {elapsed:>5.1f}s{marker}"
        )

    print(f"\nBest val TM-score: {best_tm:.4f}")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, save_path)
        print(f"Checkpoint saved → {save_path}")


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
        description="Train coarse 3D structure predictor (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML experiment config (CLI args override config values).",
    )

    data = p.add_argument_group("data")
    data.add_argument(
        "--pdb-ids", type=str, default=None,
        help="Comma-separated PDB IDs (e.g. 1UBQ,4HHB). Omit for synthetic.",
    )
    data.add_argument("--n-samples", type=int, default=128)

    model_g = p.add_argument_group("model")
    model_g.add_argument("--model", choices=["bilstm", "transformer"], default="bilstm")

    train_g = p.add_argument_group("training")
    train_g.add_argument("--epochs",     type=int,   default=10)
    train_g.add_argument("--batch-size", type=int,   default=4)
    train_g.add_argument("--lr",         type=float, default=1e-3)
    train_g.add_argument("--d-clamp",    type=float, default=10.0,
                         help="FAPE clamping distance in Angstroms")
    train_g.add_argument("--save",       type=str,   default=None)

    if pre_ns.config:
        p.set_defaults(**_load_config(pre_ns.config))

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
