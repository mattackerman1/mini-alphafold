"""
End-to-end MSA-aware 3D structure prediction (Phase 3).

Pipeline
--------
  MSA features  →  EvoformerStack  →  query_repr (B, L, c_m)
                                    →  CoordHead  →  (B, L, 3)  Cα coords
                                    →  FrameHead  →  (B, L, 3, 3) rotations
  Loss: FAPE (Frame-Aligned Point Error, AF2 §1.9.1)
  Eval: Kabsch-aligned TM-score and GDT-TS

Data sources
------------
  Synthetic (default):  random sequences with helix-like coordinates.
                         No download needed — validates the full pipeline.
  Real (--pdb-ids):     PDB IDs downloaded from RCSB; pseudo-MSAs are
                         generated automatically (no external MSA tool needed).

Usage
-----
  # Synthetic smoke-test (fast)
  python scripts/train_end_to_end.py

  # Real structures, 4-block Evoformer
  python scripts/train_end_to_end.py \\
      --pdb-ids 1UBQ,1CRN,4HHB,1L2Y,1VII \\
      --n-blocks 4 --c-m 64 --c-z 32 \\
      --epochs 30 --lr 5e-4 --n-pseudo 16 --save ckpt_phase3.pt

  # Disable triangle updates (faster, ablation)
  python scripts/train_end_to_end.py --no-triangle

  python scripts/train_end_to_end.py --help
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

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import MSADataset, collate_msa_fn
from src.data.protein_dataset import PAD_IDX, tokenize
from src.models.structure_module import StructureOutput, build_msa_structure_model
from src.training.losses import backbone_rmsd, fape_loss, make_backbone_frames


# ---------------------------------------------------------------------------
# Combined MSA + structure dataset
# ---------------------------------------------------------------------------

class MSAStructureDataset(Dataset):
    """Dataset returning (tokens, msa_features, backbone_coords) per sample.

    Wraps MSADataset to add structure coordinates.  Pseudo-MSAs are generated
    automatically from the query sequence using random mutation.

    Args:
        sequences:     List of amino-acid strings.
        coords:        List of (L, 4, 3) float32 tensors — atom order [N, CA, C, O].
        n_pseudo:      Number of pseudo-MSA homologues per query (default 8).
        mutation_rate: Per-residue substitution rate for pseudo-MSA (default 0.15).
        seed:          Optional RNG seed for reproducible pseudo-MSAs.
    """

    def __init__(
        self,
        sequences: list[str],
        coords:    list[Tensor],
        n_pseudo:      int   = 8,
        mutation_rate: float = 0.15,
        seed:          Optional[int] = None,
    ) -> None:
        assert len(sequences) == len(coords), "sequences and coords must have equal length"
        self._msa_ds = MSADataset(
            sequences,
            n_pseudo=n_pseudo,
            mutation_rate=mutation_rate,
            seed=seed,
        )
        self._coords = coords

    def __len__(self) -> int:
        return len(self._msa_ds)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        tokens, msa = self._msa_ds[idx]       # (L,), (N_seq, L, 45)
        return tokens, msa, self._coords[idx]  # (L, 4, 3)

    def __repr__(self) -> str:
        return (
            f"MSAStructureDataset(n={len(self)}, "
            f"n_pseudo={len(self._msa_ds._msa_seqs[0]) - 1})"
        )


def collate_msa_structure(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad a variable-length batch to a common sequence length.

    Returns:
        tokens:  (B, L_max) long
        msa:     (B, N_seq, L_max, 45) float32
        coords:  (B, L_max, 4, 3) float32 — padded with zeros
        lengths: (B,) long — actual sequence length per sample
    """
    tokens_list, msa_list, coords_list = zip(*batch)

    lengths = torch.tensor([t.size(0) for t in tokens_list], dtype=torch.long)
    L_max   = int(lengths.max().item())
    B       = len(tokens_list)
    N_seq   = msa_list[0].shape[0]
    feat_dim = msa_list[0].shape[2]

    # Pad tokens
    tokens = pad_sequence(tokens_list, batch_first=True, padding_value=PAD_IDX)

    # Pad MSA features: (B, N_seq, L_max, feat_dim)
    msa = torch.zeros(B, N_seq, L_max, feat_dim)
    for b, m in enumerate(msa_list):
        L_b = m.shape[1]
        msa[b, :, :L_b, :] = m

    # Pad backbone coordinates: (B, L_max, 4, 3)
    coords = torch.zeros(B, L_max, 4, 3, dtype=torch.float32)
    for b, c in enumerate(coords_list):
        L_b = c.shape[0]
        coords[b, :L_b] = c

    return tokens, msa, coords, lengths


# ---------------------------------------------------------------------------
# Synthetic data (no download)
# ---------------------------------------------------------------------------

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _helix_coords(length: int) -> Tensor:
    """Approximate alpha-helix backbone (all 4 atoms placed near Cα)."""
    coords = torch.zeros(length, 4, 3)
    r, rise, twist = 2.3, 1.5, math.radians(100)
    for i in range(length):
        angle = i * twist
        ca = torch.tensor([r * math.cos(angle), r * math.sin(angle), i * rise])
        n  = ca + torch.tensor([-0.5,  0.1, -0.4])
        c  = ca + torch.tensor([ 0.5, -0.1,  0.4])
        o  = c  + torch.tensor([ 0.0,  0.0,  1.2])
        coords[i] = torch.stack([n, ca, c, o])
    return coords


def build_synthetic_msa_dataset(
    n:            int   = 64,
    min_len:      int   = 20,
    max_len:      int   = 60,
    n_pseudo:     int   = 8,
    mutation_rate: float = 0.15,
    seed:         int   = 42,
) -> MSAStructureDataset:
    """Random sequences + approximate helix coordinates + pseudo-MSAs."""
    rng = random.Random(seed)
    sequences, coords_list = [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        seq    = "".join(rng.choice(_AA) for _ in range(length))
        c      = _helix_coords(length) + torch.randn(length, 4, 3) * 0.5
        sequences.append(seq)
        coords_list.append(c)
    return MSAStructureDataset(
        sequences, coords_list,
        n_pseudo=n_pseudo, mutation_rate=mutation_rate, seed=seed,
    )


# ---------------------------------------------------------------------------
# Real PDB data
# ---------------------------------------------------------------------------

def build_real_msa_dataset(
    pdb_ids:      list[str],
    n_pseudo:     int   = 8,
    mutation_rate: float = 0.15,
    seed:         int   = 0,
) -> MSAStructureDataset:
    """Download PDB files from RCSB and build MSAStructureDataset."""
    try:
        from src.data.pdb_parser import parse_structure_file
    except ImportError as exc:
        raise ImportError("pdb_parser requires Biopython: pip install biopython") from exc

    sequences, coords_list = [], []
    for pdb_id in pdb_ids:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"  Downloading {pdb_id.upper()}...")
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
        raise RuntimeError("No structures loaded successfully.")

    return MSAStructureDataset(
        sequences, coords_list,
        n_pseudo=n_pseudo, mutation_rate=mutation_rate, seed=seed,
    )


# ---------------------------------------------------------------------------
# Evaluation (Kabsch alignment)
# ---------------------------------------------------------------------------

def kabsch_align(P: Tensor, Q: Tensor) -> Tensor:
    """Align P onto Q via SVD (reflection-corrected)."""
    P_c = P - P.mean(dim=0, keepdim=True)
    Q_c = Q - Q.mean(dim=0, keepdim=True)
    H   = P_c.T @ Q_c
    U, _, Vt = torch.linalg.svd(H, full_matrices=True)
    d = torch.linalg.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d.item()], device=P.device))
    R = Vt.T @ D @ U.T
    return P_c @ R.T + Q.mean(dim=0, keepdim=True)


def _d0(L: int) -> float:
    return 0.5 if L < 22 else 1.24 * (L - 15) ** (1 / 3) - 1.8


def tm_score(pred_ca: Tensor, true_ca: Tensor, mask: Tensor) -> float:
    scores = []
    for b in range(pred_ca.size(0)):
        m = mask[b]
        if m.sum() < 3:
            continue
        P, Q = pred_ca[b][m], true_ca[b][m]
        d0   = _d0(Q.size(0))
        P_al = kabsch_align(P, Q)
        di   = (P_al - Q).pow(2).sum(-1).sqrt()
        scores.append((1 / (1 + (di / d0) ** 2)).mean().item())
    return sum(scores) / max(len(scores), 1)


def gdt_ts(pred_ca: Tensor, true_ca: Tensor, mask: Tensor) -> float:
    scores = []
    for b in range(pred_ca.size(0)):
        m = mask[b]
        if m.sum() < 3:
            continue
        P, Q = pred_ca[b][m], true_ca[b][m]
        P_al = kabsch_align(P, Q)
        di   = (P_al - Q).pow(2).sum(-1).sqrt()
        scores.append(
            sum((di < t).float().mean().item() for t in [1.0, 2.0, 4.0, 8.0]) / 4.0
        )
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    d_clamp:   float,
) -> tuple[float, float, float, float]:
    """Run one train or eval epoch.  Returns (fape, rmsd, tm, gdt)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_fape = 0.0
    all_pred, all_true, all_masks = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for tokens, msa, coords, lengths in loader:
            tokens = tokens.to(device)
            msa    = msa.to(device)
            coords = coords.to(device)

            seq_mask = (
                torch.arange(tokens.size(1), device=device).unsqueeze(0)
                < lengths.to(device).unsqueeze(1)
            )

            # Forward: MSA path when msa_features provided
            out: StructureOutput = model(tokens, msa_features=msa)
            true_R, true_t = make_backbone_frames(coords)

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

    n = len(loader.dataset)  # type: ignore[arg-type]

    # Pad all batches to the same L for metric aggregation
    max_L = max(t.size(1) for t in all_pred)

    def _pad_ca(t: Tensor) -> Tensor:
        # (B, L, 3) → (B, max_L, 3): pad the L dimension
        gap = max_L - t.size(1)
        return torch.nn.functional.pad(t, (0, 0, 0, gap)) if gap > 0 else t

    pred_cat = torch.cat([_pad_ca(t) for t in all_pred], dim=0)
    true_cat = torch.cat([_pad_ca(t) for t in all_true], dim=0)
    mask_cat = torch.cat(
        [torch.nn.functional.pad(m, (0, max_L - m.size(1))) for m in all_masks], dim=0
    )

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
        dataset = build_real_msa_dataset(
            pdb_list, n_pseudo=args.n_pseudo, seed=args.seed,
        )
    else:
        print(f"Building synthetic dataset ({args.n_samples} sequences)...")
        dataset = build_synthetic_msa_dataset(
            n=args.n_samples, n_pseudo=args.n_pseudo, seed=args.seed,
        )
    print(dataset)

    n_val   = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_msa_structure,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_msa_structure,
    )
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # --- model ---
    model = build_msa_structure_model(
        n_blocks=args.n_blocks,
        c_m=args.c_m,
        c_z=args.c_z,
        use_frames=True,
        use_triangle=not args.no_triangle,
    ).to(device)
    print(f"Model: {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- loop ---
    best_tm = 0.0
    hdr = (
        f"\n{'Epoch':>5}  {'FAPE':>7}  {'RMSD':>7}  "
        f"{'TM-score':>8}  {'GDT-TS':>7}  {'Time':>6}"
    )
    print(hdr)
    print("-" * 54)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run_epoch(model, train_loader, optimizer, device, args.d_clamp)
        val_fape, rmsd, tm, gdt = run_epoch(model, val_loader, None, device, args.d_clamp)
        scheduler.step()

        elapsed = time.time() - t0
        marker  = " *" if tm > best_tm else ""
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
        print(f"Checkpoint saved -> {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end MSA-aware structure prediction (Phase 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument(
        "--pdb-ids", type=str, default=None,
        help="Comma-separated PDB IDs. Omit for synthetic data.",
    )
    data.add_argument("--n-samples", type=int, default=64,
                      help="Number of synthetic sequences (ignored when --pdb-ids set).")
    data.add_argument("--n-pseudo",  type=int, default=8,
                      help="Number of pseudo-MSA homologues per query.")
    data.add_argument("--seed",      type=int, default=42)

    evo = p.add_argument_group("evoformer")
    evo.add_argument("--n-blocks",    type=int,   default=2,
                     help="Number of Evoformer blocks.")
    evo.add_argument("--c-m",         type=int,   default=64,
                     help="MSA channel width c_m.")
    evo.add_argument("--c-z",         type=int,   default=32,
                     help="Pair channel width c_z.")
    evo.add_argument("--no-triangle", action="store_true",
                     help="Disable triangle multiplicative updates.")

    train_g = p.add_argument_group("training")
    train_g.add_argument("--epochs",     type=int,   default=10)
    train_g.add_argument("--batch-size", type=int,   default=4)
    train_g.add_argument("--lr",         type=float, default=1e-3)
    train_g.add_argument("--d-clamp",    type=float, default=10.0,
                         help="FAPE clamping distance (Angstroms).")
    train_g.add_argument("--save",       type=str,   default=None,
                         help="Path to save best checkpoint (.pt).")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
