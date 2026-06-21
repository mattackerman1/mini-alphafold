"""
End-to-end MSA-aware 3D structure prediction (Phase 3).

Pipeline
--------
  MSA features  ->  EvoformerStack  ->  query_repr (B, L, c_m)
                                      ->  CoordHead  ->  (B, L, 3)  Ca coords
                                      ->  FrameHead  ->  (B, L, 3, 3) rotations
  Loss: FAPE (Frame-Aligned Point Error, AF2 §1.9.1)
  Eval: Kabsch-aligned TM-score and GDT-TS

Data sources
------------
  Synthetic (default):  random sequences with helix-like coordinates.
                         No download needed -- validates the full pipeline.
  Real (--pdb-ids):     PDB IDs downloaded from RCSB; pseudo-MSAs are
                         generated automatically (no external MSA tool needed).
  Large-scale cached (--dataset-cache):
                         Uses LargeStructureDataset with ~300 curated PDB IDs,
                         gzip-compressed disk caching, and parallel downloads.

Usage
-----
  # Synthetic smoke-test (fast)
  python scripts/train_end_to_end.py

  # Real structures, 4-block Evoformer
  python scripts/train_end_to_end.py \\
      --pdb-ids 1UBQ,1CRN,4HHB,1L2Y,1VII \\
      --n-blocks 4 --c-m 64 --c-z 32 \\
      --epochs 30 --lr 5e-4 --n-pseudo 16 --save ckpt_phase3.pt

  # Quick test with 20 cached structures (fast, validates the pipeline)
  python scripts/train_end_to_end.py \\
      --dataset-cache ~/.cache/mini_alphafold/structures \\
      --n-structures 20 --max-len 100 \\
      --epochs 5 --batch-size 2

  # Full-scale training with 300 PDB structures, TensorBoard, warmup LR
  python scripts/train_end_to_end.py \\
      --dataset-cache ~/.cache/mini_alphafold/structures \\
      --n-structures 300 --n-blocks 4 --c-m 128 --c-z 64 \\
      --epochs 100 --batch-size 8 --lr 3e-4 \\
      --num-workers 4 --pin-memory \\
      --warmup-steps 500 \\
      --log-dir runs/phase3_large \\
      --ckpt-every 10 --save ckpt_phase3_large.pt

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

import yaml
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
# TensorBoard (optional)
# ---------------------------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # type: ignore[import]
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

    class _SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback when tensorboard is not installed."""
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass


# ---------------------------------------------------------------------------
# Combined MSA + structure dataset
# ---------------------------------------------------------------------------

class MSAStructureDataset(Dataset):
    """Dataset returning (tokens, msa_features, backbone_coords) per sample.

    Supports both pseudo-MSAs (default) and real pre-cached MSAs from the
    ColabFold API (via --msa-cache).  When a cache directory is provided,
    each sequence's real MSA is loaded from disk; sequences without a cache
    entry fall back to pseudo-MSA automatically.

    Args:
        sequences:     List of amino-acid strings.
        coords:        List of (L, 4, 3) float32 tensors -- atom order [N, CA, C, O].
        n_pseudo:      Number of pseudo-MSA homologues per query (default 8).
        mutation_rate: Per-residue substitution rate for pseudo-MSA (default 0.15).
        seed:          Optional RNG seed for reproducible pseudo-MSAs.
        msa_cache_dir: Optional path to a pre-populated MSA cache directory.
                       When provided, real MSAs are loaded from cache; missing
                       entries fall back to pseudo-MSA silently.
        max_msa_depth: Maximum number of MSA sequences to use per sample.
    """

    def __init__(
        self,
        sequences: list[str],
        coords:    list[Tensor],
        n_pseudo:      int   = 8,
        mutation_rate: float = 0.15,
        seed:          Optional[int] = None,
        msa_cache_dir: Optional[str] = None,
        max_msa_depth: int = 512,
    ) -> None:
        assert len(sequences) == len(coords), "sequences and coords must have equal length"

        # Resolve MSA sequences: real from cache, or pseudo-generated
        if msa_cache_dir is not None:
            from src.data.msa_fetcher import load_cached_msa
            msa_seqs_list = []
            n_real = 0
            for seq in sequences:
                msa = load_cached_msa(
                    seq,
                    cache_dir=msa_cache_dir,
                    max_msa_depth=max_msa_depth,
                    fallback_pseudo=True,
                    n_pseudo=n_pseudo,
                )
                msa_seqs_list.append(msa)
                if not all(s == seq or "-" in s for s in msa[1:]):
                    n_real += 1
            print(f"  MSA source: {n_real}/{len(sequences)} from real cache, "
                  f"{len(sequences)-n_real} pseudo-MSA fallback")
        else:
            msa_seqs_list = None

        self._msa_ds = MSADataset(
            sequences,
            msa_sequences=msa_seqs_list,
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
        coords:  (B, L_max, 4, 3) float32 -- padded with zeros
        lengths: (B,) long -- actual sequence length per sample
    """
    tokens_list, msa_list, coords_list = zip(*batch)

    lengths = torch.tensor([t.size(0) for t in tokens_list], dtype=torch.long)
    L_max   = int(lengths.max().item())
    B       = len(tokens_list)
    N_seq   = msa_list[0].shape[0]
    feat_dim = msa_list[0].shape[2]

    tokens = pad_sequence(tokens_list, batch_first=True, padding_value=PAD_IDX)

    msa = torch.zeros(B, N_seq, L_max, feat_dim)
    for b, m in enumerate(msa_list):
        L_b = m.shape[1]
        msa[b, :, :L_b, :] = m

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
    """Approximate alpha-helix backbone (all 4 atoms placed near Ca)."""
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
# Real PDB data (small set, direct download)
# ---------------------------------------------------------------------------

def build_real_msa_dataset(
    pdb_ids:       list[str],
    n_pseudo:      int   = 8,
    mutation_rate: float = 0.15,
    seed:          int   = 0,
    msa_cache_dir: Optional[str] = None,
    max_msa_depth: int   = 512,
) -> MSAStructureDataset:
    """Download PDB files from RCSB and build MSAStructureDataset.

    Args:
        pdb_ids:       List of 4-character PDB IDs.
        n_pseudo:      Pseudo-MSA depth for sequences without a cache entry.
        mutation_rate: Mutation rate for pseudo-MSA generation.
        seed:          RNG seed.
        msa_cache_dir: Optional path to a pre-populated MSA cache directory
                       (created by scripts/prefetch_msas.py). When provided,
                       real MSAs are used; missing entries fall back to pseudo.
        max_msa_depth: Maximum MSA sequences per sample.
    """
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
            print(f"  WARNING: skipping {pdb_id} -- {exc}")
            continue
        sequences.append(data.sequence)
        coords_list.append(data.coords)
        print(f"  {pdb_id.upper()} chain {data.chain_id}: L={len(data.sequence)}")

    if not sequences:
        raise RuntimeError("No structures loaded successfully.")

    return MSAStructureDataset(
        sequences, coords_list,
        n_pseudo=n_pseudo, mutation_rate=mutation_rate, seed=seed,
        msa_cache_dir=msa_cache_dir, max_msa_depth=max_msa_depth,
    )


# ---------------------------------------------------------------------------
# Large-scale cached dataset
# ---------------------------------------------------------------------------

def build_cached_dataset(
    cache_dir:    str,
    n_pseudo:     int   = 8,
    mutation_rate: float = 0.15,
    n_workers:    int   = 8,
    seed:         int   = 42,
    min_len:      int   = 30,
    max_len:      int   = 300,
    val_fraction: float = 0.1,
    n_structures: int   = 300,
    msa_cache_dir: Optional[str] = None,
    max_msa_depth: int   = 512,
):
    """Build train/val datasets from the curated structure cache.

    Downloads missing structures in parallel, caches as gzip pickles, then
    returns (train_ds, val_ds) as LargeStructureDataset objects.

    Args:
        cache_dir:    Path to disk cache (created if absent).
        n_pseudo:     Pseudo-MSA homologues per query.
        mutation_rate: Per-residue mutation rate for pseudo-MSA.
        n_workers:    Parallel download workers.
        seed:         RNG seed for train/val split.
        min_len / max_len: Length filter applied during loading.
        val_fraction: Fraction of data reserved for validation.
        n_structures: Cap on total structures drawn from CURATED_PDB_IDS.

    Returns:
        (train_ds, val_ds) — LargeStructureDataset instances.
    """
    from src.data.structure_dataset import build_large_dataset

    print(f"Loading large-scale dataset from cache: {cache_dir}")
    train_ds, val_ds = build_large_dataset(
        cache_dir=cache_dir,
        n_pseudo=n_pseudo,
        mutation_rate=mutation_rate,
        n_workers=n_workers,
        seed=seed,
        min_len=min_len,
        max_len=max_len,
        val_fraction=val_fraction,
        n_structures=n_structures,
        msa_cache_dir=msa_cache_dir,
        max_msa_depth=max_msa_depth,
    )
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_ds, val_ds


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
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_fraction: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Step-level schedule: linearly warm up, then cosine decay to min_lr_fraction * base_lr."""
    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    d_clamp:   float,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
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
                if scheduler is not None:
                    scheduler.step()

            total_fape += loss.item() * tokens.size(0)
            all_pred.append(out.ca_coords.detach().cpu())
            all_true.append(true_t.cpu())
            all_masks.append(seq_mask.cpu())

    n = len(loader.dataset)  # type: ignore[arg-type]

    max_L = max(t.size(1) for t in all_pred)

    def _pad_ca(t: Tensor) -> Tensor:
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
    if args.dataset_cache:
        train_ds, val_ds = build_cached_dataset(
            cache_dir=args.dataset_cache,
            n_pseudo=args.n_pseudo,
            n_workers=max(1, args.num_workers),
            seed=args.seed,
            n_structures=args.n_structures,
            min_len=args.min_len,
            max_len=args.max_len,
            msa_cache_dir=args.msa_cache,
            max_msa_depth=args.max_msa_depth,
        )
    elif args.pdb_ids:
        pdb_list = [p.strip().upper() for p in args.pdb_ids.split(",")]
        print(f"Loading real PDB structures: {pdb_list}")
        if args.msa_cache:
            print(f"Using real MSA cache: {args.msa_cache}")
        dataset = build_real_msa_dataset(
            pdb_list, n_pseudo=args.n_pseudo, seed=args.seed,
            msa_cache_dir=args.msa_cache,
            max_msa_depth=args.max_msa_depth,
        )
        n_val   = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    else:
        print(f"Building synthetic dataset ({args.n_samples} sequences)...")
        dataset = build_synthetic_msa_dataset(
            n=args.n_samples, n_pseudo=args.n_pseudo, seed=args.seed,
        )
        n_val   = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    loader_kw = dict(
        batch_size=args.batch_size,
        collate_fn=collate_msa_structure,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # --- model ---
    model = build_msa_structure_model(
        n_blocks=args.n_blocks,
        c_m=args.c_m,
        c_z=args.c_z,
        use_frames=True,
        use_triangle=not args.no_triangle,
        dropout=args.dropout,
    ).to(device)
    print(f"Model: {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_steps,
                                   min_lr_fraction=args.min_lr / args.lr)

    # --- TensorBoard ---
    writer = _SummaryWriter(args.log_dir) if args.log_dir else _SummaryWriter()
    if args.log_dir:
        if _TB_AVAILABLE:
            print(f"TensorBoard log dir: {args.log_dir}")
        else:
            print(
                "WARNING: tensorboard not installed; --log-dir has no effect.\n"
                "         Install with: pip install tensorboard"
            )

    # --- loop ---
    best_tm = 0.0
    hdr = (
        f"\n{'Epoch':>5}  {'TrainFAPE':>9}  {'ValFAPE':>7}  {'RMSD':>7}  "
        f"{'TM-score':>8}  {'GDT-TS':>7}  {'LR':>8}  {'Time':>6}"
    )
    print(hdr)
    print("-" * 74)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_fape, _, _, _ = run_epoch(
            model, train_loader, optimizer, device, args.d_clamp, scheduler,
        )
        val_fape, rmsd, tm, gdt = run_epoch(
            model, val_loader, None, device, args.d_clamp,
        )

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        improved = tm > best_tm
        marker   = " *" if improved else ""
        best_tm  = max(best_tm, tm)

        print(
            f"{epoch:>5}  {train_fape:>9.4f}  {val_fape:>7.4f}  {rmsd:>7.2f}  "
            f"{tm:>8.4f}  {gdt:>7.3f}  {current_lr:>8.2e}  {elapsed:>5.1f}s{marker}"
        )

        # TensorBoard
        writer.add_scalar("loss/train_fape",  train_fape, epoch)
        writer.add_scalar("loss/val_fape",    val_fape,   epoch)
        writer.add_scalar("metrics/val_rmsd", rmsd,       epoch)
        writer.add_scalar("metrics/val_tm",   tm,         epoch)
        writer.add_scalar("metrics/val_gdt",  gdt,        epoch)
        writer.add_scalar("lr",               current_lr, epoch)

        # Best-model checkpoint (overwrites on TM improvement)
        if improved and args.save:
            best_path = Path(args.save)
            _save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_tm, args)
            print(f"  Best model saved (TM={best_tm:.4f}) -> {best_path}")

        # Periodic checkpoint
        if args.ckpt_every and (epoch % args.ckpt_every == 0):
            ckpt_base = Path(args.save or "checkpoint.pt")
            periodic_path = ckpt_base.with_stem(f"{ckpt_base.stem}_epoch{epoch:04d}")
            _save_checkpoint(periodic_path, model, optimizer, scheduler, epoch, best_tm, args)

    print(f"\nBest val TM-score: {best_tm:.4f}")
    writer.close()

    # Always write the final model state as a separate file
    if args.save:
        last_path = Path(args.save).with_stem(Path(args.save).stem + "_last")
        _save_checkpoint(last_path, model, optimizer, scheduler, args.epochs, best_tm, args)
        print(f"Last checkpoint  -> {last_path}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    best_tm: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,
            "best_tm":         best_tm,
            "args":            vars(args),
        },
        path,
    )
    print(f"  Checkpoint -> {path}")


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
        description="End-to-end MSA-aware structure prediction (Phase 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML experiment config (CLI args override config values).",
    )

    data = p.add_argument_group("data")
    data.add_argument(
        "--dataset-cache", type=str, default=None,
        help="Path to structure disk cache. Uses LargeStructureDataset (~300 curated PDBs).",
    )
    data.add_argument(
        "--pdb-ids", type=str, default=None,
        help="Comma-separated PDB IDs (small set). Omit for synthetic data.",
    )
    data.add_argument("--n-samples",    type=int, default=64,
                      help="Number of synthetic sequences (synthetic mode only).")
    data.add_argument("--n-structures", type=int, default=300,
                      help="Max structures drawn from CURATED_PDB_IDS (cached mode).")
    data.add_argument("--min-len",      type=int, default=30,
                      help="Skip structures shorter than this (cached mode).")
    data.add_argument("--max-len",      type=int, default=300,
                      help="Skip structures longer than this (cached mode).")
    data.add_argument("--n-pseudo",     type=int, default=8,
                      help="Number of pseudo-MSA homologues per query.")
    data.add_argument("--msa-cache",    type=str, default=None,
                      help=(
                          "Path to a pre-populated MSA cache directory created by "
                          "scripts/prefetch_msas.py. When provided, real MSAs are "
                          "loaded from cache instead of pseudo-MSAs. Sequences "
                          "without a cache entry fall back to pseudo-MSA silently."
                      ))
    data.add_argument("--max-msa-depth", type=int, default=512,
                      help="Maximum MSA depth (number of homologs) per sample.")
    data.add_argument("--seed",         type=int, default=42)

    evo = p.add_argument_group("evoformer")
    evo.add_argument("--n-blocks",    type=int,   default=2,
                     help="Number of Evoformer blocks.")
    evo.add_argument("--c-m",         type=int,   default=64,
                     help="MSA channel width c_m.")
    evo.add_argument("--c-z",         type=int,   default=32,
                     help="Pair channel width c_z.")
    evo.add_argument("--no-triangle", action="store_true",
                     help="Disable triangle multiplicative updates.")
    evo.add_argument("--dropout",     type=float, default=0.1,
                     help="Dropout probability in Evoformer attention and MLP layers.")

    train_g = p.add_argument_group("training")
    train_g.add_argument("--epochs",       type=int,   default=10)
    train_g.add_argument("--batch-size",   type=int,   default=4)
    train_g.add_argument("--lr",            type=float, default=1e-3)
    train_g.add_argument("--min-lr",        type=float, default=1e-5,
                         help="Minimum LR floor for cosine scheduler (default: 1e-5).")
    train_g.add_argument("--d-clamp",      type=float, default=10.0,
                         help="FAPE clamping distance (Angstroms).")
    train_g.add_argument("--warmup-steps", type=int,   default=0,
                         help="Number of linear LR warmup steps (optimizer steps).")
    train_g.add_argument("--num-workers",  type=int,   default=0,
                         help="DataLoader worker processes.")
    train_g.add_argument("--pin-memory",   action="store_true",
                         help="Pin DataLoader memory (CUDA only).")
    train_g.add_argument("--log-dir",      type=str,   default=None,
                         help="TensorBoard log directory.")
    train_g.add_argument("--ckpt-every",   type=int,   default=0,
                         help="Save a checkpoint every N epochs (0 = disabled).")
    train_g.add_argument("--save",         type=str,   default=None,
                         help="Path to save final checkpoint (.pt).")

    if pre_ns.config:
        p.set_defaults(**_load_config(pre_ns.config))

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
