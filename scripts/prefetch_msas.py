"""
Pre-fetch and cache real MSAs from the ColabFold API.

Run this once before training to populate the MSA cache. The training
script will then load MSAs from disk without any API calls.

Usage
-----
  # From a list of PDB IDs (downloads structures, extracts sequences)
  python scripts/prefetch_msas.py --pdb-ids 1UBQ,1CRN,4HHB

  # From a plain-text file of sequences (one per line)
  python scripts/prefetch_msas.py --sequence-file sequences.txt

  # From a FASTA file
  python scripts/prefetch_msas.py --fasta proteins.fasta

  # Save to a custom cache directory (default: msa_cache/)
  python scripts/prefetch_msas.py --pdb-ids 1UBQ,1CRN --cache-dir /data/msa_cache

  # Limit MSA depth (to control memory usage during training)
  python scripts/prefetch_msas.py --pdb-ids 1UBQ --max-depth 128

On Google Colab, save to Drive so the cache persists across sessions:
  python scripts/prefetch_msas.py --pdb-ids 1UBQ,1CRN \\
      --cache-dir /content/drive/MyDrive/mini-alphafold/msa_cache
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Sequence loading helpers
# ---------------------------------------------------------------------------

def _load_from_fasta(path: Path) -> list[str]:
    """Parse a FASTA file and return the sequences (no headers)."""
    sequences: list[str] = []
    current: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            elif line:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _load_from_sequence_file(path: Path) -> list[str]:
    """Load one sequence per line from a plain text file."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _fetch_sequence_from_pdb(pdb_id: str) -> str:
    """Download a PDB file from RCSB and extract the SEQRES sequence."""
    from src.data.pdb_parser import parse_structure_file

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Downloading {pdb_id.upper()} from RCSB…", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        struct = parse_structure_file(tmp_path)
        return struct.sequence
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_sequences(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return list of (label, sequence) pairs from whichever source was given."""
    pairs: list[tuple[str, str]] = []

    if args.pdb_ids:
        for pdb_id in args.pdb_ids.split(","):
            pdb_id = pdb_id.strip()
            if not pdb_id:
                continue
            try:
                seq = _fetch_sequence_from_pdb(pdb_id)
                pairs.append((pdb_id.upper(), seq))
            except Exception as exc:
                print(f"  WARNING: could not load {pdb_id}: {exc}", flush=True)

    if args.fasta:
        seqs = _load_from_fasta(Path(args.fasta))
        for i, seq in enumerate(seqs):
            pairs.append((f"fasta_{i}", seq))

    if args.sequence_file:
        seqs = _load_from_sequence_file(Path(args.sequence_file))
        for i, seq in enumerate(seqs):
            pairs.append((f"seq_{i}", seq))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    from src.data.msa_fetcher import fetch_msa, _load_cache

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"MSA cache directory: {cache_dir.resolve()}")
    print(f"ColabFold mode: {args.mode}  |  max depth: {args.max_depth}\n")

    pairs = _load_sequences(args)
    if not pairs:
        print("No sequences to process. Use --pdb-ids, --fasta, or --sequence-file.")
        sys.exit(1)

    print(f"Sequences to process: {len(pairs)}\n")

    success = 0
    skipped = 0
    failed = 0

    for i, (label, seq) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {label}  (L={len(seq)})", flush=True)

        # Check cache first
        if not args.force and _load_cache(cache_dir, seq) is not None:
            print(f"  → already cached, skipping", flush=True)
            skipped += 1
            continue

        try:
            msa = fetch_msa(
                seq,
                cache_dir=cache_dir,
                max_msa_depth=args.max_depth,
                mode=args.mode,
            )
            print(f"  → {len(msa)} sequences cached", flush=True)
            success += 1
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            failed += 1

        # Rate limit between submissions
        if i < len(pairs) - 1:
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"Done. Success: {success}  Skipped: {skipped}  Failed: {failed}")
    print(f"Cache: {cache_dir.resolve()}")
    if failed:
        print(f"WARNING: {failed} sequences failed. Re-run to retry.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-fetch real MSAs from ColabFold and cache to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("sequence sources (at least one required)")
    src.add_argument(
        "--pdb-ids", type=str, default=None,
        help="Comma-separated PDB IDs to fetch sequences from RCSB (e.g. 1UBQ,1CRN).",
    )
    src.add_argument(
        "--fasta", type=str, default=None,
        help="Path to a FASTA file of query sequences.",
    )
    src.add_argument(
        "--sequence-file", type=str, default=None,
        help="Path to a plain text file with one sequence per line.",
    )

    opt = p.add_argument_group("options")
    opt.add_argument(
        "--cache-dir", type=str, default="msa_cache",
        help="Directory to store cached MSA JSON files.",
    )
    opt.add_argument(
        "--max-depth", type=int, default=512,
        help="Maximum number of MSA sequences to cache per protein.",
    )
    opt.add_argument(
        "--mode", type=str, default="env", choices=["env", "all"],
        help="ColabFold search mode. 'env' includes metagenomics sequences.",
    )
    opt.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API submissions (rate limiting).",
    )
    opt.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if a cache entry already exists.",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
