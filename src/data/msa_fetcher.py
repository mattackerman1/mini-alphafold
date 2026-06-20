"""
ColabFold MSA fetcher with disk-based caching.

Queries the ColabFold MMseqs2 API to generate real MSAs for protein sequences,
then caches results locally so training does not require repeated API calls.

The ColabFold API is asynchronous:
  1. POST /ticket/msa  →  job ID
  2. Poll GET /ticket/{id}  →  wait for status == "COMPLETE"
  3. GET /result/download/{id}  →  tar.gz containing .a3m files

A3M format
----------
Each sequence is an aligned string where:
  - Uppercase letters and '-' are alignment columns (same width as query)
  - Lowercase letters are insertions relative to the query (removed here)

After stripping lowercase, all sequences in the MSA have the same length as
the query — ready for MSAEncoder.encode().

Public API
----------
  fetch_msa(sequence, cache_dir, ...)  →  list[str]   (aligned sequences)
  prefetch_msas(sequences, cache_dir, ...)             (bulk pre-caching)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLABFOLD_API = "https://api.colabfold.com"
DEFAULT_POLL_INTERVAL = 5       # seconds between status polls
DEFAULT_TIMEOUT = 300           # max seconds to wait for one job
DEFAULT_MAX_MSA_DEPTH = 512     # cap homologs returned (memory budget)
DEFAULT_REQUESTS_TIMEOUT = 30   # HTTP request timeout in seconds


# ---------------------------------------------------------------------------
# A3M parsing
# ---------------------------------------------------------------------------

def _parse_a3m(a3m_text: str) -> list[str]:
    """Parse an A3M-format MSA into aligned sequences.

    Strips insertion characters (lowercase letters) so every returned sequence
    has the same length as the query (the first entry).

    Args:
        a3m_text: Raw A3M file contents as a string.

    Returns:
        List of aligned strings, query first. Empty sequences are skipped.
    """
    sequences: list[str] = []
    current: list[str] = []

    for line in a3m_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                seq = "".join(current)
                # Remove lowercase insertion characters
                seq = "".join(ch for ch in seq if not ch.islower())
                if seq:
                    sequences.append(seq)
            current = []
        else:
            current.append(line)

    # Flush last entry
    if current:
        seq = "".join(current)
        seq = "".join(ch for ch in seq if not ch.islower())
        if seq:
            sequences.append(seq)

    return sequences


def _extract_a3m_from_tar(tar_bytes: bytes) -> str:
    """Extract the first .a3m file from a tar.gz byte payload.

    The ColabFold API returns a tar.gz containing one or more .a3m files.
    We prefer 'uniref.a3m' (UniRef90 hits) if present, otherwise take the
    first .a3m found.

    Args:
        tar_bytes: Raw bytes from the API download endpoint.

    Returns:
        A3M file contents as a string.

    Raises:
        ValueError: If no .a3m file is found in the archive.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        names = tar.getnames()
        # Prefer uniref.a3m, fall back to first .a3m
        a3m_names = [n for n in names if n.endswith(".a3m")]
        if not a3m_names:
            raise ValueError(f"No .a3m file in ColabFold result. Files: {names}")
        preferred = next((n for n in a3m_names if "uniref" in n), a3m_names[0])
        member = tar.getmember(preferred)
        f = tar.extractfile(member)
        if f is None:
            raise ValueError(f"Could not extract {preferred} from archive.")
        return f.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(sequence: str) -> str:
    """MD5 hex digest of the sequence — used as the cache filename."""
    return hashlib.md5(sequence.upper().encode()).hexdigest()


def _cache_path(cache_dir: Path, sequence: str) -> Path:
    return cache_dir / f"{_cache_key(sequence)}.json"


def _load_cache(cache_dir: Path, sequence: str) -> Optional[list[str]]:
    """Return cached MSA sequences, or None if not cached."""
    p = _cache_path(cache_dir, sequence)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["sequences"]
    except Exception as exc:
        logger.warning("Corrupt cache entry %s: %s", p, exc)
        return None


def _save_cache(cache_dir: Path, sequence: str, sequences: list[str]) -> None:
    """Persist MSA sequences to disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, sequence)
    p.write_text(
        json.dumps({"query": sequence, "sequences": sequences}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# ColabFold API client
# ---------------------------------------------------------------------------

def _submit_job(sequence: str, mode: str = "env") -> str:
    """Submit a sequence to the ColabFold MSA API and return the job ID.

    Args:
        sequence: Query amino acid string (no gaps).
        mode:     'env' includes environmental sequences (metagenomics);
                  'all' includes UniRef90 + BFD + MGnify + env.

    Returns:
        Job ID string.
    """
    fasta = f">query\n{sequence}"
    response = requests.post(
        f"{COLABFOLD_API}/ticket/msa",
        data={"q": fasta, "mode": mode},
        timeout=DEFAULT_REQUESTS_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    job_id = data.get("id")
    if not job_id:
        raise RuntimeError(f"ColabFold API returned no job ID: {data}")
    return job_id


def _poll_job(job_id: str, timeout: float = DEFAULT_TIMEOUT,
              poll_interval: float = DEFAULT_POLL_INTERVAL) -> None:
    """Block until the job is COMPLETE or raise on ERROR/timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(
            f"{COLABFOLD_API}/ticket/{job_id}",
            timeout=DEFAULT_REQUESTS_TIMEOUT,
        )
        response.raise_for_status()
        status = response.json().get("status", "")
        if status == "COMPLETE":
            return
        if status == "ERROR":
            raise RuntimeError(f"ColabFold job {job_id} failed with status ERROR.")
        logger.debug("Job %s status: %s — waiting %ss", job_id, status, poll_interval)
        time.sleep(poll_interval)
    raise TimeoutError(f"ColabFold job {job_id} did not complete within {timeout}s.")


def _download_result(job_id: str) -> bytes:
    """Download the tar.gz result for a completed job."""
    response = requests.get(
        f"{COLABFOLD_API}/result/download/{job_id}",
        timeout=120,   # download can be large
    )
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_msa(
    sequence: str,
    cache_dir: str | Path = "msa_cache",
    max_msa_depth: int = DEFAULT_MAX_MSA_DEPTH,
    mode: str = "env",
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[str]:
    """Fetch a real MSA for a query sequence, using disk cache when available.

    On first call for a given sequence, queries the ColabFold MMseqs2 API and
    caches the result. Subsequent calls return the cached MSA immediately.

    Args:
        sequence:      Query amino acid string (standard 1-letter codes, no gaps).
        cache_dir:     Directory for caching MSA results. Created if absent.
        max_msa_depth: Maximum number of sequences to return (query + homologs).
                       Caps memory usage for very deep MSAs.
        mode:          ColabFold search mode: 'env' (default) or 'all'.
        poll_interval: Seconds between API status polls.
        timeout:       Maximum seconds to wait for the API job to complete.

    Returns:
        List of aligned strings. sequences[0] is always the query.
        All sequences have the same length (gaps represented as '-').

    Raises:
        requests.HTTPError: On API HTTP errors.
        TimeoutError:       If the job exceeds `timeout` seconds.
        RuntimeError:       On API-reported errors.
    """
    cache_dir = Path(cache_dir)
    sequence = sequence.upper().strip()

    # Cache hit
    cached = _load_cache(cache_dir, sequence)
    if cached is not None:
        logger.debug("MSA cache hit for sequence (len=%d)", len(sequence))
        return cached[:max_msa_depth]

    # Cache miss — call API
    logger.info("Fetching MSA for sequence (len=%d) from ColabFold…", len(sequence))
    print(f"  Fetching MSA from ColabFold (L={len(sequence)})…", flush=True)

    job_id = _submit_job(sequence, mode=mode)
    logger.debug("Submitted job %s", job_id)

    _poll_job(job_id, timeout=timeout, poll_interval=poll_interval)

    tar_bytes = _download_result(job_id)
    a3m_text = _extract_a3m_from_tar(tar_bytes)
    sequences = _parse_a3m(a3m_text)

    if not sequences:
        logger.warning("Empty MSA returned for sequence (len=%d); using query only.", len(sequence))
        sequences = [sequence]

    # Ensure query is first (API always puts it first, but be defensive)
    query_clean = sequence.replace("-", "")
    if sequences[0].replace("-", "") != query_clean:
        sequences = [sequence] + [s for s in sequences if s != sequence]

    _save_cache(cache_dir, sequence, sequences)
    logger.info("  Cached %d sequences (depth capped at %d)", len(sequences), max_msa_depth)
    print(f"  Done — {len(sequences)} homologs found, cached to {cache_dir}/", flush=True)

    return sequences[:max_msa_depth]


def prefetch_msas(
    sequences: list[str],
    cache_dir: str | Path = "msa_cache",
    max_msa_depth: int = DEFAULT_MAX_MSA_DEPTH,
    mode: str = "env",
    delay_between: float = 1.0,
    skip_cached: bool = True,
) -> dict[str, list[str]]:
    """Pre-fetch and cache MSAs for a list of sequences.

    Intended to be run once before training — populates the cache so the
    training loop never needs to call the API.

    Args:
        sequences:      List of query amino acid strings.
        cache_dir:      Directory for caching results.
        max_msa_depth:  Maximum sequences per MSA.
        mode:           ColabFold search mode.
        delay_between:  Seconds to sleep between API submissions (rate limiting).
        skip_cached:    If True, skip sequences already in the cache.

    Returns:
        Dict mapping each sequence to its list of aligned MSA sequences.
    """
    cache_dir = Path(cache_dir)
    results: dict[str, list[str]] = {}
    total = len(sequences)

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        print(f"[{i+1}/{total}] sequence (L={len(seq)})", flush=True)

        if skip_cached and _load_cache(cache_dir, seq) is not None:
            print(f"  → cache hit, skipping", flush=True)
            results[seq] = _load_cache(cache_dir, seq)[:max_msa_depth]  # type: ignore[index]
            continue

        try:
            msa = fetch_msa(seq, cache_dir=cache_dir,
                            max_msa_depth=max_msa_depth, mode=mode)
            results[seq] = msa
        except Exception as exc:
            logger.error("Failed to fetch MSA for seq %d: %s", i, exc)
            print(f"  ERROR: {exc} — falling back to query only", flush=True)
            results[seq] = [seq]

        if i < total - 1:
            time.sleep(delay_between)

    print(f"\nPre-fetch complete. {len(results)}/{total} sequences cached in {cache_dir}/")
    return results


def load_cached_msa(
    sequence: str,
    cache_dir: str | Path = "msa_cache",
    max_msa_depth: int = DEFAULT_MAX_MSA_DEPTH,
    fallback_pseudo: bool = True,
    n_pseudo: int = 8,
) -> list[str]:
    """Load a cached MSA, falling back to pseudo-MSA if not cached.

    Use this in training loops to safely retrieve MSAs without blocking
    on an API call.

    Args:
        sequence:       Query amino acid string.
        cache_dir:      Cache directory to check.
        max_msa_depth:  Cap on returned sequences.
        fallback_pseudo: If True and no cache entry exists, generate a
                         pseudo-MSA rather than raising an error.
        n_pseudo:       Depth of fallback pseudo-MSA.

    Returns:
        List of aligned sequences, query first.
    """
    cached = _load_cache(Path(cache_dir), sequence.upper().strip())
    if cached is not None:
        return cached[:max_msa_depth]

    if fallback_pseudo:
        from src.data.msa_encoder import generate_pseudo_msa
        logger.warning(
            "No cached MSA for sequence (L=%d) — falling back to pseudo-MSA.", len(sequence)
        )
        return generate_pseudo_msa(sequence, n_sequences=n_pseudo)

    raise FileNotFoundError(
        f"No cached MSA found for sequence (L={len(sequence)}) in {cache_dir}. "
        "Run prefetch_msas() first, or set fallback_pseudo=True."
    )
