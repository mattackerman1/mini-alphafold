"""Tests for src/data/msa_fetcher.py — A3M parsing, caching, and API client."""

import io
import json
import tarfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data.msa_fetcher import (
    _parse_a3m,
    _extract_a3m_from_tar,
    _cache_key,
    _cache_path,
    _load_cache,
    _save_cache,
    fetch_msa,
    load_cached_msa,
    prefetch_msas,
)


# ---------------------------------------------------------------------------
# A3M parsing
# ---------------------------------------------------------------------------

SAMPLE_A3M = """\
>query
ACDEFGHIK
>homolog1
ACDE--HIK
>homolog2
ACDEFGhHIK
>homolog3
A-DEFGHiIK
"""


def test_parse_a3m_returns_query_first():
    seqs = _parse_a3m(SAMPLE_A3M)
    assert seqs[0] == "ACDEFGHIK"


def test_parse_a3m_strips_lowercase():
    """Lowercase insertion characters must be removed from all sequences."""
    seqs = _parse_a3m(SAMPLE_A3M)
    for seq in seqs:
        assert seq == seq.upper() or "-" in seq, \
            f"Lowercase not stripped: {seq!r}"
    # homolog2: ACDEFGhHIK → ACDEFGHIK (h removed)
    assert "h" not in seqs[2]
    # homolog3: A-DEFGHiIK → A-DEFGHIK (i removed)
    assert "i" not in seqs[3]


def test_parse_a3m_preserves_gaps():
    """Gap characters '-' must be preserved (they encode alignment columns)."""
    seqs = _parse_a3m(SAMPLE_A3M)
    assert "-" in seqs[1]  # homolog1 has gaps
    assert "-" in seqs[3]  # homolog3 has gap


def test_parse_a3m_query_length_matches():
    """All sequences after stripping insertions must have the same length as query."""
    seqs = _parse_a3m(SAMPLE_A3M)
    query_len = len(seqs[0])
    for seq in seqs:
        assert len(seq) == query_len, \
            f"Sequence length {len(seq)} != query length {query_len}: {seq!r}"


def test_parse_a3m_empty_returns_empty():
    assert _parse_a3m("") == []


def test_parse_a3m_single_sequence():
    a3m = ">query\nACDEF\n"
    seqs = _parse_a3m(a3m)
    assert seqs == ["ACDEF"]


# ---------------------------------------------------------------------------
# tar.gz extraction
# ---------------------------------------------------------------------------

def _make_tar(files: dict[str, str]) -> bytes:
    """Create an in-memory tar.gz with given {filename: content} mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            encoded = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    return buf.getvalue()


def test_extract_a3m_prefers_uniref():
    """When both uniref.a3m and bfd.a3m exist, prefer uniref.a3m."""
    tar_bytes = _make_tar({
        "job123/bfd.a3m": ">query\nACDEF\n",
        "job123/uniref.a3m": ">query\nMKTAY\n",
    })
    content = _extract_a3m_from_tar(tar_bytes)
    assert "MKTAY" in content


def test_extract_a3m_fallback_to_first():
    """If no uniref.a3m, return the first .a3m found."""
    tar_bytes = _make_tar({"job123/bfd.a3m": ">query\nACDEF\n"})
    content = _extract_a3m_from_tar(tar_bytes)
    assert "ACDEF" in content


def test_extract_a3m_raises_if_no_a3m():
    tar_bytes = _make_tar({"job123/result.txt": "nothing here"})
    with pytest.raises(ValueError, match="No .a3m file"):
        _extract_a3m_from_tar(tar_bytes)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def test_cache_key_is_md5_of_uppercase():
    seq = "acdef"
    expected = hashlib.md5("ACDEF".encode()).hexdigest()
    assert _cache_key(seq) == expected


def test_cache_key_case_insensitive():
    assert _cache_key("ACDEF") == _cache_key("acdef")


def test_save_and_load_cache(tmp_path):
    seq = "ACDEFGHIK"
    seqs = ["ACDEFGHIK", "ACDE--HIK", "A-DEFGHIK"]
    _save_cache(tmp_path, seq, seqs)
    loaded = _load_cache(tmp_path, seq)
    assert loaded == seqs


def test_load_cache_returns_none_if_missing(tmp_path):
    assert _load_cache(tmp_path, "ACDEFGHIK") is None


def test_save_cache_creates_dir(tmp_path):
    subdir = tmp_path / "nested" / "cache"
    _save_cache(subdir, "ACDEF", ["ACDEF"])
    assert _cache_path(subdir, "ACDEF").exists()


def test_load_cache_handles_corrupt_file(tmp_path):
    seq = "ACDEFGHIK"
    p = _cache_path(tmp_path, seq)
    p.write_text("not valid json", encoding="utf-8")
    assert _load_cache(tmp_path, seq) is None


# ---------------------------------------------------------------------------
# fetch_msa (API mocked)
# ---------------------------------------------------------------------------

def _mock_tar_response(sequences: list[str]) -> bytes:
    a3m = "\n".join(
        f">seq{i}\n{seq}" for i, seq in enumerate(sequences)
    )
    return _make_tar({"job/uniref.a3m": a3m})


def test_fetch_msa_returns_cached_on_second_call(tmp_path):
    """Second call returns cached result without hitting the API."""
    seq = "ACDEFGHIK"
    real_seqs = ["ACDEFGHIK", "ACDE--HIK"]
    _save_cache(tmp_path, seq, real_seqs)

    with patch("src.data.msa_fetcher.requests.post") as mock_post:
        result = fetch_msa(seq, cache_dir=tmp_path)
        mock_post.assert_not_called()

    assert result == real_seqs


def test_fetch_msa_calls_api_on_cache_miss(tmp_path):
    """On cache miss, the API is called and result is cached."""
    seq = "ACDEFGHIK"
    homologs = ["ACDEFGHIK", "ACDE--HIK", "A-DEFGHIK"]

    mock_submit = MagicMock()
    mock_submit.return_value.json.return_value = {"id": "job123"}
    mock_submit.return_value.raise_for_status = MagicMock()

    mock_poll = MagicMock()
    mock_poll.return_value.json.return_value = {"status": "COMPLETE"}
    mock_poll.return_value.raise_for_status = MagicMock()

    mock_download = MagicMock()
    mock_download.return_value.content = _mock_tar_response(homologs)
    mock_download.return_value.raise_for_status = MagicMock()

    def side_effect(url, **kwargs):
        if "ticket/msa" in url:
            return mock_submit.return_value
        if "ticket/job123" in url:
            return mock_poll.return_value
        if "result/download" in url:
            return mock_download.return_value
        raise ValueError(f"Unexpected URL: {url}")

    with patch("src.data.msa_fetcher.requests.post", side_effect=lambda url, **kw: mock_submit.return_value), \
         patch("src.data.msa_fetcher.requests.get", side_effect=side_effect):
        result = fetch_msa(seq, cache_dir=tmp_path)

    assert result[0] == homologs[0]
    # Verify it was cached
    assert _load_cache(tmp_path, seq) is not None


def test_fetch_msa_respects_max_depth(tmp_path):
    """max_msa_depth caps the number of returned sequences."""
    seq = "ACDEFGHIK"
    many_seqs = [seq] + [f"A{'C'*i}FGHIK"[:9] for i in range(20)]
    _save_cache(tmp_path, seq, many_seqs)

    result = fetch_msa(seq, cache_dir=tmp_path, max_msa_depth=5)
    assert len(result) <= 5


# ---------------------------------------------------------------------------
# load_cached_msa
# ---------------------------------------------------------------------------

def test_load_cached_msa_returns_cache(tmp_path):
    seq = "ACDEFGHIK"
    seqs = ["ACDEFGHIK", "ACDE--HIK"]
    _save_cache(tmp_path, seq, seqs)
    result = load_cached_msa(seq, cache_dir=tmp_path)
    assert result == seqs


def test_load_cached_msa_fallback_to_pseudo(tmp_path):
    """When cache is empty and fallback_pseudo=True, returns pseudo-MSA."""
    seq = "ACDEFGHIK"
    result = load_cached_msa(seq, cache_dir=tmp_path, fallback_pseudo=True, n_pseudo=4)
    assert len(result) == 5  # query + 4 pseudo
    assert result[0] == seq


def test_load_cached_msa_raises_without_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cached_msa("ACDEFGHIK", cache_dir=tmp_path, fallback_pseudo=False)
