"""Tests for the semantic-cache embedding backends, gates, and BLOB storage."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List

import pytest

from modelmesh.core.cache.config import SemanticCacheConfig
from modelmesh.core.cache.embedding import (
    EmbedderUnavailableError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    create_embedder,
    length_ratio,
    negation_tokens,
    numeric_tokens,
    pack_embedding,
    semantic_gate,
    unpack_embedding,
)
from modelmesh.core.storage import Storage


PARAPHRASES = [
    ("What is a hash map?", "Can you explain what a hash map is?"),
    ("How do I reverse a list in Python?", "What is the way to reverse a list in Python?"),
    (
        "Explain the difference between a mutex and a semaphore.",
        "What is the difference between a semaphore and a mutex?",
    ),
    ("How do I read a file in Python?", "What is the best way to read a file using Python?"),
]


def test_hashing_embedder_is_deterministic_normalised_and_sized() -> None:
    """The default embedder is stable, unit-length, and reports its size."""
    embedder = HashingEmbedder(256)
    first = embedder.embed("What is a hash map?")
    second = embedder.embed("What is a hash map?")
    assert embedder.dim == 256
    assert embedder.name == "hashing-256"
    assert first == second
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0, rel_tol=1e-9)


def test_embedding_blob_roundtrip() -> None:
    """float32 packing preserves the vector closely enough for cosine matching."""
    vector = HashingEmbedder(64).embed("round trip question")
    restored = unpack_embedding(pack_embedding(vector))
    assert len(restored) == len(vector)
    assert cosine_similarity(vector, restored) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("source, query", PARAPHRASES)
def test_paraphrases_clear_a_practical_threshold(source: str, query: str) -> None:
    """The hashing embedder ranks the four paraphrases above a chosen threshold."""
    embedder = HashingEmbedder(512)
    similarity = cosine_similarity(embedder.embed(source), embedder.embed(query))
    accepted, reason = semantic_gate(source, query, similarity, 0.70)
    assert similarity >= 0.70
    assert accepted is True
    assert reason == "ok"


def test_numeric_difference_is_never_a_semantic_match() -> None:
    """``2+2`` never matches ``2+3`` even when the cosine score is high."""
    assert numeric_tokens("What is 2+2?") != numeric_tokens("What is 2+3?")
    accepted, reason = semantic_gate("What is 2+2?", "What is 2+3?", 0.99, 0.0)
    assert accepted is False
    assert reason == "numeric_mismatch"


def test_negation_difference_is_never_a_semantic_match() -> None:
    """``use X`` never matches ``not use X`` even when the cosine score is high."""
    assert negation_tokens("Should I use X?") != negation_tokens("Should I not use X?")
    accepted, reason = semantic_gate("Should I use X?", "Should I not use X?", 0.99, 0.0)
    assert accepted is False
    assert reason == "negation_mismatch"


def test_below_threshold_and_length_gates() -> None:
    """Threshold and length-ratio gates reject otherwise-compatible prompts."""
    assert semantic_gate("short prompt", "short prompt", 0.40, 0.70) == (
        False,
        "below_threshold",
    )
    assert length_ratio("x" * 10, "x" * 40) == pytest.approx(0.25)
    accepted, reason = semantic_gate("x" * 10, "x" * 40, 0.99, 0.0)
    assert accepted is False
    assert reason == "length_mismatch"


def test_create_embedder_defaults_to_hashing() -> None:
    """The hashing backend is selected by default and needs no optional package."""
    embedder = create_embedder(SemanticCacheConfig(embedder="hashing", hashing_dim=128))
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 128


def test_missing_sentence_transformers_falls_back_to_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional backend logs a warning and falls back instead of crashing."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = create_embedder(
        SemanticCacheConfig(embedder="sentence_transformers", hashing_dim=64)
    )
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 64


def test_sentence_transformer_backend_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct use of the optional backend reports the missing extra clearly."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbedderUnavailableError, match="sentence-transformers is not installed"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")


def test_storage_roundtrips_the_embedding_blob(tmp_path: Path) -> None:
    """put_cache_entry/get_cache_entry preserve the embedding bytes exactly."""
    storage = Storage(db_path=tmp_path / "blob.db")
    vector = HashingEmbedder(32).embed("stored question")
    blob = pack_embedding(vector)
    storage.put_cache_entry(
        key="k",
        norm_prompt="stored question",
        context_hash="c",
        system_hash="s",
        params_hash="p",
        scope="auto",
        answer_text="answer",
        ttl_seconds=60,
        embedding=blob,
    )
    entry = storage.get_cache_entry("k")
    assert entry is not None
    assert entry["embedding"] == blob
    assert cosine_similarity(vector, unpack_embedding(entry["embedding"])) == pytest.approx(
        1.0, abs=1e-5
    )


def test_semantic_candidates_filter_hashes_and_expiry(tmp_path: Path) -> None:
    """Only matching, non-expired rows carrying an embedding are returned."""
    storage = Storage(db_path=tmp_path / "candidates.db")
    now = time.time()
    common = dict(
        context_hash="ctx",
        system_hash="sys",
        params_hash="params",
        scope="auto",
        answer_text="a",
        ttl_seconds=3600,
        embedding=pack_embedding([0.0, 1.0]),
    )
    storage.put_cache_entry(key="good", norm_prompt="good", **common)
    storage.put_cache_entry(key="bad-ctx", norm_prompt="bad ctx", **{**common, "context_hash": "x"})
    storage.put_cache_entry(key="bad-scope", norm_prompt="bad scope", **{**common, "scope": "manual"})
    storage.put_cache_entry(key="no-embed", norm_prompt="none", **{**common, "embedding": None})
    storage.put_cache_entry(
        key="expired", norm_prompt="expired", **{**common, "ttl_seconds": 10}
    )
    import sqlite3

    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("UPDATE response_cache SET created_at = 0 WHERE key = 'expired'")

    rows = storage.get_semantic_candidates("ctx", "sys", "params", "auto", limit=10, now=now)
    keys: List[str] = [row["key"] for row in rows]
    assert keys == ["good"]
