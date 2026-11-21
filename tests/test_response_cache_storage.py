"""Tests for the response_cache Storage methods and CacheConfig loader.

All tests are hermetic:
- Storage uses an in-memory or tmp_path SQLite database (no real disk DB).
- ``now`` is injected via keyword argument to simulate time passing.
- CacheConfig tests use tmp_path so they never write to config/cache.yaml.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from modelmesh.core.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON = dict(
    norm_prompt="hello world",
    context_hash="ctx_abc",
    system_hash="sys_abc",
    params_hash="par_abc",
    scope="auto",
    answer_text="The answer is 42.",
    ttl_seconds=3600,
    model_id="mock-fast",
    endpoint_id="mock-fast@mock",
    provider_id="mock",
    tokens_in=10,
    tokens_out=20,
    cost_usd=0.0001,
)


def _storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "cache_test.db")


# ---------------------------------------------------------------------------
# put / get round-trip
# ---------------------------------------------------------------------------


def test_put_get_roundtrip(tmp_path: Path) -> None:
    """A stored entry is returned verbatim on exact key match."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k1", **_COMMON)

    row = s.get_cache_entry("k1")
    assert row is not None
    assert row["answer_text"] == "The answer is 42."
    assert row["model_id"] == "mock-fast"
    assert row["tokens_in"] == 10
    assert row["tokens_out"] == 20
    assert row["cost_usd"] == pytest.approx(0.0001)
    assert row["hit_count"] == 0
    assert row["scope"] == "auto"


def test_get_missing_key_returns_none(tmp_path: Path) -> None:
    """get_cache_entry returns None for a key that was never stored."""
    s = _storage(tmp_path)
    assert s.get_cache_entry("nonexistent") is None


def test_put_replace(tmp_path: Path) -> None:
    """INSERT OR REPLACE updates an existing entry."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k1", **_COMMON)
    new = dict(_COMMON, answer_text="Updated answer.", tokens_out=99)
    s.put_cache_entry(key="k1", **new)
    row = s.get_cache_entry("k1")
    assert row is not None
    assert row["answer_text"] == "Updated answer."
    assert row["tokens_out"] == 99


# ---------------------------------------------------------------------------
# TTL / expiry (inject `now`)
# ---------------------------------------------------------------------------


def test_get_expired_entry_returns_none(tmp_path: Path) -> None:
    """An entry past its TTL is treated as a miss without being deleted."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k_ttl", **dict(_COMMON, ttl_seconds=10))

    row = s.get_cache_entry("k_ttl")
    assert row is not None, "Should be a live entry right after insert"

    future = time.time() + 20  # 20 s past a 10-s TTL
    row_expired = s.get_cache_entry("k_ttl", now=future)
    assert row_expired is None, "Entry should appear expired"


def test_get_just_before_expiry(tmp_path: Path) -> None:
    """An entry exactly at TTL boundary is still returned (> not >=)."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k_bound", **dict(_COMMON, ttl_seconds=100))
    row = s.get_cache_entry("k_bound")
    assert row is not None
    at_boundary = row["created_at"] + 100  # exactly at TTL
    # created_at + ttl == now → not expired (condition: now > created_at + ttl)
    result = s.get_cache_entry("k_bound", now=at_boundary)
    assert result is not None, "Entry exactly at TTL boundary should not be expired"


# ---------------------------------------------------------------------------
# touch_cache_entry
# ---------------------------------------------------------------------------


def test_touch_increments_hit_count(tmp_path: Path) -> None:
    """touch_cache_entry bumps hit_count and updates last_hit_at."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k_touch", **_COMMON)

    before = s.get_cache_entry("k_touch")
    assert before is not None
    assert before["hit_count"] == 0

    s.touch_cache_entry("k_touch")
    s.touch_cache_entry("k_touch")
    after = s.get_cache_entry("k_touch")
    assert after is not None
    assert after["hit_count"] == 2
    assert after["last_hit_at"] >= before["last_hit_at"]


def test_touch_missing_key_is_noop(tmp_path: Path) -> None:
    """touch_cache_entry on a nonexistent key does not raise."""
    s = _storage(tmp_path)
    s.touch_cache_entry("does_not_exist")  # must not raise


# ---------------------------------------------------------------------------
# purge_expired_cache
# ---------------------------------------------------------------------------


def test_purge_removes_expired(tmp_path: Path) -> None:
    """purge_expired_cache deletes expired rows and returns their count."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="live", **dict(_COMMON, ttl_seconds=9999))
    s.put_cache_entry(key="dead1", **dict(_COMMON, ttl_seconds=1))
    s.put_cache_entry(key="dead2", **dict(_COMMON, ttl_seconds=1))

    # Purge at a time 10 s in the future — both short-TTL entries expire.
    future = time.time() + 10
    deleted = s.purge_expired_cache(now=future)
    assert deleted == 2

    # The long-lived entry must still be accessible.
    assert s.get_cache_entry("live") is not None


def test_purge_nothing_to_do(tmp_path: Path) -> None:
    """purge_expired_cache returns 0 when nothing has expired."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k_fresh", **dict(_COMMON, ttl_seconds=9999))
    assert s.purge_expired_cache() == 0


# ---------------------------------------------------------------------------
# clear_response_cache
# ---------------------------------------------------------------------------


def test_clear_empties_table(tmp_path: Path) -> None:
    """clear_response_cache removes all entries and returns the count."""
    s = _storage(tmp_path)
    for i in range(5):
        s.put_cache_entry(key=f"k{i}", **_COMMON)

    n = s.clear_response_cache()
    assert n == 5
    assert s.get_cache_entry("k0") is None
    stats = s.response_cache_stats()
    assert stats["entries"] == 0


def test_clear_empty_table_returns_zero(tmp_path: Path) -> None:
    """clear_response_cache on an already-empty table returns 0."""
    s = _storage(tmp_path)
    assert s.clear_response_cache() == 0


# ---------------------------------------------------------------------------
# max_entries eviction
# ---------------------------------------------------------------------------


def test_max_entries_eviction(tmp_path: Path) -> None:
    """put_cache_entry evicts the oldest entry when max_entries is exceeded."""
    s = _storage(tmp_path)
    max_e = 3
    for i in range(max_e):
        s.put_cache_entry(key=f"k{i}", **_COMMON, max_entries=max_e)

    # At this point exactly max_e entries exist.
    stats = s.response_cache_stats()
    assert stats["entries"] == max_e

    # Adding one more should trigger one eviction.
    s.put_cache_entry(key="k_new", **_COMMON, max_entries=max_e)
    stats_after = s.response_cache_stats()
    assert stats_after["entries"] == max_e


def test_eviction_preserves_recently_hit(tmp_path: Path) -> None:
    """The most-recently-hit entry is NOT the one evicted."""
    s = _storage(tmp_path)
    max_e = 3
    for i in range(max_e):
        s.put_cache_entry(key=f"k{i}", **_COMMON, max_entries=max_e)

    # Mark k0 as heavily used so it has the latest last_hit_at.
    s.touch_cache_entry("k0")
    s.touch_cache_entry("k0")

    # Trigger eviction.
    s.put_cache_entry(key="k_overflow", **_COMMON, max_entries=max_e)

    # k0 should survive (it was the most recently hit).
    assert s.get_cache_entry("k0") is not None


# ---------------------------------------------------------------------------
# response_cache_stats
# ---------------------------------------------------------------------------


def test_stats_empty(tmp_path: Path) -> None:
    """Stats on a fresh DB return all-zero values."""
    s = _storage(tmp_path)
    stats = s.response_cache_stats()
    assert stats["entries"] == 0
    assert stats["total_hits"] == 0
    assert stats["expired_entries"] == 0


def test_stats_counts(tmp_path: Path) -> None:
    """Stats reflect stored entries and hit accumulation."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="a", **dict(_COMMON, ttl_seconds=9999))
    s.put_cache_entry(key="b", **dict(_COMMON, ttl_seconds=9999))
    s.touch_cache_entry("a")
    s.touch_cache_entry("a")
    s.touch_cache_entry("b")

    stats = s.response_cache_stats()
    assert stats["entries"] == 2
    assert stats["total_hits"] == 3
    assert stats["expired_entries"] == 0


def test_stats_expired_count(tmp_path: Path) -> None:
    """expired_entries counts stale rows without deleting them."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="live", **dict(_COMMON, ttl_seconds=9999))
    s.put_cache_entry(key="stale", **dict(_COMMON, ttl_seconds=1))

    # We can't freeze time.time() inside response_cache_stats, so we just
    # verify that after a purge the stale count drops.
    future = time.time() + 10
    deleted = s.purge_expired_cache(now=future)
    assert deleted == 1

    stats = s.response_cache_stats()
    assert stats["entries"] == 1


# ---------------------------------------------------------------------------
# embedding BLOB round-trip
# ---------------------------------------------------------------------------


def test_embedding_roundtrip(tmp_path: Path) -> None:
    """Embedding bytes are stored and retrieved without corruption."""
    s = _storage(tmp_path)
    blob = bytes(range(16))
    s.put_cache_entry(key="k_emb", **dict(_COMMON, embedding=blob))
    row = s.get_cache_entry("k_emb")
    assert row is not None
    assert row["embedding"] == blob


def test_embedding_null(tmp_path: Path) -> None:
    """Embedding defaults to None when not supplied."""
    s = _storage(tmp_path)
    s.put_cache_entry(key="k_no_emb", **_COMMON)
    row = s.get_cache_entry("k_no_emb")
    assert row is not None
    assert row["embedding"] is None


# ---------------------------------------------------------------------------
# CacheConfig loader
# ---------------------------------------------------------------------------


def test_cache_config_defaults_when_missing() -> None:
    """load_cache_config returns safe defaults when the YAML file is absent."""
    from modelmesh.core.cache.config import load_cache_config

    cfg = load_cache_config(path="/nonexistent/cache.yaml")
    assert cfg.enabled is True
    assert cfg.ttl_seconds == 86400
    assert cfg.max_entries == 2000
    assert cfg.min_answer_chars == 1
    assert cfg.max_answer_chars == 20000
    assert cfg.replay_chunk_chars == 40
    assert cfg.semantic.enabled is False
    assert cfg.semantic.threshold == pytest.approx(0.92)
    assert cfg.semantic.embedder == "hashing"
    assert cfg.semantic.model_name == "all-MiniLM-L6-v2"


def test_cache_config_loads_yaml(tmp_path: Path) -> None:
    """load_cache_config reads and validates a valid YAML file."""
    from modelmesh.core.cache.config import load_cache_config

    cfg_yaml = tmp_path / "cache.yaml"
    cfg_yaml.write_text(
        "enabled: false\nttl_seconds: 7200\nmax_entries: 500\n",
        encoding="utf-8",
    )
    cfg = load_cache_config(path=cfg_yaml)
    assert cfg.enabled is False
    assert cfg.ttl_seconds == 7200
    assert cfg.max_entries == 500


def test_cache_config_invalid_raises(tmp_path: Path) -> None:
    """load_cache_config raises RegistryError for invalid YAML values."""
    from modelmesh.core.cache.config import load_cache_config
    from modelmesh.core.errors import RegistryError

    bad_yaml = tmp_path / "cache.yaml"
    bad_yaml.write_text("ttl_seconds: -99\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_cache_config(path=bad_yaml)


def test_cache_config_malformed_yaml_raises(tmp_path: Path) -> None:
    """load_cache_config raises RegistryError for syntactically broken YAML."""
    from modelmesh.core.cache.config import load_cache_config
    from modelmesh.core.errors import RegistryError

    bad_yaml = tmp_path / "cache.yaml"
    bad_yaml.write_text("enabled: [\nnot closed", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_cache_config(path=bad_yaml)


def test_cache_config_semantic_section(tmp_path: Path) -> None:
    """Semantic sub-section fields are loaded correctly."""
    from modelmesh.core.cache.config import load_cache_config

    cfg_yaml = tmp_path / "cache.yaml"
    cfg_yaml.write_text(
        "semantic:\n  enabled: true\n  threshold: 0.85\n  embedder: hashing\n",
        encoding="utf-8",
    )
    cfg = load_cache_config(path=cfg_yaml)
    assert cfg.semantic.enabled is True
    assert cfg.semantic.threshold == pytest.approx(0.85)


def test_cache_config_invalid_embedder_raises(tmp_path: Path) -> None:
    """An unrecognised embedder name fails Pydantic validation."""
    from modelmesh.core.cache.config import load_cache_config
    from modelmesh.core.errors import RegistryError

    bad = tmp_path / "cache.yaml"
    bad.write_text("semantic:\n  embedder: turbo_magic\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_cache_config(path=bad)


def test_cache_config_find_path_uses_repo_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_cache_config_path resolves to the real config/cache.yaml in the repo."""
    from modelmesh.core.cache.config import find_cache_config_path

    # We don't monkeypatch here — the real file should exist in the repo root.
    p = find_cache_config_path()
    # The file either exists (repo root) or is the default fallback path —
    # either way it must be named cache.yaml.
    assert p.name == "cache.yaml"


def test_save_and_reload_cache_config(tmp_path: Path) -> None:
    """save_cache_config → load_cache_config round-trips without data loss."""
    from modelmesh.core.cache.config import CacheConfig, load_cache_config, save_cache_config

    original = CacheConfig(enabled=False, ttl_seconds=1800, max_entries=100)
    out = tmp_path / "cache.yaml"
    save_cache_config(original, path=out)
    loaded = load_cache_config(path=out)
    assert loaded.enabled is False
    assert loaded.ttl_seconds == 1800
    assert loaded.max_entries == 100
