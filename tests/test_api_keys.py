"""Tests for private API-key storage and schema upgrades."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from modelmesh.core.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    """Isolated SQLite database."""
    return Storage(db_path=tmp_path / "keys.db")


def _database_bytes(storage: Storage) -> bytes:
    """Read every on-disk byte of the SQLite database, including WAL files."""
    paths = [storage.db_path]
    paths.extend(
        sorted(storage.db_path.parent.glob(f"{storage.db_path.name}-*"))
    )
    return b"".join(path.read_bytes() for path in paths if path.exists())


def test_create_returns_mm_key_and_stores_only_a_hash(storage: Storage) -> None:
    """The plaintext key is returned once and never written to the database."""
    key_id, plaintext = storage.create_api_key("demo")

    assert key_id
    assert plaintext.startswith("mm-")
    assert len(plaintext) > 35
    assert plaintext.encode("utf-8") not in _database_bytes(storage)

    digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    with sqlite3.connect(storage.db_path) as conn:
        row = conn.execute(
            "SELECT key_hash, name, request_count FROM api_keys WHERE id = ?",
            (key_id,),
        ).fetchone()
    assert row == (digest, "demo", 0)


def test_verify_wrong_revoked_and_unknown_keys(storage: Storage) -> None:
    """Only a live key verifies; wrong and revoked keys return None."""
    key_id, plaintext = storage.create_api_key("demo")

    info = storage.verify_api_key(plaintext)
    assert info is not None
    assert info["id"] == key_id
    assert "key_hash" not in info

    assert storage.verify_api_key("mm-not-a-real-key") is None
    assert storage.revoke_api_key(key_id) is True
    assert storage.verify_api_key(plaintext) is None


def test_verify_updates_usage_counters(storage: Storage) -> None:
    """Successful verification bumps request_count and last_used_at."""
    _key_id, plaintext = storage.create_api_key("demo")
    before = storage.list_api_keys()[0]
    assert before["request_count"] == 0
    assert before["last_used_at"] is None

    storage.verify_api_key(plaintext)
    storage.verify_api_key(plaintext)
    after = storage.list_api_keys()[0]
    assert after["request_count"] == 2
    assert after["last_used_at"] is not None
    assert after["last_used_at"] > after["created_at"]


def test_list_never_exposes_hashes(storage: Storage) -> None:
    """list_api_keys omits hashes entirely."""
    _key_id, plaintext = storage.create_api_key("demo")
    digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    rows = storage.list_api_keys()

    assert len(rows) == 1
    assert "key_hash" not in rows[0]
    assert digest not in repr(rows)


def test_revoke_is_idempotent(storage: Storage) -> None:
    """Unknown and already-revoked ids cannot be revoked again."""
    key_id, _plaintext = storage.create_api_key("demo")

    assert storage.revoke_api_key("missing-id") is False
    assert storage.revoke_api_key(key_id) is True
    assert storage.revoke_api_key(key_id) is False


def test_two_keys_are_independent(storage: Storage) -> None:
    """Revoking one key leaves the other usable."""
    first_id, first = storage.create_api_key("first")
    second_id, second = storage.create_api_key("second")

    assert storage.revoke_api_key(first_id) is True
    assert storage.verify_api_key(first) is None
    info = storage.verify_api_key(second)
    assert info is not None
    assert info["id"] == second_id
    assert info["request_count"] == 1


def test_v3_database_upgrades_to_v4(tmp_path: Path) -> None:
    """An old database gains api_keys without losing existing rows."""
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE routing_log (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                conversation_id TEXT,
                message_id TEXT,
                strategy TEXT NOT NULL DEFAULT 'manual',
                chosen_model_id TEXT,
                chosen_endpoint_id TEXT,
                provider_id TEXT,
                fallback_chain_json TEXT,
                candidates_scores_json TEXT,
                features_json TEXT,
                reason TEXT,
                cache_status TEXT DEFAULT 'miss',
                ttft_ms INTEGER DEFAULT 0,
                total_latency_ms INTEGER DEFAULT 0,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cached_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0.0,
                error_category TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO routing_log (id, created_at) VALUES (?, ?)",
            ("old-turn", 1.0),
        )
        conn.execute("PRAGMA user_version = 3")

    upgraded = Storage(db_path=db_path)
    assert upgraded.list_api_keys() == []
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute(
            "SELECT id FROM routing_log WHERE id = ?", ("old-turn",)
        ).fetchone() == ("old-turn",)

    key_id, plaintext = upgraded.create_api_key("after-upgrade")
    assert upgraded.verify_api_key(plaintext) is not None
    assert upgraded.list_api_keys()[0]["id"] == key_id
