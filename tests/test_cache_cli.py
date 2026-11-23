"""Tests for `modelmesh-cli cache stats|clear|purge` and the --no-cache flag."""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Any, Dict, Iterator, List

import pytest

import modelmesh.cli as cli
from modelmesh.core.storage import Storage


def _seed_cache_row(storage: Storage) -> None:
    storage.put_cache_entry(
        key="k1",
        norm_prompt="q",
        context_hash="c",
        system_hash="s",
        params_hash="p",
        scope="auto",
        answer_text="a",
        ttl_seconds=3600,
        model_id="m",
        endpoint_id="m@e",
        provider_id="p",
    )


def _make_expired_row(storage: Storage) -> None:
    storage.put_cache_entry(
        key="k2",
        norm_prompt="q",
        context_hash="c",
        system_hash="s",
        params_hash="p",
        scope="auto",
        answer_text="a",
        ttl_seconds=10,
    )
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("UPDATE response_cache SET created_at = 0")


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: List[str], db_path: Any) -> int:
    monkeypatch.setenv("MODELMESH_DB_PATH", str(db_path))
    monkeypatch.setattr(sys, "argv", ["modelmesh-cli", *argv])
    return cli.main()


def test_cache_stats_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """`modelmesh-cli cache stats` prints the aggregate metrics as JSON."""
    db = tmp_path / "cli_stats.db"
    code = _run_cli(monkeypatch, ["cache", "stats"], db)
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_cacheable_turns"] == 0
    assert out["hits"] == 0
    assert out["misses"] == 0


def test_cache_stats_output_matches_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """`cache stats` JSON matches Storage.cache_metrics for the same database."""
    db = tmp_path / "cli_stats2.db"
    storage = Storage(db_path=db)
    _seed_cache_row(storage)
    _run_cli(monkeypatch, ["cache", "stats"], db)
    out = json.loads(capsys.readouterr().out)
    assert out == storage.cache_metrics()


def test_cache_clear_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """`modelmesh-cli cache clear` removes all cached answers."""
    db = tmp_path / "cli_clear.db"
    storage = Storage(db_path=db)
    _seed_cache_row(storage)
    assert storage.response_cache_stats()["entries"] == 1

    code = _run_cli(monkeypatch, ["cache", "clear"], db)
    assert code == 0
    out = capsys.readouterr().out
    assert "Cleared response cache: 1" in out
    assert storage.response_cache_stats()["entries"] == 0


def test_cache_purge_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """`modelmesh-cli cache purge` removes expired entries."""
    db = tmp_path / "cli_purge.db"
    storage = Storage(db_path=db)
    _make_expired_row(storage)

    code = _run_cli(monkeypatch, ["cache", "purge"], db)
    assert code == 0
    out = capsys.readouterr().out
    assert "Purged expired response-cache entries: 1" in out
    assert storage.response_cache_stats()["entries"] == 0


def test_cache_unknown_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """An unknown `cache` subcommand prints usage and exits non-zero."""
    db = tmp_path / "cli_unknown.db"
    code = _run_cli(monkeypatch, ["cache", "frobnicate"], db)
    assert code == 1
    assert "Unknown cache command" in capsys.readouterr().out


def test_no_cache_flag_passes_bypass_cache(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """`--no-cache` makes run_turn receive bypass_cache=True."""
    db = tmp_path / "cli_nocache.db"
    seen: List[Dict[str, Any]] = []

    def fake_run_turn(self: Any, **kwargs: Any) -> Iterator[Any]:
        seen.append(kwargs)
        return iter([])

    monkeypatch.setattr(cli.ChatEngine, "run_turn", fake_run_turn)
    code = _run_cli(monkeypatch, ["hello", "--no-cache"], db)
    assert code == 0
    assert len(seen) == 1
    assert seen[0]["bypass_cache"] is True


def test_normal_send_passes_bypass_cache_false(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    """Without --no-cache, run_turn receives bypass_cache=False."""
    db = tmp_path / "cli_normal.db"
    seen: List[Dict[str, Any]] = []

    def fake_run_turn(self: Any, **kwargs: Any) -> Iterator[Any]:
        seen.append(kwargs)
        return iter([])

    monkeypatch.setattr(cli.ChatEngine, "run_turn", fake_run_turn)
    code = _run_cli(monkeypatch, ["hello"], db)
    assert code == 0
    assert len(seen) == 1
    assert seen[0]["bypass_cache"] is False
