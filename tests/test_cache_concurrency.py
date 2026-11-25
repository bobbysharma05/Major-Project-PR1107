"""Concurrency tests for the response cache and its replay path."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import modelmesh.core.engine as engine_module
from modelmesh.core.cache.config import CacheConfig
from modelmesh.core.engine import ChatEngine
from modelmesh.core.providers.mock import MockAdapter
from modelmesh.core.registry import ModelRegistry
from modelmesh.core.routing.router import Router
from modelmesh.core.storage import Storage
from modelmesh.core.types import (
    ChatRequest,
    EndpointConfig,
    Message,
    ModelConfig,
    ProviderConfig,
    StreamEvent,
    StreamEventType,
)


def _registry() -> ModelRegistry:
    """Registry with one offline mock endpoint."""
    registry = ModelRegistry()
    registry.add_provider(ProviderConfig(id="mock", protocol="mock"))
    registry.add_model(
        ModelConfig(
            id="mock-cheap",
            display_name="mock-cheap",
            tier="cheap",
            capabilities=["streaming"],
            endpoints=[
                EndpointConfig(
                    id="mock-cheap@mock",
                    provider="mock",
                    api_model="mock-instant",
                    price_in_per_mtok=0.01,
                    price_out_per_mtok=0.02,
                )
            ],
        )
    )
    return registry


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    """Isolated SQLite database."""
    return Storage(db_path=tmp_path / "concurrent.db")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Count provider calls by wrapping the deterministic mock adapter."""
    recorded: List[str] = []

    class CountingAdapter(MockAdapter):
        def stream_chat(
            self,
            request: ChatRequest,
            candidate: Any,
            cancel_event: Any = None,
        ) -> Any:
            recorded.append(candidate.endpoint_id)
            yield from MockAdapter.stream_chat(self, request, candidate, cancel_event)

    monkeypatch.setattr(
        engine_module,
        "get_adapter",
        lambda protocol: CountingAdapter(),
    )
    return recorded


@pytest.fixture
def engine(storage: Storage) -> ChatEngine:
    """Engine using explicit cache config so tests never read cache.yaml."""
    registry = _registry()
    return ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=storage),
        storage=storage,
        cache_config=CacheConfig(),
    )


def _request(text: str = "What is a race condition?") -> ChatRequest:
    """Build a simple cacheable request."""
    return ChatRequest(messages=[Message.from_text("user", text)])


def _collect(chat: ChatEngine, request: ChatRequest) -> Tuple[str, Dict[str, Any]]:
    """Run a turn and return the answer text and DONE data."""
    events = list(
        chat.run_turn(request=request, strategy_name="cheapest_first")
    )
    done = next(e for e in events if e.type == StreamEventType.DONE)
    text = "".join(e.text or "" for e in events if e.type == StreamEventType.TEXT_DELTA)
    return text, done.data


def test_concurrent_identical_turns_leave_one_healthy_entry(
    engine: ChatEngine, storage: Storage, calls: List[str]
) -> None:
    """Two simultaneous turns cannot corrupt the cache or deadlock storage."""
    barrier = threading.Barrier(2)

    def turn() -> Tuple[str, Dict[str, Any]]:
        barrier.wait(timeout=10)
        return _collect(engine, _request())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(turn) for _ in range(2)]
        first, second = [future.result(timeout=20) for future in futures]

    assert first[0] == second[0]
    # Both may miss and write; INSERT OR REPLACE keeps a single final row.
    assert storage.response_cache_stats()["entries"] == 1
    assert len(calls) in (1, 2)

    calls_before = len(calls)
    replay_text, replay_done = _collect(engine, _request())
    assert replay_text == first[0]
    assert replay_done["metadata"]["cache"]["status"] == "hit"
    assert replay_done["cost_usd"] == 0.0
    assert len(calls) == calls_before


def test_replay_yields_without_holding_storage_lock(
    engine: ChatEngine, storage: Storage
) -> None:
    """A paused replay must not block another thread's storage call."""
    _collect(engine, _request("replay lock probe"))

    first_delta = threading.Event()
    release = threading.Event()
    replay_finished = threading.Event()
    errors: List[BaseException] = []

    def replay() -> None:
        try:
            for event in engine.run_turn(
                request=_request("replay lock probe"),
                strategy_name="cheapest_first",
            ):
                if (
                    event.type == StreamEventType.TEXT_DELTA
                    and not first_delta.is_set()
                ):
                    first_delta.set()
                    if not release.wait(timeout=10):
                        errors.append(TimeoutError("replay release timed out"))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            replay_finished.set()

    worker = threading.Thread(target=replay, daemon=True)
    worker.start()
    assert first_delta.wait(timeout=10)

    stats: Dict[str, Any] = {}
    stats_finished = threading.Event()

    def read_stats() -> None:
        stats.update(storage.response_cache_stats())
        stats_finished.set()

    reader = threading.Thread(target=read_stats, daemon=True)
    reader.start()
    assert stats_finished.wait(timeout=3), "storage lock was held during replay"
    assert stats["entries"] == 1

    release.set()
    assert replay_finished.wait(timeout=10)
    worker.join(timeout=5)
    reader.join(timeout=5)
    assert errors == []
