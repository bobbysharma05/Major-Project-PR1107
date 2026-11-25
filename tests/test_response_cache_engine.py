"""Engine-level tests for the exact-match response cache."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

import modelmesh.core.engine as engine_module
from modelmesh.core.cache.config import CacheConfig
from modelmesh.core.cache.keys import build_cache_key
from modelmesh.core.engine import ChatEngine
from modelmesh.core.providers.mock import MockAdapter
from modelmesh.core.registry import ModelRegistry
from modelmesh.core.routing.router import Router
from modelmesh.core.storage import Storage
from modelmesh.core.tools.registry import ToolRegistry
from modelmesh.core.types import (
    ChatRequest,
    EndpointConfig,
    ImagePart,
    Message,
    ModelConfig,
    ProviderConfig,
    StreamEvent,
    StreamEventType,
    TextPart,
    ToolCall,
    ToolSpec,
    Usage,
)


def _registry() -> ModelRegistry:
    """Registry with two offline mock models and no external dependencies."""
    registry = ModelRegistry()
    registry.add_provider(ProviderConfig(id="mock", protocol="mock"))
    for model_id, endpoint_id, price in (
        ("mock-cheap", "mock-cheap@mock", 0.01),
        ("mock-other", "mock-other@mock", 0.05),
    ):
        registry.add_model(
            ModelConfig(
                id=model_id,
                display_name=model_id,
                tier="cheap",
                capabilities=["streaming", "tools", "vision"],
                endpoints=[
                    EndpointConfig(
                        id=endpoint_id,
                        provider="mock",
                        api_model="mock-instant",
                        price_in_per_mtok=price,
                        price_out_per_mtok=price * 2,
                    )
                ],
            )
        )
    return registry


@pytest.fixture
def registry() -> ModelRegistry:
    """Fresh mock-only registry."""
    return _registry()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    """Isolated SQLite database."""
    return Storage(db_path=tmp_path / "cache.db")


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
        ) -> Iterator[StreamEvent]:
            recorded.append(candidate.endpoint_id)
            yield from MockAdapter.stream_chat(self, request, candidate, cancel_event)

    def factory(protocol: str) -> CountingAdapter:
        return CountingAdapter()

    monkeypatch.setattr(engine_module, "get_adapter", factory)
    return recorded


@pytest.fixture
def engine(registry: ModelRegistry, storage: Storage) -> ChatEngine:
    """Engine using explicit cache config so tests never read cache.yaml."""
    return ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=storage),
        storage=storage,
        cache_config=CacheConfig(),
    )


def _request(text: str = "What is a hash map?") -> ChatRequest:
    """Build a simple cacheable request."""
    return ChatRequest(messages=[Message.from_text("user", text)])


def _collect(
    chat: ChatEngine,
    request: ChatRequest,
    strategy_name: str = "cheapest_first",
    **kwargs: Any,
) -> tuple[List[StreamEvent], Dict[str, Any], str]:
    """Run a turn and return its events, DONE data and replayed answer text."""
    events = list(chat.run_turn(request=request, strategy_name=strategy_name, **kwargs))
    done = next(e for e in events if e.type == StreamEventType.DONE)
    text = "".join(e.text or "" for e in events if e.type == StreamEventType.TEXT_DELTA)
    return events, done.data, text


def _statuses(db: Storage) -> List[str]:
    """Read cache_status values in insertion order."""
    with sqlite3.connect(db.db_path) as conn:
        return [str(row[0]) for row in conn.execute(
            "SELECT cache_status FROM routing_log ORDER BY rowid"
        )]


def test_second_identical_turn_replays_without_provider_call(
    engine: ChatEngine, storage: Storage, calls: List[str]
) -> None:
    """A repeated prompt is replayed at zero cost with one routing-log hit row."""
    first_events, first_done, first_text = _collect(engine, _request(), conversation_id="c1")
    assert len(calls) == 1
    assert first_done["cost_usd"] > 0

    second_events, second_done, second_text = _collect(engine, _request(), conversation_id="c1")
    assert len(calls) == 1
    assert second_text == first_text
    assert second_done["cost_usd"] == 0.0
    assert second_done["metadata"]["cache"]["status"] == "hit"

    routed = next(e for e in second_events if e.type == StreamEventType.ROUTED)
    assert routed.data["reason"] == "Served from response cache"
    assert routed.data["metadata"]["cache"]["kind"] == "exact"
    assert routed.data["metadata"]["cache"]["similarity"] == 1.0
    assert _statuses(storage) == ["miss", "hit"]

    with sqlite3.connect(storage.db_path) as conn:
        features = conn.execute(
            "SELECT features_json FROM routing_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    assert json.loads(features)["cache"]["status"] == "hit"


@pytest.mark.parametrize("change", ["system_prompt", "temperature"])
def test_changed_input_causes_miss(
    engine: ChatEngine, calls: List[str], change: str
) -> None:
    """A different system prompt or temperature defeats the cache key."""
    _collect(engine, _request())
    changed = ChatRequest(
        messages=[Message.from_text("user", "What is a hash map?")],
        system_prompt="Use short answers." if change == "system_prompt" else None,
        temperature=0.4 if change == "temperature" else None,
    )
    _collect(engine, changed)
    assert len(calls) == 2


def test_tools_are_never_cached(engine: ChatEngine, calls: List[str]) -> None:
    """Tool-enabled requests bypass both cache lookup and storage."""
    request = ChatRequest(
        messages=[Message.from_text("user", "q")],
        tools=[ToolSpec(name="calculator", description="Math")],
    )
    _collect(engine, request)
    _collect(engine, request)
    assert len(calls) == 2


def test_attachment_is_never_cached(engine: ChatEngine, calls: List[str]) -> None:
    """Named text parts and images are attachments and are never cached."""
    request = ChatRequest(
        messages=[Message(role="user", parts=[TextPart(text="q", name="notes.txt")])]
    )
    _collect(engine, request)
    _collect(engine, request)
    image_request = ChatRequest(
        messages=[Message(role="user", parts=[ImagePart(media_type="image/png", data="abc")])]
    )
    _collect(engine, image_request)
    assert len(calls) == 3


def test_turn_with_a_tool_call_is_not_stored(
    engine: ChatEngine,
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executing a tool disqualifies the turn from being cached."""
    tool_registry = ToolRegistry()
    tool_registry.register("calculator", "Math", {"expression": {"type": "string"}}, lambda expression: 42)
    engine.tool_registry = tool_registry

    class ToolOnceAdapter(MockAdapter):
        def stream_chat(
            self,
            request: ChatRequest,
            candidate: Any,
            cancel_event: Any = None,
        ) -> Iterator[StreamEvent]:
            if not any(message.role == "tool" for message in request.messages):
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    tool_call=ToolCall(id="1", name="calculator", arguments={"expression": "6*7"}),
                )
                yield StreamEvent(type=StreamEventType.USAGE, usage=Usage(input_tokens=5, output_tokens=5))
                return
            yield from MockAdapter.stream_chat(self, request, candidate, cancel_event)

    monkeypatch.setattr(engine_module, "get_adapter", lambda protocol: ToolOnceAdapter())

    events, _done, text = _collect(engine, _request("compute 6*7"))
    assert any(event.type == StreamEventType.TOOL_RESULT for event in events)
    assert text
    assert storage.response_cache_stats()["entries"] == 0


def test_bypass_cache_calls_provider_again_and_refreshes_entry(
    engine: ChatEngine, storage: Storage, calls: List[str]
) -> None:
    """Regenerate bypasses lookup but still overwrites the stored answer."""
    _collect(engine, _request())
    _bypass_events, bypass_done, _text = _collect(engine, _request(), bypass_cache=True)
    assert len(calls) == 2
    assert bypass_done["cost_usd"] > 0
    assert bypass_done["metadata"]["cache"] == {"status": "miss"}

    _replay_events, replay_done, _text = _collect(engine, _request())
    assert len(calls) == 2
    assert replay_done["cost_usd"] == 0.0
    assert storage.response_cache_stats()["total_hits"] == 1


def test_expired_entry_is_a_miss(
    engine: ChatEngine, storage: Storage, calls: List[str]
) -> None:
    """An expired cache entry is not replayed."""
    _collect(engine, _request())
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("UPDATE response_cache SET created_at = 0")
    _collect(engine, _request())
    assert len(calls) == 2
    assert _statuses(storage) == ["miss", "miss"]


def test_storage_failure_fails_open(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """A broken cache storage layer never fails the chat turn."""

    class BrokenStorage:
        def __init__(self, inner: Storage) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def get_cache_entry(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("cache down")

        def put_cache_entry(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("cache down")

    broken = BrokenStorage(storage)
    chat = ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=broken),
        storage=broken,
        cache_config=CacheConfig(),
    )
    events, _done, text = _collect(chat, _request())
    assert text
    done_event = next(e for e in events if e.type == StreamEventType.DONE)
    assert done_event.finish_reason == "stop"
    assert len(calls) == 1
    assert not any(event.type == StreamEventType.ERROR for event in events)


def test_cancelled_turn_stores_nothing(
    engine: ChatEngine, storage: Storage, calls: List[str]
) -> None:
    """Cancelling before streaming prevents a cache write."""
    cancel = threading.Event()
    cancel.set()
    events = list(
        engine.run_turn(
            request=_request(),
            strategy_name="cheapest_first",
            cancel_event=cancel,
        )
    )
    assert not any(event.type == StreamEventType.DONE for event in events)
    assert storage.response_cache_stats()["entries"] == 0
    assert calls == []


def test_hit_with_stale_endpoint_is_treated_as_miss(
    engine: ChatEngine,
    registry: ModelRegistry,
    storage: Storage,
    calls: List[str],
) -> None:
    """A cache row whose endpoint and model no longer exist is ignored."""
    _collect(engine, _request())
    with sqlite3.connect(storage.db_path) as conn:
        stale_model = conn.execute(
            "SELECT model_id FROM response_cache LIMIT 1"
        ).fetchone()[0]

    class MissingModelRegistry:
        def __init__(self, inner: ModelRegistry) -> None:
            self._inner = inner

        def candidates(self, enabled_only: bool = True) -> List[Any]:
            return [
                c for c in self._inner.candidates(enabled_only)
                if c.model_id != stale_model
            ]

        def get_candidate(self, endpoint_id: str) -> Any:
            return None

    stale_registry = MissingModelRegistry(registry)
    chat = ChatEngine(
        registry=stale_registry,
        router=Router(registry=stale_registry, storage=storage),
        storage=storage,
        cache_config=CacheConfig(),
    )
    _collect(chat, _request())
    assert len(calls) == 2


def test_cache_hit_under_smart_clef_logs_hit_without_exception(
    engine: ChatEngine, storage: Storage, calls: List[str]
) -> None:
    """A hit replay under Smart routing logs cache_status 'hit' and never raises."""
    request = _request("smart cache question")
    parts = build_cache_key(request, "", None)
    storage.put_cache_entry(
        key=parts.key,
        norm_prompt=parts.norm_prompt,
        context_hash=parts.context_hash,
        system_hash=parts.system_hash,
        params_hash=parts.params_hash,
        scope=parts.scope,
        answer_text="cached smart answer",
        ttl_seconds=3600,
        model_id="mock-cheap",
        endpoint_id="mock-cheap@mock",
        provider_id="mock",
        tokens_in=11,
        tokens_out=7,
        cost_usd=0.0002,
        max_entries=100,
    )

    events, done, text = _collect(engine, request, strategy_name="smart_clef")
    assert calls == []  # a hit never calls a provider
    assert text == "cached smart answer"
    assert done["cost_usd"] == 0.0
    assert done["metadata"]["cache"]["status"] == "hit"
    assert _statuses(storage) == ["hit"]
    assert not any(e.type == StreamEventType.ERROR for e in events)


def test_same_prompt_hits_when_mocked_base_date_changes(
    engine: ChatEngine, calls: List[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 'Current date' line in the base prompt is not part of the key."""
    dates = iter(["Saturday, January 01, 2026", "Sunday, January 02, 2026"])

    def fake_base_prompt() -> str:
        return f"You are ModelMesh.\nCurrent date: {next(dates)}\n\nGuidelines:"

    monkeypatch.setattr(engine_module, "get_base_system_prompt", fake_base_prompt)
    _collect(engine, _request("date-independent question"))
    _events, done, text = _collect(engine, _request("date-independent question"))
    assert len(calls) == 1
    assert done["cost_usd"] == 0.0
    assert text


def test_store_answers_false_disables_cache_reads_and_writes(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """The privacy switch prevents both storing and replaying answers."""
    chat = ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=storage),
        storage=storage,
        cache_config=CacheConfig(store_answers=False),
    )
    _collect(chat, _request("private question"))
    _events, done, _text = _collect(chat, _request("private question"))

    assert len(calls) == 2
    assert done["metadata"]["cache"] == {
        "status": "not_cacheable",
        "reason": "cache_disabled",
    }
    assert storage.response_cache_stats()["entries"] == 0
