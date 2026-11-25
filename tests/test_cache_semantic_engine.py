"""Engine-level tests for semantic response-cache matching."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

import modelmesh.core.engine as engine_module
from modelmesh.core.cache.config import CacheConfig, SemanticCacheConfig
from modelmesh.core.cache.embedding import HashingEmbedder, pack_embedding, unpack_embedding
from modelmesh.core.cache.keys import build_cache_key
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

THRESHOLD = 0.70
SOURCE = "Can you explain what a hash map is?"
PARAPHRASE = "What is a hash map?"


def _registry() -> ModelRegistry:
    """Registry with two offline mock models and no external dependencies."""
    registry = ModelRegistry()
    registry.add_provider(ProviderConfig(id="mock", protocol="mock"))
    for model_id, endpoint_id in (("mock-cheap", "mock-cheap@mock"), ("mock-other", "mock-other@mock")):
        registry.add_model(
            ModelConfig(
                id=model_id,
                display_name=model_id,
                tier="cheap",
                capabilities=["streaming"],
                endpoints=[
                    EndpointConfig(
                        id=endpoint_id,
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
def registry() -> ModelRegistry:
    """Fresh mock-only registry."""
    return _registry()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    """Isolated SQLite database."""
    return Storage(db_path=tmp_path / "semantic.db")


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

    monkeypatch.setattr(engine_module, "get_adapter", lambda protocol: CountingAdapter())
    return recorded


def _engine(
    registry: ModelRegistry,
    storage: Storage,
    *,
    enabled: bool = True,
    threshold: float = THRESHOLD,
) -> ChatEngine:
    """Engine with an explicit semantic cache config (never reads cache.yaml)."""
    config = CacheConfig(
        semantic=SemanticCacheConfig(
            enabled=enabled,
            threshold=threshold,
            embedder="hashing",
            hashing_dim=512,
        )
    )
    return ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=storage),
        storage=storage,
        cache_config=config,
    )


def _request(text: str) -> ChatRequest:
    """Build a simple cacheable request."""
    return ChatRequest(messages=[Message.from_text("user", text)])


def _collect(
    chat: ChatEngine, request: ChatRequest, **kwargs: Any
) -> tuple[List[StreamEvent], Dict[str, Any], str]:
    """Run a turn and return its events, DONE data and replayed answer text."""
    events = list(chat.run_turn(request=request, strategy_name="cheapest_first", **kwargs))
    done = next(e for e in events if e.type == StreamEventType.DONE)
    text = "".join(e.text or "" for e in events if e.type == StreamEventType.TEXT_DELTA)
    return events, done.data, text


def _seed_entry(
    storage: Storage,
    source_text: str,
    answer: str = "A hash map stores key/value pairs.",
    **overrides: Any,
) -> None:
    """Store a cache row with a hashing embedding for *source_text*."""
    parts = build_cache_key(_request(source_text), "", None)
    vector = HashingEmbedder(512).embed(parts.norm_prompt)
    values: Dict[str, Any] = {
        "key": parts.key,
        "norm_prompt": parts.norm_prompt,
        "context_hash": parts.context_hash,
        "system_hash": parts.system_hash,
        "params_hash": parts.params_hash,
        "scope": parts.scope,
        "answer_text": answer,
        "ttl_seconds": 3600,
        "model_id": "mock-cheap",
        "endpoint_id": "mock-cheap@mock",
        "provider_id": "mock",
        "tokens_in": 11,
        "tokens_out": 7,
        "cost_usd": 0.0002,
        "embedding": pack_embedding(vector),
        "max_entries": 100,
    }
    values.update(overrides)
    storage.put_cache_entry(**values)


def _statuses(storage: Storage) -> List[str]:
    """Read cache_status values in insertion order."""
    with sqlite3.connect(storage.db_path) as conn:
        return [
            str(row[0])
            for row in conn.execute("SELECT cache_status FROM routing_log ORDER BY rowid")
        ]


def _last_features(storage: Storage) -> Dict[str, Any]:
    """Return the cache block from the most recent routing-log row."""
    with sqlite3.connect(storage.db_path) as conn:
        raw = conn.execute(
            "SELECT features_json FROM routing_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    return json.loads(raw)["cache"]


def test_paraphrase_is_replayed_as_a_semantic_hit(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """A near-identical prompt hits the stored answer with kind='semantic'."""
    _seed_entry(storage, SOURCE)
    chat = _engine(registry, storage)

    events, done, text = _collect(chat, _request(PARAPHRASE))
    assert calls == []
    assert text == "A hash map stores key/value pairs."
    assert done["cost_usd"] == 0.0
    block = done["metadata"]["cache"]
    assert block["status"] == "hit"
    assert block["kind"] == "semantic"
    assert block["similarity"] >= THRESHOLD
    assert _statuses(storage) == ["hit"]
    logged = _last_features(storage)
    assert logged["kind"] == "semantic"
    assert logged["similarity"] == pytest.approx(block["similarity"])
    routed = next(e for e in events if e.type == StreamEventType.ROUTED)
    assert routed.data["reason"] == "Served from response cache"


def test_semantic_hit_touches_the_matched_entry(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """A semantic hit updates the stored source entry's hit_count/last_hit_at."""
    _seed_entry(storage, SOURCE)
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("UPDATE response_cache SET last_hit_at = 0")

    chat = _engine(registry, storage)
    _events, done, _text = _collect(chat, _request(PARAPHRASE))
    assert done["metadata"]["cache"]["kind"] == "semantic"
    assert calls == []

    with sqlite3.connect(storage.db_path) as conn:
        rows = conn.execute(
            "SELECT key, hit_count, last_hit_at FROM response_cache"
        ).fetchall()
    assert len(rows) == 1  # the paraphrase itself was never stored
    _key, hit_count, last_hit_at = rows[0]
    assert hit_count == 1
    assert last_hit_at > 0


def test_numeric_change_is_a_miss_and_logs_the_near_miss(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """``2+2`` never matches ``2+3``; the rejected similarity is logged."""
    _seed_entry(storage, "What is 2+2?", answer="4")
    chat = _engine(registry, storage, threshold=0.0)

    _events, done, _text = _collect(chat, _request("What is 2+3?"))
    assert len(calls) == 1
    assert done["metadata"]["cache"]["status"] == "miss"
    near_miss = _last_features(storage)["near_miss"]
    assert near_miss["reason"] == "numeric_mismatch"
    assert near_miss["similarity"] > 0.0


def test_negation_change_is_a_miss(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """``use X`` never matches ``not use X`` even at a zero threshold."""
    _seed_entry(storage, "Should I use X?", answer="Yes.")
    chat = _engine(registry, storage, threshold=0.0)

    _events, done, _text = _collect(chat, _request("Should I not use X?"))
    assert len(calls) == 1
    assert done["metadata"]["cache"]["status"] == "miss"
    assert _last_features(storage)["near_miss"]["reason"] == "negation_mismatch"


@pytest.mark.parametrize("field", ["context_hash", "system_hash", "params_hash", "scope"])
def test_different_scope_is_never_compared(
    registry: ModelRegistry,
    storage: Storage,
    calls: List[str],
    field: str,
) -> None:
    """Entries from another context/system/params/scope are filtered out."""
    parts = build_cache_key(_request(SOURCE), "", None)
    overrides = {
        "context_hash": parts.context_hash,
        "system_hash": parts.system_hash,
        "params_hash": parts.params_hash,
        "scope": parts.scope,
    }
    overrides[field] = "different"
    _seed_entry(storage, SOURCE, **overrides)
    chat = _engine(registry, storage, threshold=0.0)

    _events, done, _text = _collect(chat, _request(PARAPHRASE))
    assert len(calls) == 1
    assert done["metadata"]["cache"]["status"] == "miss"
    assert "near_miss" not in done["metadata"]["cache"]


def test_semantic_disabled_uses_exact_matching_only(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """With semantic disabled a paraphrase misses, while the exact prompt hits."""
    _seed_entry(storage, SOURCE)
    chat = _engine(registry, storage, enabled=False)

    _events, paraphrase_done, _text = _collect(chat, _request(PARAPHRASE))
    assert len(calls) == 1
    assert paraphrase_done["metadata"]["cache"] == {"status": "miss"}

    _events, exact_done, text = _collect(chat, _request(SOURCE))
    assert len(calls) == 1
    assert text == "A hash map stores key/value pairs."
    assert exact_done["metadata"]["cache"]["kind"] == "exact"


def test_bypass_cache_skips_semantic_lookup(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """Regenerate/Retry bypasses semantic matching and calls the provider."""
    _seed_entry(storage, SOURCE)
    chat = _engine(registry, storage)

    _events, done, _text = _collect(chat, _request(PARAPHRASE), bypass_cache=True)
    assert len(calls) == 1
    assert done["cost_usd"] > 0
    assert done["metadata"]["cache"] == {"status": "miss"}


def test_miss_stores_an_embedding_when_semantic_is_enabled(
    registry: ModelRegistry, storage: Storage, calls: List[str]
) -> None:
    """The first turn stores its answer together with a float32 embedding."""
    chat = _engine(registry, storage)
    _collect(chat, _request("A completely fresh question about giraffes?"))

    with sqlite3.connect(storage.db_path) as conn:
        row = conn.execute(
            "SELECT norm_prompt, embedding FROM response_cache"
        ).fetchone()
    assert row is not None
    assert row[1] is not None
    assert len(unpack_embedding(row[1])) == 512


def test_semantic_lookup_fails_open_when_embedder_breaks(
    registry: ModelRegistry, storage: Storage, calls: List[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken embedder falls back to exact-only matching without failing the turn."""

    def broken_embedder(_config: Any) -> Any:
        raise RuntimeError("embedder down")

    monkeypatch.setattr(engine_module, "create_embedder", broken_embedder)
    _seed_entry(storage, SOURCE)
    chat = _engine(registry, storage)

    events, done, text = _collect(chat, _request(PARAPHRASE))
    assert len(calls) == 1
    assert text
    assert done["metadata"]["cache"]["status"] == "miss"
    assert not any(event.type == StreamEventType.ERROR for event in events)
