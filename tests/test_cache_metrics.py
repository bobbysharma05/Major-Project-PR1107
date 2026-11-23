"""Tests for Storage.cache_metrics: aggregation over routing_log cache statuses."""

from __future__ import annotations

import time

import pytest

from modelmesh.core.registry import ModelRegistry
from modelmesh.core.storage import Storage
from modelmesh.core.types import (
    Candidate,
    EndpointConfig,
    ModelConfig,
    ProviderConfig,
    RoutingDecision,
    Usage,
)


def _candidate(storage: Storage) -> Candidate:
    """A candidate from a minimal mock-only registry."""
    registry = ModelRegistry()
    registry.add_provider(ProviderConfig(id="mock", protocol="mock"))
    registry.add_model(
        ModelConfig(
            id="m1",
            display_name="M1",
            endpoints=[
                EndpointConfig(
                    id="m1@mock", provider="mock", api_model="x",
                    price_in_per_mtok=0.1, price_out_per_mtok=0.2,
                )
            ],
        )
    )
    return registry.candidates()[0]


def _log_turn(
    storage: Storage,
    cache_status: str,
    cache_block: dict,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cost_usd: float,
    created_at: float | None = None,
) -> None:
    """Write one routing_log row with the given cache status and features block."""
    cand = _candidate(storage)
    decision = RoutingDecision(
        chosen_candidate=cand,
        fallback_chain=[],
        strategy_name="cheapest_first",
        reason="test",
        request_features={"cache": cache_block},
    )
    if created_at is not None:
        decision.created_at = created_at
    storage.record_routing_log(
        decision,
        cand,
        Usage(input_tokens=tokens_in, output_tokens=tokens_out),
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_status=cache_status,
    )


def test_cache_metrics_aggregates_hits_misses_and_saved_costs(tmp_path) -> None:
    """Metrics report hits, misses, rates, and the saved totals from hit rows."""
    storage = Storage(db_path=tmp_path / "metrics.db")
    _log_turn(storage, "miss", {"status": "miss"}, 10, 5, 1000, 0.01)
    _log_turn(
        storage, "hit",
        {"status": "hit", "kind": "exact", "similarity": 1.0,
         "saved_cost_usd": 0.02, "saved_tokens_in": 3, "saved_tokens_out": 2},
        3, 2, 100, 0.0,
    )
    _log_turn(
        storage, "hit",
        {"status": "hit", "kind": "semantic", "similarity": 0.94,
         "saved_cost_usd": 0.03, "saved_tokens_in": 4, "saved_tokens_out": 1},
        4, 1, 200, 0.0,
    )
    _log_turn(
        storage, "not_cacheable", {"status": "not_cacheable", "reason": "tools"},
        1, 1, 50, 0.0,
    )

    metrics = storage.cache_metrics()
    assert metrics["total_cacheable_turns"] == 3
    assert metrics["hits"] == 2
    assert metrics["misses"] == 1
    assert metrics["hit_rate"] == pytest.approx(2 / 3)
    assert metrics["exact_hits"] == 1
    assert metrics["semantic_hits"] == 1
    assert metrics["saved_cost_usd"] == pytest.approx(0.05)
    assert metrics["saved_tokens_in"] == 7
    assert metrics["saved_tokens_out"] == 3
    assert metrics["avg_hit_latency_ms"] == pytest.approx(150.0)
    assert metrics["avg_miss_latency_ms"] == pytest.approx(1000.0)


def test_cache_metrics_with_no_rows(tmp_path) -> None:
    """Empty routing_log yields zeroed metrics without division errors."""
    storage = Storage(db_path=tmp_path / "empty.db")
    metrics = storage.cache_metrics()
    assert metrics["total_cacheable_turns"] == 0
    assert metrics["hits"] == 0
    assert metrics["misses"] == 0
    assert metrics["hit_rate"] == 0.0
    assert metrics["avg_hit_latency_ms"] == 0.0
    assert metrics["avg_miss_latency_ms"] == 0.0


def test_cache_metrics_since_filter(tmp_path) -> None:
    """Rows before ``since`` are excluded from the aggregation."""
    storage = Storage(db_path=tmp_path / "since.db")
    old = time.time() - 10_000
    _log_turn(storage, "miss", {"status": "miss"}, 1, 1, 100, 0.01, created_at=old)
    _log_turn(
        storage, "hit",
        {"status": "hit", "kind": "exact", "similarity": 1.0,
         "saved_cost_usd": 0.01, "saved_tokens_in": 1, "saved_tokens_out": 1},
        1, 1, 100, 0.0,
    )

    recent = storage.cache_metrics(since=old + 1)
    assert recent["hits"] == 1
    assert recent["total_cacheable_turns"] == 1
    assert recent["saved_cost_usd"] == pytest.approx(0.01)
