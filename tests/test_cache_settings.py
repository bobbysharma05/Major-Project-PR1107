"""Tests for the Cache settings tab, engine config reload, and the usage dashboard."""

from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtWidgets import QMessageBox

from modelmesh.core.cache import config as cache_config
from modelmesh.core.cache.config import CacheConfig, load_cache_config, save_cache_config
from modelmesh.core.engine import ChatEngine
from modelmesh.core.registry import ModelRegistry
from modelmesh.core.routing.router import Router
from modelmesh.core.storage import Storage
from modelmesh.core.types import (
    Candidate,
    EndpointConfig,
    ModelConfig,
    ProviderConfig,
    RoutingDecision,
    Usage,
)
from modelmesh.ui.settings.cache_tab import CacheTab
from modelmesh.ui.usage_view import UsageDialog


def _registry() -> ModelRegistry:
    """A minimal mock-only registry, same shape as other UI tests."""
    reg = ModelRegistry()
    reg.add_provider(ProviderConfig(id="mock", protocol="mock"))
    reg.add_model(
        ModelConfig(
            id="m1",
            display_name="M1",
            capabilities=["streaming"],
            endpoints=[EndpointConfig(id="m1@mock", provider="mock", api_model="x")],
        )
    )
    return reg


def _candidate(storage: Storage) -> Candidate:
    """A candidate from a fresh registry."""
    return _registry().candidates()[0]


def _log_turn(
    storage: Storage,
    cache_status: str,
    cache_block: dict,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cost_usd: float,
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
    storage.record_routing_log(
        decision,
        cand,
        Usage(input_tokens=tokens_in, output_tokens=tokens_out),
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_status=cache_status,
    )


def _patch_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, name: str = "cache.yaml"):
    """Redirect cache.yaml to a temp path so tests never touch the real file."""
    target = tmp_path / name
    monkeypatch.setattr(cache_config, "find_cache_config_path", lambda: target)
    return target


def test_cache_tab_persists_settings_and_reloads_stats(
    qapp: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tab edits are validated and saved to cache.yaml; stats reflect them."""
    _patch_cache_path(monkeypatch, tmp_path)
    storage = Storage(db_path=tmp_path / "s.db")
    tab = CacheTab(storage=storage)

    assert tab.enabled_cb.isChecked() is True
    assert tab.semantic_cb.isEnabled() is True
    assert tab.semantic_cb.isChecked() is False
    assert "Turns: 0" in tab.stats_label.text()

    tab.enabled_cb.setChecked(False)
    tab.ttl_spin.setValue(48)
    tab.save()

    cfg = load_cache_config()
    assert cfg.enabled is False
    assert cfg.ttl_seconds == 48 * 3600

    assert "Turns: 0" in tab.stats_label.text()


def test_cache_tab_saves_semantic_settings(
    qapp: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The semantic toggle, threshold, and embedder choice persist to cache.yaml."""
    _patch_cache_path(monkeypatch, tmp_path)
    storage = Storage(db_path=tmp_path / "semantic.db")
    tab = CacheTab(storage=storage)

    tab.semantic_cb.setChecked(True)
    tab.threshold_spin.setValue(0.81)
    index = tab.embedder_combo.findData("sentence_transformers")
    assert index >= 0
    tab.embedder_combo.setCurrentIndex(index)
    tab.save()

    cfg = load_cache_config()
    assert cfg.semantic.enabled is True
    assert cfg.semantic.threshold == pytest.approx(0.81)
    assert cfg.semantic.embedder == "sentence_transformers"
    assert cfg.semantic.hashing_dim == 512
    assert cfg.semantic.max_candidates == 500


def test_cache_tab_clear_deletes_entries(
    qapp: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Clear button asks for confirmation and clears cached answers."""
    _patch_cache_path(monkeypatch, tmp_path)
    storage = Storage(db_path=tmp_path / "s2.db")
    storage.put_cache_entry(
        key="k", norm_prompt="q", context_hash="c", system_hash="s",
        params_hash="p", scope="auto", answer_text="a", ttl_seconds=60,
    )
    assert storage.response_cache_stats()["entries"] == 1

    tab = CacheTab(storage=storage)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    tab.clear()
    assert storage.response_cache_stats()["entries"] == 0


def test_engine_reload_cache_config(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine picks up new cache.yaml values via reload_cache_config."""
    cfg_file = _patch_cache_path(monkeypatch, tmp_path)
    reg = _registry()
    engine = ChatEngine(registry=reg, router=Router(registry=reg), storage=Storage(db_path=tmp_path / "e.db"))
    assert engine.cache_config.enabled is True

    save_cache_config(CacheConfig(enabled=False, ttl_seconds=10, max_entries=5))
    assert cfg_file.exists()
    engine.reload_cache_config()
    assert engine.cache_config.enabled is False
    assert engine.cache_config.ttl_seconds == 10


def test_usage_dashboard_excludes_hit_rows_and_summarises_saved_cost(
    qapp: Any, tmp_path: Any
) -> None:
    """Hit rows are excluded from real token/cost totals and shown as hits."""
    storage = Storage(db_path=tmp_path / "u.db")
    _log_turn(storage, "miss", {"status": "miss"}, 10, 5, 1000, 0.01)
    _log_turn(
        storage, "hit",
        {"status": "hit", "kind": "exact", "similarity": 1.0,
         "saved_cost_usd": 0.02, "saved_tokens_in": 3, "saved_tokens_out": 2},
        3, 2, 100, 0.0,
    )

    dlg = UsageDialog(storage=storage)
    assert dlg.table.rowCount() == 1  # the hit row is not listed
    assert dlg.table.item(0, 4).text() == "10 / 5"
    assert dlg.hits_label.text().startswith("<strong>Cache Hits:</strong> 1")
    assert "<strong>Saved Cost:</strong> $0.02" in dlg.hits_label.text()


def test_settings_dialog_registers_cache_tab(qapp: Any) -> None:
    """The Settings dialog contains a Cache tab."""
    from modelmesh.ui.settings.settings_dialog import SettingsDialog

    dlg = SettingsDialog(registry=_registry(), storage=None)
    assert any(
        dlg.tabs.tabText(i) == "Cache" for i in range(dlg.tabs.count())
    )
    assert dlg.cache_tab is not None
