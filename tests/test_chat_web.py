"""Chat transcript (QtWebEngine) safety: untrusted text must render as text, never as HTML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtCore import QEventLoop, QTimer

from modelmesh.ui.chat_view import ChatView

PAYLOAD = '<img src=x onerror="window.__pwned=1">'


def _wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _eval(view: ChatView, script: str) -> Any:
    result: dict = {}
    loop = QEventLoop()

    def done(value: Any) -> None:
        result["v"] = value
        loop.quit()

    view.page().runJavaScript(script, done)
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    return result.get("v")


@pytest.fixture
def view(qapp: Any) -> ChatView:
    v = ChatView()
    for _ in range(100):
        if v._is_page_loaded:
            break
        _wait(50)
    if not v._is_page_loaded:
        pytest.skip("QtWebEngine page did not load in this environment")
    return v


def test_untrusted_strings_render_as_text(view: ChatView) -> None:
    view.add_user_message("u1", PAYLOAD)
    view.start_assistant_message("a1", PAYLOAD, PAYLOAD, "smart_clef", PAYLOAD, {"notices": [PAYLOAD]})
    view.add_tool_card("a1", "call_0", PAYLOAD, {"q": PAYLOAD})
    view.update_tool_result("a1", "call_0", "fetch_url", PAYLOAD, None, True, 5)
    view.add_tool_card("a1", "call_1", "read_text_file", {})
    view.update_tool_result("a1", "call_1", "read_text_file", None, PAYLOAD, False, 5)
    view.show_fallback_notice(PAYLOAD, PAYLOAD, PAYLOAD)
    view.finish_assistant_message("a1", {
        "model_id": PAYLOAD,
        "metadata": {
            "strategy": "smart_clef", "need": 0.5, "task": PAYLOAD,
            "decision_source": "heuristic_fallback", "decision_error": PAYLOAD,
            "candidates": [{"model_id": PAYLOAD, "q": 1, "admitted": True}],
        },
    })
    view.show_error("a1", PAYLOAD, "unknown")
    _wait(300)

    assert _eval(view, "document.querySelectorAll('#chat-container img').length") == 0
    assert _eval(view, "window.__pwned === 1") is False
    text = _eval(view, "document.getElementById('chat-container').innerText")
    assert text.count("onerror") >= 8  # shown literally in every place it was passed


def test_heuristic_fallback_notice_and_attribution_shown(view: ChatView) -> None:
    meta = {"strategy": "smart_clef", "need": 0.4, "decision_source": "heuristic_fallback",
            "decision_error": "Circuit breaker open", "candidates": []}
    view.start_assistant_message("a2", "m", "p", "smart_clef", "r", meta)
    view.finish_assistant_message("a2", {"model_id": "m", "metadata": meta})
    _wait(300)
    text = _eval(view, "document.getElementById('msg-a2').innerText")
    assert "Heuristic fallback" in text
    assert "decision service was unavailable" in text
    assert "Model scores: Artificial Analysis" in _eval(
        view, "document.getElementById('insights-a2').innerText || document.getElementById('insights-a2').textContent"
    )


def test_tool_cards_are_scoped_per_message(view: ChatView) -> None:
    for msg in ("m1", "m2"):
        view.start_assistant_message(msg, "m", "p", "manual", "")
        view.add_tool_card(msg, "call_0", "calculator", {})
    view.update_tool_result("m2", "call_0", "calculator", "42", None, True, 1)
    _wait(300)
    first = _eval(view, "document.getElementById('msg-m1').querySelector('.tool-status').textContent")
    second = _eval(view, "document.getElementById('msg-m2').querySelector('.tool-status').textContent")
    assert first == "Running..."
    assert second.startswith("Success")


def test_streaming_closes_tilde_and_long_fences(view: ChatView) -> None:
    view.start_assistant_message("s1", "m", "p", "manual", "")
    view.append_token("s1", "Here:\n~~~python\nx = 1\n")
    _wait(300)
    assert _eval(view, "document.getElementById('content-s1').querySelectorAll('pre').length") == 1


def test_code_is_highlighted_once_reply_finishes(view: ChatView) -> None:
    view.start_assistant_message("h1", "m", "p", "manual", "")
    view.append_token("h1", "```python\ndef f():\n    return 1\n```\n")
    _wait(300)
    assert _eval(view, "document.querySelectorAll('#content-h1 code[data-highlighted]').length") == 0
    view.finish_assistant_message("h1", {"model_id": "m"})
    _wait(300)
    assert _eval(view, "document.querySelectorAll('#content-h1 code[data-highlighted]').length") == 1


def test_cache_hit_shows_cached_badge_and_zero_cost(view: ChatView) -> None:
    """An exact hit shows the cached badge and a zero cost in the footer."""
    meta = {
        "strategy": "cheapest_first",
        "cache": {"status": "hit", "kind": "exact", "similarity": 1.0},
    }
    view.start_assistant_message("c1", "m", "p", "cheapest_first", "Served from response cache", meta)
    view.append_token("c1", "cached answer")
    view.finish_assistant_message("c1", {
        "model_id": "m", "provider_id": "p", "strategy": "cheapest_first",
        "metadata": meta, "cost_usd": 0.0, "tokens_in": 10, "tokens_out": 4,
    })
    _wait(300)

    footer = _eval(view, "document.getElementById('meta-c1').innerText")
    assert "cached" in footer
    assert "$0" in footer


def test_semantic_cache_hit_shows_similarity_in_badge(view: ChatView) -> None:
    """A semantic hit includes its similarity in the cached badge text."""
    meta = {
        "strategy": "cheapest_first",
        "cache": {"status": "hit", "kind": "semantic", "similarity": 0.94},
    }
    view.start_assistant_message("c2", "m", "p", "cheapest_first", "Served from response cache", meta)
    view.finish_assistant_message("c2", {"model_id": "m", "metadata": meta, "cost_usd": 0.0})
    _wait(300)

    footer = _eval(view, "document.getElementById('meta-c2').innerText")
    assert "cached (semantic 0.94)" in footer


def test_cache_miss_shows_no_cached_badge(view: ChatView) -> None:
    """A miss renders the normal cost and no cached badge."""
    meta = {"strategy": "cheapest_first", "cache": {"status": "miss"}}
    view.start_assistant_message("c3", "m", "p", "cheapest_first", "r", meta)
    view.finish_assistant_message("c3", {"model_id": "m", "metadata": meta, "cost_usd": 0.001})
    _wait(300)

    footer = _eval(view, "document.getElementById('meta-c3').innerText")
    assert "cached" not in footer
    assert "$0.00100" in footer


def test_cache_badge_restored_from_saved_conversation(view: ChatView, tmp_path: Path) -> None:
    """The cached badge and Details line survive a reload from storage."""
    from modelmesh.core.storage import Storage
    from modelmesh.core.types import Message
    from modelmesh.ui.transcript import render_conversation

    storage = Storage(tmp_path / "restore.db")
    conv = "restore-conv"
    storage.create_conversation(conv_id=conv)
    user = Message.from_text("user", "question")
    user.created_at = 1000.0
    assistant = Message.from_text("assistant", "answer")
    assistant.created_at = 1001.0
    assistant.meta = {
        "strategy": "smart_clef",
        "reason": "Served from response cache",
        "routing": {
            "strategy": "smart_clef",
            "need": 0.5,
            "task": "chat_general",
            "cache": {"status": "hit", "kind": "exact", "similarity": 1.0},
        },
    }
    storage.save_message(user, conversation_id=conv)
    storage.save_message(assistant, conversation_id=conv, cost=0.0, status="complete")

    render_conversation(view, storage.get_messages(conv))
    _wait(300)

    footer = _eval(view, f"document.getElementById('meta-{assistant.id}').innerText")
    assert "cached" in footer
    assert "$0" in footer
    details = _eval(view, f"document.getElementById('insights-{assistant.id}').textContent")
    assert "Cache: hit" in details
