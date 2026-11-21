"""Tests for the pure response-cache key and eligibility helpers."""

from __future__ import annotations

import json

import pytest

from modelmesh.core.cache.keys import (
    build_cache_key,
    is_cacheable_request,
    normalize_prompt,
    split_request,
)
from modelmesh.core.types import (
    ChatRequest,
    ImagePart,
    Message,
    TextPart,
    ToolCall,
    ToolSpec,
)


def _request(text: str = "question") -> ChatRequest:
    """Build a minimal cacheable request."""
    return ChatRequest(messages=[Message.from_text("user", text)])


def test_normalize_prompt_collapses_whitespace_case_and_unicode() -> None:
    """NFKC, whitespace folding and casefolding make equivalent prompts equal."""
    assert normalize_prompt("  Hello\u00a0WORLD\n again\t ") == "hello world again"
    assert normalize_prompt("ＨＥＬＬＯ") == normalize_prompt("hello")
    assert normalize_prompt("Straße") == normalize_prompt("STRASSE")


def test_normalized_variants_share_one_cache_key() -> None:
    """Superficial prompt differences do not create extra cache entries."""
    first = build_cache_key(_request("  Hello\u00a0WORLD  "), "system", None)
    second = build_cache_key(_request("hello world"), "system", None)
    assert first.key == second.key
    assert first.norm_prompt == "hello world"


def test_system_prompt_changes_key() -> None:
    """The effective system prompt is part of the cache key."""
    first = build_cache_key(_request("q"), "system-a", None)
    second = build_cache_key(_request("q"), "system-b", None)
    assert first.key != second.key


def test_temperature_changes_key() -> None:
    """Different sampling temperatures must not share an answer."""
    request = _request("q")
    request.temperature = 0.7
    assert build_cache_key(request, "system", None).key != build_cache_key(
        _request("q"), "system", None
    ).key


def test_max_tokens_changes_key() -> None:
    """Different output limits are different requests."""
    request = _request("q")
    request.max_tokens = 128
    assert build_cache_key(request, "system", None).key != build_cache_key(
        _request("q"), "system", None
    ).key


def test_extra_params_change_key() -> None:
    """Generation extras are included in the parameter hash."""
    request = _request("q")
    request.extra_params = {"top_p": 0.9}
    assert build_cache_key(request, "system", None).key != build_cache_key(
        _request("q"), "system", None
    ).key


def test_pinned_scope_changes_key() -> None:
    """A pinned model/endpoint must not reuse an auto-routed answer."""
    auto = build_cache_key(_request("q"), "system", None)
    pinned = build_cache_key(_request("q"), "system", "model-a:endpoint-1")
    assert pinned.scope == "model-a:endpoint-1"
    assert auto.scope == "auto"
    assert pinned.key != auto.key


def test_earlier_context_changes_key() -> None:
    """A different earlier conversation produces a different context hash."""
    first = ChatRequest(
        messages=[
            Message.from_text("user", "old question"),
            Message.from_text("assistant", "old answer"),
            Message.from_text("user", "q"),
        ]
    )
    second = ChatRequest(
        messages=[
            Message.from_text("user", "different question"),
            Message.from_text("assistant", "different answer"),
            Message.from_text("user", "q"),
        ]
    )
    assert (
        build_cache_key(first, "system", None).context_hash
        != build_cache_key(second, "system", None).context_hash
    )


def test_two_turn_conversation_differs_from_same_last_message_first() -> None:
    """The same last message asked first is a different cache entry."""
    two_turn = ChatRequest(
        messages=[
            Message.from_text("user", "first question"),
            Message.from_text("assistant", "first answer"),
            Message.from_text("user", "repeat me"),
        ]
    )
    first_turn = _request("repeat me")
    assert build_cache_key(two_turn, "system", None).key != build_cache_key(
        first_turn, "system", None
    ).key


def test_strategy_does_not_affect_key() -> None:
    """The routing strategy is not part of the cache key."""
    first = _request("q")
    second = _request("q")
    first.strategy = "cheapest_first"  # type: ignore[attr-defined]
    second.strategy = "expensive_first"  # type: ignore[attr-defined]
    assert build_cache_key(first, "system", None).key == build_cache_key(
        second, "system", None
    ).key


def test_build_cache_key_is_deterministic() -> None:
    """Repeated calls with equivalent requests return identical key parts."""
    first = build_cache_key(_request("q"), "system", "endpoint-1")
    second = build_cache_key(_request("q"), "system", "endpoint-1")
    assert first == second
    assert first.key == second.key


def test_split_request_includes_roles_and_text_in_order() -> None:
    """split_request returns a canonical context of earlier messages only."""
    request = ChatRequest(
        messages=[
            Message.from_text("user", " First  question "),
            Message.from_text("assistant", " Answer "),
            Message.from_text("user", "  SECOND\tQuestion  "),
        ]
    )
    prompt, context = split_request(request)
    assert prompt == "second question"
    assert json.loads(context) == [
        {"role": "user", "text": " First  question "},
        {"role": "assistant", "text": " Answer "},
    ]


def test_split_request_without_messages_is_empty() -> None:
    """An empty request yields an empty prompt and empty context list."""
    assert split_request(ChatRequest(messages=[])) == ("", "[]")


@pytest.mark.parametrize(
    "candidate_request, reason",
    [
        (
            ChatRequest(
                messages=[Message.from_text("user", "q")],
                tools=[ToolSpec(name="calculator", description="Math")],
            ),
            "tools",
        ),
        (
            ChatRequest(
                messages=[
                    Message(role="user", parts=[ImagePart(media_type="image/png", data="abc")])
                ]
            ),
            "attachments",
        ),
        (
            ChatRequest(
                messages=[
                    Message(role="user", parts=[TextPart(text="file body", name="notes.txt")])
                ]
            ),
            "attachments",
        ),
        (ChatRequest(messages=[]), "no_user_message"),
        (ChatRequest(messages=[Message.from_text("assistant", "not a question")]), "no_user_message"),
    ],
)
def test_non_cacheable_requests_report_reason(
    candidate_request: ChatRequest, reason: str
) -> None:
    """Tools, attachments and missing user turns are refused with clear reasons."""
    ok, actual = is_cacheable_request(candidate_request)
    assert ok is False
    assert actual == reason


def test_tool_call_history_is_not_cacheable() -> None:
    """Earlier tool calls make the whole conversation non-cacheable."""
    request = ChatRequest(
        messages=[
            Message(
                role="assistant",
                parts=[TextPart(text="calling a tool")],
                tool_calls=[ToolCall(id="1", name="calculator", arguments={})],
            ),
            Message.from_text("user", "q"),
        ]
    )
    assert is_cacheable_request(request) == (False, "tool_history")


def test_tool_result_history_is_not_cacheable() -> None:
    """A prior tool result is also a tool history and is refused."""
    request = ChatRequest(
        messages=[
            Message(role="tool", parts=[TextPart(text="42")], tool_call_id="1"),
            Message.from_text("user", "q"),
        ]
    )
    assert is_cacheable_request(request) == (False, "tool_history")


def test_image_in_history_is_not_cacheable() -> None:
    """Attachments anywhere in the conversation block caching."""
    request = ChatRequest(
        messages=[
            Message.from_text("user", "look at this"),
            Message(
                role="assistant",
                parts=[ImagePart(media_type="image/png", data="abc")],
            ),
            Message.from_text("user", "q"),
        ]
    )
    assert is_cacheable_request(request) == (False, "attachments")


def test_named_text_in_history_is_not_cacheable() -> None:
    """A named TextPart in an earlier message is an attachment."""
    request = ChatRequest(
        messages=[
            Message(role="user", parts=[TextPart(text="file body", name="a.txt")]),
            Message.from_text("user", "q"),
        ]
    )
    assert is_cacheable_request(request) == (False, "attachments")


def test_simple_request_is_cacheable() -> None:
    """A plain user turn is accepted."""
    assert is_cacheable_request(_request("q")) == (True, "ok")
