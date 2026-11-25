"""Tests for the OpenAI-compatible gateway (offline, mock-only)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import modelmesh.core.engine as engine_module  # noqa: E402
from modelmesh.core.cache.config import CacheConfig  # noqa: E402
from modelmesh.core.engine import ChatEngine  # noqa: E402
from modelmesh.core.errors import AuthError  # noqa: E402
from modelmesh.core.providers.mock import MockAdapter  # noqa: E402
from modelmesh.core.registry import ModelRegistry  # noqa: E402
from modelmesh.core.routing.router import Router  # noqa: E402
from modelmesh.core.storage import Storage  # noqa: E402
from modelmesh.core.types import (  # noqa: E402
    ChatRequest,
    EndpointConfig,
    ModelConfig,
    ProviderConfig,
    StreamEvent,
)
from modelmesh.gateway.app import create_app  # noqa: E402


def _registry() -> ModelRegistry:
    """Registry with two offline mock models."""
    registry = ModelRegistry()
    registry.add_provider(ProviderConfig(id="mock", protocol="mock"))
    for model_id, price in (("mock-cheap", 0.01), ("mock-other", 0.05)):
        registry.add_model(
            ModelConfig(
                id=model_id,
                display_name=model_id,
                tier="cheap",
                capabilities=["streaming"],
                endpoints=[
                    EndpointConfig(
                        id=f"{model_id}@mock",
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
def storage(tmp_path: Path) -> Storage:
    """Isolated SQLite database."""
    return Storage(db_path=tmp_path / "gateway.db")


@pytest.fixture
def registry() -> ModelRegistry:
    """Fresh mock-only registry."""
    return _registry()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Count provider calls by wrapping the deterministic mock adapter."""

    class CountingAdapter(MockAdapter):
        def stream_chat(
            self,
            request: ChatRequest,
            candidate: Any,
            cancel_event: Any = None,
        ) -> Iterator[StreamEvent]:
            calls.append(candidate.endpoint_id)
            yield from MockAdapter.stream_chat(self, request, candidate, cancel_event)

    recorded = calls = []
    monkeypatch.setattr(engine_module, "get_adapter", lambda protocol: CountingAdapter())
    return recorded


@pytest.fixture
def engine(registry: ModelRegistry, storage: Storage) -> ChatEngine:
    """Engine with explicit cache config so tests never read cache.yaml."""
    return ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=storage),
        storage=storage,
        cache_config=CacheConfig(),
    )


@pytest.fixture
def plaintext_key(storage: Storage) -> str:
    """Create one private API key and return its plaintext."""
    return storage.create_api_key("gateway-test")[1]


@pytest.fixture
def auth(plaintext_key: str) -> Dict[str, str]:
    """Authorization header for the generated key."""
    return {"Authorization": f"Bearer {plaintext_key}"}


@pytest.fixture
def client(engine: ChatEngine, storage: Storage) -> TestClient:
    """Test client with injected engine/storage and no real config."""
    return TestClient(create_app(engine, storage))


def _chat(
    client: TestClient,
    headers: Dict[str, str],
    messages: Optional[List[Dict[str, Any]]] = None,
    **extra: Any,
) -> Any:
    """POST a chat completion with a simple user message by default."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello gateway"}]
    payload: Dict[str, Any] = {"messages": messages}
    payload.update(extra)
    return client.post("/v1/chat/completions", headers=headers, json=payload)


def test_health_is_public(client: TestClient) -> None:
    """The liveness probe needs no key."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_rejects_missing_bad_and_revoked_keys(
    client: TestClient, storage: Storage, plaintext_key: str, auth: Dict[str, str]
) -> None:
    """Missing, invalid and revoked keys all return OpenAI-shaped 401s."""
    for headers in (
        {},
        {"Authorization": "Bearer mm-wrong-key"},
        {"Authorization": "Basic abc"},
    ):
        response = _chat(client, headers)
        assert response.status_code == 401
        assert response.json()["error"] == {
            "message": "Invalid or missing API key.",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }

    key_id = storage.list_api_keys()[0]["id"]
    assert storage.revoke_api_key(key_id) is True
    revoked = _chat(client, auth)
    assert revoked.status_code == 401
    assert revoked.json()["error"]["code"] == "invalid_api_key"


def test_valid_key_returns_exact_openai_shape(
    client: TestClient, auth: Dict[str, str]
) -> None:
    """A successful completion has the documented envelope and extension."""
    response = _chat(client, auth)
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"id", "object", "created", "model", "choices", "usage", "modelmesh"}
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == "mock-cheap"
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    extension = body["modelmesh"]
    assert set(extension) == {
        "strategy", "endpoint_id", "provider_id", "cost_usd", "latency_sec", "cache"
    }
    assert extension["endpoint_id"] == "mock-cheap@mock"
    assert extension["provider_id"] == "mock"
    assert extension["cache"]["status"] == "miss"


def test_second_identical_request_is_a_free_cache_hit(
    client: TestClient, auth: Dict[str, str], calls: List[str]
) -> None:
    """The gateway uses the response cache like any other engine caller."""
    first = _chat(client, auth, messages=[{"role": "user", "content": "cache me"}])
    second = _chat(client, auth, messages=[{"role": "user", "content": "cache me"}])

    assert first.status_code == second.status_code == 200
    assert len(calls) == 1
    assert second.json()["modelmesh"]["cache"]["status"] == "hit"
    assert second.json()["modelmesh"]["cost_usd"] == 0.0
    assert second.json()["choices"][0]["message"]["content"] == (
        first.json()["choices"][0]["message"]["content"]
    )


def test_gateway_requests_do_not_save_conversations(
    client: TestClient, storage: Storage, auth: Dict[str, str]
) -> None:
    """No chat messages are stored; usage still lands in routing_log."""
    assert _chat(client, auth).status_code == 200
    with sqlite3.connect(storage.db_path) as conn:
        conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        rows = conn.execute(
            "SELECT COUNT(*), SUM(conversation_id IS NULL) FROM routing_log"
        ).fetchone()
    assert conversations == 0
    assert rows[0] == 1
    assert rows[1] == 1


def test_stream_true_is_rejected(client: TestClient, auth: Dict[str, str]) -> None:
    """Streaming is not implemented and returns a 400."""
    response = _chat(client, auth, stream=True)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "stream_unsupported"


def test_empty_and_non_user_last_message_are_rejected(
    client: TestClient, auth: Dict[str, str]
) -> None:
    """Empty message lists and assistant-final turns are invalid."""
    empty = _chat(client, auth, messages=[])
    assert empty.status_code == 400
    last_assistant = _chat(
        client,
        auth,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    assert last_assistant.status_code == 400


def test_unknown_model_is_404(client: TestClient, auth: Dict[str, str]) -> None:
    """A pinned model id must exist in the registry."""
    response = _chat(client, auth, model="not-a-model")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_manual_model_name_is_rejected(client: TestClient, auth: Dict[str, str]) -> None:
    """The manual strategy cannot be selected via the model field."""
    response = _chat(client, auth, model="manual")
    assert response.status_code == 400


def test_pinned_model_and_strategy_name_work(
    client: TestClient, auth: Dict[str, str]
) -> None:
    """A real model id pins; a strategy name selects that strategy."""
    pinned = _chat(client, auth, model="mock-other")
    assert pinned.status_code == 200
    assert pinned.json()["model"] == "mock-other"
    assert pinned.json()["modelmesh"]["endpoint_id"] == "mock-other@mock"

    strategy = _chat(client, auth, model="cheapest_first")
    assert strategy.status_code == 200
    assert strategy.json()["modelmesh"]["strategy"] == "cheapest_first"


def test_system_message_reaches_the_request(
    engine: ChatEngine,
    storage: Storage,
    auth: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System messages are joined into ChatRequest.system_prompt."""
    captured: Dict[str, Any] = {}
    original = engine.run_turn

    def spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(engine, "run_turn", spy)
    client = TestClient(create_app(engine, storage))
    response = _chat(
        client,
        auth,
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "system", "content": "Use Python."},
            {"role": "user", "content": "Write a loop."},
        ],
    )
    assert response.status_code == 200
    assert captured["request"].system_prompt == "Be brief.\n\nUse Python."
    assert [msg.role for msg in captured["request"].messages] == ["user"]
    assert captured["conversation_id"] is None


def test_text_parts_are_accepted_but_images_are_not(
    client: TestClient, auth: Dict[str, str]
) -> None:
    """Only list-of-text content is accepted."""
    parts = _chat(
        client,
        auth,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "parts"},
                ],
            }
        ],
    )
    assert parts.status_code == 200

    image = _chat(
        client,
        auth,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.invalid/a.png"},
                    }
                ],
            }
        ],
    )
    assert image.status_code == 400
    assert image.json()["error"]["message"] == "only text content is supported"


def test_unsupported_role_is_rejected(client: TestClient, auth: Dict[str, str]) -> None:
    """Roles outside system/user/assistant are refused."""
    response = _chat(
        client, auth, messages=[{"role": "tool", "content": "result"}]
    )
    assert response.status_code == 400


def test_models_endpoint_requires_auth_and_lists_auto(
    client: TestClient, auth: Dict[str, str]
) -> None:
    """/v1/models is authenticated and advertises ``auto`` plus model ids."""
    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = [item["id"] for item in body["data"]]
    assert "auto" in ids
    assert "mock-cheap" in ids
    assert all(item["object"] == "model" for item in body["data"])


def test_provider_error_is_502_without_secret_text(
    engine: ChatEngine,
    storage: Storage,
    auth: Dict[str, str],
    plaintext_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider failures never echo raw details or the API key."""

    class FailingAdapter(MockAdapter):
        def stream_chat(
            self,
            request: ChatRequest,
            candidate: Any,
            cancel_event: Any = None,
        ) -> Iterator[StreamEvent]:
            raise AuthError("topsecret-upstream-value-987654", provider_id="mock")

    monkeypatch.setattr(
        engine_module, "get_adapter", lambda protocol: FailingAdapter()
    )
    client = TestClient(create_app(engine, storage))
    response = _chat(client, auth)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"
    assert "topsecret-upstream-value-987654" not in response.text
    assert plaintext_key not in response.text


def test_authorization_header_is_never_logged(
    client: TestClient, auth: Dict[str, str], plaintext_key: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The bearer token must never reach captured logs."""
    with caplog.at_level(logging.INFO):
        assert _chat(client, auth).status_code == 200
    assert plaintext_key not in caplog.text
    assert "Bearer" not in caplog.text
