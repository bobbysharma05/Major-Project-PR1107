"""FastAPI application factory for the ModelMesh gateway.

The factory takes an already-built engine and storage, so tests inject a
mock-only registry and a temporary database without touching real config.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from modelmesh.core.errors import ProviderError, RoutingError, friendly_error
from modelmesh.core.types import ChatRequest, Message, StreamEventType, Usage

from .errors import (
    bad_request,
    internal_error,
    model_not_found,
    no_available_model,
    unauthorized,
    upstream_error,
)
from .schemas import ChatCompletionRequest, MessageSpec, TextPartSpec

logger = logging.getLogger(__name__)

STRATEGY_NAMES = frozenset(
    {"random", "cheapest_first", "expensive_first", "smart_clef"}
)
ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


def _authenticate(request: Request, storage: Any) -> Optional[Dict[str, Any]]:
    """Verify the bearer token from the Authorization header.

    The header value is never logged.

    Args:
        request: Incoming HTTP request.
        storage: Storage instance holding hashed API keys.

    Returns:
        The verified key record, or ``None`` when authentication fails.
    """
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return None
    plaintext = header[7:].strip()
    if not plaintext:
        return None
    return storage.verify_api_key(plaintext)


def _content_text(content: Union[str, List[TextPartSpec]]) -> Optional[str]:
    """Extract plain text from string or part-list content.

    Returns:
        The text, or ``None`` when any part is not a plain text part.
    """
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content:
        if part.type != "text" or part.text is None:
            return None
        parts.append(part.text)
    return "".join(parts)


def _resolve_model(
    model: str,
    default_strategy: str,
    registry: Any,
) -> tuple[str, Optional[str], Optional[Any]]:
    """Map the OpenAI ``model`` field to a strategy and optional pin.

    Returns:
        ``(strategy_name, pinned_model_id, error_response)``.  Exactly one of
        ``pinned_model_id`` and ``error_response`` is non-None.
    """
    if model == "auto":
        return default_strategy, None, None
    if model in STRATEGY_NAMES:
        return model, None, None
    if model == "manual":
        return (
            "manual",
            None,
            bad_request(
                "The 'manual' model is not supported by the gateway; "
                "pin a specific model id instead."
            ),
        )
    if registry.get_model(model) is None:
        return "manual", None, model_not_found(model)
    return "manual", model, None


def _cache_block(done_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the cache status block from DONE event metadata."""
    metadata = done_data.get("metadata") or {}
    cache = metadata.get("cache") or {}
    block: Dict[str, Any] = {"status": str(cache.get("status") or "miss")}
    if cache.get("kind") is not None:
        block["kind"] = cache["kind"]
    if cache.get("similarity") is not None:
        block["similarity"] = cache["similarity"]
    return block


def _success_body(
    done_data: Dict[str, Any],
    text: str,
    usage: Usage,
    strategy: str,
    fallback_model: str,
) -> Dict[str, Any]:
    """Build the OpenAI response body with the ModelMesh extension."""
    candidate = done_data.get("candidate")
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    model_id = getattr(candidate, "model_id", None) or fallback_model
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "modelmesh": {
            "strategy": str(done_data.get("strategy") or strategy),
            "endpoint_id": getattr(candidate, "endpoint_id", None),
            "provider_id": getattr(candidate, "provider_id", None),
            "cost_usd": float(done_data.get("cost_usd") or 0.0),
            "latency_sec": float(done_data.get("latency_sec") or 0.0),
            "cache": _cache_block(done_data),
        },
    }


def create_app(
    engine: Any,
    storage: Any,
    default_strategy: str = "cheapest_first",
) -> FastAPI:
    """Create the gateway application with injected engine and storage.

    Args:
        engine: A configured :class:`~modelmesh.core.engine.ChatEngine`.
        storage: Storage instance used for hashed API keys and routing logs.
        default_strategy: Strategy used when the request model is ``auto``.

    Returns:
        A ready-to-use FastAPI application.
    """
    app = FastAPI(title="ModelMesh Gateway", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    def validation_handler(_request: Request, _exc: RequestValidationError) -> Any:
        """Convert FastAPI's 422 validation errors to OpenAI-shaped 400s."""
        return bad_request("Invalid request body.")

    @app.get("/health")
    def health() -> Dict[str, str]:
        """Unauthenticated liveness probe."""
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models(request: Request) -> Any:
        """List ``auto`` plus every enabled logical model id."""
        if _authenticate(request, storage) is None:
            return unauthorized()
        model_ids = {"auto"}
        for candidate in engine.registry.candidates(enabled_only=True):
            model_ids.add(candidate.model_id)
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "owned_by": "modelmesh"}
                for model_id in sorted(model_ids)
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: Request, body: ChatCompletionRequest
    ) -> Any:
        """Run one non-streaming chat completion through the engine."""
        if _authenticate(request, storage) is None:
            return unauthorized()
        if not body.messages:
            return bad_request("messages must not be empty.")
        if body.stream:
            return bad_request("streaming is not supported yet.", "stream_unsupported")
        if body.messages[-1].role != "user":
            return bad_request("the last message must have role 'user'.")

        system_parts: List[str] = []
        messages: List[Message] = []
        for spec in body.messages:
            if spec.role not in ALLOWED_ROLES:
                return bad_request(f"unsupported message role: {spec.role}")
            text = _content_text(spec.content)
            if text is None:
                return bad_request("only text content is supported")
            if spec.role == "system":
                system_parts.append(text)
            else:
                messages.append(Message.from_text(spec.role, text))

        strategy, pinned_model, error = _resolve_model(
            body.model, default_strategy, engine.registry
        )
        if error is not None:
            return error

        chat_request = ChatRequest(
            messages=messages,
            system_prompt="\n\n".join(system_parts) or None,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )

        text_parts: List[str] = []
        done_data: Dict[str, Any] = {}
        usage: Optional[Usage] = None
        try:
            events = engine.run_turn(
                request=chat_request,
                candidate=None,
                strategy_name=strategy,
                pinned_model_id=pinned_model,
                conversation_id=None,
            )
            for event in events:
                if event.type == StreamEventType.TEXT_DELTA:
                    text_parts.append(event.text or "")
                elif event.type == StreamEventType.USAGE:
                    usage = event.usage
                elif event.type == StreamEventType.DONE:
                    done_data = event.data or {}
                    usage = event.usage or usage
                elif event.type == StreamEventType.ERROR:
                    return upstream_error()
        except RoutingError:
            return no_available_model()
        except ProviderError as exc:
            # friendly_error returns a user-safe message with no raw details.
            logger.warning("Gateway provider failure: %s", friendly_error(exc)[0])
            return upstream_error()
        except Exception:
            logger.exception("Gateway chat completion failed")
            return internal_error()

        if not done_data:
            return upstream_error()

        final_usage = usage or Usage()
        return _success_body(
            done_data,
            "".join(text_parts),
            final_usage,
            strategy,
            body.model,
        )

    return app
