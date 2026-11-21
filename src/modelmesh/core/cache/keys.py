"""Pure helpers for response-cache keys and request eligibility.

This module performs no I/O: it only inspects canonical request objects and
hashes strings.  Keeping the logic here lets the engine use it without pulling
in PyQt, configuration files, or the database.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from modelmesh.core.types import ChatRequest, ImagePart, TextPart


@dataclass(frozen=True)
class CacheKeyParts:
    """The immutable pieces used to build (and store) a response-cache key."""

    key: str
    norm_prompt: str
    context_hash: str
    system_hash: str
    params_hash: str
    scope: str


def _canonical_json(value: Any) -> str:
    """Serialise *value* deterministically for hashing.

    Args:
        value: Plain JSON-compatible data.

    Returns:
        A compact JSON string with sorted keys and stable Unicode output.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt(text: str) -> str:
    """Normalise prompt text so superficial differences share a cache key.

    Applies Unicode NFKC, strips leading/trailing whitespace, collapses every
    internal whitespace run to one space, and casefolds the result.  No stemming,
    punctuation removal, or other aggressive rewriting is performed.

    Args:
        text: Raw prompt text.

    Returns:
        The normalised prompt text.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split()).strip()


def is_cacheable_request(request: ChatRequest) -> tuple[bool, str]:
    """Decide whether a request may use the response cache.

    Args:
        request: The canonical chat request.

    Returns:
        ``(True, "ok")`` when the request is cacheable, otherwise
        ``(False, reason)`` where reason is a short machine-readable label.
    """
    if request.tools:
        return False, "tools"

    for message in request.messages:
        for part in message.parts:
            if isinstance(part, ImagePart):
                return False, "attachments"
            if isinstance(part, TextPart) and part.name is not None:
                return False, "attachments"

    if not request.messages:
        return False, "no_user_message"
    if request.messages[-1].role != "user":
        return False, "no_user_message"

    for message in request.messages:
        if message.tool_calls or message.role == "tool" or message.tool_call_id is not None:
            return False, "tool_history"

    return True, "ok"


def split_request(request: ChatRequest) -> tuple[str, str]:
    """Split a request into its prompt text and canonical context string.

    Args:
        request: The canonical chat request.

    Returns:
        A tuple of the normalised last user message text and a canonical JSON
        string containing the role and text of every earlier message, in order.
        When there is no user message as the last message, the prompt is empty.
    """
    earlier = request.messages[:-1] if request.messages else []
    context = [
        {"role": message.role, "text": message.text_content()}
        for message in earlier
    ]

    last = request.messages[-1] if request.messages else None
    prompt = ""
    if last is not None and last.role == "user":
        prompt = normalize_prompt(last.text_content())

    return prompt, _canonical_json(context)


def build_cache_key(
    request: ChatRequest,
    effective_system_prompt: Optional[str],
    scope_model: Optional[str],
) -> CacheKeyParts:
    """Build the cache key for an eligible request.

    The key hashes the normalised last user text, the earlier conversation
    context, the effective system prompt, generation parameters, and the pinned
    model/endpoint id (or ``"auto"``).  The routing strategy is deliberately not
    part of the key.

    Args:
        request: The canonical chat request.
        effective_system_prompt: System prompt actually used for the turn.
        scope_model: Pinned model/endpoint id, or ``None`` for automatic routing.

    Returns:
        The key plus each component hash, ready for storage.
    """
    norm_prompt, context_json = split_request(request)
    context_hash = _sha256(context_json)
    system_hash = _sha256(effective_system_prompt or "")
    params_hash = _sha256(
        _canonical_json(
            {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "extra_params": request.extra_params,
            }
        )
    )
    scope = scope_model if scope_model else "auto"

    key = _sha256(
        _canonical_json(
            {
                "prompt": norm_prompt,
                "context_hash": context_hash,
                "system_hash": system_hash,
                "params_hash": params_hash,
                "scope": scope,
            }
        )
    )

    return CacheKeyParts(
        key=key,
        norm_prompt=norm_prompt,
        context_hash=context_hash,
        system_hash=system_hash,
        params_hash=params_hash,
        scope=scope,
    )
