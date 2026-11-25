"""Pydantic request schemas for the OpenAI-compatible gateway."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict


class TextPartSpec(BaseModel):
    """One content part; only plain text parts are accepted by the gateway."""

    model_config = ConfigDict(extra="ignore")

    type: str
    text: Optional[str] = None


class MessageSpec(BaseModel):
    """An OpenAI-style chat message (role plus string or parts content)."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: Union[str, List[TextPartSpec]]


class ChatCompletionRequest(BaseModel):
    """Body of ``POST /v1/chat/completions``; unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    model: str = "auto"
    messages: List[MessageSpec]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
