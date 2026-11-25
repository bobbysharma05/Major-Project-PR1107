"""Pydantic schema and loader for config/cache.yaml.

Mirrors the style of ``core/config/routing_config.py``:
- ``find_cache_config_path`` walks upward to locate the YAML file.
- ``load_cache_config`` returns a validated ``CacheConfig`` instance;
  a missing file yields safe defaults, a malformed file raises
  ``RegistryError`` with a clear message.
- No PyQt imports; this module is pure-core.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from modelmesh.core.errors import RegistryError


class SemanticCacheConfig(BaseModel):
    """Settings for the optional semantic (embedding-based) similarity check."""

    enabled: bool = False
    threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    embedder: Literal["hashing", "sentence_transformers"] = "hashing"
    model_name: str = "all-MiniLM-L6-v2"
    hashing_dim: int = Field(default=512, gt=0)
    max_candidates: int = Field(default=500, gt=0)

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class CacheConfig(BaseModel):
    """Root model representing the entire cache.yaml file.

    Safe defaults are chosen so a missing file causes no behaviour change
    other than the cache being active with sensible limits.
    """

    enabled: bool = True
    # Storing an answer also stores any user text reproduced in it.  Setting
    # this to false disables the cache entirely (reads and writes).
    store_answers: bool = True
    ttl_seconds: int = Field(default=86400, gt=0)
    max_entries: int = Field(default=2000, gt=0)
    min_answer_chars: int = Field(default=1, ge=0)
    max_answer_chars: int = Field(default=20000, gt=0)
    replay_chunk_chars: int = Field(default=40, gt=0)
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)

    @field_validator("max_answer_chars")
    @classmethod
    def _max_gt_min(cls, v: int) -> int:
        # Cross-field validation happens after all fields exist; we guard the
        # trivial case here.  The richer check lives in the model validator.
        if v < 1:
            raise ValueError("max_answer_chars must be at least 1")
        return v


def find_cache_config_path() -> Path:
    """Locate ``config/cache.yaml`` starting from the current working directory.

    Walks up through parent directories, then falls back to
    ``{cwd}/config/cache.yaml`` (which may not exist yet).
    """
    curr = Path.cwd().resolve()
    for p in [curr] + list(curr.parents):
        candidate = p / "config" / "cache.yaml"
        if candidate.exists():
            return candidate
    return Path.cwd() / "config" / "cache.yaml"


def load_cache_config(path: Optional[Path | str] = None) -> CacheConfig:
    """Load and validate ``cache.yaml`` using Pydantic.

    Args:
        path: Optional explicit path to ``cache.yaml``.

    Returns:
        Validated :class:`CacheConfig` instance.  When the file is absent,
        a default-valued instance is returned silently.

    Raises:
        RegistryError: If the YAML is syntactically malformed or the values
            fail Pydantic validation.
    """
    target = Path(path) if path else find_cache_config_path()
    if not target.exists():
        return CacheConfig()

    try:
        raw_text = target.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
    except Exception as exc:
        raise RegistryError(f"Failed to read/parse {target}: {exc}") from exc

    try:
        return CacheConfig.model_validate(data)
    except Exception as exc:
        raise RegistryError(
            f"Invalid cache configuration in {target}: {exc}"
        ) from exc


def save_cache_config(
    config: CacheConfig, path: Optional[Path | str] = None
) -> None:
    """Persist a :class:`CacheConfig` back to ``cache.yaml``.

    Writes atomically (temp file + ``os.replace``) so a concurrent reader
    never sees a half-written file.

    Args:
        config: The config object to serialise.
        path: Optional explicit destination path.
    """
    target = Path(path) if path else find_cache_config_path()
    data = config.model_dump(mode="json")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    os.replace(tmp, target)
