"""Dependency-free embedding backends and similarity gates for the response cache.

The semantic cache needs a *pluggable* way to compare prompts.  The default
:class:`HashingEmbedder` uses only the standard library (``zlib`` hashing over
word unigrams, word bigrams, and character 3-grams) so the feature works on a
bare install.  :class:`SentenceTransformerEmbedder` is an optional upgrade; it
lazily imports ``sentence_transformers`` and raises a clear error when the
package is missing so :func:`create_embedder` can fall back to hashing.

Everything here is pure: no database, no Qt, no network.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from abc import ABC, abstractmethod
from array import array
from typing import Any, Iterable, List, Tuple

from modelmesh.core.cache.keys import normalize_prompt

logger = logging.getLogger(__name__)

# Feature weights for the hashing embedder.  Unigrams carry the topic,
# bigrams capture short phrases, and character 3-grams add robustness to
# typos/morphology without dominating the score.  Function words are dropped
# before hashing so paraphrases like "what is X" / "can you explain X" align.
_FEATURE_WEIGHTS = {"w": 1.0, "b": 0.7, "c": 0.05}
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NEGATION_WORDS = frozenset({"not", "no", "never", "don't", "cannot"})
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did what how why when
    where who which can could would should shall will may might i you he she
    it we they me my your our their to of in on at for with and or but
    between explain way best use using please tell give about
    """.split()
)
_MIN_LENGTH_RATIO = 0.5
_MAX_LENGTH_RATIO = 2.0


class EmbedderUnavailableError(RuntimeError):
    """Raised when an optional embedding backend cannot be loaded."""


class Embedder(ABC):
    """A deterministic mapping from text to an L2-normalised float vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier (recorded in logs/diagnostics)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector length produced by :meth:`embed`."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an L2-normalised embedding for *text*."""

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free signed feature-hashing embedder.

    Tokens are normalised and common function words removed, then word
    unigrams, word bigrams, and character 3-grams are hashed into a fixed
    number of buckets with a stable sign.  The result is L2-normalised, so
    cosine similarity is a plain dot product.
    """

    def __init__(self, dim: int = 512) -> None:
        """Create an embedder with *dim* hash buckets.

        Args:
            dim: Vector length.  Must be positive.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def name(self) -> str:
        """Backend identifier including the vector size."""
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        """Vector length produced by this embedder."""
        return self._dim

    def embed(self, text: str) -> List[float]:
        """Embed *text* using signed feature hashing.

        Args:
            text: Raw prompt text.

        Returns:
            An L2-normalised list of ``dim`` floats.  Empty/featureless input
            returns an all-zero vector.
        """
        vector = [0.0] * self._dim
        for feature, weight in _iter_features(text):
            bucket, sign = _hash_feature(feature, self._dim)
            vector[bucket] += sign * weight
        return _l2_normalize(vector)


class SentenceTransformerEmbedder(Embedder):
    """Optional sentence-transformers backend (lazy import, optional extra)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load *model_name* through ``sentence_transformers``.

        Args:
            model_name: Name of a sentence-transformers model.

        Raises:
            EmbedderUnavailableError: If the optional package is not installed
                or the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # ImportError, or a broken optional install
            raise EmbedderUnavailableError(
                "sentence-transformers is not installed; "
                "install the optional 'semantic' extra to use it"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbedderUnavailableError(
                f"Could not load sentence-transformers model {model_name!r}: {exc}"
            ) from exc
        self._name = f"sentence_transformers:{model_name}"
        dimension = self._model.get_sentence_embedding_dimension()
        self._dim = int(dimension) if dimension else 0

    @property
    def name(self) -> str:
        """Backend identifier including the model name."""
        return self._name

    @property
    def dim(self) -> int:
        """Vector length reported by the underlying model."""
        return self._dim

    def embed(self, text: str) -> List[float]:
        """Embed *text* with the underlying sentence-transformers model."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def create_embedder(config: Any) -> Embedder:
    """Build the embedder requested by a semantic cache config.

    Args:
        config: A ``SemanticCacheConfig`` (or any object exposing ``embedder``,
            ``model_name``, and ``hashing_dim``).

    Returns:
        A ready-to-use :class:`Embedder`.  If the requested
        ``sentence_transformers`` backend is unavailable, a warning is logged
        and a :class:`HashingEmbedder` is returned instead.
    """
    backend = str(getattr(config, "embedder", "hashing") or "hashing")
    if backend == "sentence_transformers":
        model_name = str(getattr(config, "model_name", "all-MiniLM-L6-v2"))
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:
            logger.warning(
                "Semantic embedder %r unavailable (%s); falling back to hashing.",
                backend,
                exc,
            )
    dim = int(getattr(config, "hashing_dim", 512) or 512)
    return HashingEmbedder(dim=dim)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Return the cosine similarity of two vectors using the standard library.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        A value in ``[-1.0, 1.0]``; ``0.0`` when either vector is empty,
        zero-length, or the lengths differ.
    """
    left = list(a)
    right = list(b)
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = math.fsum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(math.fsum(x * x for x in left))
    norm_right = math.sqrt(math.fsum(y * y for y in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def pack_embedding(vector: Iterable[float]) -> bytes:
    """Serialise a vector to little-endian-agnostic float32 bytes for SQLite.

    Args:
        vector: Embedding values.

    Returns:
        Raw bytes produced by ``array('f')``.
    """
    return array("f", [float(value) for value in vector]).tobytes()


def unpack_embedding(blob: bytes) -> List[float]:
    """Deserialise float32 bytes produced by :func:`pack_embedding`.

    Args:
        blob: Raw bytes from the ``response_cache.embedding`` column.

    Returns:
        The embedding as a list of floats.

    Raises:
        ValueError: If *blob* is not a whole number of float32 values.
    """
    values = array("f")
    values.frombytes(blob)
    return list(values)


def numeric_tokens(text: str) -> frozenset[str]:
    """Return the set of digit/decimal tokens in *text* (normalised first)."""
    return frozenset(_NUMBER_RE.findall(normalize_prompt(text)))


def negation_tokens(text: str) -> frozenset[str]:
    """Return the set of negation words present in *text* (normalised first)."""
    words = set(_TOKEN_RE.findall(normalize_prompt(text)))
    return frozenset(words & _NEGATION_WORDS)


def length_ratio(a: str, b: str) -> float:
    """Return ``len(a) / len(b)`` for normalised prompts, guarding zero length."""
    left = len(normalize_prompt(a))
    right = len(normalize_prompt(b))
    if left == 0 and right == 0:
        return 1.0
    if right == 0:
        return math.inf
    return left / right


def semantic_gate(
    prompt_a: str,
    prompt_b: str,
    similarity: float,
    threshold: float,
) -> Tuple[bool, str]:
    """Decide whether two prompts are safe to treat as the same question.

    Even a high cosine score is not enough: numbers, negations, and prompt
    length must agree, so ``"2+2"`` never matches ``"2+3"`` and ``"use X"``
    never matches ``"do not use X"``.

    Args:
        prompt_a: Normalised incoming prompt.
        prompt_b: Normalised stored prompt.
        similarity: Cosine similarity between their embeddings.
        threshold: Minimum accepted similarity.

    Returns:
        ``(True, "ok")`` when the pair may be a semantic hit, otherwise
        ``(False, reason)`` with a short machine-readable reason.
    """
    if similarity < threshold:
        return False, "below_threshold"
    if numeric_tokens(prompt_a) != numeric_tokens(prompt_b):
        return False, "numeric_mismatch"
    if negation_tokens(prompt_a) != negation_tokens(prompt_b):
        return False, "negation_mismatch"
    ratio = length_ratio(prompt_a, prompt_b)
    if not (_MIN_LENGTH_RATIO <= ratio <= _MAX_LENGTH_RATIO):
        return False, "length_mismatch"
    return True, "ok"


def _iter_features(text: str) -> Iterable[Tuple[str, float]]:
    """Yield ``(feature_key, weight)`` pairs for the hashing embedder.

    Function words are removed first; if only stopwords remain the feature list
    is empty and the resulting vector is all zeros (never a semantic hit).
    """
    words = [
        word
        for word in _TOKEN_RE.findall(normalize_prompt(text))
        if word not in _STOPWORDS
    ]
    for word in words:
        yield f"w:{word}", _FEATURE_WEIGHTS["w"]
    for first, second in zip(words, words[1:]):
        yield f"b:{first} {second}", _FEATURE_WEIGHTS["b"]
    compact = " ".join(words)
    for index in range(max(0, len(compact) - 2)):
        yield f"c:{compact[index:index + 3]}", _FEATURE_WEIGHTS["c"]


def _hash_feature(feature: str, dim: int) -> Tuple[int, float]:
    """Map *feature* to a stable bucket index and ``+1``/``-1`` sign."""
    digest = zlib.crc32(feature.encode("utf-8")) & 0xFFFFFFFF
    index = digest % dim
    sign = 1.0 if (digest >> 31) & 1 else -1.0
    return index, sign


def _l2_normalize(vector: List[float]) -> List[float]:
    """Return *vector* scaled to unit L2 norm (or unchanged when all-zero)."""
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
