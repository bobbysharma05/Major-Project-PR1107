"""Reproducible evaluation harness for the ModelMesh response cache.

The harness runs four phases against a clean, temporary SQLite database:

* **A — cold pass**: every golden prompt is sent with ``bypass_cache=True`` so
  the turn is guaranteed to miss while still storing an answer *and* an
  embedding (semantic matching is enabled during this phase).
* **B — exact repeat**: the same golden prompts are sent again; the exact cache
  should serve all of them.
* **C — paraphrase pass**: one natural paraphrase per golden prompt is sent with
  semantic matching disabled, then enabled at each threshold.
* **D — near-miss pass**: prompts that look similar but mean something different
  are sent under the same disabled/enabled sweep.  Every hit here is a false
  hit.

Phases C and D use a read-only cache wrapper, so a miss cannot add a new entry
and contaminate later thresholds.  The golden entries created in phase A are
the only semantic candidates for the whole sweep.

Usage::

    python scripts/cache_eval.py --provider mock
    python scripts/cache_eval.py --provider live --yes-i-know-this-spends-quota

Live mode refuses to run without the confirmation flag and only uses the
configured free Groq endpoint.  It never prints the API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modelmesh.core.cache.config import CacheConfig, SemanticCacheConfig
from modelmesh.core.engine import ChatEngine
from modelmesh.core.keys import has_provider_auth
from modelmesh.core.registry import ModelRegistry, load_default_registry
from modelmesh.core.routing.router import Router
from modelmesh.core.storage import Storage
from modelmesh.core.types import Candidate, ChatRequest, Message, StreamEventType

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.80, 0.90, 0.92, 0.95)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_prompts.json"
PARAPHRASE_PATH = REPO_ROOT / "tests" / "fixtures" / "paraphrase_prompts.json"
NEAR_MISS_PATH = REPO_ROOT / "tests" / "fixtures" / "near_miss_prompts.json"


class ReadOnlyCache:
    """Delegate every Storage call except writes.

    Phases C and D must compare against the phase-A golden entries only.  A
    miss is therefore allowed to read/touch the cache but not to add a row.
    """

    def __init__(self, inner: Storage) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def put_cache_entry(self, *args: Any, **kwargs: Any) -> None:
        """Ignore cache writes during read-only evaluation phases."""
        return None


def _load_items(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON fixture list and check the minimum required fields."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    for item in data:
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
            raise ValueError(f"{path} contains an entry without id/prompt: {item!r}")
    return data


def load_and_validate_fixtures() -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Load golden/paraphrase/near-miss fixtures and validate their linkage.

    Returns:
        ``(golden, paraphrases, near_misses, golden_by_id)``.

    Raises:
        ValueError: If counts, ids, source links, or ``same_meaning`` flags are
            inconsistent.
    """
    golden = _load_items(GOLDEN_PATH)
    paraphrases = _load_items(PARAPHRASE_PATH)
    near_misses = _load_items(NEAR_MISS_PATH)

    if len(golden) != 30:
        raise ValueError(f"expected 30 golden prompts, found {len(golden)}")
    if len(paraphrases) != len(golden):
        raise ValueError(
            f"expected {len(golden)} paraphrases, found {len(paraphrases)}"
        )
    if len(near_misses) != 15:
        raise ValueError(f"expected 15 near-miss prompts, found {len(near_misses)}")

    golden_by_id = {item["id"]: item for item in golden}
    if len(golden_by_id) != len(golden):
        raise ValueError("golden prompt ids are not unique")

    paraphrase_sources = set()
    for item in paraphrases:
        source = item.get("source_id")
        if source not in golden_by_id:
            raise ValueError(f"{item['id']} references unknown source {source!r}")
        if item.get("same_meaning") is not True:
            raise ValueError(f"{item['id']} must have same_meaning=true")
        paraphrase_sources.add(source)
    missing = set(golden_by_id) - paraphrase_sources
    if missing:
        raise ValueError(f"golden prompts without a paraphrase: {sorted(missing)}")

    for item in near_misses:
        source = item.get("source_id")
        if source not in golden_by_id:
            raise ValueError(f"{item['id']} references unknown source {source!r}")
        if item.get("same_meaning") is not False:
            raise ValueError(f"{item['id']} must have same_meaning=false")

    return golden, paraphrases, near_misses, golden_by_id


def _select_candidate(registry: ModelRegistry, provider: str) -> Candidate:
    """Pick the mock candidate, or the configured free Groq endpoint for live."""
    candidates = registry.candidates()
    if provider == "mock":
        pool = [candidate for candidate in candidates if candidate.provider_id == "mock"]
        if not pool:
            raise RuntimeError("no enabled mock candidate found in providers.yaml")
    else:
        groq = registry.get_provider("groq")
        if groq is None:
            raise RuntimeError("no 'groq' provider found in providers.yaml")
        catalog = groq.catalog
        if catalog is None or not (catalog.assume_free or catalog.free_only):
            raise RuntimeError(
                "the configured Groq provider is not marked free "
                "(catalog.assume_free or catalog.free_only); refusing to run live"
            )
        pool = [candidate for candidate in candidates if candidate.provider_id == "groq"]
        if not pool:
            raise RuntimeError("no enabled Groq endpoint found in providers.yaml")

    # Cheapest first keeps live evaluation on the free endpoint even if the
    # config grows more Groq models later.
    return min(
        pool,
        key=lambda candidate: (
            candidate.price_in_per_mtok + candidate.price_out_per_mtok,
            candidate.endpoint_id,
        ),
    )


def build_registry(provider: str) -> Tuple[ModelRegistry, Candidate]:
    """Load the registry and select the only endpoint the evaluation may use.

    Args:
        provider: ``"mock"`` or ``"live"``.

    Returns:
        The registry and the pinned candidate for every turn.

    Raises:
        RuntimeError: If the requested provider is unavailable or not free.
    """
    registry = load_default_registry()
    if provider == "live":
        # Read the key only from the process environment.  This script never
        # opens or parses .env, and it never logs the key.
        groq = registry.get_provider("groq")
        if groq is None or not has_provider_auth(groq.auth_env):
            raise RuntimeError(
                "GROQ_API_KEY is not set in the environment; export it before "
                "running live mode"
            )
    return registry, _select_candidate(registry, provider)


def make_cache_config(*, semantic_enabled: bool, threshold: float) -> CacheConfig:
    """Build an explicit cache config so the repo's cache.yaml cannot interfere."""
    return CacheConfig(
        enabled=True,
        ttl_seconds=86_400 * 30,
        max_entries=10_000,
        min_answer_chars=1,
        max_answer_chars=100_000,
        replay_chunk_chars=40,
        semantic=SemanticCacheConfig(
            enabled=semantic_enabled,
            threshold=threshold,
            embedder="hashing",
            hashing_dim=512,
            max_candidates=500,
        ),
    )


def make_engine(
    registry: ModelRegistry,
    storage: Any,
    *,
    semantic_enabled: bool,
    threshold: float,
) -> ChatEngine:
    """Create an engine with a pinned cache configuration."""
    return ChatEngine(
        registry=registry,
        router=Router(registry=registry, storage=storage),
        storage=storage,
        cache_config=make_cache_config(
            semantic_enabled=semantic_enabled, threshold=threshold
        ),
    )


def run_turn(
    engine: ChatEngine,
    candidate: Candidate,
    prompt: str,
    *,
    bypass_cache: bool,
) -> Dict[str, Any]:
    """Execute one turn and return a plain-data record of what happened.

    Raises:
        RuntimeError: If the engine reports an error or never emits DONE.
    """
    text_parts: List[str] = []
    done = None
    usage = None
    for event in engine.run_turn(
        request=ChatRequest(messages=[Message.from_text("user", prompt)]),
        candidate=candidate,
        strategy_name="manual",
        bypass_cache=bypass_cache,
    ):
        if event.type == StreamEventType.TEXT_DELTA:
            text_parts.append(event.text or "")
        elif event.type == StreamEventType.USAGE:
            usage = event.usage
        elif event.type == StreamEventType.DONE:
            done = event
        elif event.type == StreamEventType.ERROR:
            raise RuntimeError(f"provider error: {event.data}")

    if done is None:
        raise RuntimeError("turn ended without a DONE event")

    answer = "".join(text_parts)
    final_usage = done.usage or usage
    data = done.data or {}
    metadata = data.get("metadata") or {}
    cache_block = metadata.get("cache") or {}
    status = str(cache_block.get("status") or "unknown")
    kind = cache_block.get("kind")
    similarity = cache_block.get("similarity")
    return {
        "status": status,
        "hit": status == "hit",
        "kind": kind,
        "similarity": float(similarity) if similarity is not None else None,
        "latency_ms": round(float(data.get("latency_sec") or 0.0) * 1000.0, 3),
        "cost_usd": round(float(data.get("cost_usd") or 0.0), 6),
        "tokens_in": int(getattr(final_usage, "input_tokens", 0) or 0),
        "tokens_out": int(getattr(final_usage, "output_tokens", 0) or 0),
        "saved_cost_usd": round(float(cache_block.get("saved_cost_usd") or 0.0), 6),
        "saved_tokens_in": int(cache_block.get("saved_tokens_in") or 0),
        "saved_tokens_out": int(cache_block.get("saved_tokens_out") or 0),
        "answer": answer,
    }


def run_phase(
    engine: ChatEngine,
    candidate: Candidate,
    items: Sequence[Dict[str, Any]],
    *,
    phase: str,
    run: str,
    bypass_cache: bool,
    provider: str,
    golden_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run one phase/run over *items* and return per-turn records."""
    records: List[Dict[str, Any]] = []
    for item in items:
        source_id = str(item.get("source_id") or item["id"])
        source_prompt = str(golden_by_id[source_id]["prompt"])
        record = run_turn(
            engine,
            candidate,
            str(item["prompt"]),
            bypass_cache=bypass_cache,
        )
        answer = record.pop("answer")
        prompt = str(item["prompt"])
        record.update(
            {
                "phase": phase,
                "run": run,
                "id": str(item["id"]),
                "prompt": prompt,
                "source_id": source_id,
                "same_meaning": bool(item.get("same_meaning", True)),
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "answer_preview": answer[:160],
            }
        )
        if provider == "mock":
            # The mock adapter embeds the last user prompt in its answer, so a
            # replay of a different prompt is directly observable.
            record["answer_contains_query"] = prompt in answer
            record["answer_contains_source"] = source_prompt in answer
        records.append(record)
    return records


def summarize(
    records: Sequence[Dict[str, Any]],
    *,
    phase: str,
    run: str,
    semantic_enabled: bool,
    threshold: Optional[float],
) -> Dict[str, Any]:
    """Aggregate one run into the metrics shown in the report."""
    turns = len(records)
    hits = [record for record in records if record["hit"]]
    exact_hits = [record for record in hits if record["kind"] == "exact"]
    semantic_hits = [record for record in hits if record["kind"] == "semantic"]
    false_hits = [record for record in hits if record["same_meaning"] is False]
    similarities = [
        record["similarity"]
        for record in semantic_hits
        if record["similarity"] is not None
    ]
    return {
        "phase": phase,
        "run": run,
        "semantic_enabled": semantic_enabled,
        "threshold": threshold,
        "turns": turns,
        "hits": len(hits),
        "hit_rate": round(len(hits) / turns, 6) if turns else 0.0,
        "exact_hits": len(exact_hits),
        "semantic_hits": len(semantic_hits),
        "semantic_hit_rate": round(len(semantic_hits) / turns, 6) if turns else 0.0,
        "false_hits": len(false_hits),
        "false_hit_rate": round(len(false_hits) / turns, 6) if turns else 0.0,
        "avg_latency_ms": round(
            sum(record["latency_ms"] for record in records) / turns, 3
        )
        if turns
        else 0.0,
        "total_cost_usd": round(sum(record["cost_usd"] for record in records), 6),
        "saved_cost_usd": round(sum(record["saved_cost_usd"] for record in records), 6),
        "saved_tokens_in": sum(record["saved_tokens_in"] for record in records),
        "saved_tokens_out": sum(record["saved_tokens_out"] for record in records),
        "saved_tokens_total": sum(
            record["saved_tokens_in"] + record["saved_tokens_out"]
            for record in records
        ),
        "avg_semantic_similarity": round(sum(similarities) / len(similarities), 6)
        if similarities
        else None,
    }


def run_evaluation(
    *,
    provider: str,
    thresholds: Sequence[float],
    golden: Sequence[Dict[str, Any]],
    paraphrases: Sequence[Dict[str, Any]],
    near_misses: Sequence[Dict[str, Any]],
    golden_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all four phases and return the JSON-ready result payload."""
    registry, candidate = build_registry(provider)
    with tempfile.TemporaryDirectory(prefix="modelmesh-cache-eval-") as temp_dir:
        storage = Storage(db_path=Path(temp_dir) / "eval.db")

        # Phase A: guaranteed cold pass.  bypass_cache=True means the read path
        # is skipped, but the store path still writes an answer + embedding.
        cold_engine = make_engine(
            registry, storage, semantic_enabled=True, threshold=thresholds[0]
        )
        records_a = run_phase(
            cold_engine,
            candidate,
            golden,
            phase="A_cold",
            run="cold",
            bypass_cache=True,
            provider=provider,
            golden_by_id=golden_by_id,
        )

        # Phase B: exact repeats against the entries written in phase A.
        exact_engine = make_engine(
            registry, storage, semantic_enabled=False, threshold=thresholds[0]
        )
        records_b = run_phase(
            exact_engine,
            candidate,
            golden,
            phase="B_exact_repeat",
            run="exact_repeat",
            bypass_cache=False,
            provider=provider,
            golden_by_id=golden_by_id,
        )

        entries = storage.response_cache_stats()["entries"]
        if entries != len(golden):
            raise RuntimeError(
                f"expected {len(golden)} warmed cache entries, found {entries}"
            )

        # Phases C/D read the golden entries and never write new ones.
        read_only = ReadOnlyCache(storage)
        c_runs: Dict[str, Dict[str, Any]] = {}
        d_runs: Dict[str, Dict[str, Any]] = {}
        records_c: List[Dict[str, Any]] = []
        records_d: List[Dict[str, Any]] = []

        baseline_c_engine = make_engine(
            registry, read_only, semantic_enabled=False, threshold=thresholds[0]
        )
        baseline_c = run_phase(
            baseline_c_engine,
            candidate,
            paraphrases,
            phase="C_paraphrase",
            run="semantic_disabled",
            bypass_cache=False,
            provider=provider,
            golden_by_id=golden_by_id,
        )
        records_c.extend(baseline_c)
        c_runs["disabled"] = summarize(
            baseline_c,
            phase="C_paraphrase",
            run="semantic_disabled",
            semantic_enabled=False,
            threshold=None,
        )

        baseline_d_engine = make_engine(
            registry, read_only, semantic_enabled=False, threshold=thresholds[0]
        )
        baseline_d = run_phase(
            baseline_d_engine,
            candidate,
            near_misses,
            phase="D_near_miss",
            run="semantic_disabled",
            bypass_cache=False,
            provider=provider,
            golden_by_id=golden_by_id,
        )
        records_d.extend(baseline_d)
        d_runs["disabled"] = summarize(
            baseline_d,
            phase="D_near_miss",
            run="semantic_disabled",
            semantic_enabled=False,
            threshold=None,
        )

        for threshold in thresholds:
            key = f"{threshold:.2f}"
            c_engine = make_engine(
                registry, read_only, semantic_enabled=True, threshold=threshold
            )
            c_records = run_phase(
                c_engine,
                candidate,
                paraphrases,
                phase="C_paraphrase",
                run=f"semantic_{key}",
                bypass_cache=False,
                provider=provider,
                golden_by_id=golden_by_id,
            )
            records_c.extend(c_records)
            c_runs[key] = summarize(
                c_records,
                phase="C_paraphrase",
                run=f"semantic_{key}",
                semantic_enabled=True,
                threshold=threshold,
            )

            d_engine = make_engine(
                registry, read_only, semantic_enabled=True, threshold=threshold
            )
            d_records = run_phase(
                d_engine,
                candidate,
                near_misses,
                phase="D_near_miss",
                run=f"semantic_{key}",
                bypass_cache=False,
                provider=provider,
                golden_by_id=golden_by_id,
            )
            records_d.extend(d_records)
            d_runs[key] = summarize(
                d_records,
                phase="D_near_miss",
                run=f"semantic_{key}",
                semantic_enabled=True,
                threshold=threshold,
            )

        sweep = [
            {
                "threshold": threshold,
                "paraphrase_semantic_hit_rate": c_runs[f"{threshold:.2f}"][
                    "semantic_hit_rate"
                ],
                "near_miss_false_hit_rate": d_runs[f"{threshold:.2f}"][
                    "false_hit_rate"
                ],
            }
            for threshold in thresholds
        ]

        mock_failures = 0
        if provider == "mock":
            for record in records_a + records_b + records_c + records_d:
                if record["hit"] and not record.get("answer_contains_source"):
                    mock_failures += 1
                if not record["hit"] and not record.get("answer_contains_query"):
                    mock_failures += 1

        return {
            "provider": provider,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "thresholds": list(thresholds),
            "fixtures": {
                "golden": len(golden),
                "paraphrase": len(paraphrases),
                "near_miss": len(near_misses),
            },
            "candidate": {
                "endpoint_id": candidate.endpoint_id,
                "model_id": candidate.model_id,
                "provider_id": candidate.provider_id,
            },
            "database": "temporary SQLite DB (deleted after the run)",
            "mock_consistency_failures": mock_failures,
            "phases": {
                "A_cold": summarize(
                    records_a,
                    phase="A_cold",
                    run="cold",
                    semantic_enabled=True,
                    threshold=thresholds[0],
                ),
                "B_exact_repeat": summarize(
                    records_b,
                    phase="B_exact_repeat",
                    run="exact_repeat",
                    semantic_enabled=False,
                    threshold=thresholds[0],
                ),
                "C_paraphrase": {"disabled": c_runs["disabled"], "thresholds": c_runs},
                "D_near_miss": {"disabled": d_runs["disabled"], "thresholds": d_runs},
            },
            "threshold_sweep": sweep,
            "records": records_a + records_b + records_c + records_d,
        }


def _pct(value: Optional[float]) -> str:
    """Format a 0..1 rate as a percentage."""
    return "—" if value is None else f"{value * 100:.1f}%"


def _money(value: float) -> str:
    """Format a USD amount with six decimals."""
    return f"${value:.6f}"


def _ms(value: float) -> str:
    """Format milliseconds with one decimal."""
    return f"{value:.1f}"


def _tokens(value: int) -> str:
    """Format an integer token count with thousands separators."""
    return f"{value:,}"


def render_markdown(results: Dict[str, Any]) -> str:
    """Render the evaluation payload as a report-ready Markdown table set."""
    phases = results["phases"]
    a = phases["A_cold"]
    b = phases["B_exact_repeat"]
    c_disabled = phases["C_paraphrase"]["disabled"]
    d_disabled = phases["D_near_miss"]["disabled"]
    c_thresholds = phases["C_paraphrase"]["thresholds"]
    d_thresholds = phases["D_near_miss"]["thresholds"]
    thresholds = results["thresholds"]

    lines: List[str] = [
        "# ModelMesh Response Cache Evaluation",
        "",
        f"- Provider: `{results['provider']}`",
        f"- Candidate: `{results['candidate']['endpoint_id']}` "
        f"({results['candidate']['model_id']})",
        f"- Generated: {results['generated_at']}",
        f"- Golden prompts: {results['fixtures']['golden']}",
        f"- Paraphrases: {results['fixtures']['paraphrase']}",
        f"- Near misses: {results['fixtures']['near_miss']}",
        f"- Thresholds: {', '.join(f'{t:.2f}' for t in thresholds)}",
        "- Cache: isolated temporary SQLite DB, clean for this run",
        "",
        "## Phase A — cold pass",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Turns | {a['turns']} |",
        f"| Hits | {a['hits']} ({_pct(a['hit_rate'])}) |",
        f"| Avg latency | {_ms(a['avg_latency_ms'])} ms |",
        f"| Total cost | {_money(a['total_cost_usd'])} |",
        f"| Tokens saved | {_tokens(a['saved_tokens_total'])} |",
        "",
        "## Phase B — exact repeat pass",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Turns | {b['turns']} |",
        f"| Hits | {b['hits']} ({_pct(b['hit_rate'])}) |",
        f"| Exact hits | {b['exact_hits']} |",
        f"| Avg latency | {_ms(b['avg_latency_ms'])} ms |",
        f"| Total cost | {_money(b['total_cost_usd'])} |",
        f"| Tokens saved | {_tokens(b['saved_tokens_total'])} |",
        "",
        "## Phase C — paraphrase pass",
        "",
        "| Semantic | Threshold | Hits | Semantic hits | Semantic hit rate | "
        "Avg latency (ms) | Total cost | Tokens saved |",
        "|---|---|---|---|---|---|---|---|",
        f"| disabled | — | {c_disabled['hits']} | {c_disabled['semantic_hits']} | "
        f"{_pct(c_disabled['semantic_hit_rate'])} | {_ms(c_disabled['avg_latency_ms'])} | "
        f"{_money(c_disabled['total_cost_usd'])} | {_tokens(c_disabled['saved_tokens_total'])} |",
    ]
    for threshold in thresholds:
        row = c_thresholds[f"{threshold:.2f}"]
        lines.append(
            f"| enabled | {threshold:.2f} | {row['hits']} | {row['semantic_hits']} | "
            f"{_pct(row['semantic_hit_rate'])} | {_ms(row['avg_latency_ms'])} | "
            f"{_money(row['total_cost_usd'])} | {_tokens(row['saved_tokens_total'])} |"
        )

    lines.extend(
        [
            "",
            "## Phase D — near-miss pass",
            "",
            "| Semantic | Threshold | Hits | False hits | False-hit rate | "
            "Avg latency (ms) | Total cost | Tokens saved |",
            "|---|---|---|---|---|---|---|---|",
            f"| disabled | — | {d_disabled['hits']} | {d_disabled['false_hits']} | "
            f"{_pct(d_disabled['false_hit_rate'])} | {_ms(d_disabled['avg_latency_ms'])} | "
            f"{_money(d_disabled['total_cost_usd'])} | {_tokens(d_disabled['saved_tokens_total'])} |",
        ]
    )
    for threshold in thresholds:
        row = d_thresholds[f"{threshold:.2f}"]
        lines.append(
            f"| enabled | {threshold:.2f} | {row['hits']} | {row['false_hits']} | "
            f"{_pct(row['false_hit_rate'])} | {_ms(row['avg_latency_ms'])} | "
            f"{_money(row['total_cost_usd'])} | {_tokens(row['saved_tokens_total'])} |"
        )

    lines.extend(
        [
            "",
            "## Threshold sweep",
            "",
            "| Threshold | Paraphrase semantic hit rate | Near-miss false-hit rate |",
            "|---|---|---|",
        ]
    )
    for point in results["threshold_sweep"]:
        lines.append(
            f"| {point['threshold']:.2f} | "
            f"{_pct(point['paraphrase_semantic_hit_rate'])} | "
            f"{_pct(point['near_miss_false_hit_rate'])} |"
        )

    default_threshold = 0.92 if 0.92 in thresholds else thresholds[-1]
    default_key = f"{default_threshold:.2f}"
    default_c = c_thresholds[default_key]
    default_d = d_thresholds[default_key]
    false_ids = [
        record["id"]
        for record in results["records"]
        if record["phase"] == "D_near_miss"
        and record["run"] == f"semantic_{default_key}"
        and record["hit"]
    ]
    missed_ids = [
        record["id"]
        for record in results["records"]
        if record["phase"] == "C_paraphrase"
        and record["run"] == f"semantic_{default_key}"
        and not record["hit"]
    ]

    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- At the default threshold ({default_threshold:.2f}), semantic matching "
            f"served {default_c['semantic_hits']}/{default_c['turns']} paraphrases "
            f"and produced {default_d['false_hits']}/{default_d['turns']} false "
            f"hits.",
            f"- Near misses that slipped through at {default_threshold:.2f}: "
            + (", ".join(f"`{item}`" for item in false_ids) if false_ids else "none")
            + ".",
            f"- Paraphrases still missed at {default_threshold:.2f}: "
            f"{len(missed_ids)}/{results['fixtures']['paraphrase']}.",
            "- The hashing embedder is lexical: long prompts can exceed a high "
            "threshold after a single entity swap, while short natural "
            "paraphrases can score below 0.80. The numeric and negation gates "
            "stop changed numbers and explicit negations, but not entity swaps.",
            "- Semantic matching is disabled by default in `config/cache.yaml`; "
            "these numbers are a reason to keep it opt-in until a neural "
            "embedder is configured.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/cache_eval.py --provider mock",
            "# live mode (requires GROQ_API_KEY in the environment):",
            "python scripts/cache_eval.py --provider live --yes-i-know-this-spends-quota",
            "```",
            "",
            "## Notes",
            "",
            "- Phase A uses `bypass_cache=True` so every turn is a guaranteed miss; "
            "semantic matching is enabled during this phase so each stored answer "
            "also gets an embedding.",
            "- Phases C and D use a read-only cache wrapper: a miss cannot add an "
            "entry, so every threshold is measured against the same 30 golden entries.",
            "- A near-miss hit is counted as a false hit because every near-miss "
            "fixture has `same_meaning: false`.",
            "- The dependency-free `HashingEmbedder` is used; the stopword list and "
            "feature weights are unchanged from Step 6.",
        ]
    )
    if results["provider"] == "mock":
        lines.append(
            f"- Mock mode: the mock adapter embeds the last user prompt in its "
            f"answer. Consistency checks failed {results['mock_consistency_failures']} "
            f"time(s)."
        )
    else:
        lines.append(
            "- Live mode: only the configured free Groq endpoint was used; the "
            "evaluation never falls back to another provider."
        )
    lines.append("")
    return "\n".join(lines)


def _parse_thresholds(raw: str) -> Tuple[float, ...]:
    """Parse a comma-separated threshold list, validating the 0..1 range."""
    values: List[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold out of range: {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one threshold is required")
    return tuple(values)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evaluation CLI.

    Args:
        argv: Arguments after the script name (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on evaluation failure, 2 on a
        refused/invalid invocation.
    """
    parser = argparse.ArgumentParser(
        description="Reproducible evaluation harness for the ModelMesh response cache."
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "live"),
        default="mock",
        help="Use the offline mock provider or the configured free Groq endpoint.",
    )
    parser.add_argument(
        "--yes-i-know-this-spends-quota",
        action="store_true",
        dest="confirm_live",
        help="Required for --provider live; confirms real provider quota may be used.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLDS),
        help="Comma-separated cosine thresholds for the semantic sweep.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for cache_eval.json and cache_eval.md.",
    )
    args = parser.parse_args(argv)

    if args.provider == "live" and not args.confirm_live:
        print(
            "Refusing to run live evaluation: pass "
            "--yes-i-know-this-spends-quota to confirm that real provider quota "
            "may be used.",
            file=sys.stderr,
        )
        return 2

    try:
        thresholds = _parse_thresholds(args.thresholds)
        golden, paraphrases, near_misses, golden_by_id = load_and_validate_fixtures()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Invalid evaluation input: {exc}", file=sys.stderr)
        return 2

    if args.provider == "live":
        max_calls = len(golden) + (1 + len(thresholds)) * (
            len(paraphrases) + len(near_misses)
        )
        print(
            f"Live mode: up to {max_calls} provider calls against the configured "
            f"free Groq endpoint.",
            file=sys.stderr,
        )

    try:
        results = run_evaluation(
            provider=args.provider,
            thresholds=thresholds,
            golden=golden,
            paraphrases=paraphrases,
            near_misses=near_misses,
            golden_by_id=golden_by_id,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cache_eval.json"
    md_path = output_dir / "cache_eval.md"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(results), encoding="utf-8")

    phases = results["phases"]
    a = phases["A_cold"]
    b = phases["B_exact_repeat"]
    print(f"ModelMesh cache evaluation ({results['provider']})")
    print(f"  candidate: {results['candidate']['endpoint_id']}")
    print(f"  phase A cold: {a['hits']}/{a['turns']} hits ({_pct(a['hit_rate'])})")
    print(f"  phase B exact: {b['hits']}/{b['turns']} hits ({_pct(b['hit_rate'])})")
    for point in results["threshold_sweep"]:
        print(
            f"  threshold {point['threshold']:.2f}: paraphrase semantic hit rate "
            f"{_pct(point['paraphrase_semantic_hit_rate'])}, near-miss false-hit rate "
            f"{_pct(point['near_miss_false_hit_rate'])}"
        )
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
