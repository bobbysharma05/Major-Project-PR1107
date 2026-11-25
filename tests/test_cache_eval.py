"""Tests for the reproducible cache-evaluation harness and its fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "cache_eval.py"


def _load_script() -> Any:
    """Import ``scripts/cache_eval.py`` as a module without a package install."""
    spec = importlib.util.spec_from_file_location("cache_eval_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("could not load scripts/cache_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_eval = _load_script()


def test_fixtures_link_every_golden_prompt() -> None:
    """Every paraphrase and near miss must reference a real golden prompt."""
    golden, paraphrases, near_misses, golden_by_id = (
        cache_eval.load_and_validate_fixtures()
    )

    assert len(golden) == 30
    assert len(paraphrases) == 30
    assert len(near_misses) == 15
    assert {item["source_id"] for item in paraphrases} == set(golden_by_id)
    assert all(item["same_meaning"] is True for item in paraphrases)
    assert all(item["same_meaning"] is False for item in near_misses)
    assert len({item["prompt"] for item in near_misses}) == len(near_misses)


def test_parse_thresholds_rejects_out_of_range_values() -> None:
    """The CLI threshold parser accepts a list and rejects invalid values."""
    assert cache_eval._parse_thresholds("0.80, 0.90,0.92") == (0.80, 0.90, 0.92)
    with pytest.raises(ValueError):
        cache_eval._parse_thresholds("1.20")
    with pytest.raises(ValueError):
        cache_eval._parse_thresholds(" , ")


def test_read_only_cache_ignores_writes_but_delegates_reads() -> None:
    """Phases C/D must not add entries, but every other call still works."""
    calls: List[Tuple[str, Tuple[Any, ...]]] = []

    class FakeStorage:
        def put_cache_entry(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("put", args))

        def get_cache_entry(self, key: str) -> str:
            calls.append(("get", (key,)))
            return "entry"

    wrapper = cache_eval.ReadOnlyCache(FakeStorage())
    wrapper.put_cache_entry("key", "answer")
    assert wrapper.get_cache_entry("key") == "entry"
    assert calls == [("get", ("key",))]


def test_live_mode_requires_explicit_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    """``--provider live`` refuses to spend quota without the confirmation flag."""
    assert cache_eval.main(["--provider", "live"]) == 2
    assert "--yes-i-know-this-spends-quota" in capsys.readouterr().err
