"""Stdlib-only smoke test for a running ModelMesh gateway.

Usage::

    python scripts/gateway_smoke.py --base-url http://127.0.0.1:8000 --key mm-...

The key is sent in a header but is never printed or written anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def http_json(
    base_url: str,
    method: str,
    path: str,
    *,
    key: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """Make one HTTP request and return ``(status, body, error)``.

    Args:
        base_url: Gateway root, for example ``http://127.0.0.1:8000``.
        method: HTTP method.
        path: Path starting with ``/``.
        key: Optional bearer token; never included in the error string.
        payload: Optional JSON body.

    Returns:
        Status code, parsed JSON body (when available) and an error message.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        status = exc.code
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"

    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return status, None, "response was not valid JSON"
    return status, body, None


def main(argv: Optional[List[str]] = None) -> int:
    """Run all smoke checks and return 0 only when every check passes."""
    parser = argparse.ArgumentParser(description="ModelMesh gateway smoke test.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key", required=True, help="A private API key (never printed).")
    args = parser.parse_args(argv)

    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        """Record and print one check result."""
        results.append((name, ok, detail))
        label = "PASS" if ok else "FAIL"
        print(f"{label} {name}: {detail}")

    status, body, error = http_json(args.base_url, "GET", "/health")
    record(
        "health",
        status == 200 and bool(body) and body.get("status") == "ok",
        f"HTTP {status}" + (f" ({error})" if error else ""),
    )

    status, body, error = http_json(
        args.base_url,
        "POST",
        "/v1/chat/completions",
        payload={
            "model": "auto",
            "messages": [{"role": "user", "content": "unauthorized smoke check"}],
        },
    )
    no_key_ok = (
        status == 401
        and bool(body)
        and (body.get("error") or {}).get("code") == "invalid_api_key"
    )
    record("no-key 401", no_key_ok, f"HTTP {status}" + (f" ({error})" if error else ""))

    status, body, error = http_json(
        args.base_url, "GET", "/v1/models", key=args.key
    )
    model_ids = [item.get("id") for item in (body or {}).get("data", [])] if body else []
    record(
        "models list",
        status == 200 and bool(body) and body.get("object") == "list" and "auto" in model_ids,
        f"HTTP {status}, {len(model_ids)} model(s)"
        + (f" ({error})" if error else ""),
    )

    prompt = f"Smoke test {uuid.uuid4().hex}: reply with one short sentence."
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
    }
    status, body, error = http_json(
        args.base_url, "POST", "/v1/chat/completions", key=args.key, payload=payload
    )
    choices = (body or {}).get("choices") or []
    message = choices[0].get("message") if choices else {}
    shape_ok = (
        status == 200
        and bool(body)
        and body.get("object") == "chat.completion"
        and isinstance(body.get("created"), int)
        and isinstance(body.get("usage"), dict)
        and isinstance(body.get("modelmesh"), dict)
        and isinstance(message.get("content"), str)
    )
    record(
        "completion shape",
        shape_ok,
        f"HTTP {status}" + (f" ({error})" if error else ""),
    )

    status_2, body_2, error_2 = http_json(
        args.base_url, "POST", "/v1/chat/completions", key=args.key, payload=payload
    )
    first_cache = ((body or {}).get("modelmesh") or {}).get("cache") or {}
    second_cache = ((body_2 or {}).get("modelmesh") or {}).get("cache") or {}
    cost = ((body_2 or {}).get("modelmesh") or {}).get("cost_usd")
    cache_ok = (
        status_2 == 200
        and first_cache.get("status") == "miss"
        and second_cache.get("status") == "hit"
        and cost == 0
    )
    record(
        "cache replay",
        cache_ok,
        f"first={first_cache.get('status')}, second={second_cache.get('status')}, "
        f"cost_usd={cost}"
        + (f" ({error_2})" if error_2 else ""),
    )

    passed = sum(1 for _name, ok, _detail in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
