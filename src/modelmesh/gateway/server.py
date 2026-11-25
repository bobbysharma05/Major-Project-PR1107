"""Server entry point for the ModelMesh OpenAI-compatible gateway."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from modelmesh.core.engine import ChatEngine
from modelmesh.core.keys import load_env
from modelmesh.core.registry import load_default_registry
from modelmesh.core.routing.router import Router
from modelmesh.core.storage import Storage

INSTALL_HINT = 'Install the gateway extra: pip install -e ".[gateway]"'


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start the gateway with an engine built exactly like ``cli.py``.

    Args:
        argv: Arguments after the command name (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on clean shutdown, 1 on a startup failure.
    """
    try:
        import uvicorn

        from .app import create_app
    except ImportError as exc:
        print(
            f"Gateway dependencies are missing ({exc}). {INSTALL_HINT}",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        description="ModelMesh OpenAI-compatible HTTP gateway."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="Path to providers.yaml/config dir")
    parser.add_argument("--env-file", default=None, help="Path to .env secrets file")
    parser.add_argument(
        "--default-strategy",
        default="cheapest_first",
        help="Strategy used when the request model is 'auto'.",
    )
    args = parser.parse_args(argv)

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"WARNING: gateway is bound to {args.host!r}, which is not loopback. "
            "Anyone who reaches this port can use your private API keys. "
            "Use a reverse proxy with TLS and rate limiting.",
            file=sys.stderr,
        )

    env_path = Path(args.env_file) if args.env_file else None
    load_env(env_path)
    config_dir = Path(args.config) if args.config else None

    storage = Storage()
    storage.purge_expired_cache()
    registry = load_default_registry(config_dir)
    router = Router(registry=registry, storage=storage)
    engine = ChatEngine(registry=registry, router=router, storage=storage)

    app = create_app(
        engine, storage, default_strategy=args.default_strategy
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
