"""CLI entry point — run the RAG REST API with the configurator mounted.

Usage::

    python -m lasagna.configurator
    # or
    python -m lasagna.configurator --port 8000
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

from loguru import logger


def _find_available_port(start: int, max_attempts: int = 10) -> int:
    """Find the first available port starting from *start*.

    Tries up to *max_attempts* consecutive ports.  Raises ``RuntimeError``
    if none are free.
    """
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available port in range {start}-{start + max_attempts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lasagna Configurator")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1). 0.0.0.0 exposes an unauthenticated API.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install it with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    # Ensure the configurator UI is mounted via env var
    os.environ.setdefault("RAG_CONFIGURATOR", "true")

    # Find an available port (handles conflicts gracefully)
    port = _find_available_port(args.port)
    if port != args.port:
        logger.info(f"Port {args.port} in use, using {port} instead")

    uvicorn.run(
        "lasagna.api.rest.app:app",
        host=args.host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
