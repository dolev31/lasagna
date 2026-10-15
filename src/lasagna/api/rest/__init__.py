"""REST API for Lasagna.

Provides a versioned HTTP API (``/api/v1/``) for document management,
semantic search, citation retrieval, health probes, and system
administration.

Usage::

    from lasagna.api.rest import create_app

    app = create_app()  # FastAPI instance ready for uvicorn
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import create_app

__all__ = ["create_app"]


def __getattr__(name: str):
    # Resolved on first use: importing `.app` builds the whole application, so
    # importing lasagna.api.rest.auth_mode (the CLI and MCP server do, to check
    # a bind address) built it too.
    if name == "create_app":
        from .app import create_app

        globals()[name] = create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
