"""RAG Configurator — standalone UI for managing RAG profiles.

Mount on any FastAPI app via ``mount_configurator(app)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

CONFIGURATOR_DIR = Path(__file__).parent
FRONTEND_DIST = CONFIGURATOR_DIR / "frontend" / "dist"


def mount_configurator(app: "FastAPI", prefix: str = "/configurator") -> None:
    """Register configurator API routes and serve the built frontend.

    Args:
        app: The FastAPI application to mount on.
        prefix: URL prefix for the configurator (default ``/configurator``).
    """
    from .routes import router

    app.include_router(router, prefix=prefix)

    # Serve built frontend as static files (if dist/ exists)
    if FRONTEND_DIST.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            prefix,
            StaticFiles(directory=str(FRONTEND_DIST), html=True),
            name="configurator-frontend",
        )
