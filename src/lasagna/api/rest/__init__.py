"""REST API for RAG Knowledge Base.

Provides a versioned HTTP API (``/api/v1/``) for document management,
semantic search, citation retrieval, health probes, and system
administration.

Usage::

    from lasagna.api.rest import create_app

    app = create_app()  # FastAPI instance ready for uvicorn
"""

from .app import create_app

__all__ = ["create_app"]
