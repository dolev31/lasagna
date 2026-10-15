"""API layer - Main RAGBackend facade.

This layer provides the primary entry point for the RAG system:
- RAGBackend: Orchestrates all components (processing, retrieval, storage)

This is the main interface for external consumers.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rag_backend import RAGBackend, get_rag_backend, initialize, reset_instance, close_backend

__all__ = [
    "RAGBackend",
    "get_rag_backend",
    "initialize",
    "reset_instance",
    "close_backend",
]


def __getattr__(name: str):
    # Resolved on first use. Importing any lasagna.api submodule, even the
    # small auth_mode, ran this file, and an eager import of rag_backend here
    # loaded torch and the embedding stack: `lasagna-mcp --help` took 8 s.
    if name in __all__:
        from . import rag_backend

        value = getattr(rag_backend, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
