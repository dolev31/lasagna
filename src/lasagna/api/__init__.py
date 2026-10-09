"""API layer - Main RAGBackend facade.

This layer provides the primary entry point for the RAG system:
- RAGBackend: Orchestrates all components (processing, retrieval, storage)

This is the main interface for external consumers.
"""

from .rag_backend import RAGBackend, get_rag_backend, initialize, reset_instance, close_backend

__all__ = [
    "RAGBackend",
    "get_rag_backend",
    "initialize",
    "reset_instance",
    "close_backend",
]
