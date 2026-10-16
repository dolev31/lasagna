"""Embedding generation and caching subsystem.

This module provides embedding generation using sentence-transformers models
with caching support for improved performance.
"""

from .manager import (
    EmbeddingManager,
    EmbeddingCache,
    get_embedding_manager,
    reset_embedding_manager,
)

__all__ = [
    "EmbeddingManager",
    "EmbeddingCache",
    "get_embedding_manager",
    "reset_embedding_manager",
]
