"""Chunking subsystem for document processing.

This module provides intelligent text chunking strategies
including semantic chunking that preserves document structure.
"""

from .semantic import (
    SemanticChunker,
    ChunkingStrategy,
    Chunk,
    chunk_document,
)

__all__ = [
    "SemanticChunker",
    "ChunkingStrategy",
    "Chunk",
    "chunk_document",
]
