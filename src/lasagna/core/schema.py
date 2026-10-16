"""Schema definitions for Lasagna.

This module defines Pydantic models for documents, chunks, and metadata
used throughout the RAG backend.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

# pymilvus is optional -- only needed for the Milvus backend, so it is imported
# where it is used (see create_rag_document_schema) rather than here.
#
# Importing it at module scope made the comment above untrue: this module is
# reached from `lasagna.core.__init__`, so *any* use of the package pulled
# pymilvus in. That mattered beyond import weight, because
# `pymilvus/settings.py` calls `load_dotenv()` at import time -- so merely
# importing a Lasagna pipeline read whatever `.env` sat above site-packages and
# injected its contents into `os.environ`. On a developer machine that is an
# unrelated project's live credentials appearing in this process.
#
# Found by the E2E core-journey phase, which asserts that importing Lasagna
# injects no credential environment variables.


class DocumentStatus(str, Enum):
    """Status of document processing.

    Values are synchronized with frontend (knowledge.ts).
    IMPORTANT: Keep these values in sync with:
    - src/frontend_workspaces/agentic_chat/src/types/knowledge.ts

    Lifecycle transitions:
      UPLOADING → PROCESSING → INDEXING → READY
      Any state  → ERROR
      READY      → SOFT_DELETED  (archive / grace-period before GC hard-deletes)
      ERROR      → SOFT_DELETED  (auto-scheduled after error_retention_days)
    """

    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXING = "indexing"  # Added to match frontend
    READY = "ready"
    ERROR = "error"  # Changed from FAILED to match frontend
    SOFT_DELETED = "soft_deleted"  # Archived; GC will hard-delete after grace window

    # Alias for backward compatibility (deprecated, use ERROR instead)
    @classmethod
    def _missing_(cls, value):
        """Handle backward compatibility for 'failed' status."""
        if value == "failed":
            return cls.ERROR
        return None

    @classmethod
    def is_terminal(cls, status: "DocumentStatus") -> bool:
        """Return True for statuses where no further processing is expected."""
        return status in (cls.READY, cls.ERROR, cls.SOFT_DELETED)


class Document(BaseModel):
    """Document metadata for RAG system."""

    document_id: str = Field(description="Unique identifier for the document")
    session_id: str = Field(description="Session/namespace this document belongs to")
    filename: str = Field(description="Original filename")
    file_type: str = Field(description="File extension (e.g., '.pdf', '.txt')")
    file_size: int = Field(description="File size in bytes")
    status: DocumentStatus = Field(
        default=DocumentStatus.PROCESSING, description="Processing status"
    )
    chunks_count: int = Field(default=0, description="Number of chunks created from this document")
    upload_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Upload timestamp"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )

    # Lifecycle fields
    deleted_at: Optional[datetime] = Field(
        default=None, description="Timestamp when soft-deleted; None = active"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="TTL expiry timestamp; None = no auto-expiry"
    )
    collection_name: Optional[str] = Field(
        default=None, description="Vector collection this doc was written to (for orphan detection)"
    )
    duplicate_of: Optional[str] = Field(
        default=None,
        description="Set on an upload whose content the session already held: that document's id",
    )

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DocumentChunk(BaseModel):
    """A chunk of text from a document with its embedding."""

    content: str = Field(description="The text content of this chunk")
    document_id: str = Field(description="ID of the parent document")
    chunk_index: int = Field(description="Position of this chunk in the document (0-based)")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable")

    # Metadata for retrieval
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "This is a sample chunk of text from a document.",
                "document_id": "doc_123456",
                "chunk_index": 0,
                "page_number": 1,
                "metadata": {"filename": "example.pdf", "session_id": "session_abc"},
            }
        }
    )


class RecordedDocumentChunk(DocumentChunk):
    """A document chunk that has been stored in Milvus."""

    chunk_id: str = Field(description="Unique chunk identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp"
    )


class DocumentMetadata(BaseModel):
    """Detailed metadata for a document, including storage info."""

    document_id: str
    session_id: str
    filename: str
    file_path: str = Field(description="Path to stored file on disk")
    file_type: str
    file_size: int
    status: DocumentStatus
    chunks_count: int
    upload_date: datetime
    error_message: Optional[str] = None

    # Processing details
    processing_time: Optional[float] = Field(
        default=None, description="Time taken to process (seconds)"
    )
    text_length: Optional[int] = Field(default=None, description="Total characters extracted")

    # Backend tracking
    vector_backend_name: Optional[str] = Field(
        default="milvus", description="Backend that stores this document's vectors"
    )
    storage_backend_name: Optional[str] = Field(
        default="local", description="Backend that stores this document's files"
    )

    # Lifecycle fields
    deleted_at: Optional[datetime] = Field(
        default=None, description="Timestamp when soft-deleted; None = active"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="TTL expiry timestamp; None = no auto-expiry"
    )
    collection_name: Optional[str] = Field(
        default=None, description="Vector collection this doc was written to"
    )


class SearchResult(BaseModel):
    """Result from a RAG search query."""

    chunk: RecordedDocumentChunk
    score: float = Field(description="Similarity score (0-1)")
    document: Document = Field(description="Parent document metadata")


# =============================================================================
# Abstract (backend-agnostic) schema
# =============================================================================


def create_abstract_rag_schema(embedding_dim: int = 1024):
    """Create a backend-agnostic schema for RAG document collections.

    This schema works with any VectorStoreAdapter backend (Qdrant, Pinecone, etc.).

    Args:
        embedding_dim: Dimension of embedding vectors

    Returns:
        CollectionSchema from vector_backends.base
    """
    from lasagna.vector_backends.base import (
        CollectionSchema as AbstractCollectionSchema,
        FieldSchema as AbstractFieldSchema,
    )

    return AbstractCollectionSchema(
        fields=[
            AbstractFieldSchema(name="chunk_id", dtype="VARCHAR", max_length=128, is_primary=True),
            AbstractFieldSchema(name="document_id", dtype="VARCHAR", max_length=128),
            AbstractFieldSchema(name="session_id", dtype="VARCHAR", max_length=128),
            AbstractFieldSchema(name="content", dtype="VARCHAR", max_length=65535),
            AbstractFieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=embedding_dim),
            AbstractFieldSchema(name="metadata", dtype="JSON"),
        ],
        description="Lasagna RAG document collection",
    )


# =============================================================================
# Milvus-specific schema (only available when pymilvus is installed)
# =============================================================================


def create_rag_document_schema(embedding_dim: int = 1024, enable_native_bm25: bool = False):
    """Create Milvus schema with configurable embedding dimension.

    Args:
        embedding_dim: Dimension of embedding vectors (default 1024 for BGE-large)
        enable_native_bm25: If True, adds sparse vector field for native BM25
            (requires Milvus Standalone/Distributed, not supported in Milvus Lite)

    Returns:
        pymilvus CollectionSchema for RAG documents

    Raises:
        ImportError: If pymilvus is not installed
    """
    try:
        # Never a bare import: pymilvus reads the nearest .env into os.environ.
        from lasagna.vector_backends._pymilvus import from_pymilvus

        MilvusCollectionSchema, DataType, MilvusFieldSchema = from_pymilvus(
            "CollectionSchema", "DataType", "FieldSchema"
        )
    except ImportError as exc:
        raise ImportError(
            "pymilvus is required for create_rag_document_schema(). "
            "Use create_abstract_rag_schema() for non-Milvus backends."
        ) from exc

    fields = [
        MilvusFieldSchema(
            name="chunk_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="Unique chunk identifier",
        ),
        MilvusFieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Parent document ID",
        ),
        MilvusFieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=4000,  # Increased for larger chunks
            description="Text content of the chunk",
        ),
        MilvusFieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
            description="Dense vector embedding of the content",
        ),
        MilvusFieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="JSON metadata including session_id, chunk_index, page_number, filename, stable_id",
        ),
    ]

    # Add sparse vector field for native BM25 if enabled
    # Note: Only supported in Milvus Standalone/Distributed, NOT Milvus Lite
    if enable_native_bm25:
        fields.append(
            MilvusFieldSchema(
                name="sparse_embedding",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="Sparse BM25 vector for full-text search (native)",
            )
        )

    description = "RAG document chunks with embeddings for semantic search"
    if enable_native_bm25:
        description += " - includes native BM25 sparse vectors"

    return MilvusCollectionSchema(fields=fields, description=description)


# Default schema, kept for backwards compatibility.
#
# Resolved lazily (PEP 562). Building it at module scope meant every import of
# this module constructed a Milvus schema, which imported pymilvus, which calls
# `load_dotenv()` -- so importing Lasagna at all read a `.env` from above
# site-packages into `os.environ`. Nothing in the package or its tests reads
# this name, but it stays reachable so an external caller does not break.
#
# `None` when pymilvus is not installed, matching the previous behaviour.
def __getattr__(name: str):
    if name == "rag_document_schema":
        try:
            value = create_rag_document_schema(1024)
        except ImportError:
            value = None
        globals()["rag_document_schema"] = value  # cache; __getattr__ won't run again
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# SQLite table creation SQL
# Canonical DDL — must match database.py:CREATE_DOCUMENTS_TABLE
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS rag_documents (
    document_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER DEFAULT 0,
    file_type TEXT DEFAULT '',
    status TEXT DEFAULT 'processing',
    chunks_count INTEGER DEFAULT 0,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    processing_time REAL,
    text_length INTEGER,
    preview_text TEXT,
    vector_backend_name TEXT DEFAULT 'milvus',
    storage_backend_name TEXT DEFAULT 'local',
    deleted_at TIMESTAMP DEFAULT NULL,
    expires_at TIMESTAMP DEFAULT NULL,
    collection_name TEXT DEFAULT NULL
);
"""

# GC audit log — records each phase of each GC run
CREATE_GC_AUDIT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS gc_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    docs_scanned INTEGER DEFAULT 0,
    docs_cleaned INTEGER DEFAULT 0,
    bytes_freed INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    details TEXT DEFAULT '{}'
);
"""

CREATE_GC_AUDIT_RUN_INDEX = """
CREATE INDEX IF NOT EXISTS idx_gc_audit_run_id ON gc_audit_log(run_id);
"""

CREATE_GC_AUDIT_STARTED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_gc_audit_started ON gc_audit_log(started_at);
"""

# Distributed advisory lock — prevents duplicate GC runs across processes
CREATE_GC_LOCK_TABLE = """
CREATE TABLE IF NOT EXISTS gc_lock (
    lock_name TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    acquired_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL
);
"""

# Migration SQL to add preview_text column to existing tables
ADD_PREVIEW_TEXT_COLUMN = """
ALTER TABLE rag_documents ADD COLUMN preview_text TEXT;
"""

# Migration SQL to add lifecycle columns to existing tables
ADD_LIFECYCLE_COLUMNS = [
    "ALTER TABLE rag_documents ADD COLUMN deleted_at TIMESTAMP DEFAULT NULL;",
    "ALTER TABLE rag_documents ADD COLUMN expires_at TIMESTAMP DEFAULT NULL;",
    "ALTER TABLE rag_documents ADD COLUMN collection_name TEXT DEFAULT NULL;",
]

CREATE_DOCUMENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_session_id ON rag_documents(session_id);
"""

CREATE_DOCUMENTS_STATUS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_status ON rag_documents(status);
"""

CREATE_DOCUMENTS_DELETED_AT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_documents_deleted_at ON rag_documents(deleted_at)
WHERE deleted_at IS NOT NULL;
"""

CREATE_DOCUMENTS_EXPIRES_AT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_documents_expires_at ON rag_documents(expires_at)
WHERE expires_at IS NOT NULL;
"""
