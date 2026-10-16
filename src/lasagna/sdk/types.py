"""Type definitions for the Knowledge SDK.

Contains all enums, dataclasses, and type aliases used throughout the SDK.
These types provide type-safe configuration and result handling.

Example:
    ```python
    from lasagna.sdk.types import (
        ChunkingStrategy,
        Document,
        SearchResult,
    )

    # Use enums for type-safe configuration
    strategy = ChunkingStrategy.SEMANTIC

    # Access search results with typed attributes
    for result in results:
        print(f"Score: {result.score}, Content: {result.chunk.content}")
    ```
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Iterator, Union
from datetime import datetime


# =============================================================================
# ENUMS - Type-safe configuration values
# =============================================================================


class _StrValueEnum(str, Enum):
    """A string enum that prints as its value on every supported Python.

    Python 3.11 changed how a ``(str, Enum)`` member formats: on 3.10
    ``f"{DocumentStatus.READY}"`` gave ``ready`` and on 3.12 it gives
    ``DocumentStatus.READY``, so the same code printed differently by
    interpreter. Defining both methods makes ``str()`` and f-strings give the
    value everywhere, as ``enum.StrEnum`` (3.11+) does.
    """

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(str(self.value), format_spec)


class ChunkingStrategy(_StrValueEnum):
    """Available chunking strategies for document processing.

    Attributes:
        FIXED: Fixed-size chunks with configurable overlap
        SEMANTIC: Semantically coherent chunks based on similarity
        RECURSIVE: Recursive character text splitting
        SENTENCE_WINDOW: Sliding window over sentences
        STRUCTURE_AWARE: Sections split at headers, tables kept whole, each
            section chunked semantically (needs preserve_headers or
            preserve_tables; with both off it chunks as SEMANTIC)
        HIERARCHICAL: Parent chunks split into children (needs
            enable_hierarchical; with it off it chunks as SEMANTIC)
    """

    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE_WINDOW = "sentence_window"
    STRUCTURE_AWARE = "structure_aware"
    HIERARCHICAL = "hierarchical"


class EmbeddingProvider(_StrValueEnum):
    """Supported embedding model providers.

    Attributes:
        HUGGINGFACE: HuggingFace sentence-transformers models
        OPENAI: OpenAI embedding models (ada-002, etc.)
    """

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class OCREngine(_StrValueEnum):
    """Available OCR engines for scanned document processing.

    Attributes:
        EASYOCR: EasyOCR engine (default, good accuracy)
        TESSERACT: Tesseract OCR (faster, widely available)
        PADDLEOCR: PaddleOCR (good for multilingual)
    """

    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"


class HyDEStrategy(_StrValueEnum):
    """HyDE (Hypothetical Document Embeddings) strategies.

    HyDE generates hypothetical answers to improve retrieval quality.

    Attributes:
        OFF: HyDE disabled
        FIRST_HOP: Only apply HyDE on first search iteration
        ADAPTIVE: Automatically decide based on query complexity
        ALWAYS: Always generate hypothetical document
    """

    OFF = "off"
    FIRST_HOP = "first_hop"
    ADAPTIVE = "adaptive"
    ALWAYS = "always"


class ChunkRetrievalStrategy(_StrValueEnum):
    """Strategies for determining how many chunks to retrieve.

    Attributes:
        FIXED: Return exactly N chunks
        ABOVE_THRESHOLD: Return all chunks above score threshold
        ADAPTIVE: Dynamically adjust based on score distribution
        AGENT_DECIDES: Let the agentic RAG decide
    """

    FIXED = "fixed"
    ABOVE_THRESHOLD = "above_threshold"
    ADAPTIVE = "adaptive"
    AGENT_DECIDES = "agent_decides"


class EnrichmentStrategy(_StrValueEnum):
    """Document awareness enrichment strategies.

    Attributes:
        FIRST_TOKENS: Extract first N tokens from document
        LLM_SUMMARY: Generate LLM summary of document
        HYBRID: Combine both based on document size
    """

    FIRST_TOKENS = "first_tokens"
    LLM_SUMMARY = "llm_summary"
    HYBRID = "hybrid"


class CitationFormat(_StrValueEnum):
    """Citation formatting styles.

    Attributes:
        STABLE_ID: Hash-based stable citation IDs (e.g., [abc123])
        INDEX: Sequential 1, 2, 3... numbering (e.g., [1], [2])
        MARKDOWN: Full markdown with source info (e.g., [Source: doc.pdf, p.5])
        NONE: No citation markers in output
    """

    STABLE_ID = "stable_id"
    INDEX = "index"
    MARKDOWN = "markdown"
    NONE = "none"


class CitationScoreMethod(_StrValueEnum):
    """Score normalization methods for citations.

    Attributes:
        MINMAX: Min-max normalization to [0, 1] range
        SIGMOID: Sigmoid normalization for smoother distribution
        CLAMP: Clamp raw scores to [0, 1]
    """

    MINMAX = "minmax"
    SIGMOID = "sigmoid"
    CLAMP = "clamp"


class VectorBackend(_StrValueEnum):
    """Supported vector database backends.

    Attributes:
        MILVUS: Milvus vector database (default)
        PINECONE: Pinecone managed service
        QDRANT: Qdrant vector database
        WEAVIATE: Weaviate vector database
        PGVECTOR: PostgreSQL pgvector extension
        OPENSEARCH: OpenSearch search engine (Lucene BM25 + k-NN vectors)
        MEMORY: In-memory backend (for testing)
    """

    MILVUS = "milvus"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"
    OPENSEARCH = "opensearch"
    MEMORY = "memory"


class DocumentStatus(_StrValueEnum):
    """Document processing status.

    Attributes:
        PENDING: Document queued for processing
        PROCESSING: Document being processed (text extraction, chunking)
        INDEXING: Chunks being indexed in vector database
        READY: Document fully indexed and searchable
        FAILED: Processing failed (check error_message)
        SOFT_DELETED: Archived; recoverable until garbage collection removes it
    """

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    # The backend reports it for archived documents. Without it, archive()
    # archived the document and then failed converting the result.
    SOFT_DELETED = "soft_deleted"

    # Backward compatibility with legacy status names
    @classmethod
    def _missing_(cls, value: str) -> "DocumentStatus":
        """Handle legacy status values."""
        legacy_map = {
            "uploading": cls.PENDING,
            "indexed": cls.READY,
            "error": cls.FAILED,
        }
        if value in legacy_map:
            return legacy_map[value]
        return None


# =============================================================================
# DATA CLASSES - Structured data types
# =============================================================================


@dataclass
class Document:
    """Document metadata.

    Represents a document in the knowledge base with all its metadata.

    Attributes:
        document_id: Unique identifier for the document
        filename: Original filename
        session_id: Session that owns this document
        status: Current processing status
        file_path: Path where document is stored
        file_type: File extension (e.g., '.pdf', '.txt')
        file_size: File size in bytes
        chunk_count: Number of chunks created from document
        created_at: Upload timestamp
        metadata: Additional custom metadata
    """

    document_id: str
    filename: str
    session_id: str
    status: DocumentStatus
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    chunk_count: int = 0
    created_at: Optional[datetime] = None
    error_message: Optional[str] = None
    preview_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunks_count(self) -> int:
        """Alias for backend compatibility (backend uses chunks_count)."""
        return self.chunk_count


@dataclass
class DocumentChunk:
    """A chunk of document content.

    Represents a single chunk extracted from a document.

    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: Parent document ID
        content: Text content of the chunk
        chunk_index: Position in document (0-based)
        page_number: Page number if applicable
        section_header: Section header if detected
        metadata: Additional chunk metadata
    """

    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Citation tracking information.

    Used to track and verify citations in RAG responses.

    Attributes:
        citation_id: Unique citation identifier (stable hash)
        chunk_id: Referenced chunk ID
        document_id: Source document ID
        filename: Source filename for display
        page_number: Page number if available
        content_preview: Preview of cited content
        score: Retrieval score (0-1)
        highlight: Highlighted text span
    """

    citation_id: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    content_preview: Optional[str] = None
    score: float = 0.0
    highlight: Optional[str] = None
    highlights: Optional[List[str]] = None
    hop_index: int = 0


@dataclass
class SearchResult:
    """A single search result.

    Contains a chunk with its relevance score and parent document.

    Attributes:
        chunk: The retrieved chunk
        document: Parent document metadata
        score: Relevance score (0-1, higher is better)
        citation_id: Optional citation ID for tracking
        highlight: Highlighted matching text
    """

    chunk: DocumentChunk
    document: Document
    score: float
    citation_id: Optional[str] = None
    highlight: Optional[str] = None
    rerank_score: Optional[float] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


@dataclass
class SearchResults:
    """Collection of search results.

    Provides iteration and helper methods for working with results.

    Attributes:
        results: List of search results
        query: Original search query
        total_chunks_searched: Total chunks in search scope
        search_time_ms: Search duration in milliseconds
        degraded_stages: Stages that fell back to a lesser result during this
            search (``hybrid``, ``hyde``, ...), as the REST API reports them;
            empty when every stage ran as configured.

    Example:
        ```python
        results = await client.search("revenue")

        # Iterate over results
        for result in results:
            print(f"[{result.score:.2f}] {result.chunk.content[:100]}")

        # Get formatted context
        context = results.to_context()
        ```
    """

    results: List[SearchResult]
    query: str
    total_chunks_searched: int = 0
    search_time_ms: float = 0.0
    citations: List[Citation] = field(default_factory=list)
    degraded_stages: List[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[SearchResult]:
        """Iterate over search results."""
        return iter(self.results)

    def __len__(self) -> int:
        """Get number of results."""
        return len(self.results)

    def __getitem__(self, index: int) -> SearchResult:
        """Get result by index."""
        return self.results[index]

    def to_context(self, max_chunks: int = 5) -> str:
        """Convert to formatted context string for LLM injection.

        Args:
            max_chunks: Maximum number of chunks to include

        Returns:
            Formatted context string with source citations
        """
        if not self.results:
            return "No relevant information found."

        parts = []
        for idx, result in enumerate(self.results[:max_chunks]):
            source = result.document.filename
            page = f" (page {result.chunk.page_number})" if result.chunk.page_number else ""
            parts.append(f"[Source {idx + 1}: {source}{page}]\n{result.chunk.content}")

        return "\n\n".join(parts)

    def get_documents(self) -> List[Document]:
        """Get unique documents from results."""
        seen = set()
        docs = []
        for result in self.results:
            if result.document.document_id not in seen:
                seen.add(result.document.document_id)
                docs.append(result.document)
        return docs


@dataclass
class UploadResult:
    """Result of document upload operation.

    Attributes:
        document_id: ID of uploaded document
        filename: Original filename
        status: Processing status
        chunk_count: Number of chunks created
        processing_time_ms: Processing duration
        message: Status message
    """

    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int
    processing_time_ms: float
    message: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if upload was successful."""
        return self.status in (DocumentStatus.READY, DocumentStatus.INDEXING)


@dataclass
class HopData:
    """Data from a single search hop in agentic RAG.

    Attributes:
        iteration: Hop number (1-based)
        query: Query used for this hop
        chunks_retrieved: Number of chunks retrieved
        refined_query: Refined query if applicable
        coverage_score: Coverage score for this hop
    """

    iteration: int
    query: str
    chunks_retrieved: int
    refined_query: Optional[str] = None
    coverage_score: Optional[float] = None
    hop_time_ms: float = 0.0


@dataclass
class AgenticSearchResults:
    """Results from agentic multi-hop search.

    Contains the final context, all citations, and hop data.

    Attributes:
        final_context: Formatted context with citations
        citations: All citations used
        hops: Data for each search hop
        total_iterations: Number of search iterations
        total_chunks_retrieved: Total unique chunks retrieved
        search_time_ms: Total search duration
        results: Underlying search results
    """

    final_context: str
    citations: List[Citation]
    hops: List[HopData]
    total_iterations: int
    total_chunks_retrieved: int
    search_time_ms: float
    results: Optional[SearchResults] = None
    refined_answer: Optional[str] = None

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get citation by ID."""
        for citation in self.citations:
            if citation.citation_id == citation_id:
                return citation
        return None


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================


@dataclass
class ChunkCiteRetrieval:
    """Configuration for chunk citation retrieval strategy.

    Controls how many chunks are returned and how they are selected
    when generating cited responses.

    Attributes:
        strategy: Retrieval strategy to use
        fixed_size: Number of chunks for FIXED strategy
        threshold_score: Minimum score for ABOVE_THRESHOLD strategy
        max_chunks: Maximum chunks for ADAPTIVE strategy
        min_chunks: Minimum chunks for ADAPTIVE strategy
    """

    strategy: ChunkRetrievalStrategy = ChunkRetrievalStrategy.ADAPTIVE
    fixed_size: int = 5
    threshold_score: float = 0.5
    max_chunks: int = 7
    min_chunks: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.value
            if isinstance(self.strategy, ChunkRetrievalStrategy)
            else self.strategy,
            "fixed_size": self.fixed_size,
            "threshold_score": self.threshold_score,
            "max_chunks": self.max_chunks,
            "min_chunks": self.min_chunks,
        }


@dataclass
class DocumentAwareness:
    """Configuration for document awareness in agent prompts.

    Controls how document metadata and previews are injected into
    agent context to improve response quality and grounding.

    Attributes:
        enabled: Whether document awareness is active
        enrichment_strategy: How to generate document previews
        first_tokens_count: Tokens to extract for FIRST_TOKENS strategy
        include_filename: Include filename in context
        include_upload_date: Include upload date in context
        include_file_type: Include file type in context
        summary_model: LLM model for LLM_SUMMARY strategy
        summary_max_tokens: Max tokens for generated summaries
        summary_prompt: Prompt template for summary generation
        hybrid_llm_threshold_kb: File size threshold (KB) for HYBRID strategy
        max_documents_in_context: Maximum documents to include
        cache_snippets: Cache generated snippets
        snippet_cache_ttl: Cache TTL in seconds
    """

    enabled: bool = True
    enrichment_strategy: Union[EnrichmentStrategy, str] = EnrichmentStrategy.FIRST_TOKENS
    first_tokens_count: int = 150
    include_filename: bool = True
    include_upload_date: bool = True
    include_file_type: bool = True
    summary_model: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "gpt-6-luna"))
    summary_max_tokens: int = 100
    summary_prompt: str = (
        "Summarize this document in 1-2 sentences, focusing on the main topic and key information:"
    )
    hybrid_llm_threshold_kb: int = 100
    max_documents_in_context: int = 10
    cache_snippets: bool = True
    snippet_cache_ttl: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "enrichment_strategy": self.enrichment_strategy.value
            if isinstance(self.enrichment_strategy, EnrichmentStrategy)
            else self.enrichment_strategy,
            "first_tokens_count": self.first_tokens_count,
            "include_filename": self.include_filename,
            "include_upload_date": self.include_upload_date,
            "include_file_type": self.include_file_type,
            "summary_model": self.summary_model,
            "summary_max_tokens": self.summary_max_tokens,
            "summary_prompt": self.summary_prompt,
            "hybrid_llm_threshold_kb": self.hybrid_llm_threshold_kb,
            "max_documents_in_context": self.max_documents_in_context,
            "cache_snippets": self.cache_snippets,
            "snippet_cache_ttl": self.snippet_cache_ttl,
        }


# =============================================================================
# TYPE ALIASES
# =============================================================================

# For backward compatibility and convenience
DocumentList = List[Document]
ChunkList = List[DocumentChunk]
CitationList = List[Citation]
