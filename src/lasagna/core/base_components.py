"""Base component ABCs for the RAG pipeline.

These abstract classes define the interface for each swappable component
in the pipeline. Researchers can implement custom versions of any component
and register them via the component registry.

Example:
    from lasagna.core.base_components import BaseReranker, ScoredDocument
    from lasagna.core.registry import registry

    @registry.register("reranker", "my_novel_reranker")
    class MyNovelReranker(BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            # Custom reranking logic
            return sorted(documents, key=lambda d: d.score, reverse=True)[:top_k]

    # The registry is also available from:
    #   from lasagna.retrieval.registry import component_registry
    #   from lasagna.sdk import component_registry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lasagna.agentic.agentic_rag import AgenticRAGResult
    from lasagna.retrieval.search.ranker import RerankedResult


# ---------------------------------------------------------------------------
# Data classes for component I/O
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A document chunk produced by a chunker.

    This is the canonical Chunk definition used across the RAG pipeline.
    All fields are optional except ``content`` to support both simple
    fixed-size chunking and advanced structure-aware strategies.
    """

    content: str
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_number: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    section_header: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    chunk_type: str = "text"


@dataclass
class ScoredDocument:
    """A document with a relevance score (from retrieval or reranking)."""

    content: str
    score: float
    document_id: str = ""
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """A document after extraction/processing."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: Optional[List[str]] = None
    tables: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Base component ABCs
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Abstract base class for embedding models.

    Implement this to integrate a custom embedding model into the pipeline.
    """

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of texts to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        ...

    @abstractmethod
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """Encode a single query (may apply query-specific prefixing).

        Args:
            query: Search query text

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this embedder."""
        return self.__class__.__name__

    # Non-abstract convenience methods used by HyDE multi-doc flow.
    # Override in subclasses for optimised batch encoding.

    def encode_documents_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[np.ndarray]:
        """Encode multiple documents. Default delegates to encode()."""
        result = self.encode(texts)
        return [result[i] for i in range(len(texts))]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        dot = float(np.dot(embedding1, embedding2))
        norm1 = float(np.linalg.norm(embedding1))
        norm2 = float(np.linalg.norm(embedding2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm1 * norm2)))


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Full document text
            **kwargs: Strategy-specific parameters (chunk_size, overlap, etc.)

        Returns:
            List of Chunk objects
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseReranker(ABC):
    """Abstract base class for reranking models."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        """Rerank documents by relevance to the query.

        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of results to return

        Returns:
            Reranked list of ScoredDocument (sorted by score descending)
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            collection: Collection/index name to search
            top_k: Number of results to return

        Returns:
            List of ScoredDocument (sorted by score descending)
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors/extractors."""

    @abstractmethod
    def extract(self, file_path: str, **kwargs) -> ProcessedDocument:
        """Extract text and metadata from a file.

        Args:
            file_path: Path to the document file

        Returns:
            ProcessedDocument with extracted text and metadata
        """
        ...

    @property
    def supported_extensions(self) -> List[str]:
        """File extensions this processor can handle."""
        return []

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseQueryEnhancer(ABC):
    """Abstract base class for query enhancement strategies."""

    @abstractmethod
    def enhance(self, query: str, **kwargs) -> List[str]:
        """Enhance/expand a query into one or more variations.

        Args:
            query: Original search query

        Returns:
            List of enhanced query strings (may include the original)
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseGenerator(ABC):
    """Abstract base class for answer generation: the last stage.

    Receives the question and the contexts the earlier stages selected, and
    returns the answer text. ``llm`` is the run's generator model (a LangChain
    chat model, called with ``invoke``); a strategy that brings its own model
    may ignore it. ``instructions`` carries harness-level requirements the
    strategy must honour, such as permission to abstain on a benchmark with
    unanswerable questions.
    """

    @abstractmethod
    def generate(
        self, query: str, contexts: List[str], *, llm: Any = None, instructions: str = "", **kwargs
    ) -> str:
        """Answer ``query`` from ``contexts``.

        Args:
            query: The question.
            contexts: Passage texts, best first.
            llm: The generator model, when the strategy uses the run's model.
            instructions: Extra requirements to include in the prompt.

        Returns:
            The answer text.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Agentic pipeline ABC
# ---------------------------------------------------------------------------


@dataclass
class AgenticPipelineInput:
    """Standardized input for agentic RAG pipelines.

    Passed to ``BaseAgenticPipeline.run()`` with all context needed to
    execute a retrieval pipeline.
    """

    question: str
    session_id: str
    top_k: int = 5
    document_filter: Optional[List[str]] = None
    trace_id: Optional[str] = None
    routing_decision: Any = None
    session_config: Any = None  # RAGConfig instance
    llm: Any = None  # BaseChatModel instance


class BaseAgenticPipeline(ABC):
    """Abstract base class for pluggable agentic RAG pipelines.

    Implement this to register a completely custom agentic retrieval
    strategy.  The pipeline receives a standardized input and must
    produce an ``AgenticRAGResult``.

    Example::

        from lasagna.core.base_components import (
            BaseAgenticPipeline, AgenticPipelineInput,
        )
        from lasagna.core.registry import registry

        @registry.register("agentic_pipeline", "my_custom_rag")
        class MyCustomRAG(BaseAgenticPipeline):
            def __init__(self, session_config=None, **kwargs):
                self.config = session_config

            async def run(self, input: AgenticPipelineInput) -> "AgenticRAGResult":
                # Custom retrieval logic ...
                from lasagna.agentic.agentic_rag import AgenticRAGResult
                return AgenticRAGResult(
                    formatted_context="...",
                    chunks=[...],
                    citations=[...],
                    agent_summary="...",
                    search_count=1,
                )
    """

    @abstractmethod
    async def run(self, input: AgenticPipelineInput) -> "AgenticRAGResult":
        """Execute the agentic retrieval pipeline.

        Args:
            input: Standardized pipeline input with question, config,
                LLM, etc.

        Returns:
            ``AgenticRAGResult`` with formatted context, chunks, citations,
            hop data, and metrics.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Conversion utilities: bridge between public ABC types and internal types
# ---------------------------------------------------------------------------


def scored_document_from_dict(d: Dict[str, Any]) -> ScoredDocument:
    """Convert an internal search result dict to a ``ScoredDocument``.

    Handles the various dict layouts produced by vector backends
    (Milvus, pgvector, etc.) and the search orchestrator.

    Uses explicit ``is not None`` checks so that falsy values like
    ``score=0.0`` or ``content=""`` are preserved correctly.
    """
    # Content: prefer top-level, fall back to nested entity, then repr
    content = d.get("content")
    if content is None:
        content = (d.get("entity") or {}).get("content")
    if content is None:
        content = str(d)

    # Score: first non-None value wins (0.0 is a valid score)
    score_raw = d.get("score")
    if score_raw is None:
        score_raw = d.get("distance")
    if score_raw is None:
        score_raw = d.get("combined_score")
    if score_raw is None:
        score_raw = 0.0
    score = float(score_raw)

    # Chunk ID: prefer "id", fall back to "chunk_id"
    chunk_id_raw = d.get("id")
    if chunk_id_raw is None:
        chunk_id_raw = d.get("chunk_id", "")

    metadata = d.get("metadata")

    return ScoredDocument(
        content=content,
        score=score,
        document_id=d.get("document_id", ""),
        chunk_id=str(chunk_id_raw),
        metadata=metadata if metadata is not None else {},
    )


def scored_document_to_dict(sd: ScoredDocument) -> Dict[str, Any]:
    """Convert a ``ScoredDocument`` to the internal dict format."""
    return {
        "content": sd.content,
        "score": sd.score,
        "document_id": sd.document_id,
        "id": sd.chunk_id,
        "chunk_id": sd.chunk_id,
        "metadata": sd.metadata,
    }


def scored_document_from_reranked(r: RerankedResult) -> ScoredDocument:
    """Convert a ``RerankedResult`` to a ``ScoredDocument``."""
    return ScoredDocument(
        content=r.content,
        score=r.normalized_score,
        document_id=r.document_id,
        chunk_id=r.chunk_id,
        metadata=r.metadata,
    )
