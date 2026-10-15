"""Lasagna: a framework for controlled RAG evaluation.

Each stage of a retrieval-augmented generation pipeline sits behind a fixed
interface, so one component can be swapped while everything else is held fixed
and the measured difference is attributable to the change. The same pipeline
serves documents to agents through an SDK, a REST API and an MCP server.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     PUBLIC API (this module)                     │
    ├─────────────────────────────────────────────────────────────────┤
    │  core/           │ config, schema, session management           │
    │  api/            │ RAGBackend - main orchestrator               │
    │  processing/     │ Document processor, OCR, semantic chunking   │
    │  retrieval/      │ Hybrid search, reranking, query enhancement  │
    │  infrastructure/ │ Embedding generation, caching                │
    │  vector_backends/│ Vector database abstraction (BYOVDB)         │
    │  agentic/        │ Multi-hop retrieval, citations, awareness    │
    │  providers/      │ Abstraction for standalone deployments       │
    │  utils/          │ Monitoring, logging, tokenization            │
    └─────────────────────────────────────────────────────────────────┘

Quick Start (host application):
    ```python
    from lasagna import RAGBackend, get_rag_backend

    backend = get_rag_backend()
    doc = backend.upload_document("file.pdf", session_id="user123")
    results = backend.search("query", session_id="user123")
    ```

Quick Start (Standalone/External agents):
    ```python
    from lasagna.standalone import create_rag_backend
    from pathlib import Path

    backend = create_rag_backend(
        db_dir=Path("/data/rag/dbs"),
        upload_dir=Path("/data/rag/uploads"),
        llm_provider="openai",
    )
    backend.upload_document("/path/to/doc.pdf", session_id="my_session")
    results = backend.search("query", session_id="my_session")
    ```

For most uses, prefer the async SDK: ``from lasagna.sdk import KnowledgeClient``.

Lazy imports: All public names are lazily loaded on first access to avoid
pulling heavy dependencies (langchain, sentence-transformers, pymilvus, etc.)
at package-import time.  This lets lightweight consumers (tests, CLI helpers)
``import lasagna`` without triggering the full dependency tree.

Logging: as a library, Lasagna stays silent unless asked. Its loguru messages
are disabled here and turned on by the ``lasagna`` and ``lasagna-mcp``
commands, the REST app, and the experiment CLI. To see them when using the
package from your own code, call ``loguru.logger.enable("lasagna")``.
"""

import importlib as _importlib
import os as _os
from typing import TYPE_CHECKING

from loguru import logger as _logger

_logger.disable("lasagna")

# ragas, the judge library behind the evaluation harness, posts a usage event
# (its version, the LLM provider and model name, a random id) to its authors
# each time a judge is built, unless RAGAS_DO_NOT_TRACK is "true". People run
# this on private corpora, so it is off unless they choose otherwise: set
# RAGAS_DO_NOT_TRACK=false before importing lasagna to turn it back on.
_os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# Mapping: public_name -> (submodule, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # One-line components: @lasagna.reranker above a plain function.
    "reranker": (".components", "reranker"),
    "query_enhancer": (".components", "query_enhancer"),
    "chunker": (".components", "chunker"),
    "embedder": (".components", "embedder"),
    "retriever": (".components", "retriever"),
    "document_processor": (".components", "document_processor"),
    "generator": (".components", "generator"),
    # A LangChain or LlamaIndex component as a stage.
    "from_langchain": (".sdk.integrations.inbound", "from_langchain"),
    "from_llamaindex": (".sdk.integrations.inbound", "from_llamaindex"),
    # Core
    "RAGConfig": (".core", "RAGConfig"),
    "get_rag_config": (".core", "get_rag_config"),
    "Document": (".core", "Document"),
    "DocumentChunk": (".core", "DocumentChunk"),
    "DocumentStatus": (".core", "DocumentStatus"),
    "DocumentMetadata": (".core", "DocumentMetadata"),
    "KnowledgeConfigManager": (".core", "KnowledgeConfigManager"),
    "get_knowledge_config_manager": (".core", "get_knowledge_config_manager"),
    # API
    "RAGBackend": (".api", "RAGBackend"),
    "get_rag_backend": (".api", "get_rag_backend"),
    "initialize": (".api", "initialize"),
    "reset_instance": (".api", "reset_instance"),
    # Processing
    "DocumentProcessor": (".processing", "DocumentProcessor"),
    "OCRProcessor": (".processing", "OCRProcessor"),
    "get_ocr_processor": (".processing", "get_ocr_processor"),
    "is_scanned_pdf": (".processing", "is_scanned_pdf"),
    "SemanticChunker": (".processing", "SemanticChunker"),
    "chunk_document": (".processing", "chunk_document"),
    # Retrieval - Search
    "HybridSearcher": (".retrieval.search", "HybridSearcher"),
    "get_hybrid_searcher": (".retrieval.search", "get_hybrid_searcher"),
    "CombinedReranker": (".retrieval.search", "CombinedReranker"),
    "get_reranker": (".retrieval.search", "get_reranker"),
    # Retrieval - Enhancement
    "QueryEnhancer": (".retrieval.enhancement", "QueryEnhancer"),
    "get_query_enhancer": (".retrieval.enhancement", "get_query_enhancer"),
    "QueryClassifier": (".retrieval.enhancement", "QueryClassifier"),
    "get_query_classifier": (".retrieval.enhancement", "get_query_classifier"),
    "CoverageScorer": (".retrieval.enhancement", "CoverageScorer"),
    "get_coverage_scorer": (".retrieval.enhancement", "get_coverage_scorer"),
    # Infrastructure
    "EmbeddingManager": (".infrastructure.embedding", "EmbeddingManager"),
    "get_embedding_manager": (".infrastructure.embedding", "get_embedding_manager"),
    "SearchCache": (".infrastructure.cache", "SearchCache"),
    "get_search_cache": (".infrastructure.cache", "get_search_cache"),
    # Storage
    "VectorStoreAdapter": (".vector_backends", "VectorStoreAdapter"),
    "FilterExpression": (".vector_backends", "FilterExpression"),
    "get_vector_backend": (".vector_backends", "get_backend"),
    "get_available_backends": (".vector_backends", "get_available_backends"),
    "register_backend": (".vector_backends", "register_backend"),
    # Agentic
    "AgenticRAG": (".agentic", "AgenticRAG"),
    "AgenticRAGResult": (".agentic", "AgenticRAGResult"),
    "AgenticRAGMetrics": (".agentic", "AgenticRAGMetrics"),
    "HopData": (".agentic", "HopData"),
    "RetrievedChunkData": (".agentic", "RetrievedChunkData"),
    "search_knowledge_base": (".agentic", "search_knowledge_base"),
    "mark_chunk_for_citation": (".agentic", "mark_chunk_for_citation"),
    "estimate_tokens": (".agentic", "estimate_tokens"),
    "refine_chunk": (".agentic", "refine_chunk"),
    "CitationManager": (".agentic", "CitationManager"),
    "Citation": (".agentic", "Citation"),
    "get_citation_manager": (".agentic", "get_citation_manager"),
    "DocumentInfo": (".agentic", "DocumentInfo"),
    "DocumentSnippet": (".agentic", "DocumentSnippet"),
    "get_document_awareness_context": (".agentic", "get_document_awareness_context"),
    # Standalone
    "create_rag_backend": (".standalone", "create_rag_backend"),
    "create_agentic_rag": (".standalone", "create_agentic_rag"),
    # Utilities & Observability
    "METRICS": (".utils", "METRICS"),
    "TRACER": (".utils", "TRACER"),
    "configure_metrics": (".utils", "configure_metrics"),
    "configure_tracing": (".utils", "configure_tracing"),
    "log_json": (".utils", "log_json"),
    "hash_text": (".utils", "hash_text"),
    "trunc_text": (".utils", "trunc_text"),
    "resolve_model_settings": (".utils", "resolve_model_settings"),
}

# The same names, imported for type checkers only (never at runtime). Resolved
# lazily through __getattr__, every one of them was `Any` to a user's type
# checker. tests/test_lazy_names_are_typed.py keeps this block in step with
# _LAZY_IMPORTS.
if TYPE_CHECKING:
    from .components import (
        reranker,
        query_enhancer,
        chunker,
        embedder,
        retriever,
        document_processor,
        generator,
    )
    from .sdk.integrations.inbound import from_langchain, from_llamaindex
    from .core import (
        RAGConfig,
        get_rag_config,
        Document,
        DocumentChunk,
        DocumentStatus,
        DocumentMetadata,
        KnowledgeConfigManager,
        get_knowledge_config_manager,
    )
    from .api import (
        RAGBackend,
        get_rag_backend,
        initialize,
        reset_instance,
    )
    from .processing import (
        DocumentProcessor,
        OCRProcessor,
        get_ocr_processor,
        is_scanned_pdf,
        SemanticChunker,
        chunk_document,
    )
    from .retrieval.search import (
        HybridSearcher,
        get_hybrid_searcher,
        CombinedReranker,
        get_reranker,
    )
    from .retrieval.enhancement import (
        QueryEnhancer,
        get_query_enhancer,
        QueryClassifier,
        get_query_classifier,
        CoverageScorer,
        get_coverage_scorer,
    )
    from .infrastructure.embedding import (
        EmbeddingManager,
        get_embedding_manager,
    )
    from .infrastructure.cache import (
        SearchCache,
        get_search_cache,
    )
    from .vector_backends import (
        VectorStoreAdapter,
        FilterExpression,
        get_backend as get_vector_backend,
        get_available_backends,
        register_backend,
    )
    from .agentic import (
        AgenticRAG,
        AgenticRAGResult,
        AgenticRAGMetrics,
        HopData,
        RetrievedChunkData,
        search_knowledge_base,
        mark_chunk_for_citation,
        estimate_tokens,
        refine_chunk,
        CitationManager,
        Citation,
        get_citation_manager,
        DocumentInfo,
        DocumentSnippet,
        get_document_awareness_context,
    )
    from .standalone import (
        create_rag_backend,
        create_agentic_rag,
    )
    from .utils import (
        METRICS,
        TRACER,
        configure_metrics,
        configure_tracing,
        log_json,
        hash_text,
        trunc_text,
        resolve_model_settings,
    )


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | globals().keys())


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # One-line components
    "reranker",
    "query_enhancer",
    "chunker",
    "embedder",
    "retriever",
    "document_processor",
    "generator",
    "from_langchain",
    "from_llamaindex",
    # Core
    "RAGConfig",
    "get_rag_config",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "DocumentMetadata",
    "KnowledgeConfigManager",
    "get_knowledge_config_manager",
    # API
    "RAGBackend",
    "get_rag_backend",
    "initialize",
    "reset_instance",
    # Processing
    "DocumentProcessor",
    "OCRProcessor",
    "get_ocr_processor",
    "is_scanned_pdf",
    "SemanticChunker",
    "chunk_document",
    # Retrieval - Search
    "HybridSearcher",
    "get_hybrid_searcher",
    "CombinedReranker",
    "get_reranker",
    # Retrieval - Enhancement
    "QueryEnhancer",
    "get_query_enhancer",
    "QueryClassifier",
    "get_query_classifier",
    "CoverageScorer",
    "get_coverage_scorer",
    # Infrastructure
    "EmbeddingManager",
    "get_embedding_manager",
    "SearchCache",
    "get_search_cache",
    # Storage
    "VectorStoreAdapter",
    "FilterExpression",
    "get_vector_backend",
    "get_available_backends",
    "register_backend",
    # Agentic
    "AgenticRAG",
    "AgenticRAGResult",
    "AgenticRAGMetrics",
    "HopData",
    "RetrievedChunkData",
    "search_knowledge_base",
    "mark_chunk_for_citation",
    "estimate_tokens",
    "refine_chunk",
    "CitationManager",
    "Citation",
    "get_citation_manager",
    "DocumentInfo",
    "DocumentSnippet",
    "get_document_awareness_context",
    # Standalone
    "create_rag_backend",
    "create_agentic_rag",
    # Utilities & Observability
    "METRICS",
    "TRACER",
    "configure_metrics",
    "configure_tracing",
    "log_json",
    "hash_text",
    "trunc_text",
    "resolve_model_settings",
]
# The version, at the import path everyone tries first.
#
# `lasagna.__version__` raised AttributeError while `lasagna.sdk.__version__`
# worked -- so the obvious way to report which version produced a result was the
# one that did not exist.
from lasagna.sdk._version import __version__, __version_info__  # noqa: E402,F401
