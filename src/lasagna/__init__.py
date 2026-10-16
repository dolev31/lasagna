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
import warnings as _warnings
from typing import TYPE_CHECKING

from loguru import logger as _logger

_logger.disable("lasagna")

# ragas, the judge library behind the evaluation harness, posts a usage event
# (its version, the LLM provider and model name, a random id) to its authors
# each time a judge is built, unless RAGAS_DO_NOT_TRACK is "true". People run
# this on private corpora, so it is off unless they choose otherwise: set
# RAGAS_DO_NOT_TRACK=false before importing lasagna to turn it back on.
_os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# On macOS, newer transformers make torch log a note about a torch feature
# Lasagna never uses (see _quiet.hide_torch_redirects_note). Dropped here,
# before anything imports transformers.
from lasagna._quiet import hide_torch_redirects_note as _hide_torch_redirects_note  # noqa: E402

_hide_torch_redirects_note()

# The public surface: public_name -> (submodule, attribute). Everything here is
# in __all__ and in the TYPE_CHECKING block below.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # One-line components: @lasagna.reranker above a plain function.
    "reranker": (".components", "reranker"),
    "query_enhancer": (".components", "query_enhancer"),
    "chunker": (".components", "chunker"),
    "embedder": (".components", "embedder"),
    "retriever": (".components", "retriever"),
    "document_processor": (".components", "document_processor"),
    "generator": (".components", "generator"),
    "register": (".components", "register"),
    "registry": (".core.registry", "registry"),
    # A LangChain or LlamaIndex component as a stage.
    "from_langchain": (".sdk.integrations.inbound", "from_langchain"),
    "from_llamaindex": (".sdk.integrations.inbound", "from_llamaindex"),
    # Configuration and the SDK client
    "RAGConfig": (".core", "RAGConfig"),
    "KnowledgeClient": (".sdk", "KnowledgeClient"),
    "KnowledgeConfig": (".sdk", "KnowledgeConfig"),
    # Evaluation
    "Experiment": (".knowledge_evaluation_sdk.experiment", "Experiment"),
    "ExperimentConfig": (".knowledge_evaluation_sdk.experiment", "ExperimentConfig"),
    # Serving
    "RAGBackend": (".api", "RAGBackend"),
    "get_rag_backend": (".api", "get_rag_backend"),
    "create_rag_backend": (".standalone", "create_rag_backend"),
    # Vector stores
    "VectorStoreAdapter": (".vector_backends", "VectorStoreAdapter"),
    "register_backend": (".vector_backends", "register_backend"),
}

# Exported from here before 0.1.0 and still importable for one release, with a
# DeprecationWarning that names the module to import each one from. Most are
# internals (a chunker class, a tracer, a token estimator) that a top-level
# export presented as API.
_DEPRECATED_IMPORTS: dict[str, tuple[str, str]] = {
    "get_rag_config": (".core", "get_rag_config"),
    "Document": (".core", "Document"),
    "DocumentChunk": (".core", "DocumentChunk"),
    "DocumentStatus": (".core", "DocumentStatus"),
    "DocumentMetadata": (".core", "DocumentMetadata"),
    "KnowledgeConfigManager": (".core", "KnowledgeConfigManager"),
    "get_knowledge_config_manager": (".core", "get_knowledge_config_manager"),
    "initialize": (".api", "initialize"),
    "reset_instance": (".api", "reset_instance"),
    "DocumentProcessor": (".processing", "DocumentProcessor"),
    "OCRProcessor": (".processing", "OCRProcessor"),
    "get_ocr_processor": (".processing", "get_ocr_processor"),
    "is_scanned_pdf": (".processing", "is_scanned_pdf"),
    "SemanticChunker": (".processing", "SemanticChunker"),
    "chunk_document": (".processing", "chunk_document"),
    "HybridSearcher": (".retrieval.search", "HybridSearcher"),
    "get_hybrid_searcher": (".retrieval.search", "get_hybrid_searcher"),
    "CombinedReranker": (".retrieval.search", "CombinedReranker"),
    "get_reranker": (".retrieval.search", "get_reranker"),
    "QueryEnhancer": (".retrieval.enhancement", "QueryEnhancer"),
    "get_query_enhancer": (".retrieval.enhancement", "get_query_enhancer"),
    "QueryClassifier": (".retrieval.enhancement", "QueryClassifier"),
    "get_query_classifier": (".retrieval.enhancement", "get_query_classifier"),
    "CoverageScorer": (".retrieval.enhancement", "CoverageScorer"),
    "get_coverage_scorer": (".retrieval.enhancement", "get_coverage_scorer"),
    "EmbeddingManager": (".infrastructure.embedding", "EmbeddingManager"),
    "get_embedding_manager": (".infrastructure.embedding", "get_embedding_manager"),
    "SearchCache": (".infrastructure.cache", "SearchCache"),
    "get_search_cache": (".infrastructure.cache", "get_search_cache"),
    "FilterExpression": (".vector_backends", "FilterExpression"),
    "get_vector_backend": (".vector_backends", "get_backend"),
    "get_available_backends": (".vector_backends", "get_available_backends"),
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
    "create_agentic_rag": (".standalone", "create_agentic_rag"),
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
# _LAZY_IMPORTS. The deprecated names are left out, so a type checker flags them.
if TYPE_CHECKING:
    from .components import (
        reranker,
        query_enhancer,
        chunker,
        embedder,
        retriever,
        document_processor,
        generator,
        register,
    )
    from .sdk.integrations.inbound import from_langchain, from_llamaindex
    from .core.registry import registry
    from .core import RAGConfig
    from .sdk import KnowledgeClient, KnowledgeConfig
    from .knowledge_evaluation_sdk.experiment import Experiment, ExperimentConfig
    from .api import RAGBackend, get_rag_backend
    from .standalone import create_rag_backend
    from .vector_backends import VectorStoreAdapter, register_backend


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val  # cache for subsequent access
        return val
    if name in _DEPRECATED_IMPORTS:
        module_path, attr_name = _DEPRECATED_IMPORTS[name]
        mod = _importlib.import_module(module_path, __name__)
        # Not cached, so every use is reported. stacklevel=2 is the user's line
        # for `from lasagna import X` too: warn() skips importlib's frames.
        _warnings.warn(
            f"`lasagna.{name}` is deprecated and will be removed in 0.2.0. "
            f"Use `from {mod.__name__} import {attr_name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | globals().keys())


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [*_LAZY_IMPORTS, "__version__"]

# The version, at the import path everyone tries first.
#
# `lasagna.__version__` raised AttributeError while `lasagna.sdk.__version__`
# worked -- so the obvious way to report which version produced a result was the
# one that did not exist.
from lasagna.sdk._version import __version__, __version_info__  # noqa: E402,F401
