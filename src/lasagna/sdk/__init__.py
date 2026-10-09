"""Lasagna SDK - RAG capabilities for any application.

This SDK provides a clean, minimal API for knowledge base operations including
document ingestion, semantic search, and retrieval-augmented generation.

The SDK wraps the existing RAG backend with a simple, type-safe interface
following patterns from the main the host application SDK.

Quick Start:
    ```python
    from lasagna.sdk import KnowledgeClient

    # Create client with defaults
    client = KnowledgeClient()

    # Upload a document
    result = await client.upload("./document.pdf")
    print(f"Uploaded: {result.document_id}")

    # Search
    results = await client.search_documents("What is the revenue?")
    for r in results:
        print(f"[{r.score:.2f}] {r.chunk.content[:100]}")

    # Get context for LLM injection
    context = await client.get_context("key findings")
    ```

With Configuration:
    ```python
    from lasagna.sdk import (
        KnowledgeClient,
        KnowledgeConfig,
        ChunkingStrategy,
    )

    # Custom configuration
    config = KnowledgeConfig(
        enable_reranking=True,
        enable_hyde=True,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=800,
    )
    client = KnowledgeClient(config=config)

    # Or use presets
    from lasagna.sdk import config_for_quality
    client = KnowledgeClient(config=config_for_quality())
    ```

LangChain Integration:
    ```python
    from lasagna.sdk import KnowledgeClient
    from lasagna.sdk.integrations import KnowledgeRetriever

    client = KnowledgeClient()
    retriever = KnowledgeRetriever(client, top_k=5)

    # Use with LangChain chains
    docs = await retriever.ainvoke("query")
    ```

Standalone Usage (standalone):
    ```python
    from lasagna.sdk import (
        KnowledgeClient,
        config_for_standalone,
    )

    client = KnowledgeClient(
        config=config_for_standalone(
            db_path="./my_data/vectors.db",
            upload_dir="./my_data/uploads",
        )
    )
    ```

For comprehensive documentation, see:
- README.md - SDK overview and documentation index
"""

import importlib
import re

from ._version import __version__, __version_info__

# =============================================================================
# TIER 1: Always imported (lightweight — no heavy ML dependencies)
# =============================================================================

from .client import KnowledgeClient

from .config import (
    KnowledgeConfig,
    config_for_speed,
    config_for_standard,
    config_for_balanced,
    config_for_max_quality,
    config_for_quality,  # backward compat alias for config_for_max_quality
    config_for_standalone,
    config_for_production,
)

from .types import (
    # Enums
    ChunkingStrategy,
    EmbeddingProvider,
    OCREngine,
    HyDEStrategy,
    ChunkRetrievalStrategy,
    EnrichmentStrategy,
    CitationFormat,
    CitationScoreMethod,
    VectorBackend,
    DocumentStatus,
    # Dataclasses
    Document,
    DocumentChunk,
    SearchResult,
    SearchResults,
    Citation,
    UploadResult,
    HopData,
    AgenticSearchResults,
    ChunkCiteRetrieval,
    DocumentAwareness,
)

from .exceptions import (
    # Base
    KnowledgeError,
    # Document
    DocumentError,
    DocumentNotFoundError,
    UploadError,
    ChunkingError,
    ExtractionError,
    # Search
    SearchError,
    QueryError,
    EmbeddingError,
    RetrievalError,
    # Citation
    CitationError,
    CitationNotFoundError,
    # Configuration
    ConfigurationError,
    ProfileNotFoundError,
    # Backend
    BackendError,
    BackendConnectionError,
    BackendTimeoutError,
    # Agentic
    AgenticError,
    MaxIterationsError,
    # Evaluation
    EvaluationError,
    JudgeError,
    # Integration
    IntegrationError,
)

# =============================================================================
# TIER 2+: Lazy-loaded via __getattr__ (heavy deps loaded on first access)
#
# These imports pull in torch, sentence-transformers, pymilvus, etc.
# By deferring them, `from lasagna.sdk import KnowledgeClient`
# stays fast and lightweight.
# =============================================================================

_LAZY_IMPORTS = {
    # Primary (MCP client)
    "MCPKnowledgeClient": (".integrations.mcp_client", "MCPKnowledgeClient"),
    # Managers
    "DocumentManager": (".managers", "DocumentManager"),
    "SearchManager": (".managers", "SearchManager"),
    "ConfigManager": (".managers", "ConfigManager"),
    "CitationManager": (".managers", "CitationManager"),
    # Backends
    "get_backend": (".backends", "get_backend"),
    "list_backends": (".backends", "list_backends"),
    "get_available_backends": (".backends", "get_available_backends"),
    "register_backend": (".backends", "register_backend"),
    "VectorBackendBase": (".backends", "VectorBackendBase"),
    "BackendCapabilities": (".backends", "BackendCapabilities"),
    "MilvusBackend": (".backends", "MilvusBackend"),
    "PineconeBackend": (".backends", "PineconeBackend"),
    "QdrantBackend": (".backends", "QdrantBackend"),
    "WeaviateBackend": (".backends", "WeaviateBackend"),
    "FilterExpression": (".backends", "FilterExpression"),
    # Processing
    "SemanticChunker": (".processing", "SemanticChunker"),
    "FixedChunker": (".processing", "FixedChunker"),
    "DocumentExtractor": (".processing", "DocumentExtractor"),
    "OCRProcessor": (".processing", "OCRProcessor"),
    "chunk_document": (".processing", "chunk_document"),
    # Retrieval
    "EmbeddingModel": (".retrieval", "EmbeddingModel"),
    "HybridSearcher": (".retrieval", "HybridSearcher"),
    "Reranker": (".retrieval", "Reranker"),
    "QueryEnhancer": (".retrieval", "QueryEnhancer"),
    "QueryClassifier": (".retrieval", "QueryClassifier"),
    "CoverageScorer": (".retrieval", "CoverageScorer"),
    # Agentic
    "AgenticRAGWrapper": (".agentic", "AgenticRAGWrapper"),
    "create_rag_tools": (".agentic", "create_rag_tools"),
    "AgenticRAG": (".agentic", "AgenticRAG"),
    # Evaluation
    "KnowledgeEvaluator": (".evaluation", "KnowledgeEvaluator"),
    "EvalConfig": (".evaluation", "EvalConfig"),
    "EvalResult": (".evaluation", "EvalResult"),
    # Component Registry (for researchers)
    "ComponentRegistry": (".component_registry", "ComponentRegistry"),
    "component_registry": (".component_registry", "component_registry"),
    "get_component_registry": (".component_registry", "get_component_registry"),
    # Base Components (for researchers to extend)
    "BaseEmbedder": ("lasagna.core.base_components", "BaseEmbedder"),
    "BaseChunker": ("lasagna.core.base_components", "BaseChunker"),
    "BaseReranker": ("lasagna.core.base_components", "BaseReranker"),
    "BaseRetriever": ("lasagna.core.base_components", "BaseRetriever"),
    "BaseDocumentProcessor": ("lasagna.core.base_components", "BaseDocumentProcessor"),
    "BaseQueryEnhancer": ("lasagna.core.base_components", "BaseQueryEnhancer"),
    "BaseAgenticPipeline": ("lasagna.core.base_components", "BaseAgenticPipeline"),
    "AgenticPipelineInput": ("lasagna.core.base_components", "AgenticPipelineInput"),
    "ScoredDocument": ("lasagna.core.base_components", "ScoredDocument"),
    "ProcessedDocument": ("lasagna.core.base_components", "ProcessedDocument"),
    # Pipeline (for researchers)
    "KnowledgePipeline": ("lasagna.core.pipeline", "KnowledgePipeline"),
    "PipelineResult": ("lasagna.core.pipeline", "PipelineResult"),
    # Registry singleton
    "registry": ("lasagna.core.registry", "registry"),
    # RAGAS Evaluation
    "RAGASEvaluator": ("lasagna.knowledge_evaluation_sdk.ragas_evaluator", "RAGASEvaluator"),
    # Ablation Studies
    "sweep": ("lasagna.knowledge_evaluation_sdk.ablation", "sweep"),
    "sweep_pipelines": ("lasagna.knowledge_evaluation_sdk.ablation", "sweep_pipelines"),
    "sweep_nodes": ("lasagna.knowledge_evaluation_sdk.ablation", "sweep_nodes"),
    "AblationResults": ("lasagna.knowledge_evaluation_sdk.ablation", "AblationResults"),
    "grid_search": ("lasagna.knowledge_evaluation_sdk.ablation", "grid_search"),
    # IR Metrics
    "ndcg_at_k": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "ndcg_at_k"),
    "precision_at_k": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "precision_at_k"),
    "recall_at_k": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "recall_at_k"),
    "f1_at_k": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "f1_at_k"),
    "hit_rate_at_k": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "hit_rate_at_k"),
    "mrr": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "mrr"),
    "map_score": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "map_score"),
    "corpus_metrics": ("lasagna.knowledge_evaluation_sdk.ir_metrics", "corpus_metrics"),
    # Benchmark Datasets
    "load_benchmark": ("lasagna.knowledge_evaluation_sdk.datasets", "load_benchmark"),
    "list_benchmarks": ("lasagna.knowledge_evaluation_sdk.datasets", "list_benchmarks"),
    # Benchmark Framework
    "run_benchmark": ("lasagna.knowledge_evaluation_sdk.experiment", "run_benchmark"),
    "Experiment": ("lasagna.knowledge_evaluation_sdk.experiment", "Experiment"),
    "ExperimentConfig": ("lasagna.knowledge_evaluation_sdk.experiment", "ExperimentConfig"),
    "BenchmarkResults": ("lasagna.knowledge_evaluation_sdk.results", "BenchmarkResults"),
    "ReportGenerator": ("lasagna.knowledge_evaluation_sdk.reporting", "ReportGenerator"),
    "BaseBenchmarkAdapter": ("lasagna.knowledge_evaluation_sdk.benchmarks", "BaseBenchmarkAdapter"),
    "benchmark_registry": ("lasagna.knowledge_evaluation_sdk.benchmarks", "benchmark_registry"),
    "BenchmarkInfo": ("lasagna.knowledge_evaluation_sdk.benchmarks", "BenchmarkInfo"),
    "BenchmarkSample": ("lasagna.knowledge_evaluation_sdk.benchmarks", "BenchmarkSample"),
    # Component Test Harness
    "ComponentTestHarness": ("lasagna.core.testing", "ComponentTestHarness"),
    "ValidationReport": ("lasagna.core.testing", "ValidationReport"),
    # Data types from base components
    "Chunk": ("lasagna.core.base_components", "Chunk"),
    # Contextual Retrieval
    "ContextualEnricher": (
        "lasagna.processing.contextual.contextual_enricher",
        "ContextualEnricher",
    ),
    # Storage Backends
    "get_storage_backend": ("lasagna.storage", "get_storage_backend"),
    "register_storage_backend": ("lasagna.storage", "register_storage_backend"),
    "get_available_storage_backends": ("lasagna.storage", "get_available_storage_backends"),
    "StorageBackend": ("lasagna.storage", "StorageBackend"),
    "StorageCapabilities": ("lasagna.storage", "StorageCapabilities"),
    "StorageFileInfo": ("lasagna.storage", "StorageFileInfo"),
    "LocalStorageBackend": ("lasagna.storage", "LocalStorageBackend"),
    # Utilities
    "with_retry": (".utils.retry", "with_retry"),
    "with_retry_sync": (".utils.retry", "with_retry_sync"),
    "RetryConfig": (".utils.retry", "RetryConfig"),
    "RETRY_BACKEND": (".utils.retry", "RETRY_BACKEND"),
    "RETRY_SEARCH": (".utils.retry", "RETRY_SEARCH"),
}


#: Which extra provides which optional dependency, so the error can say what to
#: install rather than only what is missing.
_EXTRA_FOR_MODULE = (
    ("knowledge_evaluation_sdk", "eval"),
    ("ragas", "eval"),
    ("datasets", "eval"),
    ("pinecone", "vector-backends"),
    ("weaviate", "vector-backends"),
    ("qdrant", "vector-backends"),
    ("opensearch", "opensearch"),
    ("pgvector", "pgvector"),
    ("psycopg", "pgvector"),
)


class _MissingDependency:
    """Placeholder for a name whose optional dependency is not installed.

    Importing it succeeds; using it raises with the package that is missing and
    the extra that provides it. That ordering matters: the names in this table
    are re-exported by documentation and IDEs, and an import that raises would
    make `from lasagna.sdk import X` fail for anyone who only wanted to read the
    signature.
    """

    __slots__ = ("_name", "_module_path", "_cause")

    def __init__(self, name: str, module_path: str, cause: BaseException) -> None:
        self._name = name
        self._module_path = module_path
        self._cause = cause

    def _hint(self) -> str:
        extra = next(
            (e for marker, e in _EXTRA_FOR_MODULE if marker in self._module_path),
            None,
        )
        # ModuleNotFoundError carries `.name`; a plain ImportError raised from
        # inside a package does not, so fall back to parsing the message.
        missing = getattr(self._cause, "name", None)
        if not missing:
            match = re.search(r"No module named '([^']+)'", str(self._cause))
            missing = match.group(1) if match else "an optional dependency"
        install = f'pip install "lasagna-rag[{extra}]"' if extra else f"pip install {missing}"
        return f"'{self._name}' needs {missing}, which is not installed. Install it with: {install}"

    def __call__(self, *args, **kwargs):
        raise ImportError(self._hint()) from self._cause

    def __getattr__(self, attr: str):
        raise ImportError(self._hint()) from self._cause

    def __repr__(self) -> str:  # helpful in a traceback or a REPL
        return f"<unavailable: {self._name} ({self._hint()})>"

    def __bool__(self) -> bool:
        # Falsy, so `if lasagna.sdk.run_benchmark:` still reads as "absent"
        # for anyone who was relying on the old None.
        return False


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        try:
            mod = importlib.import_module(module_path, __package__)
            val = getattr(mod, attr)
        except ImportError as exc:
            # Optional dependency not installed (e.g. pymilvus, pinecone).
            #
            # Binding None here let `from lasagna.sdk import run_benchmark`
            # succeed without the `eval` extra and fail later with
            # "'NoneType' object is not callable" -- a message that names
            # neither the missing package nor the extra that provides it. The
            # sentinel keeps the import working (so `hasattr` and introspection
            # behave) and raises something actionable at the point of use.
            import logging

            logging.getLogger(__name__).debug(
                "Optional import %s unavailable (missing dependency for %s)",
                name,
                module_path,
            )
            val = _MissingDependency(name, module_path, exc)
        except AttributeError:
            # Module found but attribute missing — likely a bug in the lazy-import table.
            raise AttributeError(
                f"{module_path!r} has no attribute {attr!r}. "
                f"Check the _LAZY_IMPORTS mapping for '{name}'."
            )
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Primary
    "KnowledgeClient",
    "MCPKnowledgeClient",
    # Config
    "KnowledgeConfig",
    "config_for_speed",
    "config_for_standard",
    "config_for_balanced",
    "config_for_max_quality",
    "config_for_quality",  # backward compat alias
    "config_for_standalone",
    "config_for_production",
    # Types - Enums
    "ChunkingStrategy",
    "EmbeddingProvider",
    "OCREngine",
    "HyDEStrategy",
    "ChunkRetrievalStrategy",
    "EnrichmentStrategy",
    "CitationFormat",
    "CitationScoreMethod",
    "VectorBackend",
    "DocumentStatus",
    # Types - Dataclasses
    "Document",
    "DocumentChunk",
    "SearchResult",
    "SearchResults",
    "Citation",
    "UploadResult",
    "HopData",
    "AgenticSearchResults",
    "ChunkCiteRetrieval",
    "DocumentAwareness",
    # Managers
    "DocumentManager",
    "SearchManager",
    "ConfigManager",
    "CitationManager",
    # Backends
    "get_backend",
    "list_backends",
    "get_available_backends",
    "register_backend",
    "VectorBackendBase",
    "BackendCapabilities",
    "MilvusBackend",
    "PineconeBackend",
    "QdrantBackend",
    "WeaviateBackend",
    "FilterExpression",
    # Processing
    "SemanticChunker",
    "FixedChunker",
    "DocumentExtractor",
    "OCRProcessor",
    "chunk_document",
    # Retrieval
    "EmbeddingModel",
    "HybridSearcher",
    "Reranker",
    "QueryEnhancer",
    "QueryClassifier",
    "CoverageScorer",
    # Agentic
    "AgenticRAGWrapper",
    "create_rag_tools",
    "AgenticRAG",
    # Evaluation
    "KnowledgeEvaluator",
    "EvalConfig",
    "EvalResult",
    # Exceptions
    "KnowledgeError",
    "DocumentError",
    "DocumentNotFoundError",
    "UploadError",
    "ChunkingError",
    "ExtractionError",
    "SearchError",
    "QueryError",
    "EmbeddingError",
    "RetrievalError",
    "CitationError",
    "CitationNotFoundError",
    "ConfigurationError",
    "ProfileNotFoundError",
    "BackendError",
    "BackendConnectionError",
    "BackendTimeoutError",
    "AgenticError",
    "MaxIterationsError",
    "EvaluationError",
    "JudgeError",
    "IntegrationError",
    # Component Registry (for researchers)
    "ComponentRegistry",
    "component_registry",
    "get_component_registry",
    # Base Components (for researchers to extend)
    "BaseEmbedder",
    "BaseChunker",
    "BaseReranker",
    "BaseRetriever",
    "BaseDocumentProcessor",
    "BaseQueryEnhancer",
    "BaseAgenticPipeline",
    "AgenticPipelineInput",
    "ScoredDocument",
    "ProcessedDocument",
    # Pipeline (for researchers)
    "KnowledgePipeline",
    "PipelineResult",
    "registry",
    # RAGAS Evaluation
    "RAGASEvaluator",
    # Ablation Studies
    "sweep",
    "sweep_pipelines",
    "sweep_nodes",
    "AblationResults",
    "grid_search",
    # IR Metrics
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "hit_rate_at_k",
    "mrr",
    "map_score",
    "corpus_metrics",
    # Benchmark Datasets
    "load_benchmark",
    "list_benchmarks",
    # Benchmark Framework
    "run_benchmark",
    "Experiment",
    "ExperimentConfig",
    "BenchmarkResults",
    "ReportGenerator",
    "BaseBenchmarkAdapter",
    "benchmark_registry",
    "BenchmarkInfo",
    "BenchmarkSample",
    # Component Test Harness
    "ComponentTestHarness",
    "ValidationReport",
    # Data types
    "Chunk",
    # Contextual Retrieval
    "ContextualEnricher",
    # Storage Backends
    "get_storage_backend",
    "register_storage_backend",
    "get_available_storage_backends",
    "StorageBackend",
    "StorageCapabilities",
    "StorageFileInfo",
    "LocalStorageBackend",
    # Utilities
    "with_retry",
    "with_retry_sync",
    "RetryConfig",
    "RETRY_BACKEND",
    "RETRY_SEARCH",
]
