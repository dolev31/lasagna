"""The public API is a promise; this test is where it is written down.

`lasagna.__all__` and `lasagna.sdk.__all__` are what users import. Today either
could gain or lose a name without anything failing, and the first anyone would
know is a downstream ImportError after upgrading.

These lists are not a style rule. Changing one is a deliberate act: adding a
name is a new public commitment, removing one is a breaking change that needs a
major version and a changelog entry. Update the list in the same commit that
changes the export, so the diff shows the API moving.
"""

import pytest

import lasagna
import lasagna.sdk


EXPECTED_LASAGNA = {
    # One-line components (lasagna.components)
    "chunker",
    "document_processor",
    "generator",
    "embedder",
    "query_enhancer",
    "reranker",
    "retriever",
    # LangChain and LlamaIndex components as stages (lasagna.sdk.integrations.inbound)
    "from_langchain",
    "from_llamaindex",
    "AgenticRAG",
    "AgenticRAGMetrics",
    "AgenticRAGResult",
    "Citation",
    "CitationManager",
    "CombinedReranker",
    "CoverageScorer",
    "Document",
    "DocumentChunk",
    "DocumentInfo",
    "DocumentMetadata",
    "DocumentProcessor",
    "DocumentSnippet",
    "DocumentStatus",
    "EmbeddingManager",
    "FilterExpression",
    "HopData",
    "HybridSearcher",
    "KnowledgeConfigManager",
    "METRICS",
    "OCRProcessor",
    "QueryClassifier",
    "QueryEnhancer",
    "RAGBackend",
    "RAGConfig",
    "RetrievedChunkData",
    "SearchCache",
    "SemanticChunker",
    "TRACER",
    "VectorStoreAdapter",
    "chunk_document",
    "configure_metrics",
    "configure_tracing",
    "create_agentic_rag",
    "create_rag_backend",
    "estimate_tokens",
    "get_available_backends",
    "get_citation_manager",
    "get_coverage_scorer",
    "get_document_awareness_context",
    "get_embedding_manager",
    "get_hybrid_searcher",
    "get_knowledge_config_manager",
    "get_ocr_processor",
    "get_query_classifier",
    "get_query_enhancer",
    "get_rag_backend",
    "get_rag_config",
    "get_reranker",
    "get_search_cache",
    "get_vector_backend",
    "hash_text",
    "initialize",
    "is_scanned_pdf",
    "log_json",
    "mark_chunk_for_citation",
    "refine_chunk",
    "register_backend",
    "reset_instance",
    "resolve_model_settings",
    "search_knowledge_base",
    "trunc_text",
}

EXPECTED_SDK = {
    "AblationResults",
    "AgenticError",
    "AgenticPipelineInput",
    "AgenticRAG",
    "AgenticRAGWrapper",
    "AgenticSearchResults",
    "BackendCapabilities",
    "BackendConnectionError",
    "BackendError",
    "BackendTimeoutError",
    "BaseAgenticPipeline",
    "BaseBenchmarkAdapter",
    "BaseChunker",
    "BaseDocumentProcessor",
    "BaseEmbedder",
    "BaseQueryEnhancer",
    "BaseReranker",
    "BaseRetriever",
    "BenchmarkInfo",
    "BenchmarkResults",
    "BenchmarkSample",
    "Chunk",
    "ChunkCiteRetrieval",
    "ChunkRetrievalStrategy",
    "ChunkingError",
    "ChunkingStrategy",
    "Citation",
    "CitationError",
    "CitationFormat",
    "CitationManager",
    "CitationNotFoundError",
    "CitationScoreMethod",
    "ComponentRegistry",
    "ComponentTestHarness",
    "ConfigManager",
    "ConfigurationError",
    "ContextualEnricher",
    "CoverageScorer",
    "Document",
    "DocumentAwareness",
    "DocumentChunk",
    "DocumentError",
    "DocumentExtractor",
    "DocumentManager",
    "DocumentNotFoundError",
    "DocumentStatus",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingProvider",
    "EnrichmentStrategy",
    "EvalConfig",
    "EvalResult",
    "EvaluationError",
    "Experiment",
    "ExperimentConfig",
    "ExtractionError",
    "FilterExpression",
    "FixedChunker",
    "HopData",
    "HyDEStrategy",
    "HybridSearcher",
    "IntegrationError",
    "JudgeError",
    "KnowledgeClient",
    "KnowledgeConfig",
    "KnowledgeError",
    "KnowledgeEvaluator",
    "KnowledgePipeline",
    "LocalStorageBackend",
    "MCPKnowledgeClient",
    "MaxIterationsError",
    "MilvusBackend",
    "OCREngine",
    "OCRProcessor",
    "PineconeBackend",
    "PipelineResult",
    "ProcessedDocument",
    "ProfileNotFoundError",
    "QdrantBackend",
    "QueryClassifier",
    "QueryEnhancer",
    "QueryError",
    "RAGASEvaluator",
    "RETRY_BACKEND",
    "RETRY_SEARCH",
    "ReportGenerator",
    "Reranker",
    "RetrievalError",
    "RetryConfig",
    "ScoredDocument",
    "SearchError",
    "SearchManager",
    "SearchResult",
    "SearchResults",
    "SemanticChunker",
    "StorageBackend",
    "StorageCapabilities",
    "StorageFileInfo",
    "UploadError",
    "UploadResult",
    "ValidationReport",
    "VectorBackend",
    "VectorBackendBase",
    "WeaviateBackend",
    "__version__",
    "__version_info__",
    "benchmark_registry",
    "chunk_document",
    "component_registry",
    "config_for_balanced",
    "config_for_max_quality",
    "config_for_production",
    "config_for_quality",
    "config_for_speed",
    "config_for_standalone",
    "config_for_standard",
    "corpus_metrics",
    "create_rag_tools",
    "f1_at_k",
    "get_available_backends",
    "get_available_storage_backends",
    "get_backend",
    "get_component_registry",
    "get_storage_backend",
    "grid_search",
    "hit_rate_at_k",
    "list_backends",
    "list_benchmarks",
    "load_benchmark",
    "map_score",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "register_backend",
    "register_storage_backend",
    "registry",
    "run_benchmark",
    "sweep",
    "sweep_nodes",
    "sweep_pipelines",
    "with_retry",
    "with_retry_sync",
}


class TestPublicSurface:
    def test_lasagna_exports_are_unchanged(self):
        actual = set(lasagna.__all__)
        added, removed = actual - EXPECTED_LASAGNA, EXPECTED_LASAGNA - actual
        assert not (added or removed), (
            f"lasagna.__all__ changed.\n"
            f"  added:   {sorted(added)}\n"
            f"  removed: {sorted(removed)}  <-- breaking for anyone importing these\n"
            f"Update EXPECTED_LASAGNA in this file if the change is intended."
        )

    def test_sdk_exports_are_unchanged(self):
        actual = set(lasagna.sdk.__all__)
        added, removed = actual - EXPECTED_SDK, EXPECTED_SDK - actual
        assert not (added or removed), (
            f"lasagna.sdk.__all__ changed.\n"
            f"  added:   {sorted(added)}\n"
            f"  removed: {sorted(removed)}  <-- breaking for anyone importing these\n"
            f"Update EXPECTED_SDK in this file if the change is intended."
        )


class TestExportsResolve:
    """An exported name that raises on access is worse than an absent one.

    Both modules resolve lazily through __getattr__, so a typo or a moved
    symbol stays invisible until a user touches that specific name.
    """

    @pytest.mark.parametrize("name", sorted(lasagna.__all__))
    def test_lasagna_export_resolves(self, name):
        assert getattr(lasagna, name) is not None or True

    @pytest.mark.parametrize("name", sorted(lasagna.sdk.__all__))
    def test_sdk_export_resolves(self, name):
        assert getattr(lasagna.sdk, name) is not None or True
