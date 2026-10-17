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
    "register",
    "registry",
    "RAGConfig",
    "KnowledgeClient",
    "KnowledgeConfig",
    "Experiment",
    "ExperimentConfig",
    "RAGBackend",
    "get_rag_backend",
    "create_rag_backend",
    "VectorStoreAdapter",
    "register_backend",
    "__version__",
}

#: Exported from `lasagna` before 0.1.0. Still importable for one release, each
#: with a DeprecationWarning naming the module to import it from instead.
DEPRECATED_LASAGNA = {
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
    "RetrievedChunkData",
    "SearchCache",
    "SemanticChunker",
    "TRACER",
    "chunk_document",
    "configure_metrics",
    "configure_tracing",
    "create_agentic_rag",
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
    "reset_instance",
    "resolve_model_settings",
    "search_knowledge_base",
    "trunc_text",
}

EXPECTED_SDK = {
    "__version__",
    "AgenticSearchResults",
    "BackendConnectionError",
    "BackendError",
    "ChunkingStrategy",
    "ChunkRetrievalStrategy",
    "Citation",
    "config_for_balanced",
    "config_for_max_quality",
    "config_for_production",
    "config_for_speed",
    "config_for_standalone",
    "config_for_standard",
    "ConfigurationError",
    "Document",
    "DocumentChunk",
    "DocumentError",
    "DocumentNotFoundError",
    "DocumentProcessingError",
    "DocumentStatus",
    "DocumentsLeftBehind",
    "EmbeddingProvider",
    "EmbeddingSizeConflict",
    "HyDEStrategy",
    "KnowledgeClient",
    "KnowledgeConfig",
    "KnowledgeError",
    "MCPKnowledgeClient",
    "ProfileCannotLoad",
    "ProfileNotFoundError",
    "ProfileSwitchInProgress",
    "ReindexFailed",
    "SearchError",
    "SearchResult",
    "SearchResults",
    "SettingsNotSaved",
    "UnknownProfileError",
    "UploadError",
    "UploadResult",
    "VectorBackend",
}

#: In ``lasagna.sdk.__all__`` before 0.1.0, left out of it since (it listed
#: about 160 names). Each still imports from ``lasagna.sdk``, with no warning:
#: only ``import *`` and the documented surface changed.
NOT_EXPORTED_SDK = {
    "__version_info__",
    "AblationResults",
    "AgenticError",
    "AgenticPipelineInput",
    "AgenticRAG",
    "AgenticRAGWrapper",
    "BackendCapabilities",
    "BackendTimeoutError",
    "BaseAgenticPipeline",
    "BaseBenchmarkAdapter",
    "BaseChunker",
    "BaseDocumentProcessor",
    "BaseEmbedder",
    "BaseQueryEnhancer",
    "BaseReranker",
    "BaseRetriever",
    "benchmark_registry",
    "BenchmarkInfo",
    "BenchmarkResults",
    "BenchmarkSample",
    "Chunk",
    "chunk_document",
    "ChunkCiteRetrieval",
    "ChunkingError",
    "CitationError",
    "CitationFormat",
    "CitationManager",
    "CitationNotFoundError",
    "CitationScoreMethod",
    "component_registry",
    "ComponentRegistry",
    "ComponentTestHarness",
    "config_for_quality",
    "ConfigManager",
    "ContextualEnricher",
    "corpus_metrics",
    "CoverageScorer",
    "create_rag_tools",
    "DocumentAwareness",
    "DocumentExtractor",
    "DocumentManager",
    "EmbeddingError",
    "EmbeddingModel",
    "EnrichmentStrategy",
    "EvalConfig",
    "EvalResult",
    "EvaluationError",
    "Experiment",
    "ExperimentConfig",
    "ExtractionError",
    "f1_at_k",
    "FilterExpression",
    "FixedChunker",
    "get_available_backends",
    "get_available_storage_backends",
    "get_backend",
    "get_component_registry",
    "get_storage_backend",
    "grid_search",
    "hit_rate_at_k",
    "HopData",
    "HybridSearcher",
    "IntegrationError",
    "JudgeError",
    "KnowledgeEvaluator",
    "KnowledgePipeline",
    "list_backends",
    "list_benchmarks",
    "load_benchmark",
    "LocalStorageBackend",
    "map_score",
    "MaxIterationsError",
    "MilvusBackend",
    "mrr",
    "ndcg_at_k",
    "OCREngine",
    "OCRProcessor",
    "PineconeBackend",
    "PipelineResult",
    "precision_at_k",
    "ProcessedDocument",
    "QdrantBackend",
    "QueryClassifier",
    "QueryEnhancer",
    "QueryError",
    "RAGASEvaluator",
    "recall_at_k",
    "register_backend",
    "register_storage_backend",
    "registry",
    "ReportGenerator",
    "Reranker",
    "RetrievalError",
    "RETRY_BACKEND",
    "RETRY_SEARCH",
    "RetryConfig",
    "run_benchmark",
    "ScoredDocument",
    "SearchManager",
    "SemanticChunker",
    "StorageBackend",
    "StorageCapabilities",
    "StorageFileInfo",
    "sweep",
    "sweep_nodes",
    "sweep_pipelines",
    "ValidationReport",
    "VectorBackendBase",
    "WeaviateBackend",
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


class TestTheSdkStillImportsWhatItNoLongerLists:
    def test_the_two_lists_are_apart(self):
        assert not NOT_EXPORTED_SDK & set(lasagna.sdk.__all__)

    @pytest.mark.parametrize("name", sorted(NOT_EXPORTED_SDK))
    def test_it_still_imports(self, name):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert hasattr(lasagna.sdk, name)


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


class TestDeprecatedExports:
    """The names dropped from `lasagna.__all__` in 0.1.0 keep working for a release.

    Fifty-six of them were internals (a chunker class, a tracer, a token
    estimator) presented as top-level API. Each still resolves, to the same
    object, and says which module to import it from.
    """

    def test_the_list_is_the_one_written_here(self):
        assert set(lasagna._DEPRECATED_IMPORTS) == DEPRECATED_LASAGNA
        assert not DEPRECATED_LASAGNA & set(lasagna.__all__)

    @pytest.mark.parametrize("name", sorted(DEPRECATED_LASAGNA))
    def test_it_resolves_and_names_its_module(self, name):
        import importlib

        module, attr = lasagna._DEPRECATED_IMPORTS[name]
        home = importlib.import_module(module, "lasagna")
        with pytest.warns(DeprecationWarning) as caught:
            value = getattr(lasagna, name)
        assert value is getattr(home, attr)
        assert f"from {home.__name__} import {attr}" in str(caught[0].message)
        assert name not in dir(lasagna)

    def test_the_warning_points_at_the_importing_line(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from lasagna import Document  # noqa: F401

        # Two lookups (import's hasattr, then the binding); both name this file.
        assert caught and {w.filename for w in caught} == {__file__}
