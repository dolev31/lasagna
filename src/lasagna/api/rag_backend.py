"""RAG Backend — Public Facade.

Thin facade over four specialized managers that implement the RAG pipeline.
All public methods delegate to the appropriate internal manager:

    - CollectionManager  (_collection_manager.py)  — vector backend lifecycle
    - DocumentPipeline   (_document_pipeline.py)    — upload / process / delete
    - SearchOrchestrator (_search_orchestrator.py)  — search / rerank / cite
    - AutoMerger         (_auto_merger.py)          — hierarchical chunk merging

Features (all configurable via settings.lasagna.toml):
    - BGE-large embeddings with query prefixing
    - Semantic chunking with structure preservation
    - Hybrid search (dense + BM25)
    - Cross-encoder reranking with MMR diversity
    - Stable citation tracking
    - HyDE query expansion
    - OCR support for scanned documents

Usage:
    from lasagna import RAGBackend

    backend = RAGBackend()
    doc = backend.upload_document("file.pdf", session_id="user123")
    results = backend.search("query", session_id="user123")
"""

import threading
from typing import List, Optional, Dict, Any, Tuple

from loguru import logger

from ..vector_backends import VectorStoreAdapter
from lasagna.core.schema import (
    Document,
    DocumentMetadata,
    SearchResult,
)
from lasagna.core.config import rag_config, RAGConfig

# Processing layer - initialized in _do_init
from ..processing.document_processor import DocumentProcessor
from ..processing.chunker.semantic import SemanticChunker

# Infrastructure layer
from ..infrastructure.embedding import get_embedding_manager, EmbeddingManager
from ..infrastructure.cache import get_search_cache

# Retrieval layer
from ..retrieval.search.hybrid import get_hybrid_searcher, HybridSearcher
from ..retrieval.search.ranker import get_reranker, CombinedReranker
from ..retrieval.enhancement.enhancer import get_query_enhancer, QueryEnhancer

# Agentic layer - citations
from lasagna.agentic.citation_manager import get_citation_manager, CitationManager, Citation

# Extracted modules
from ._validation import is_valid_document_id
from ._auto_merger import AutoMerger
from ._collection_manager import CollectionManager
from ._document_pipeline import DocumentPipeline
from ._search_orchestrator import SearchOrchestrator

# Lifecycle management
from ..lifecycle import DocumentLifecycleManager, LifecyclePolicy
from ..lifecycle._results import BulkDeleteResult, ConsistencyReport, GCReport, RepairReport

# Module-level singleton instance and lock for thread safety
_instance: Optional["RAGBackend"] = None
_instance_lock = threading.Lock()


class RAGBackend:
    """Primary RAG Backend with configurable SOTA features.

    This is the main RAG backend for Lasagna's knowledge base system.
    All features are configurable via settings.lasagna.toml.

    Features (all toggleable):
    - Embeddings: BGE-large (1024 dim) with query prefixing
    - Chunking: Semantic chunking with structure awareness
    - Search: Hybrid dense + BM25 with configurable alpha
    - Reranking: Cross-encoder with MMR diversity
    - Citations: Stable hash-based tracking
    - Query Enhancement: HyDE expansion
    - OCR: Support for scanned documents

    Example:
        backend = RAGBackend()
        doc = backend.upload_document("file.pdf", session_id="user123")
        results = backend.search("query", session_id="user123")
    """

    def __new__(cls):
        """Ensure only one instance exists (thread-safe singleton pattern)."""
        global _instance
        if _instance is None:
            with _instance_lock:
                # Double-check locking pattern
                if _instance is None:
                    _instance = super().__new__(cls)
                    _instance._initialized = False
        return _instance

    def __init__(self):
        """Initialize RAG backend with all SOTA components.

        Thread-safe: uses _instance_lock to prevent concurrent initialization.
        The lock is acquired here (not just in __new__) because two threads can
        both get the same instance from __new__ and then both enter __init__.
        """
        if getattr(self, "_initialized", False):
            return

        with _instance_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            self._initialized = True
            try:
                self._do_init()
            except Exception as e:
                self._initialized = False
                logger.error(f"Failed to initialize RAG Backend: {e}", exc_info=True)
                raise

    def _do_init(self):
        """Perform actual initialization."""
        logger.info("Initializing RAG Backend...")

        # Configure observability (no-op by default, Prometheus/tracing opt-in)
        from lasagna.utils.monitoring import configure_metrics
        from lasagna.utils.tracing import configure_tracing

        configure_metrics(enable_prometheus=rag_config.enable_prometheus)
        configure_tracing(
            enable=rag_config.enable_tracing,
            service_name=rag_config.tracing_service_name,
            exporter=rag_config.tracing_exporter,
            endpoint=rag_config.tracing_endpoint,
        )

        # Configure multi-tenancy (no-op by default)
        from lasagna.core.tenant import configure_tenant_manager
        from lasagna.core.quota import configure_quota_manager

        configure_tenant_manager(isolation_mode=rag_config.tenant_isolation)
        configure_quota_manager(enabled=rag_config.quota_enabled)

        # Configure rate limiting (no-op by default)
        from lasagna.core.rate_limiter import configure_rate_limiter, RateLimiterConfig

        configure_rate_limiter(
            config=RateLimiterConfig(
                enabled=rag_config.rate_limiting_enabled,
                strategy=rag_config.rate_limiting_strategy,
                requests_per_minute=rag_config.rate_limiting_rpm,
                burst_capacity=rag_config.rate_limiting_burst,
            )
        )

        # Initialize SOTA embedding manager.
        #
        # Passing the config explicitly rather than relying on the factory's
        # `None` fallback, so that its config-drift detection can fire: a second
        # backend init after the SDK bridged a different embedding model must
        # reload the manager rather than silently keep the first one.
        self.embedding_manager: EmbeddingManager = get_embedding_manager(rag_config)

        # Initialize vector backend and collection manager
        vector_backend: VectorStoreAdapter = CollectionManager.create_vector_backend()
        vector_backend.connect()

        self._collection_manager = CollectionManager(
            default_backend=vector_backend,
            default_backend_name=rag_config.vector_backend,
            default_backend_config=rag_config.vector_backend_config or {},
            embedding_manager=self.embedding_manager,
        )
        # Backward-compat alias
        self._vector_backend = self._collection_manager._default_backend

        # Initialize semantic chunker
        self.chunker = SemanticChunker(embedding_manager=self.embedding_manager)

        # Initialize document processor
        self.document_processor = DocumentProcessor()

        # Initialize hybrid search
        self.hybrid_searcher: HybridSearcher = get_hybrid_searcher()

        # Initialize reranker (check component registry for custom override)
        self.reranker: CombinedReranker = self._create_reranker()

        # Initialize citation manager
        self.citation_manager: CitationManager = get_citation_manager()

        # Initialize query enhancer (check component registry for custom override)
        self.query_enhancer: QueryEnhancer = self._create_query_enhancer()

        # Initialize auto-merger for hierarchical chunking
        self._auto_merger = AutoMerger()

        # Initialize storage backend
        from lasagna.storage import get_storage_backend

        self._storage = get_storage_backend()

        # Initialize document pipeline
        self._document_pipeline = DocumentPipeline(
            document_processor=self.document_processor,
            embedding_manager=self.embedding_manager,
            chunker=self.chunker,
            hybrid_searcher=self.hybrid_searcher,
            collection_manager=self._collection_manager,
            storage=self._storage,
        )

        # Initialize search orchestrator
        self._search_orchestrator = SearchOrchestrator(
            embedding_manager=self.embedding_manager,
            hybrid_searcher=self.hybrid_searcher,
            reranker=self.reranker,
            query_enhancer=self.query_enhancer,
            citation_manager=self.citation_manager,
            collection_manager=self._collection_manager,
            document_pipeline=self._document_pipeline,
            auto_merger=self._auto_merger,
            default_backend=self._vector_backend,
        )

        # Initialize collection with correct dimensions
        self._collection_manager.initialize_collection()
        self._collection_manager.mark_collection_initialized(
            rag_config.vector_backend, rag_config.collection_name
        )

        # Initialize SQLite database
        self._initialize_database()

        # Initialize document lifecycle manager (soft-delete, restore, bulk ops, GC)
        self._lifecycle_manager = self._create_lifecycle_manager()

        # Connect hybrid searcher to vector backend for native BM25 detection
        if hasattr(self.hybrid_searcher, "set_vector_backend"):
            self.hybrid_searcher.set_vector_backend(
                self._vector_backend, rag_config.collection_name
            )
        else:
            self.hybrid_searcher.set_milvus_client(self._vector_backend, rag_config.collection_name)

        # Log backend capabilities for debugging
        caps = self._vector_backend.capabilities
        logger.info(
            f"RAG Backend initialized: "
            f"embedding_dim={self.embedding_manager.dimension}, "
            f"default embedder={self.embedding_manager.model_name} (loads on first use), "
            f"backend={self._vector_backend.backend_name}"
        )
        logger.info(
            f"Backend capabilities: "
            f"hybrid_search={caps.supports_hybrid_search}, "
            f"native_bm25={caps.supports_native_bm25}, "
            f"json_filters={caps.supports_json_filters}, "
            f"max_batch={caps.max_batch_size}"
        )

    # =========================================================================
    # Component Registry Integration
    # =========================================================================

    def _create_reranker(self) -> CombinedReranker:
        """Create reranker, checking component registry for custom override."""
        custom_reranker = rag_config.custom_reranker
        if custom_reranker:
            try:
                from ..retrieval.registry import get_component_registry

                registry = get_component_registry()
                if registry.get_reranker(custom_reranker):
                    logger.info(f"Using custom reranker from registry: {custom_reranker}")
                    return registry.create_reranker(
                        custom_reranker,
                        embedding_manager=self.embedding_manager,
                    )
                logger.warning(
                    f"Custom reranker '{custom_reranker}' not found in registry, using default"
                )
            except Exception as e:
                logger.warning(f"Failed to load custom reranker '{custom_reranker}': {e}")
        return get_reranker(embedding_manager=self.embedding_manager)

    def _create_query_enhancer(self) -> QueryEnhancer:
        """Create query enhancer, checking component registry for custom override."""
        custom_enhancer = rag_config.custom_query_enhancer
        if custom_enhancer:
            try:
                from ..retrieval.registry import get_component_registry

                registry = get_component_registry()
                if registry.get_enhancer(custom_enhancer):
                    logger.info(f"Using custom query enhancer from registry: {custom_enhancer}")
                    return registry.create_enhancer(custom_enhancer)
                logger.warning(
                    f"Custom enhancer '{custom_enhancer}' not found in registry, using default"
                )
            except Exception as e:
                logger.warning(f"Failed to load custom enhancer '{custom_enhancer}': {e}")
        return get_query_enhancer()

    def _create_lifecycle_manager(self) -> DocumentLifecycleManager:
        """Wire the DocumentLifecycleManager with all dependencies and start the lifecycle scheduler."""
        from lasagna.core.database import get_rag_database
        from lasagna.lifecycle.scheduler import LifecycleScheduler

        db = get_rag_database()
        policy = LifecyclePolicy.from_env()
        manager = DocumentLifecycleManager(
            db=db,
            pipeline=self._document_pipeline,
            backend=self._vector_backend,
            policy=policy,
            collection_name=rag_config.collection_name,
        )
        # Start the lifecycle scheduler (APScheduler in dev, Celery Beat in production)
        self._lifecycle_scheduler = LifecycleScheduler(gc=manager._gc, policy=policy)
        self._lifecycle_scheduler.start()
        return manager

    # =========================================================================
    # Delegations to CollectionManager (backward compatibility)
    # =========================================================================

    def _get_effective_config(
        self, config: RAGConfig, backend: VectorStoreAdapter = None
    ) -> RAGConfig:
        """Get effective config with capability-based feature degradation."""
        return self._collection_manager.get_effective_config(config, backend)

    def _should_use_hybrid_search(
        self, config: RAGConfig, backend: VectorStoreAdapter = None
    ) -> bool:
        """Check if hybrid search should be used based on config and capabilities."""
        return self._collection_manager.should_use_hybrid_search(config, backend)

    def get_collection_embedding_dim(
        self, collection_name: str, backend: VectorStoreAdapter = None
    ) -> Optional[int]:
        """Get the embedding dimension from an existing vector collection."""
        return self._collection_manager.get_collection_embedding_dim(collection_name, backend)

    # Backward-compat alias
    _get_collection_embedding_dim = get_collection_embedding_dim

    def reload_embedding_manager(self, config: RAGConfig) -> None:
        """Reload embedding manager if the configured model changes."""
        self._collection_manager.maybe_reload_embedding_manager(config)
        # Sync back any changes to local references
        self.embedding_manager = self._collection_manager.embedding_manager
        self.chunker.embedding_manager = self.embedding_manager
        self.reranker = self._create_reranker()

    # Backward-compat alias
    _maybe_reload_embedding_manager = reload_embedding_manager

    def _collection_has_documents(
        self, collection_name: str, backend: VectorStoreAdapter = None
    ) -> bool:
        """Check if a vector collection has any documents."""
        return self._collection_manager.collection_has_documents(collection_name, backend)

    def force_rebuild_collection(self, confirm: bool = False) -> bool:
        """Force rebuild the collection, DELETING ALL EXISTING DOCUMENTS."""
        return self._collection_manager.force_rebuild_collection(
            confirm=confirm, hybrid_searcher=self.hybrid_searcher
        )

    def ensure_collection_for_config(self, config: RAGConfig) -> None:
        """Ensure a collection exists for the provided config."""
        return self._collection_manager.ensure_collection_for_config(config)

    # Backward-compat alias
    _ensure_collection_for_config = ensure_collection_for_config

    def recreate_collection_for_config(
        self, config: RAGConfig, backend: VectorStoreAdapter = None
    ) -> None:
        """Drop and recreate the collection for the provided config."""
        return self._collection_manager.recreate_collection_for_config(config, backend)

    # Backward-compat alias
    _recreate_collection_for_config = recreate_collection_for_config

    def _ensure_backend_collection(self, backend: VectorStoreAdapter, config: RAGConfig) -> None:
        """Ensure collection exists and is indexed on the given backend."""
        return self._collection_manager.ensure_backend_collection(backend, config)

    def _initialize_database(self):
        """Initialize the metadata database (SQLite or PostgreSQL).

        Uses the unified RAGDatabase which handles:
        - Table creation (rag_documents, citations)
        - Index creation
        - Schema migrations
        """
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()
        db_info = getattr(rag_db, "db_path", None) or getattr(rag_db, "_dsn", "unknown")
        logger.info(f"Metadata database initialized: {db_info}")

    # =========================================================================
    # Delegations to DocumentPipeline
    # =========================================================================

    def upload_document(
        self,
        file_path: str,
        session_id: str,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
        _chunker_override: Any = None,
        _document_processor_override: Any = None,
    ) -> Document:
        """Upload and process a document with SOTA pipeline.

        ``_chunker_override`` takes a ``BaseChunker`` instance, mirroring the
        ``_reranker_override`` the search path already accepts. Underscored for
        the same reason: it is the swap seam an experiment uses, not part of the
        everyday API.
        """
        return self._document_pipeline.upload_document(
            file_path,
            session_id,
            filename,
            document_id=document_id,
            _chunker_override=_chunker_override,
            _document_processor_override=_document_processor_override,
        )

    def process_document(
        self,
        document: Document,
        file_path: str,
        config: Optional[RAGConfig] = None,
    ):
        """Process document with SOTA pipeline."""
        return self._document_pipeline.process_document(document, file_path, config)

    # Backward-compat alias
    _process_document = process_document

    def get_session_config(self, session_id: Optional[str] = None) -> RAGConfig:
        """Get a session-aware RAGConfig instance.

        When a session ID is provided, the returned config will include
        any session-specific overrides configured via the UI.

        Args:
            session_id: Optional session identifier

        Returns:
            RAGConfig instance (session-aware if session_id provided)
        """
        if session_id:
            return rag_config.with_session(session_id)
        return rag_config

    # Backward-compat alias
    _get_session_config = get_session_config

    def _resolve_backend(self, session_id: Optional[str] = None) -> VectorStoreAdapter:
        """Resolve the vector backend for a given session."""
        return self._collection_manager.resolve_backend(session_id)

    def get_vector_backend(self, session_id: Optional[str] = None) -> VectorStoreAdapter:
        """Public API for getting the session-appropriate vector backend."""
        return self._collection_manager.get_vector_backend(session_id)

    # =========================================================================
    # Delegations to SearchOrchestrator
    # =========================================================================

    def search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        min_score: float = -100.0,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _from_routing: bool = False,
        _reranker_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
        _embedder_override: Optional[Any] = None,
        query_config_overrides: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _agentic_pipeline_override: Optional[Any] = None,
        _graph_node_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search with SOTA retrieval pipeline.

        The three trailing overrides are what KnowledgePipeline passes for a
        swapped retriever, agentic pipeline or graph node; without them here
        those swaps raised TypeError before reaching the orchestrator.
        """
        return self._search_orchestrator.search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
            use_hyde=use_hyde,
            use_reranking=use_reranking,
            trace_id=trace_id,
            routing_tier=routing_tier,
            _from_routing=_from_routing,
            _reranker_override=_reranker_override,
            _enhancer_override=_enhancer_override,
            _embedder_override=_embedder_override,
            query_config_overrides=query_config_overrides,
            _retriever_override=_retriever_override,
            _agentic_pipeline_override=_agentic_pipeline_override,
            _graph_node_overrides=_graph_node_overrides,
        )

    def search_with_citations(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> Tuple[List[SearchResult], List[Citation], str]:
        """Search and return results with citations and formatted context."""
        return self._search_orchestrator.search_with_citations(
            query=query,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_ids,
            trace_id=trace_id,
            routing_tier=routing_tier,
            _reranker_override=_reranker_override,
            _enhancer_override=_enhancer_override,
        )

    async def async_search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        min_score: float = -100.0,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Async search — suitable for FastAPI async endpoints."""
        return await self._search_orchestrator.async_search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
            use_hyde=use_hyde,
            use_reranking=use_reranking,
            trace_id=trace_id,
            routing_tier=routing_tier,
            _reranker_override=_reranker_override,
            _enhancer_override=_enhancer_override,
        )

    async def async_search_with_citations(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> Tuple[List[SearchResult], List[Citation], str]:
        """Async search with citations — suitable for FastAPI async endpoints."""
        return await self._search_orchestrator.async_search_with_citations(
            query=query,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_ids,
            trace_id=trace_id,
            routing_tier=routing_tier,
            _reranker_override=_reranker_override,
            _enhancer_override=_enhancer_override,
        )

    def search_multi_query(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        n_variations: int = 3,
        trace_id: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Search with query expansion and RRF-based result fusion."""
        return self._search_orchestrator.search_multi_query(
            query=query,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_ids,
            n_variations=n_variations,
            trace_id=trace_id,
            _reranker_override=_reranker_override,
            _enhancer_override=_enhancer_override,
        )

    async def async_agentic_search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        routing_decision: Any = None,
    ):
        """Agentic multi-hop search with iterative retrieval.

        Uses an LLM agent to decompose complex queries, perform multiple
        retrieval rounds, and synthesise results with citations.

        Args:
            routing_decision: Pre-computed RoutingDecision from upstream.
                Forwarded to ``AgenticRAG.retrieve()`` so the structured graph
                can skip re-classification.

        Returns:
            AgenticRAGResult with formatted_context, citations, chunks, metrics.
        """
        from lasagna.agentic.agentic_rag import AgenticRAG
        from lasagna.providers import get_llm_provider

        config = self.get_session_config(session_id)
        llm = get_llm_provider().get_llm()
        agent = AgenticRAG(
            llm=llm,
            max_iterations=getattr(config, "max_agent_iterations", 3),
            session_config=config,
        )
        return await agent.retrieve(
            question=query,
            session_id=session_id,
            document_ids=document_ids,
            top_k=top_k or getattr(config, "rerank_top_k", 5),
            trace_id=trace_id,
            routing_decision=routing_decision,
        )

    def get_documents(self, session_id: str) -> List[DocumentMetadata]:
        """Get all documents for a session."""
        return self._document_pipeline.get_documents(session_id)

    def get_document_by_id(self, document_id: str, session_id: str) -> Optional[DocumentMetadata]:
        """Get a single document by ID (O(1) primary-key lookup with session verification)."""
        return self._document_pipeline.get_document_by_id(document_id, session_id)

    def delete_document(
        self, document_id: str, session_id: str, *, keep_file: bool = False
    ) -> bool:
        """Delete a document and its chunks."""
        return self._document_pipeline.delete_document(document_id, session_id, keep_file=keep_file)

    def clear_session_state(self, session_id: str) -> Dict[str, Any]:
        """Clear all session-level state (cache, BM25, config).

        Note: Does NOT delete documents - use delete_document() for that.
        This is for clearing caches and in-memory state only.

        Args:
            session_id: Session ID to clear

        Returns:
            Dict with cleanup results:
            {
                "cache_invalidated": bool,
                "bm25_cleared": bool,
                "session_cleared": bool,
            }
        """
        result = {
            "cache_invalidated": False,
            "bm25_cleared": False,
            "session_cleared": False,
        }

        # 1. Clear BM25 hybrid search index
        if rag_config.enable_hybrid_search and self.hybrid_searcher:
            try:
                self.hybrid_searcher.clear_session(session_id)
                result["bm25_cleared"] = True
                logger.debug(f"BM25 index cleared for session: {session_id}")
            except Exception as e:
                logger.warning(f"BM25 clear failed for session {session_id}: {e}")

        # 2. Invalidate search cache
        if rag_config.enable_search_cache:
            try:
                search_cache = get_search_cache(rag_config)
                invalidated = search_cache.invalidate_session(session_id)
                result["cache_invalidated"] = True
                logger.debug(
                    f"Search cache invalidated: {invalidated} entries for session {session_id}"
                )
            except Exception as e:
                logger.warning(f"Cache invalidation failed for session {session_id}: {e}")

        # 3. Clear session config state
        try:
            from ..core.session_config import get_knowledge_config_manager

            config_manager = get_knowledge_config_manager()
            config_manager.delete_session_state(session_id)
            result["session_cleared"] = True
            logger.debug(f"Session config state cleared: {session_id}")
        except Exception as e:
            logger.warning(f"Session config clear failed for session {session_id}: {e}")

        # 4. Clear document awareness snippet cache
        try:
            from ..agentic.document_awareness import clear_session_cache

            clear_session_cache(session_id)
            logger.debug(f"Document awareness cache cleared: {session_id}")
        except Exception as e:
            logger.warning(f"Document awareness cache clear failed for session {session_id}: {e}")

        logger.info(f"Session state cleared: {session_id} - {result}")
        return result

    def _get_document_metadata(self, document_id: str) -> Optional[Document]:
        """Get document metadata from SQLite."""
        return self._document_pipeline._get_document_metadata(document_id)

    def _get_documents_metadata_batch(self, document_ids: List[str]) -> Dict[str, Document]:
        """Get document metadata for multiple documents in a single query."""
        return self._document_pipeline.get_documents_metadata_batch(document_ids)

    @staticmethod
    def _is_valid_document_id(document_id: str) -> bool:
        """Validate document ID format to prevent injection attacks."""
        return is_valid_document_id(document_id)

    def get_context(self, search_results: List[SearchResult], max_chunks: int = 5) -> str:
        """Build context string from search results."""
        if not search_results:
            return ""

        context_parts = []
        for idx, result in enumerate(search_results[:max_chunks]):
            context_parts.append(
                f"[Source {idx + 1}: {result.document.filename}, "
                f"chunk {result.chunk.chunk_index + 1}]\n{result.chunk.content}"
            )

        return "\n\n".join(context_parts)

    def get_document_chunks(
        self,
        document_id: str,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        return self._document_pipeline.get_document_chunks(
            document_id, session_id, limit=limit, offset=offset
        )

    def get_document_file_path(self, document_id: str, session_id: str) -> Optional[str]:
        """Get the file path for a document."""
        return self._document_pipeline.get_document_file_path(document_id, session_id)

    # =========================================================================
    # Delegations to DocumentLifecycleManager
    # =========================================================================

    def soft_delete_document(self, document_id: str, session_id: str) -> dict:
        """Archive (soft-delete) a document.

        The document is marked as SOFT_DELETED and its GC grace window starts.
        Vectors and the stored file are NOT removed immediately — the GC handles
        that after ``policy.soft_delete_grace_hours``.  Quota is NOT freed here.

        Args:
            document_id: Document to archive.
            session_id: Session the document belongs to (authorization).

        Returns:
            Updated document metadata dict.

        Raises:
            ValueError: If the document does not exist or belongs to a different session.
        """
        return self._lifecycle_manager.soft_delete(document_id, session_id)

    def restore_document(self, document_id: str, session_id: str) -> dict:
        """Restore a soft-deleted document within the grace window.

        Args:
            document_id: Soft-deleted document to restore.
            session_id: Session the document belongs to (authorization).

        Returns:
            Updated document metadata dict (status=READY).

        Raises:
            ValueError: If the document is not soft-deleted.
            DocumentExpiredError: If the grace window has already expired.
        """
        return self._lifecycle_manager.restore(document_id, session_id)

    def bulk_delete_session(self, session_id: str, *, confirm: bool = False) -> BulkDeleteResult:
        """Efficiently delete ALL documents in a session.

        Uses a single vector-level delete (or partition drop for Milvus) followed
        by a batch DB delete.  Much faster than calling ``delete_document()`` in a loop.

        Args:
            session_id: Session whose documents should be purged.
            confirm: Must be ``True`` — acts as a safety guard against accidental calls.

        Returns:
            BulkDeleteResult with counts and bytes freed.
        """
        return self._lifecycle_manager.bulk_delete_session(session_id, confirm=confirm)

    def bulk_delete_documents(
        self,
        document_ids: List[str],
        session_id: str,
    ) -> BulkDeleteResult:
        """Delete a specific list of documents belonging to a session.

        Args:
            document_ids: IDs of documents to delete.
            session_id: Session authorization — documents not in this session are skipped.

        Returns:
            BulkDeleteResult with counts.
        """
        return self._lifecycle_manager.bulk_delete(document_ids, session_id)

    def run_gc(self, *, session_id: Optional[str] = None) -> GCReport:
        """Trigger a manual garbage collection cycle.

        Runs all 6 GC phases:
        1. Hard-delete expired soft-deleted documents
        2. Expire documents past their TTL
        3. Clean up old ERROR documents
        4. Clean up expired citations
        5. Clean up idle sessions
        6. Reconcile quota counters

        Args:
            session_id: Optional — limit GC to a single session.

        Returns:
            GCReport with per-phase results.
        """
        return self._lifecycle_manager.run_gc(session_id=session_id)

    def detect_orphans(
        self,
        session_id: Optional[str] = None,
    ) -> ConsistencyReport:
        """Detect orphaned data between the metadata DB and vector store.

        Returns a report of DB-only records (failed ingestions) and
        vector-only records (leaked vectors).  Does NOT make any changes.

        Args:
            session_id: Optional — limit scan to a single session.

        Returns:
            ConsistencyReport with db_only and vec_only orphan lists.
        """
        return self._lifecycle_manager.detect_orphans(session_id=session_id)

    def repair_orphans(
        self,
        report: ConsistencyReport,
        *,
        dry_run: bool = True,
    ) -> RepairReport:
        """Repair orphans found in a ConsistencyReport.

        Args:
            report: Output from ``detect_orphans()``.
            dry_run: If True (default), no changes are made — useful for previewing.

        Returns:
            RepairReport describing what was (or would be) repaired.
        """
        return self._lifecycle_manager.repair_orphans(report, dry_run=dry_run)

    def get_gc_audit_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent GC audit log entries.

        Reads the ``gc_audit_log`` table through the database backend rather
        than via direct SQL in caller code (avoids accessing private DB attrs).

        Args:
            limit: Maximum number of entries to return (most recent first).

        Returns:
            List of dicts, one per GC phase run, with keys:
            ``run_id``, ``phase``, ``started_at``, ``completed_at``,
            ``docs_scanned``, ``docs_cleaned``, ``bytes_freed``, ``errors``.
        """
        from lasagna.core.database import get_rag_database

        try:
            return get_rag_database().get_gc_audit_log(limit=limit)
        except NotImplementedError:
            # DB backend doesn't implement get_gc_audit_log yet — return empty.
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        stats = {
            "embedding_model": self.embedding_manager.model_name,
            "embedding_dim": self.embedding_manager.dimension,
            "collection_name": rag_config.collection_name,
            "index_type": rag_config.index_type,
            "metric_type": rag_config.metric_type,
            "features": {
                "hybrid_search": rag_config.enable_hybrid_search,
                "reranking": rag_config.enable_reranking,
                "mmr": rag_config.enable_mmr,
                "hyde": rag_config.enable_hyde,
                "ocr": rag_config.enable_ocr,
            },
        }

        # Add hybrid search mode info
        hybrid_info = self.hybrid_searcher.get_mode_info()
        stats["hybrid_search_mode"] = hybrid_info

        # Add cache stats if available
        cache_stats = self.embedding_manager.get_cache_stats()
        if cache_stats:
            stats["embedding_cache"] = cache_stats

        return stats


def get_rag_backend() -> RAGBackend:
    """Get or create the RAG backend singleton.

    Returns:
        The RAGBackend singleton instance
    """
    return RAGBackend()


def initialize() -> RAGBackend:
    """Pre-initialize the RAG backend singleton.

    Returns:
        The initialized RAGBackend singleton instance
    """
    logger.info("Pre-initializing RAG Backend...")
    return RAGBackend()


def close_backend() -> None:
    """Close the RAG backend singleton and release all resources.

    Call this on application shutdown to properly release:
    - Vector backend connections (Milvus, Weaviate, pgvector)
    - Embedding manager resources (GPU memory, thread pools)
    - Database connections (SQLite, PostgreSQL pools)

    Safe to call multiple times — no-op if already closed.
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            return
        backend = _instance
        _instance = None

    # Close outside the lock to avoid holding it during slow I/O
    logger.info("Closing RAG backend resources...")
    if hasattr(backend, "_lifecycle_scheduler") and backend._lifecycle_scheduler is not None:
        try:
            backend._lifecycle_scheduler.stop()
        except Exception as e:
            logger.warning(f"Lifecycle scheduler stop failed: {e}")

    if hasattr(backend, "_vector_backend") and backend._vector_backend is not None:
        try:
            if hasattr(backend._vector_backend, "disconnect"):
                backend._vector_backend.disconnect()
        except Exception as e:
            logger.warning(f"Vector backend disconnect failed: {e}")

    if hasattr(backend, "embedding_manager") and backend.embedding_manager is not None:
        try:
            if hasattr(backend.embedding_manager, "close"):
                backend.embedding_manager.close()
        except Exception as e:
            logger.warning(f"Embedding manager close failed: {e}")

    logger.info("RAG backend closed")


def reset_instance():
    """Reset the RAG backend singleton instance.

    CRITICAL FIX: Uses lock to prevent race condition with get_rag_backend().
    Without the lock, a reset during creation could leave the system in an
    inconsistent state.

    Prefer ``close_backend()`` for production shutdown — it properly
    releases vector connections and GPU resources before resetting.
    """
    global _instance
    with _instance_lock:
        if _instance is not None:
            logger.warning("Resetting RAG Backend singleton instance")
            _instance = None
