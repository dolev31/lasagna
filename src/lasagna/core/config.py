"""Configuration management for RAG Knowledge Base system.

This module provides comprehensive configuration for the SOTA RAG implementation
including embedding models, chunking strategies, hybrid search, reranking, and more.

The configuration supports session-scoped overrides that allow users to modify
RAG settings via the UI without permanently altering the settings.lasagna.toml file.

Provider Abstraction:
    This module uses the provider abstraction layer for configuration sources.
    Auto-configures from environment variables (RAG_DB_DIR, RAG_UPLOAD_DIR, etc.).
    For custom configuration, call configure_providers() before instantiating RAGConfig.
"""

import copy
import os
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dynaconf import Dynaconf
from loguru import logger
from lasagna.utils.logging_utils import configure_knowledge_logging

if TYPE_CHECKING:
    from lasagna.providers import ConfigProvider


class RAGConfig:
    """RAG system configuration manager with SOTA defaults.

    Supports session-scoped overrides that can be applied via the UI.
    When a session ID is provided, the configuration will include any
    session-specific overrides.

    Provider Support:
        - When config_provider is None (default): Uses auto-detected provider
        - Auto-configures from environment variables on first access
        - For custom configuration: Pass a ConfigProvider implementation
    """

    def __init__(self, config_provider: Optional["ConfigProvider"] = None):
        """Initialize RAG configuration.

        Args:
            config_provider: Optional custom configuration provider.
                           If None, uses auto-detected provider from environment
                           variables (RAG_DB_DIR, RAG_UPLOAD_DIR, API keys).
        """
        self._config_provider = config_provider
        self._knowledge_settings = self._load_knowledge_settings()
        self._env_overrides: Dict[str, Any] = {}  # Populated in _load_config
        self._config = self._load_config()
        self._session_id: Optional[str] = None
        self._query_overrides: Dict[str, Any] = {}  # Per-query ephemeral overrides
        configure_knowledge_logging(self.log_level)

    def _get_provider(self) -> "ConfigProvider":
        """Get the configuration provider (lazy initialization).

        Returns:
            ConfigProvider instance
        """
        if self._config_provider is None:
            from lasagna.providers import get_config_provider

            self._config_provider = get_config_provider()
        return self._config_provider

    def with_session(self, session_id: Optional[str]) -> "RAGConfig":
        """Create a config instance bound to a specific session.
        This allows retrieval of session-specific overrides.

        Args:
            session_id: The session identifier

        Returns:
            A new RAGConfig instance bound to the session
        """
        config = RAGConfig.__new__(RAGConfig)
        config._config_provider = self._config_provider
        config._config = copy.deepcopy(self._config)
        config._env_overrides = copy.deepcopy(self._env_overrides)
        config._knowledge_settings = self._knowledge_settings
        config._session_id = session_id
        config._query_overrides = {}
        return config

    def settings_fingerprint(self) -> str:
        """A short hash of everything that can differ between two searches.

        Base settings are process-wide; what varies between calls is the
        session's resolved config and the per-query overrides. The search cache
        keys on this, so changing any setting (a reranker model, a threshold,
        the candidate depth) is a cache miss rather than a stale result.
        """
        import hashlib
        import json as _json

        parts: Dict[str, Any] = {"query": getattr(self, "_query_overrides", None) or {}}
        if self._session_id:
            try:
                from lasagna.providers import get_session_provider

                parts["session"] = get_session_provider().get_resolved_config(self._session_id)
            except Exception:
                parts["session"] = {"_unresolved": self._session_id}
        blob = _json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_query_overrides(self, overrides: Dict[str, Any]) -> "RAGConfig":
        """Create a config clone with per-query ephemeral overrides (highest priority).

        These overrides are scoped to a single query call and are never written
        to the session provider.  They take effect above session overrides,
        environment variables, and TOML settings.

        Args:
            overrides: Flat dict of config keys to override (e.g.
                ``{"enable_reranking": False, "hybrid_alpha": 0.5}``).

        Returns:
            A new RAGConfig instance that inherits all current settings plus
            the given overrides.  The original instance is not modified.
        """
        config = RAGConfig.__new__(RAGConfig)
        config._config_provider = self._config_provider
        # Shallow copy: _config values are primitives (str/int/bool/dict-of-primitives).
        # A shallow copy is sufficient to prevent mutation of the original dict while
        # avoiding the overhead of deep-copying large nested structures.
        config._config = dict(self._config)
        config._env_overrides = self._env_overrides  # read-only in practice
        config._knowledge_settings = self._knowledge_settings
        config._session_id = self._session_id
        config._query_overrides = dict(overrides)
        return config

    def _get_session_override(self, *keys: str, default: Any = None) -> Any:
        """Get a value from session overrides if available.

        Args:
            *keys: The nested keys to look up (e.g., 'chunk_size' or 'document_awareness', 'enabled')
            default: Default value if not found

        Returns:
            The session override value, or None if not found
        """
        # Per-query ephemeral overrides take highest priority (set via with_query_overrides()).
        # Use flat leaf-key lookup — matches the flat dicts used by the eval SDK.
        if getattr(self, "_query_overrides", None):
            leaf_key = keys[-1] if len(keys) > 1 else keys[0]
            if leaf_key in self._query_overrides:
                return self._query_overrides[leaf_key]

        if not self._session_id:
            return None

        def _lookup(resolved: Dict[str, Any], path: Tuple[str, ...]) -> Tuple[Optional[Any], bool]:
            value: Any = resolved
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None, False
            return value, True

        try:
            from lasagna.providers import get_session_provider

            session_provider = get_session_provider()
            resolved = session_provider.get_resolved_config(self._session_id)

            # Navigate nested keys
            value, ok = _lookup(resolved, keys)
            if ok:
                return value

            # Session overrides are stored flat (no leading 'rag') for UI settings.
            if keys and keys[0] == "rag":
                value, ok = _lookup(resolved, keys[1:])
                if ok:
                    return value

                if len(keys) > 1:
                    value, ok = _lookup(resolved, (keys[-1],))
                    if ok:
                        return value
            return None
        except Exception as e:
            logger.debug(f"Could not get session override for {keys}: {e}")
            return None

    def _load_knowledge_settings(self) -> Dynaconf:
        """Load settings from settings.lasagna.toml (or RAG_SETTINGS_FILE override).

        The config file is located in the knowledge module folder, making the
        knowledge service self-contained and usable as an external service.

        When the ``RAG_SETTINGS_FILE`` environment variable is set, that file
        is loaded instead of the default ``settings.lasagna.toml``.  The
        path can be absolute or relative to the knowledge module directory.

        Returns:
            Dynaconf settings object
        """
        knowledge_module_dir = Path(__file__).parent.parent  # knowledge/ folder

        # Allow overriding the settings file via environment variable
        settings_file_env = os.environ.get("RAG_SETTINGS_FILE")
        if settings_file_env:
            override_path = Path(settings_file_env)
            if not override_path.is_absolute():
                override_path = knowledge_module_dir / override_path
            if override_path.exists():
                logger.info(f"Loading settings from RAG_SETTINGS_FILE: {override_path}")
                return Dynaconf(
                    root_path=str(override_path.parent),
                    settings_files=[str(override_path)],
                )
            logger.warning(
                f"RAG_SETTINGS_FILE={settings_file_env} not found, falling back to default"
            )

        knowledge_toml_path = knowledge_module_dir / "settings.lasagna.toml"

        if knowledge_toml_path.exists():
            logger.debug(f"Loading settings from knowledge module: {knowledge_toml_path}")
            return Dynaconf(
                root_path=str(knowledge_module_dir),
                settings_files=[str(knowledge_toml_path)],
            )

        logger.debug("settings.lasagna.toml not found in knowledge module, using defaults only")
        return Dynaconf()

    def _load_config(self) -> Dict[str, Any]:
        """Load RAG configuration from settings.

        Uses the config provider to get RAG settings and paths.

        Returns:
            Configuration dictionary
        """
        # Get RAG-specific settings from provider (settings.memory.rag in the host application)
        try:
            provider = self._get_provider()
            rag_settings = provider.get_rag_settings()
        except RuntimeError:
            # No provider configured - use empty settings
            logger.warning("No config provider, using defaults only")
            rag_settings = {}

        # Ensure rag_settings is a dict
        if hasattr(rag_settings, "to_dict"):
            rag_settings = rag_settings.to_dict()
        elif not isinstance(rag_settings, dict):
            rag_settings = {}

        # SOTA Default configuration
        default_config = {
            # =================================================================
            # Vector Database
            # =================================================================
            "vector_backend": "milvus",  # Supported: milvus, pinecone, qdrant, weaviate, pgvector, opensearch
            "vector_backend_config": {},  # Backend-specific configuration
            "collection_name": "rag_documents_v2",
            "embedding_dim": 1024,  # BGE-large
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "index_params": {"M": 16, "efConstruction": 256},
            "search_params": {"ef": 128},
            # =================================================================
            # Embedding Model
            # =================================================================
            "embedding_provider": "huggingface",
            "embedding_model": "BAAI/bge-m3",  # SOTA: dense+sparse+ColBERT, 8K context, 100+ languages
            "embedding_model_fallback": "BAAI/bge-large-en-v1.5",
            "embedding_backend": "auto",  # 'auto', 'sentence-transformers', 'flag-embedding'
            "normalize_embeddings": True,
            "batch_size": 32,
            "use_gpu": True,
            "embedding_truncate_dim": None,  # None = use full dim; int = truncate+renorm (Matryoshka)
            # =================================================================
            # Chunking
            # =================================================================
            "chunking_strategy": "semantic",  # fixed, semantic, recursive, sentence_window
            "chunk_size": 800,
            "chunk_overlap": 0,  # Research: no measurable benefit from overlap (arXiv:2601.14123)
            "semantic_threshold": 0.75,
            "min_chunk_size": 200,
            "max_chunk_size": 1500,
            "window_size": 3,
            "sentence_overlap": 1,
            "enable_hierarchical": True,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.5,
            "enable_late_chunking": False,  # Late chunking: embed full doc, pool per chunk (Jina, arxiv:2409.04701)
            "preserve_tables": True,
            "preserve_headers": True,
            "extract_metadata": True,
            "max_file_size": 52428800,  # 50MB
            "max_content_length": 4000,  # Max chars per chunk in storage
            # =================================================================
            # Hybrid Search
            # =================================================================
            "enable_hybrid_search": True,
            "hybrid_alpha": 0.7,
            "sparse_method": "bm25",
            "default_top_k": 20,  # Research: 75+ introduces hard negatives (ICLR 2025)
            "similarity_threshold": 0.3,
            "rrf_k": 60,  # RRF constant (higher = more weight to lower ranks)
            "enable_bm25_stemming": True,  # Enable Porter stemming for BM25
            "bm25_max_sessions": 1000,  # Max in-memory BM25 session indices
            "bm25_session_ttl_seconds": 86400,  # Idle TTL for session indices (24h)
            # =================================================================
            # Reranking
            # =================================================================
            "enable_reranking": True,
            "reranker_model": "mixedbread-ai/mxbai-rerank-large-v1",  # BEIR NDCG@10 = 57.49
            "rerank_top_k": 5,  # Research: context cliff at ~2,500 tokens
            "reranker_batch_size": 16,
            "custom_reranker": "",  # Registry name for custom reranker (empty = use default)
            "custom_query_enhancer": "",  # Registry name for custom query enhancer (empty = use default)
            # Empty = resolve from MODEL_NAME / the configured provider. Never
            # name a specific vendor route here: a gateway-prefixed default like
            # "<vendor>/<model>" is rejected outright by every other provider, so
            # it fails for anyone who did not happen to share our deployment.
            "llm_reranker_model": "",  # Model for LLM listwise reranker
            "llm_reranker_window_size": 10,  # Sliding window size for listwise reranking
            "colbert_model": "colbert-ir/colbertv2.0",  # ColBERT late-interaction model
            "enable_context_reordering": True,  # Lost-in-the-middle fix: place best results at start/end
            "enable_context_compression": False,  # Compress chunks to reduce noise/tokens
            "compression_method": "extractive",  # 'extractive' | 'abstractive'
            "compression_ratio": 0.3,  # Keep ~30% of original length
            "enable_graph_retrieval": False,  # GraphRAG entity-linked retrieval
            "graph_max_hops": 2,  # Max traversal depth in knowledge graph
            "graph_entity_types": ["PERSON", "ORG", "CONCEPT", "LOCATION", "TECHNOLOGY", "EVENT"],
            "graph_extraction_model": "",  # Model for entity extraction (empty = configured default)
            # Per-node agentic overrides, keyed by node name. Empty means "use
            # the shipped prompt / the pipeline default model". These are what
            # make an agentic technique testable: without them the prompt that
            # decides whether an answer is grounded cannot be varied, and two
            # runs using different prompts recorded nothing to tell them apart.
            "agentic_prompts": {},
            "agentic_models": {},
            "enable_raptor": False,  # RAPTOR: recursive summary tree indexing (arXiv:2401.18059)
            "raptor_max_cluster_size": 10,  # Max chunks per cluster in RAPTOR tree
            "raptor_max_levels": 5,  # Max tree depth
            # raptor_retrieval_mode: reserved for future tree-traversal query path.
            # Currently all RAPTOR nodes are searched via standard vector search (collapsed mode).
            "enable_speculative_rag": False,  # Speculative RAG: parallel drafts + verification
            "speculative_n_drafts": 3,  # Number of parallel draft answers
            "speculative_subset_size": 3,  # Context chunks per draft
            "enable_mmr": True,
            "mmr_lambda": 0.5,  # SOTA default (LangChain, LlamaIndex). Pre-MMR filter ensures only relevant items.
            "mmr_score_normalization": "sigmoid",  # Options: 'sigmoid', 'minmax', 'sigmoid_minmax', 'none'. Plain sigmoid preserves cross-encoder calibration.
            # =================================================================
            # Query Enhancement
            # =================================================================
            "enable_hyde": False,
            "hyde_model": "default",
            "hyde_n_docs": 3,  # Number of hypothetical docs for multi-HyDE (1 = legacy single-doc)
            "hyde_embedding_strategy": "average",  # Options: 'average', 'max_similarity'
            "enable_query_expansion": False,
            "max_query_variations": 3,
            "enable_query_decomposition": False,
            "enable_query_classification": True,  # Enable adaptive query-type classification
            "agentic_hyde_strategy": "adaptive",
            "enable_contextual_retrieval": False,  # Anthropic-style chunk enrichment at index time
            "enable_rag_fusion": False,  # Multi-query generation + RRF fusion
            "rag_fusion_n_queries": 3,  # Number of query variations for RAG Fusion
            "enable_crag": False,  # Corrective RAG: LLM grades retrieval quality
            "crag_web_search_fallback": False,  # Fall back to web search if retrieval quality is low
            # =================================================================
            # Query Routing (Adaptive-RAG)
            # =================================================================
            "enable_query_routing": True,  # Route queries to FAST/STANDARD/COMPREHENSIVE tiers
            "routing_strategy": "rule_based",  # 'rule_based' (deterministic) or 'adaptive' (confidence-aware)
            "routing_default_tier": "standard",  # Default tier when routing is disabled: fast, standard, comprehensive
            "routing_confidence_threshold": 0.5,  # Adaptive strategy escalates tier below this confidence
            "routing_allow_tier_override": True,  # Allow API callers to force a routing tier
            "routing_log_decisions": True,  # Log routing decisions for observability
            # =================================================================
            # Citation Tracking
            # =================================================================
            "citation_format": "stable_id",
            "citation_id_length": 8,
            "verify_citations": True,
            "track_citations_db": True,
            "show_page_numbers": True,
            "show_confidence_scores": False,
            "include_source_preview": True,
            "citation_db_max_age_hours": 168,  # 1 week
            "citation_db_max_per_session": 1000,
            "citation_normalize_scores": True,
            "citation_score_method": "minmax",  # 'minmax', 'sigmoid', 'clamp'
            # =================================================================
            # File Types
            # =================================================================
            "supported_extensions": [
                ".pdf",
                ".docx",
                ".doc",
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".html",
                ".xml",
                ".pptx",
                ".xlsx",
                ".rtf",
                ".epub",
            ],
            # =================================================================
            # Storage
            # =================================================================
            "upload_dir_name": "rag_uploads",
            "db_name": "rag.db",  # Unified database (consolidated from rag_metadata.db + rag_citations.db)
            # =================================================================
            # Parallel Upload
            # =================================================================
            "max_concurrent_uploads": 5,  # Maximum parallel document uploads (1-20)
            # =================================================================
            # OCR
            # =================================================================
            "enable_ocr": False,
            "ocr_engine": "easyocr",
            "ocr_languages": ["en"],
            "ocr_confidence_threshold": 0.5,
            # =================================================================
            # Processing
            # =================================================================
            "extract_images": False,
            "preserve_formatting": True,
            "async_processing": True,
            "processing_timeout": 300,
            # =================================================================
            # Docling Document Processing
            # =================================================================
            "docling_enabled": True,  # Docling is default; falls back to legacy if not installed
            "docling_mode": "auto",  # 'auto', 'local', 'production'
            "docling_pdf_backend": "docling_parse",  # 'docling_parse' or 'pypdfium2'
            "docling_ocr_enabled": True,  # Enable OCR within Docling
            "docling_table_structure": True,  # Enable TableFormer table extraction
            "docling_max_tokens": 512,  # Max tokens per chunk (HybridChunker)
            "docling_timeout": 300,  # Per-document timeout (seconds)
            # =================================================================
            # Caching
            # =================================================================
            "enable_embedding_cache": True,
            "embedding_cache_size": 10000,
            "enable_search_cache": True,
            "search_cache_ttl": 300,
            "search_cache_max_size": 1000,  # Max cached search results
            # =================================================================
            # Security Limits
            # =================================================================
            "max_document_ids_filter": 100,  # Max document IDs in filter to prevent expression overflow
            # =================================================================
            # Observability
            # =================================================================
            "enable_prometheus": False,  # No-op metrics by default; True activates Prometheus
            "enable_tracing": False,  # No-op tracing by default; True activates OpenTelemetry
            "tracing_exporter": "otlp",  # 'otlp', 'jaeger', 'zipkin', 'console'
            "tracing_endpoint": "http://localhost:4317",  # OTLP gRPC endpoint
            "tracing_service_name": "rag-knowledge",
            "log_level": "INFO",
            # =================================================================
            # Multi-Tenancy
            # =================================================================
            "tenant_isolation": "none",  # 'none' = metadata filter only; 'partition' = native backend partitions
            "quota_enabled": False,  # Per-tenant resource quotas
            # =================================================================
            # Rate Limiting
            # =================================================================
            "rate_limiting_enabled": False,  # Adaptive rate limiter
            "rate_limiting_strategy": "adaptive",  # 'adaptive' or 'fixed'
            "rate_limiting_rpm": 60.0,  # Base requests per minute per tenant
            "rate_limiting_burst": 10.0,  # Max burst token capacity
            "rate_limiting_burst_multiplier": 2.0,  # Capacity multiplier when idle
            "rate_limiting_overload_threshold": 0.8,  # System load that triggers throttling
            "rate_limiting_idle_threshold": 0.3,  # System load below which bursting is allowed
            # =================================================================
            # Database Backend
            # =================================================================
            "database_backend": "sqlite",  # 'sqlite' (default) or 'postgresql'
            "database_dsn": "",  # PostgreSQL DSN (when database_backend='postgresql')
            "database_pool_min": 2,  # Min connection pool size (PostgreSQL)
            "database_pool_max": 10,  # Max connection pool size (PostgreSQL)
            # =================================================================
            # Redis / Distributed Cache
            # =================================================================
            "cache_backend": "memory",  # 'memory' (default) or 'redis'
            "redis_url": "redis://localhost:6379",  # Redis connection URL
            "redis_pool_size": 10,  # Redis connection pool size
            # =================================================================
            # Celery / Background Processing
            # =================================================================
            "celery_enabled": False,  # False = synchronous processing (default)
            "celery_broker_url": "redis://localhost:6379/0",
            "celery_result_backend": "redis://localhost:6379/1",
            # =================================================================
            # Connection Pooling
            # =================================================================
            "connection_pool_size": 1,  # Vector backend pool size (1 = no pooling)
            # =================================================================
            # File Storage Backend
            # =================================================================
            "storage_backend": "local",  # 'local' (default), 's3', 'fsspec'
            "storage_backend_config": {},  # Backend-specific configuration
            # =================================================================
            # Fast Ingestion (eval-only pipeline tuning)
            # =================================================================
            "ingest_extract_workers": min(os.cpu_count() or 4, 8),  # Parallel chunking workers
            "ingest_embed_batch_size": 64,  # Cross-doc embedding batch size
            "ingest_insert_batch_size": 500,  # Chunks per vector store insert call
        }

        # Merge with settings from TOML (settings.memory.rag takes precedence over defaults)
        config = {**default_config, **rag_settings}

        # Also merge settings from settings.lasagna.toml if available
        # This ensures settings.lasagna.toml values are used when settings.memory.rag doesn't override them
        knowledge_rag = self._knowledge_settings.get("rag", {})
        if hasattr(knowledge_rag, "to_dict"):
            knowledge_rag = knowledge_rag.to_dict()
        elif not isinstance(knowledge_rag, dict):
            knowledge_rag = {}

        # 12-factor: Load overrides from RAG_* environment variables.
        # Only keys that exist in default_config are accepted.
        env_overrides = self._load_env_overrides(default_config)
        self._env_overrides = env_overrides  # Store for priority lookup in _get_knowledge_setting

        # Merge order: defaults < knowledge.toml < settings.memory.rag < env vars
        config = {**default_config, **knowledge_rag, **rag_settings, **env_overrides}

        # Resolve paths using provider
        try:
            provider = self._get_provider()
            upload_dir = provider.upload_dir
            db_dir = provider.db_dir
        except RuntimeError:
            # No provider - use current directory fallback
            logger.warning("No config provider, using current directory for paths")
            cwd = Path(os.getcwd())
            upload_dir = cwd / "rag_uploads"
            db_dir = cwd / "dbs"

        config["upload_dir"] = str(upload_dir)

        db_name = config.get("db_name", "rag.db")
        config["db_path"] = str(db_dir / db_name)

        # Unified database: citation_db_path now points to same file as db_path
        # Kept for backward compatibility - code can still reference citation_db_path
        config["citation_db_path"] = config["db_path"]

        # Ensure directories exist
        os.makedirs(config["upload_dir"], exist_ok=True)
        os.makedirs(str(db_dir), exist_ok=True)

        logger.info(
            f"RAG config initialized: upload_dir={config['upload_dir']}, db_path={config['db_path']}"
        )

        return config

    @staticmethod
    def _load_env_overrides(default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load overrides from ``RAG_*`` environment variables (12-factor).

        Only keys that exist in *default_config* are accepted.  Values are
        coerced to the type of the default (bool, int, float, str, list).

        Example:
            ``RAG_DATABASE_BACKEND=postgresql``  →  ``{'database_backend': 'postgresql'}``
            ``RAG_ENABLE_RERANKING=true``         →  ``{'enable_reranking': True}``
        """
        overrides: Dict[str, Any] = {}
        for env_key, env_value in os.environ.items():
            if not env_key.startswith("RAG_"):
                continue
            config_key = env_key[4:].lower()  # Strip RAG_ prefix
            if config_key not in default_config:
                continue
            default_value = default_config[config_key]
            try:
                if isinstance(default_value, bool):
                    overrides[config_key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(default_value, int):
                    overrides[config_key] = int(env_value)
                elif isinstance(default_value, float):
                    overrides[config_key] = float(env_value)
                elif isinstance(default_value, (list, dict)):
                    import json as _json

                    overrides[config_key] = _json.loads(env_value)
                else:
                    overrides[config_key] = env_value
            except (ValueError, TypeError) as exc:
                logger.warning(f"Ignoring invalid env override {env_key}={env_value!r}: {exc}")

        # Validate numeric ranges for known keys
        _VALID_RANGES: Dict[str, tuple] = {
            "chunk_size": (50, 100_000),
            "chunk_overlap": (0, 50_000),
            "default_top_k": (1, 1000),
            "rerank_top_k": (1, 1000),
            "embedding_dim": (32, 8192),
            "hybrid_alpha": (0.0, 1.0),
            "mmr_lambda": (0.0, 1.0),
            "similarity_threshold": (-1.0, 1.0),
            "connection_pool_size": (1, 500),
            "routing_confidence_threshold": (0.0, 1.0),
        }
        for key, value in list(overrides.items()):
            if key in _VALID_RANGES and isinstance(value, (int, float)):
                lo, hi = _VALID_RANGES[key]
                if not (lo <= value <= hi):
                    logger.warning(
                        f"Ignoring out-of-range env override RAG_{key.upper()}={value} "
                        f"(valid: {lo}–{hi})"
                    )
                    del overrides[key]

        if overrides:
            logger.info(f"Applied {len(overrides)} RAG_* env overrides: {list(overrides.keys())}")
        return overrides

    def _get_knowledge_setting(self, *keys, default=None):
        """Safely get nested setting from knowledge settings.

        Priority order (highest to lowest):
        1. Session overrides (per-session UI changes)
        2. Environment variable overrides (RAG_* 12-factor)
        3. TOML settings (settings.lasagna.toml)
        4. default parameter (typically self._config[key], which merges
           defaults < knowledge.toml < settings.memory.rag < env vars)
        """
        # 1. Check session overrides first (highest priority)
        session_value = self._get_session_override(*keys)
        if session_value is not None:
            return session_value

        # 2. Check env var overrides — use the leaf key for flat lookup
        #    e.g., keys=('rag', 'enable_reranking') → leaf='enable_reranking'
        if self._env_overrides and len(keys) >= 1:
            leaf_key = keys[-1] if len(keys) > 1 else keys[0]
            if leaf_key in self._env_overrides:
                return self._env_overrides[leaf_key]

        # 3. Fall back to TOML settings
        obj = self._knowledge_settings
        for key in keys:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj if obj is not None else default

    # =========================================================================
    # Vector Database Properties
    # =========================================================================
    @property
    def vector_backend(self) -> str:
        """Vector database backend to use.

        Supported backends:
            - 'milvus' (default): Milvus Lite, Standalone, or Distributed
            - 'pinecone': Pinecone managed service
            - 'qdrant': Qdrant vector database
            - 'weaviate': Weaviate vector database
            - 'pgvector': PostgreSQL pgvector extension
            - 'opensearch': OpenSearch search engine (Lucene BM25 + k-NN vectors)

        Set via:
            - Environment variable: RAG_VECTOR_BACKEND (legacy: LASAGNA_VECTOR_BACKEND)
            - Config file: settings.lasagna.toml [rag] vector_backend
        """
        return self._get_knowledge_setting(
            "rag", "vector_backend", default=self._config["vector_backend"]
        )

    @property
    def vector_backend_config(self) -> Dict[str, Any]:
        """Backend-specific configuration.

        For Milvus:
            - db_path: Path to Milvus Lite database
            - uri: Milvus server URI
            - token: Authentication token

        For Pinecone:
            - api_key: Pinecone API key
            - environment: Pinecone environment
            - index_name: Index name

        For Qdrant:
            - url: Qdrant server URL
            - api_key: API key (optional)
            - collection_name: Collection name

        For Weaviate:
            - url: Weaviate server URL
            - api_key: API key (optional)
        """
        return self._get_knowledge_setting(
            "rag", "vector_backend_config", default=self._config["vector_backend_config"]
        )

    @property
    def collection_name(self) -> str:
        return self._get_knowledge_setting(
            "rag", "collection_name", default=self._config["collection_name"]
        )

    @property
    def embedding_dim(self) -> int:
        return self._get_knowledge_setting(
            "rag", "embedding_dim", default=self._config["embedding_dim"]
        )

    @property
    def embedding_truncate_dim(self) -> Optional[int]:
        """Matryoshka dimension truncation. None = use full dim."""
        return self._get_knowledge_setting(
            "rag", "embedding_truncate_dim", default=self._config["embedding_truncate_dim"]
        )

    @property
    def metric_type(self) -> str:
        return self._get_knowledge_setting(
            "rag", "metric_type", default=self._config["metric_type"]
        )

    @property
    def index_type(self) -> str:
        return self._get_knowledge_setting("rag", "index_type", default=self._config["index_type"])

    @property
    def index_params(self) -> Dict[str, Any]:
        return self._get_knowledge_setting(
            "rag", "index_params", default=self._config["index_params"]
        )

    @property
    def search_params(self) -> Dict[str, Any]:
        return self._get_knowledge_setting(
            "rag", "search_params", default=self._config["search_params"]
        )

    # =========================================================================
    # Embedding Model Properties
    # =========================================================================
    @property
    def embedding_provider(self) -> str:
        return self._get_knowledge_setting(
            "rag", "embedding_provider", default=self._config["embedding_provider"]
        )

    @property
    def embedding_model(self) -> str:
        return self._get_knowledge_setting(
            "rag", "embedding_model", default=self._config["embedding_model"]
        )

    @property
    def embedding_model_fallback(self) -> str:
        return self._get_knowledge_setting(
            "rag", "embedding_model_fallback", default=self._config["embedding_model_fallback"]
        )

    @property
    def embedding_backend(self) -> str:
        """Embedding backend: 'auto', 'sentence-transformers', or 'flag-embedding'."""
        return self._get_knowledge_setting(
            "rag", "embedding_backend", default=self._config["embedding_backend"]
        )

    @property
    def normalize_embeddings(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "normalize_embeddings", default=self._config["normalize_embeddings"]
        )

    @property
    def batch_size(self) -> int:
        return self._get_knowledge_setting("rag", "batch_size", default=self._config["batch_size"])

    @property
    def use_gpu(self) -> bool:
        # A hardware decision, not a research dimension: an explicit
        # RAG_USE_GPU wins over profile and per-query overrides. Profiles
        # carry use_gpu from the settings file, so as a per-query override it
        # outranked the operator's RAG_USE_GPU=false and moved models onto a
        # GPU backend the operator had turned off.
        env = os.environ.get("RAG_USE_GPU", "").strip().lower()
        if env:
            return env in ("1", "true", "yes", "on")
        return self._get_knowledge_setting("rag", "use_gpu", default=self._config["use_gpu"])

    # =========================================================================
    # Chunking Properties
    # =========================================================================
    @property
    def chunking_strategy(self) -> str:
        return self._get_knowledge_setting(
            "rag", "chunking_strategy", default=self._config["chunking_strategy"]
        )

    @property
    def chunk_size(self) -> int:
        return self._get_knowledge_setting("rag", "chunk_size", default=self._config["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return self._get_knowledge_setting(
            "rag", "chunk_overlap", default=self._config["chunk_overlap"]
        )

    @property
    def semantic_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "semantic_threshold", default=self._config["semantic_threshold"]
        )

    @property
    def min_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "min_chunk_size", default=self._config["min_chunk_size"]
        )

    @property
    def max_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "max_chunk_size", default=self._config["max_chunk_size"]
        )

    @property
    def enable_hierarchical(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_hierarchical", default=self._config["enable_hierarchical"]
        )

    @property
    def parent_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "parent_chunk_size", default=self._config["parent_chunk_size"]
        )

    @property
    def child_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "child_chunk_size", default=self._config["child_chunk_size"]
        )

    @property
    def auto_merge_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "auto_merge_threshold", default=self._config["auto_merge_threshold"]
        )

    @property
    def enable_late_chunking(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_late_chunking", default=self._config["enable_late_chunking"]
        )

    @property
    def preserve_tables(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "preserve_tables", default=self._config["preserve_tables"]
        )

    @property
    def preserve_headers(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "preserve_headers", default=self._config["preserve_headers"]
        )

    @property
    def max_content_length(self) -> int:
        """Maximum content length per chunk in storage (characters)."""
        return self._get_knowledge_setting(
            "rag", "max_content_length", default=self._config["max_content_length"]
        )

    @property
    def window_size(self) -> int:
        """Number of sentences per chunk in sentence_window strategy."""
        return self._get_knowledge_setting(
            "rag", "window_size", default=self._config["window_size"]
        )

    @property
    def sentence_overlap(self) -> int:
        """Number of sentences to overlap in sentence_window strategy."""
        return self._get_knowledge_setting(
            "rag", "sentence_overlap", default=self._config["sentence_overlap"]
        )

    # =========================================================================
    # Hybrid Search Properties
    # =========================================================================
    @property
    def enable_hybrid_search(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_hybrid_search", default=self._config["enable_hybrid_search"]
        )

    @property
    def hybrid_alpha(self) -> float:
        return self._get_knowledge_setting(
            "rag", "hybrid_alpha", default=self._config["hybrid_alpha"]
        )

    @property
    def sparse_method(self) -> str:
        return self._get_knowledge_setting(
            "rag", "sparse_method", default=self._config["sparse_method"]
        )

    @property
    def rrf_k(self) -> int:
        """RRF constant for Reciprocal Rank Fusion.

        Higher values give more weight to lower-ranked results.
        Typical range: 20-100, default: 60.
        """
        return self._get_knowledge_setting("rag", "rrf_k", default=self._config["rrf_k"])

    @property
    def enable_bm25_stemming(self) -> bool:
        """Enable Porter stemming for BM25.

        Improves recall by matching word stems (e.g., "running" matches "run").
        """
        return self._get_knowledge_setting(
            "rag", "enable_bm25_stemming", default=self._config["enable_bm25_stemming"]
        )

    @property
    def bm25_max_sessions(self) -> int:
        """Maximum number of in-memory BM25 session indices to keep."""
        return self._get_knowledge_setting(
            "rag", "bm25_max_sessions", default=self._config["bm25_max_sessions"]
        )

    @property
    def bm25_session_ttl_seconds(self) -> int:
        """Idle TTL for BM25 session indices in seconds."""
        return self._get_knowledge_setting(
            "rag", "bm25_session_ttl_seconds", default=self._config["bm25_session_ttl_seconds"]
        )

    # =========================================================================
    # Reranking Properties
    # =========================================================================
    @property
    def enable_reranking(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_reranking", default=self._config["enable_reranking"]
        )

    @property
    def reranker_model(self) -> str:
        return self._get_knowledge_setting(
            "rag", "reranker_model", default=self._config["reranker_model"]
        )

    @property
    def custom_reranker(self) -> str:
        """Registry name for a custom reranker component (empty string = use default)."""
        return self._get_knowledge_setting(
            "rag", "custom_reranker", default=self._config["custom_reranker"]
        )

    @property
    def custom_query_enhancer(self) -> str:
        """Registry name for a custom query enhancer component (empty string = use default)."""
        return self._get_knowledge_setting(
            "rag", "custom_query_enhancer", default=self._config["custom_query_enhancer"]
        )

    @property
    def llm_reranker_model(self) -> str:
        """Model for LLM listwise reranker (empty config = configured default)."""
        from lasagna.utils.model_utils import get_default_model_name

        return (
            self._get_knowledge_setting(
                "rag", "llm_reranker_model", default=self._config["llm_reranker_model"]
            )
            or get_default_model_name()
        )

    @property
    def llm_reranker_window_size(self) -> int:
        """Sliding window size for listwise reranking."""
        return self._get_knowledge_setting(
            "rag", "llm_reranker_window_size", default=self._config["llm_reranker_window_size"]
        )

    @property
    def colbert_model(self) -> str:
        """ColBERT late-interaction model name."""
        return self._get_knowledge_setting(
            "rag", "colbert_model", default=self._config["colbert_model"]
        )

    @property
    def rerank_top_k(self) -> int:
        return self._get_knowledge_setting(
            "rag", "rerank_top_k", default=self._config["rerank_top_k"]
        )

    @property
    def enable_context_reordering(self) -> bool:
        """Lost-in-the-middle fix: reorder results so best are at start and end."""
        return self._get_knowledge_setting(
            "rag", "enable_context_reordering", default=self._config["enable_context_reordering"]
        )

    @property
    def enable_graph_retrieval(self) -> bool:
        """Enable GraphRAG entity-linked retrieval."""
        return self._get_knowledge_setting(
            "rag", "enable_graph_retrieval", default=self._config["enable_graph_retrieval"]
        )

    @property
    def graph_max_hops(self) -> int:
        """Max traversal depth in knowledge graph."""
        return self._get_knowledge_setting(
            "rag", "graph_max_hops", default=self._config["graph_max_hops"]
        )

    @property
    def graph_entity_types(self) -> list:
        """Entity types for graph extraction."""
        return self._get_knowledge_setting(
            "rag", "graph_entity_types", default=self._config["graph_entity_types"]
        )

    @property
    def agentic_prompts(self) -> dict:
        """Per-node prompt overrides for the agentic graph, keyed by node name.

        See ``lasagna.agentic.prompts.describe_nodes()`` for the nodes and the
        placeholders each one supplies.
        """
        return self._get_knowledge_setting("rag", "agentic_prompts", default={}) or {}

    @property
    def agentic_models(self) -> dict:
        """Per-node model overrides for the agentic graph, keyed by node name.

        Nodes do different jobs -- a grounding check is a classification, a
        generation is not -- so measuring a small model on the gates against a
        large one on generation is a normal experiment.
        """
        return self._get_knowledge_setting("rag", "agentic_models", default={}) or {}

    @property
    def graph_extraction_model(self) -> str:
        """Model for entity extraction (empty config = configured default)."""
        from lasagna.utils.model_utils import get_default_model_name

        return (
            self._get_knowledge_setting(
                "rag", "graph_extraction_model", default=self._config["graph_extraction_model"]
            )
            or get_default_model_name()
        )

    @property
    def enable_raptor(self) -> bool:
        """Enable RAPTOR recursive summary tree indexing."""
        return self._get_knowledge_setting(
            "rag", "enable_raptor", default=self._config["enable_raptor"]
        )

    @property
    def raptor_max_cluster_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "raptor_max_cluster_size", default=self._config["raptor_max_cluster_size"]
        )

    @property
    def raptor_max_levels(self) -> int:
        return self._get_knowledge_setting(
            "rag", "raptor_max_levels", default=self._config["raptor_max_levels"]
        )

    # raptor_retrieval_mode: removed — no query-time consumer exists.
    # All RAPTOR nodes are searched via standard vector search (collapsed mode).
    # Restore this property when tree-traversal retrieval is wired into the search orchestrator.

    @property
    def enable_context_compression(self) -> bool:
        """Enable context compression to reduce noise/tokens after reranking."""
        return self._get_knowledge_setting(
            "rag", "enable_context_compression", default=self._config["enable_context_compression"]
        )

    @property
    def compression_method(self) -> str:
        """Compression method: 'extractive' or 'abstractive'."""
        return self._get_knowledge_setting(
            "rag", "compression_method", default=self._config["compression_method"]
        )

    @property
    def compression_ratio(self) -> float:
        """Target compression ratio (0.3 = keep 30%)."""
        return self._get_knowledge_setting(
            "rag", "compression_ratio", default=self._config["compression_ratio"]
        )

    @property
    def enable_mmr(self) -> bool:
        return self._get_knowledge_setting("rag", "enable_mmr", default=self._config["enable_mmr"])

    @property
    def mmr_lambda(self) -> float:
        return self._get_knowledge_setting("rag", "mmr_lambda", default=self._config["mmr_lambda"])

    @property
    def mmr_score_normalization(self) -> str:
        """MMR score normalization strategy.

        Options:
            - 'sigmoid': Apply sigmoid to each score (recommended, preserves calibration)
            - 'minmax': Apply min-max scaling
            - 'sigmoid_minmax': Apply sigmoid then minmax (destroys calibration)
            - 'none': No normalization (legacy behavior)
        """
        return self._get_knowledge_setting(
            "rag", "mmr_score_normalization", default=self._config["mmr_score_normalization"]
        )

    @property
    def reranker_batch_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "reranker_batch_size", default=self._config["reranker_batch_size"]
        )

    # =========================================================================
    # Query Enhancement Properties
    # =========================================================================
    @property
    def enable_hyde(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_hyde", default=self._config["enable_hyde"]
        )

    @property
    def hyde_model(self) -> str:
        return self._get_knowledge_setting("rag", "hyde_model", default=self._config["hyde_model"])

    @property
    def hyde_n_docs(self) -> int:
        """Number of hypothetical documents for multi-HyDE.

        Set to 1 for legacy single-document behavior.
        Recommended: 3-5 for better retrieval quality.
        """
        return self._get_knowledge_setting(
            "rag", "hyde_n_docs", default=self._config["hyde_n_docs"]
        )

    @property
    def hyde_embedding_strategy(self) -> str:
        """Strategy for combining multi-HyDE embeddings.

        Options:
            - 'average': Average all embeddings (recommended)
            - 'max_similarity': Use embedding most similar to query
        """
        return self._get_knowledge_setting(
            "rag", "hyde_embedding_strategy", default=self._config["hyde_embedding_strategy"]
        )

    @property
    def enable_query_classification(self) -> bool:
        """Enable adaptive query-type classification.

        When enabled, queries are classified into types (factual, conceptual,
        navigational, comparative) and retrieval parameters are adjusted accordingly.
        """
        return self._get_knowledge_setting(
            "rag",
            "enable_query_classification",
            default=self._config["enable_query_classification"],
        )

    @property
    def enable_query_expansion(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_query_expansion", default=self._config["enable_query_expansion"]
        )

    @property
    def max_query_variations(self) -> int:
        return self._get_knowledge_setting(
            "rag", "max_query_variations", default=self._config["max_query_variations"]
        )

    @property
    def enable_query_decomposition(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_query_decomposition", default=self._config["enable_query_decomposition"]
        )

    @property
    def agentic_hyde_strategy(self) -> str:
        return self._get_knowledge_setting(
            "rag",
            "agentic_hyde_strategy",
            default=self._config.get("agentic_hyde_strategy", "adaptive"),
        )

    @property
    def enable_contextual_retrieval(self) -> bool:
        """Enable Anthropic-style contextual retrieval.

        When enabled, each chunk is enriched with LLM-generated context
        at index time, reducing retrieval failure rates by ~67%.
        Reference: https://www.anthropic.com/news/contextual-retrieval
        """
        return self._get_knowledge_setting(
            "rag",
            "enable_contextual_retrieval",
            default=self._config["enable_contextual_retrieval"],
        )

    @property
    def enable_rag_fusion(self) -> bool:
        """Enable RAG Fusion: multi-query generation + Reciprocal Rank Fusion.

        Generates multiple query variations and merges results using RRF,
        improving comprehensiveness by ~8-10% accuracy.
        """
        return self._get_knowledge_setting(
            "rag", "enable_rag_fusion", default=self._config["enable_rag_fusion"]
        )

    @property
    def rag_fusion_n_queries(self) -> int:
        """Number of query variations to generate for RAG Fusion."""
        return self._get_knowledge_setting(
            "rag", "rag_fusion_n_queries", default=self._config["rag_fusion_n_queries"]
        )

    @property
    def enable_crag(self) -> bool:
        """Enable Corrective RAG: LLM grades retrieval quality post-search.

        If retrieval quality is poor, the query is refined or the system
        falls back to web search (when crag_web_search_fallback is enabled).
        """
        return self._get_knowledge_setting(
            "rag", "enable_crag", default=self._config["enable_crag"]
        )

    @property
    def crag_web_search_fallback(self) -> bool:
        """Enable web search fallback for Corrective RAG.

        When CRAG determines retrieval quality is low, fall back to
        web search to supplement results.
        """
        return self._get_knowledge_setting(
            "rag", "crag_web_search_fallback", default=self._config["crag_web_search_fallback"]
        )

    # =========================================================================
    # Query Routing Properties (Adaptive-RAG)
    # =========================================================================

    @property
    def enable_query_routing(self) -> bool:
        """Enable intelligent query routing to FAST/STANDARD/COMPREHENSIVE tiers."""
        return self._get_knowledge_setting(
            "rag", "enable_query_routing", default=self._config["enable_query_routing"]
        )

    @property
    def routing_strategy(self) -> str:
        """Routing strategy: 'rule_based' (deterministic) or 'adaptive' (confidence-aware)."""
        return self._get_knowledge_setting(
            "rag", "routing_strategy", default=self._config["routing_strategy"]
        )

    @property
    def routing_default_tier(self) -> str:
        """Default routing tier when routing is disabled."""
        return self._get_knowledge_setting(
            "rag", "routing_default_tier", default=self._config["routing_default_tier"]
        )

    @property
    def routing_confidence_threshold(self) -> float:
        """Confidence threshold below which the adaptive strategy escalates the tier."""
        return self._get_knowledge_setting(
            "rag",
            "routing_confidence_threshold",
            default=self._config["routing_confidence_threshold"],
        )

    @property
    def routing_allow_tier_override(self) -> bool:
        """Allow API callers to force a specific routing tier."""
        return self._get_knowledge_setting(
            "rag",
            "routing_allow_tier_override",
            default=self._config["routing_allow_tier_override"],
        )

    @property
    def routing_log_decisions(self) -> bool:
        """Log routing decisions for observability."""
        return self._get_knowledge_setting(
            "rag", "routing_log_decisions", default=self._config["routing_log_decisions"]
        )

    # =========================================================================
    # Observability Properties
    # =========================================================================
    @property
    def enable_prometheus(self) -> bool:
        """Enable Prometheus metrics collection.

        When True, all existing METRICS.counter() / METRICS.histogram() calls
        throughout the search pipeline emit real Prometheus metrics.
        Requires: pip install prometheus-client>=0.20.0
        """
        return self._get_knowledge_setting(
            "rag", "enable_prometheus", default=self._config["enable_prometheus"]
        )

    @property
    def enable_tracing(self) -> bool:
        """Enable OpenTelemetry distributed tracing.

        When True, search pipeline stages are wrapped in trace spans for
        end-to-end latency visibility (viewable in Jaeger, Zipkin, etc.).
        Requires: pip install opentelemetry-api opentelemetry-sdk
        """
        return self._get_knowledge_setting(
            "rag", "enable_tracing", default=self._config["enable_tracing"]
        )

    @property
    def tracing_exporter(self) -> str:
        """Tracing exporter: 'otlp', 'jaeger', 'zipkin', or 'console'."""
        return self._get_knowledge_setting(
            "rag", "tracing_exporter", default=self._config["tracing_exporter"]
        )

    @property
    def tracing_endpoint(self) -> str:
        """OTLP/Jaeger/Zipkin collector endpoint."""
        return self._get_knowledge_setting(
            "rag", "tracing_endpoint", default=self._config["tracing_endpoint"]
        )

    @property
    def tracing_service_name(self) -> str:
        """Service name reported in traces."""
        return self._get_knowledge_setting(
            "rag", "tracing_service_name", default=self._config["tracing_service_name"]
        )

    # =========================================================================
    # Multi-Tenancy Properties
    # =========================================================================
    @property
    def tenant_isolation(self) -> str:
        """Tenant isolation mode: 'none' or 'partition'.

        - ``none``: Current metadata-filter behaviour (default, single-user).
        - ``partition``: Native backend partitions (Milvus, Pinecone) with
          metadata filter as defense-in-depth.
        """
        return self._get_knowledge_setting(
            "rag", "tenant_isolation", default=self._config["tenant_isolation"]
        )

    @property
    def quota_enabled(self) -> bool:
        """Enable per-tenant resource quotas (documents, storage, rate limits)."""
        return self._get_knowledge_setting(
            "rag", "quota_enabled", default=self._config["quota_enabled"]
        )

    # =========================================================================
    # Rate Limiting Properties
    # =========================================================================
    @property
    def rate_limiting_enabled(self) -> bool:
        """Enable adaptive rate limiting per tenant."""
        return self._get_knowledge_setting(
            "rag", "rate_limiting_enabled", default=self._config["rate_limiting_enabled"]
        )

    @property
    def rate_limiting_strategy(self) -> str:
        """Rate limiting strategy: 'adaptive' or 'fixed'."""
        return self._get_knowledge_setting(
            "rag", "rate_limiting_strategy", default=self._config["rate_limiting_strategy"]
        )

    @property
    def rate_limiting_rpm(self) -> float:
        """Base requests per minute per tenant."""
        return float(
            self._get_knowledge_setting(
                "rag", "rate_limiting_rpm", default=self._config["rate_limiting_rpm"]
            )
        )

    @property
    def rate_limiting_burst(self) -> float:
        """Maximum burst token capacity per tenant."""
        return float(
            self._get_knowledge_setting(
                "rag", "rate_limiting_burst", default=self._config["rate_limiting_burst"]
            )
        )

    @property
    def rate_limiting_burst_multiplier(self) -> float:
        """Capacity multiplier when system is idle (allows bursting)."""
        return float(
            self._get_knowledge_setting(
                "rag",
                "rate_limiting_burst_multiplier",
                default=self._config["rate_limiting_burst_multiplier"],
            )
        )

    @property
    def rate_limiting_overload_threshold(self) -> float:
        """System load score that triggers throttling (0.0–1.0)."""
        return float(
            self._get_knowledge_setting(
                "rag",
                "rate_limiting_overload_threshold",
                default=self._config["rate_limiting_overload_threshold"],
            )
        )

    @property
    def rate_limiting_idle_threshold(self) -> float:
        """System load score below which bursting is allowed (0.0–1.0)."""
        return float(
            self._get_knowledge_setting(
                "rag",
                "rate_limiting_idle_threshold",
                default=self._config["rate_limiting_idle_threshold"],
            )
        )

    # =========================================================================
    # Database Backend Properties
    # =========================================================================
    @property
    def database_backend(self) -> str:
        """Database backend: 'sqlite' (default) or 'postgresql'."""
        return self._get_knowledge_setting(
            "rag", "database_backend", default=self._config["database_backend"]
        )

    @property
    def database_dsn(self) -> str:
        """PostgreSQL connection string (DSN).  Required when database_backend='postgresql'."""
        return self._get_knowledge_setting(
            "rag", "database_dsn", default=self._config["database_dsn"]
        )

    @property
    def database_pool_min(self) -> int:
        """Minimum database connection pool size (PostgreSQL)."""
        return int(
            self._get_knowledge_setting(
                "rag", "database_pool_min", default=self._config["database_pool_min"]
            )
        )

    @property
    def database_pool_max(self) -> int:
        """Maximum database connection pool size (PostgreSQL)."""
        return int(
            self._get_knowledge_setting(
                "rag", "database_pool_max", default=self._config["database_pool_max"]
            )
        )

    # =========================================================================
    # Redis / Distributed Cache Properties
    # =========================================================================
    @property
    def cache_backend(self) -> str:
        """Cache backend: 'memory' (default) or 'redis'."""
        return self._get_knowledge_setting(
            "rag", "cache_backend", default=self._config["cache_backend"]
        )

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return self._get_knowledge_setting("rag", "redis_url", default=self._config["redis_url"])

    @property
    def redis_pool_size(self) -> int:
        """Redis connection pool size."""
        return int(
            self._get_knowledge_setting(
                "rag", "redis_pool_size", default=self._config["redis_pool_size"]
            )
        )

    # =========================================================================
    # Celery / Background Processing Properties
    # =========================================================================
    @property
    def celery_enabled(self) -> bool:
        """Enable Celery for background document processing."""
        return self._get_knowledge_setting(
            "rag", "celery_enabled", default=self._config["celery_enabled"]
        )

    @property
    def celery_broker_url(self) -> str:
        """Celery broker URL (typically Redis)."""
        return self._get_knowledge_setting(
            "rag", "celery_broker_url", default=self._config["celery_broker_url"]
        )

    @property
    def celery_result_backend(self) -> str:
        """Celery result backend URL (typically Redis)."""
        return self._get_knowledge_setting(
            "rag", "celery_result_backend", default=self._config["celery_result_backend"]
        )

    # =========================================================================
    # Connection Pooling Properties
    # =========================================================================
    @property
    def connection_pool_size(self) -> int:
        """Vector backend connection pool size.  1 = no pooling (default)."""
        return int(
            self._get_knowledge_setting(
                "rag", "connection_pool_size", default=self._config["connection_pool_size"]
            )
        )

    # =========================================================================
    # File Storage Backend Properties
    # =========================================================================
    @property
    def storage_backend(self) -> str:
        """File storage backend: 'local' (default), 's3', or 'fsspec'.

        Supported backends:
            - 'local': Local filesystem (zero-config, built-in)
            - 's3': S3-compatible (AWS S3, MinIO, R2, Backblaze, Wasabi, 15+ services)
            - 'fsspec': Universal (GCS, Azure, FTP, SFTP, HDFS, 30+ backends)

        Set via:
            - Environment variable: RAG_STORAGE_BACKEND
            - Config file: settings.lasagna.toml [rag] storage_backend
        """
        return self._get_knowledge_setting(
            "rag", "storage_backend", default=self._config["storage_backend"]
        )

    @property
    def storage_backend_config(self) -> Dict[str, Any]:
        """Backend-specific storage configuration.

        For S3:
            - bucket: S3 bucket name (required)
            - prefix: Key prefix (default: "rag-documents")
            - endpoint_url: S3-compatible endpoint (for MinIO, R2, etc.)
            - region_name: AWS region (default: "us-east-1")

        For fsspec:
            - protocol: Filesystem protocol (e.g. "gcs", "az", "ftp")
            - base_path: Base path within the filesystem
            - storage_options: Dict of fsspec storage options

        Set via:
            - Environment variable: RAG_STORAGE_BACKEND_CONFIG (JSON string)
            - Config file: settings.lasagna.toml [rag] storage_backend_config
        """
        return self._get_knowledge_setting(
            "rag", "storage_backend_config", default=self._config["storage_backend_config"]
        )

    # =========================================================================
    # Logging Properties
    # =========================================================================
    @property
    def log_level(self) -> str:
        value = self._get_knowledge_setting("rag", "log_level", default=self._config["log_level"])
        return str(value).upper()

    # =========================================================================
    # Citation Properties
    # =========================================================================
    @property
    def citation_format(self) -> str:
        return self._get_knowledge_setting(
            "rag", "citation_format", default=self._config["citation_format"]
        )

    @property
    def citation_id_length(self) -> int:
        return self._get_knowledge_setting(
            "rag", "citation_id_length", default=self._config["citation_id_length"]
        )

    @property
    def verify_citations(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "verify_citations", default=self._config["verify_citations"]
        )

    @property
    def track_citations_db(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "track_citations_db", default=self._config["track_citations_db"]
        )

    @property
    def show_page_numbers(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "show_page_numbers", default=self._config["show_page_numbers"]
        )

    @property
    def citation_db_path(self) -> str:
        return self._config["citation_db_path"]

    @property
    def citation_db_max_age_hours(self) -> int:
        return self._get_knowledge_setting(
            "rag", "citation_db_max_age_hours", default=self._config["citation_db_max_age_hours"]
        )

    @property
    def citation_db_max_per_session(self) -> int:
        return self._get_knowledge_setting(
            "rag",
            "citation_db_max_per_session",
            default=self._config["citation_db_max_per_session"],
        )

    @property
    def citation_normalize_scores(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "citation_normalize_scores", default=self._config["citation_normalize_scores"]
        )

    @property
    def citation_score_method(self) -> str:
        return self._get_knowledge_setting(
            "rag", "citation_score_method", default=self._config["citation_score_method"]
        )

    # =========================================================================
    # Search Properties
    # =========================================================================
    @property
    def default_top_k(self) -> int:
        return self._get_knowledge_setting(
            "rag", "default_top_k", default=self._config["default_top_k"]
        )

    @property
    def similarity_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "similarity_threshold", default=self._config["similarity_threshold"]
        )

    @property
    def max_file_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "max_file_size", default=self._config["max_file_size"]
        )

    @property
    def supported_extensions(self) -> List[str]:
        return self._get_knowledge_setting(
            "rag", "supported_extensions", default=self._config["supported_extensions"]
        )

    # =========================================================================
    # Storage Properties
    # =========================================================================
    @property
    def upload_dir(self) -> str:
        return self._config["upload_dir"]

    @property
    def db_path(self) -> str:
        return self._config["db_path"]

    @property
    def max_concurrent_uploads(self) -> int:
        """Maximum number of parallel document uploads (1-20)."""
        return self._get_knowledge_setting(
            "rag", "max_concurrent_uploads", default=self._config["max_concurrent_uploads"]
        )

    # =========================================================================
    # OCR Properties
    # =========================================================================
    @property
    def enable_ocr(self) -> bool:
        return self._get_knowledge_setting("rag", "enable_ocr", default=self._config["enable_ocr"])

    @property
    def ocr_engine(self) -> str:
        return self._get_knowledge_setting("rag", "ocr_engine", default=self._config["ocr_engine"])

    @property
    def ocr_languages(self) -> List[str]:
        return self._get_knowledge_setting(
            "rag", "ocr_languages", default=self._config["ocr_languages"]
        )

    @property
    def ocr_confidence_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "ocr_confidence_threshold", default=self._config["ocr_confidence_threshold"]
        )

    # =========================================================================
    # Processing Properties
    # =========================================================================
    @property
    def extract_images(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "extract_images", default=self._config["extract_images"]
        )

    @property
    def preserve_formatting(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "preserve_formatting", default=self._config["preserve_formatting"]
        )

    @property
    def async_processing(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "async_processing", default=self._config["async_processing"]
        )

    # =========================================================================
    # Docling Properties
    # =========================================================================
    @property
    def docling_enabled(self) -> bool:
        """Enable Docling for structure-aware document processing."""
        return self._get_knowledge_setting(
            "rag", "docling_enabled", default=self._config["docling_enabled"]
        )

    @property
    def docling_mode(self) -> str:
        """Docling deployment mode: 'auto', 'local', or 'production'."""
        return self._get_knowledge_setting(
            "rag", "docling_mode", default=self._config["docling_mode"]
        )

    @property
    def docling_pdf_backend(self) -> str:
        """Docling PDF backend: 'docling_parse' or 'pypdfium2'."""
        return self._get_knowledge_setting(
            "rag", "docling_pdf_backend", default=self._config["docling_pdf_backend"]
        )

    @property
    def docling_ocr_enabled(self) -> bool:
        """Enable OCR within Docling pipeline."""
        return self._get_knowledge_setting(
            "rag", "docling_ocr_enabled", default=self._config["docling_ocr_enabled"]
        )

    @property
    def docling_table_structure(self) -> bool:
        """Enable TableFormer table structure extraction."""
        return self._get_knowledge_setting(
            "rag", "docling_table_structure", default=self._config["docling_table_structure"]
        )

    @property
    def docling_max_tokens(self) -> int:
        """Max tokens per chunk for Docling HybridChunker."""
        return int(
            self._get_knowledge_setting(
                "rag", "docling_max_tokens", default=self._config["docling_max_tokens"]
            )
        )

    @property
    def docling_timeout(self) -> int:
        """Per-document processing timeout in seconds."""
        return int(
            self._get_knowledge_setting(
                "rag", "docling_timeout", default=self._config["docling_timeout"]
            )
        )

    # =========================================================================
    # Caching Properties
    # =========================================================================
    @property
    def enable_embedding_cache(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_embedding_cache", default=self._config["enable_embedding_cache"]
        )

    @property
    def embedding_cache_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "embedding_cache_size", default=self._config["embedding_cache_size"]
        )

    @property
    def enable_search_cache(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_search_cache", default=self._config["enable_search_cache"]
        )

    @property
    def search_cache_ttl(self) -> int:
        return self._get_knowledge_setting(
            "rag", "search_cache_ttl", default=self._config["search_cache_ttl"]
        )

    @property
    def search_cache_max_size(self) -> int:
        """Maximum number of cached search results."""
        return self._get_knowledge_setting(
            "rag", "search_cache_max_size", default=self._config["search_cache_max_size"]
        )

    @property
    def max_document_ids_filter(self) -> int:
        """Maximum document IDs allowed in filter expressions.

        HIGH PRIORITY FIX: Prevents Milvus filter expression overflow when
        too many document IDs are passed, which can cause query failures.
        """
        return self._get_knowledge_setting(
            "rag", "max_document_ids_filter", default=self._config["max_document_ids_filter"]
        )

    # =========================================================================
    # Agentic RAG Properties (existing)
    # =========================================================================
    @property
    def use_agentic_rag(self) -> bool:
        return self._get_knowledge_setting("rag", "use_agentic_rag", default=True)

    @property
    def use_final_search(self) -> bool:
        return self._get_knowledge_setting("rag", "use_final_search", default=False)

    @property
    def chunk_cite_retrieval(self) -> Dict[str, Any]:
        ccr = self._get_knowledge_setting("rag", "chunk_cite_retrieval", default=None)
        if ccr:
            if hasattr(ccr, "to_dict"):
                return ccr.to_dict()
            return ccr
        return {
            "strategy": "adaptive",
            "fixed_size": 5,
            "threshold_score": 0.5,
            "max_chunks": 20,
            "min_chunks": 2,
            "score_gap_buffer": 2,
            "min_gap_ratio": 0.15,
        }

    @property
    def chunk_refinement_method(self) -> str:
        return self._get_knowledge_setting("rag", "chunk_refinement_method", default="auto")

    @property
    def refinement_model(self) -> str:
        from lasagna.utils.model_utils import get_default_model_name

        return self._get_knowledge_setting(
            "rag", "refinement_model", default=get_default_model_name()
        )

    @property
    def max_agent_iterations(self) -> int:
        """Maximum number of search iterations for agentic RAG."""
        return self._get_knowledge_setting("rag", "max_agent_iterations", default=3)

    @property
    def agent_temperature(self) -> float:
        """Temperature for agent/refinement model (low for determinism)."""
        return self._get_knowledge_setting("rag", "agent_temperature", default=0.1)

    @property
    def max_query_length(self) -> int:
        """Maximum query length in tokens for refinement queries."""
        return self._get_knowledge_setting("rag", "max_query_length", default=12)

    @property
    def auto_refine_avg_length_threshold(self) -> int:
        """Average chunk length threshold to trigger auto-refinement."""
        return self._get_knowledge_setting("rag", "auto_refine_avg_length_threshold", default=1000)

    @property
    def auto_refine_avg_score_threshold(self) -> float:
        """Average score threshold below which to trigger auto-refinement."""
        return self._get_knowledge_setting("rag", "auto_refine_avg_score_threshold", default=0.5)

    @property
    def auto_refine_chunk_count_threshold(self) -> int:
        """Chunk count threshold above which to trigger auto-refinement."""
        return self._get_knowledge_setting("rag", "auto_refine_chunk_count_threshold", default=7)

    # =========================================================================
    # Structured Graph Properties (Phase 1d)
    # =========================================================================
    @property
    def use_structured_graph(self) -> bool:
        """Use structured StateGraph instead of flat ReAct agent."""
        return self._get_knowledge_setting("rag", "use_structured_graph", default=False)

    @property
    def agentic_pipeline(self) -> str:
        """Name of the registered agentic pipeline to use.

        Precedence (highest to lowest):
          1. Session override
          2. TOML / env var ``RAG_AGENTIC_PIPELINE``
          3. Backward compat: ``"default"`` when ``use_structured_graph=True``
          4. ``"legacy_react"`` (default)
        """
        override = self._get_session_override("rag", "agentic_pipeline")
        if override:
            return str(override)
        val = self._get_knowledge_setting("rag", "agentic_pipeline", default=None)
        if val:
            return str(val)
        if self.use_structured_graph:
            return "default"
        return "legacy_react"

    @property
    def enable_hallucination_check(self) -> bool:
        """Enable post-generation hallucination grounding check (ISSUP)."""
        return self._get_knowledge_setting("rag", "enable_hallucination_check", default=False)

    @property
    def enable_answer_quality_check(self) -> bool:
        """Enable post-generation answer quality check (ISUSE)."""
        return self._get_knowledge_setting("rag", "enable_answer_quality_check", default=False)

    @property
    def max_quality_check_attempts(self) -> int:
        """Maximum retry attempts for quality gate failures."""
        return self._get_knowledge_setting("rag", "max_quality_check_attempts", default=2)

    @property
    def enable_speculative_rag(self) -> bool:
        """Enable speculative RAG: parallel draft generation + verification."""
        return self._get_knowledge_setting("rag", "enable_speculative_rag", default=False)

    @property
    def speculative_n_drafts(self) -> int:
        """Number of parallel draft answers for speculative RAG."""
        return self._get_knowledge_setting("rag", "speculative_n_drafts", default=3)

    @property
    def speculative_subset_size(self) -> int:
        """Number of context chunks per draft in speculative RAG."""
        return self._get_knowledge_setting("rag", "speculative_subset_size", default=3)

    @property
    def coverage_scorer_mode(self) -> str:
        """Coverage assessment mode: 'keyword' (fast) or 'llm' (accurate)."""
        return self._get_knowledge_setting("rag", "coverage_scorer_mode", default="keyword")

    @property
    def final_search_mode(self) -> str:
        """Final search behavior: 'merge' (combine hop + final) or 'replace' (legacy)."""
        return self._get_knowledge_setting("rag", "final_search_mode", default="merge")

    # =========================================================================
    # Document Awareness Properties
    # =========================================================================
    @property
    def document_awareness_enabled(self) -> bool:
        """Whether document awareness is enabled for agent prompts."""
        return self._get_knowledge_setting("rag", "document_awareness", "enabled", default=True)

    @property
    def enrichment_strategy(self) -> str:
        """Document enrichment strategy: first_tokens, llm_summary, or hybrid."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "enrichment_strategy", default="first_tokens"
        )

    @property
    def first_tokens_count(self) -> int:
        """Number of tokens to extract from document start."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "first_tokens_count", default=150
        )

    @property
    def include_filename(self) -> bool:
        """Include filename in document snippet."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "include_filename", default=True
        )

    @property
    def include_upload_date(self) -> bool:
        """Include upload date in document snippet."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "include_upload_date", default=True
        )

    @property
    def include_file_type(self) -> bool:
        """Include file type in document snippet."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "include_file_type", default=True
        )

    @property
    def summary_model(self) -> str:
        """Model for generating LLM summaries."""
        from lasagna.utils.model_utils import get_default_model_name

        return self._get_knowledge_setting(
            "rag", "document_awareness", "summary_model", default=get_default_model_name()
        )

    @property
    def summary_max_tokens(self) -> int:
        """Max tokens for document summaries."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "summary_max_tokens", default=100
        )

    @property
    def summary_prompt(self) -> str:
        """Prompt for generating document summaries."""
        default = "Summarize this document in 1-2 sentences, focusing on the main topic and key information:"
        return self._get_knowledge_setting(
            "rag", "document_awareness", "summary_prompt", default=default
        )

    @property
    def hybrid_llm_threshold_kb(self) -> int:
        """Documents larger than this (KB) get LLM summary in hybrid mode."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "hybrid_llm_threshold_kb", default=100
        )

    @property
    def max_documents_in_context(self) -> int:
        """Maximum documents to include in agent context."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "max_documents_in_context", default=10
        )

    @property
    def cache_snippets(self) -> bool:
        """Whether to cache document snippets."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "cache_snippets", default=True
        )

    @property
    def snippet_cache_ttl(self) -> int:
        """Cache lifetime for snippets in seconds."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "snippet_cache_ttl", default=3600
        )

    # =========================================================================
    # Fast Ingestion
    # =========================================================================
    @property
    def ingest_extract_workers(self) -> int:
        """Number of parallel workers for text extraction/chunking in fast ingest."""
        return int(
            self._get_knowledge_setting(
                "rag", "ingest_extract_workers", default=self._config["ingest_extract_workers"]
            )
        )

    @property
    def ingest_embed_batch_size(self) -> int:
        """Cross-document embedding batch size for fast ingest."""
        return int(
            self._get_knowledge_setting(
                "rag", "ingest_embed_batch_size", default=self._config["ingest_embed_batch_size"]
            )
        )

    @property
    def ingest_insert_batch_size(self) -> int:
        """Chunks per vector store insert call in fast ingest."""
        return int(
            self._get_knowledge_setting(
                "rag", "ingest_insert_batch_size", default=self._config["ingest_insert_batch_size"]
            )
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)

    def set_override(self, key: str, value: Any) -> None:
        """Set a runtime config override (same priority as RAG_* env vars).

        Writes into the env-override layer (priority 2 in
        ``_get_knowledge_setting``), so it beats TOML and defaults but is
        overridden by explicit session overrides.

        Note: ``with_session()`` deep-copies ``_env_overrides``, so this
        override is visible to session copies created *after* this call
        but does not propagate to copies that already exist.

        Args:
            key: Config key (e.g., ``'vector_backend'``, ``'enable_reranking'``)
            value: Config value
        """
        self._env_overrides[key] = value

    def clear_overrides(self) -> None:
        """Remove all runtime overrides, reload from ``RAG_*`` env vars only.

        Uses ``_config`` values as the defaults dict for ``_load_env_overrides``
        so that key-validation and type-coercion match the original load.
        This is safe because ``_config`` is never mutated after ``__init__``.
        """
        default_keys = {
            k for k in self._config if k not in ("upload_dir", "db_path", "citation_db_path")
        }
        self._env_overrides = self._load_env_overrides({k: self._config[k] for k in default_keys})


# ---------------------------------------------------------------------------
# Thread-safe lazy singleton
# ---------------------------------------------------------------------------
_rag_config: Optional[RAGConfig] = None
_rag_config_lock = Lock()


def get_rag_config() -> RAGConfig:
    """Get or create the global RAGConfig singleton.

    Uses double-check locking for thread-safe lazy initialization.
    Prefer this over the module-level ``rag_config`` name in new code.
    """
    global _rag_config
    if _rag_config is None:
        with _rag_config_lock:
            if _rag_config is None:
                _rag_config = RAGConfig()
    return _rag_config


def _reset_rag_config() -> None:
    """Reset the global RAGConfig singleton (for testing only)."""
    global _rag_config
    with _rag_config_lock:
        _rag_config = None


def reset_rag_config() -> None:
    """Reset the global RAGConfig singleton.

    After calling this, the next access to ``rag_config`` (or call to
    ``get_rag_config()``) will create a fresh instance that re-reads
    environment variables, TOML settings, and provider configuration.
    """
    _reset_rag_config()


def __getattr__(name: str):
    """Module-level __getattr__ for backward-compatible ``rag_config`` access.

    ``from lasagna.core.config import rag_config`` triggers
    this on first access, lazily creating the singleton.
    """
    if name == "rag_config":
        return get_rag_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
