"""Vector backend and collection lifecycle management.

Handles backend resolution (per-session switching), collection creation,
schema migration, index management, and embedding model reload.
"""

import time
import threading
from typing import Any, Optional

from loguru import logger

from lasagna.core.config import rag_config, RAGConfig
from lasagna.core.schema import (
    create_abstract_rag_schema,
)
from ..vector_backends import (
    get_backend,
    VectorStoreAdapter,
)
from ..infrastructure.embedding import get_embedding_manager, EmbeddingManager


def _dense_index_type(backend: Any, config: Any) -> str:
    """The dense-vector index to build: FLAT on Milvus Lite, else the setting.

    Milvus Lite 3.x builds an HNSW (or AUTOINDEX) index with faiss inside the
    calling process when a collection is loaded. On macOS, once torch has
    started its own OpenMP runtime, that build aborts with "OMP: Error #15" or
    deadlocks, so any search from a new process crashed once a collection held
    a few hundred chunks. FLAT avoids the build and is exact, which also keeps
    approximate-search noise out of measured differences; Lite corpora are
    small enough for brute force.
    """
    if getattr(backend, "is_lite", False):
        return "FLAT"
    return "HNSW" if config.index_type == "HNSW" else (config.index_type or "AUTOINDEX")


class CollectionManager:
    """Manages vector backend lifecycle, collection creation, and schema migration.

    Owns:
    - Default backend instance and per-session backend resolution
    - Initialized collections tracking (bounded LRU)
    - Collection creation, indexing, schema migration
    - Embedding dimension mismatch detection and reload
    """

    def __init__(
        self,
        default_backend: VectorStoreAdapter,
        default_backend_name: str,
        default_backend_config: dict,
        embedding_manager: EmbeddingManager,
    ):
        self._default_backend = default_backend
        self._default_backend_name = default_backend_name
        self._default_backend_config = default_backend_config
        self.embedding_manager = embedding_manager

        # Bounded LRU set: (backend_name, collection_name) pairs that have been initialized.
        # Max 500 entries prevents unbounded memory growth in multi-tenant deployments.
        self._initialized_collections: dict = {}  # key → timestamp for LRU eviction
        self._MAX_INITIALIZED_COLLECTIONS = 500
        self._collection_init_lock = threading.Lock()

    # =========================================================================
    # Backend Resolution
    # =========================================================================

    def resolve_backend(self, session_id: Optional[str] = None) -> VectorStoreAdapter:
        """Resolve the vector backend for a given session.

        Fast path: no session or no override → returns default backend (zero overhead).
        Slow path: session has override → get from BackendRegistry (cached), ensure collection.

        Args:
            session_id: Optional session identifier

        Returns:
            VectorStoreAdapter instance for the session's configured backend
        """
        if not session_id:
            return self._default_backend

        config = self._get_session_config(session_id)
        backend_name = config.vector_backend
        backend_config = config.vector_backend_config or {}

        # Fast path: matches default backend
        if (
            backend_name == self._default_backend_name
            and backend_config == self._default_backend_config
        ):
            return self._default_backend

        # Get or create backend from registry (cached by name+config hash)
        backend = get_backend(backend_name, **backend_config)
        if not backend.is_connected():
            backend.connect()

        # Lazy collection initialization
        collection_key = (backend_name, config.collection_name)
        if collection_key not in self._initialized_collections:
            self.ensure_backend_collection(backend, config)

        return backend

    def get_vector_backend(self, session_id: Optional[str] = None) -> VectorStoreAdapter:
        """Public API for getting the session-appropriate vector backend."""
        return self.resolve_backend(session_id)

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    def get_effective_config(
        self, config: RAGConfig, backend: VectorStoreAdapter = None
    ) -> RAGConfig:
        """Get effective config with capability-based feature degradation."""
        backend = backend or self._default_backend
        caps = backend.capabilities

        if config.enable_hybrid_search and not caps.supports_hybrid_search:
            logger.warning(
                f"Backend '{backend.backend_name}' does not support hybrid search. "
                f"Falling back to dense-only search."
            )
            config = (
                RAGConfig(**{**config.__dict__, "enable_hybrid_search": False})
                if hasattr(config, "__dict__")
                else config
            )

        if not caps.supports_json_filters:
            logger.debug(
                f"Backend '{backend.backend_name}' has limited JSON filter support. "
                f"Complex filters may be simplified."
            )

        return config

    def should_use_hybrid_search(
        self, config: RAGConfig, backend: VectorStoreAdapter = None
    ) -> bool:
        """Check if hybrid search should be used based on config and capabilities."""
        if not config.enable_hybrid_search:
            return False

        backend = backend or self._default_backend
        caps = backend.capabilities
        if not caps.supports_hybrid_search and not caps.supports_native_bm25:
            return True  # HybridSearcher handles fallback

        return True

    # =========================================================================
    # Collection Lifecycle
    # =========================================================================

    def initialize_collection(self):
        """Initialize vector collection with SOTA configuration."""
        collection_name = rag_config.collection_name
        embedding_dim = self.embedding_manager.dimension

        if not self._default_backend.has_collection(collection_name):
            logger.info(f"Creating vector collection: {collection_name} (dim={embedding_dim})")
            self.create_collection_on_backend(
                self._default_backend, collection_name, embedding_dim, rag_config
            )
            self.create_index_on_backend(self._default_backend, collection_name, rag_config)
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            if self.needs_schema_migration(self._default_backend, collection_name):
                self.recreate_collection_for_config(rag_config)
            else:
                self.ensure_index_exists(collection_name)

    def create_collection_index(self, collection_name: str):
        """Create index on collection — delegates to the unified backend-agnostic method."""
        self.create_index_on_backend(self._default_backend, collection_name, rag_config)

    def ensure_collection_for_config(self, config: RAGConfig) -> None:
        """Ensure a collection exists for the provided config."""
        collection_name = config.collection_name
        embedding_dim = config.embedding_dim

        if self._default_backend.has_collection(collection_name):
            return

        logger.info(f"Creating vector collection: {collection_name} (dim={embedding_dim})")
        self.create_collection_on_backend(
            self._default_backend, collection_name, embedding_dim, config
        )
        self.create_index_on_backend(self._default_backend, collection_name, config)

    def recreate_collection_for_config(
        self, config: RAGConfig, backend: VectorStoreAdapter = None
    ) -> None:
        """Drop and recreate the collection for the provided config."""
        backend = backend or self._default_backend
        collection_name = config.collection_name
        if backend.has_collection(collection_name):
            logger.warning(f"Dropping vector collection for rebuild: {collection_name}")
            backend.drop_collection(collection_name)
        collection_key = (backend.backend_name, collection_name)
        self._initialized_collections.pop(collection_key, None)
        self.ensure_backend_collection(backend, config)

    def ensure_backend_collection(self, backend: VectorStoreAdapter, config: RAGConfig) -> None:
        """Ensure collection exists and is indexed on the given backend.

        Thread-safe, idempotent, and fully backend-agnostic.
        Uses bounded LRU tracking to prevent unbounded memory growth.
        """
        collection_name = config.collection_name
        collection_key = (backend.backend_name, collection_name)

        with self._collection_init_lock:
            if collection_key in self._initialized_collections:
                self._initialized_collections[collection_key] = time.monotonic()
                return

            if not backend.has_collection(collection_name):
                embedding_dim = self.embedding_manager.dimension
                logger.info(
                    f"Creating collection '{collection_name}' on {backend.backend_name} "
                    f"(dim={embedding_dim})"
                )
                self.create_collection_on_backend(backend, collection_name, embedding_dim, config)
                self.create_index_on_backend(backend, collection_name, config)
            elif self.needs_schema_migration(backend, collection_name):
                self.recreate_collection_for_config(config, backend=backend)
            else:
                logger.info(
                    f"Collection '{collection_name}' already exists on {backend.backend_name}"
                )

            self.mark_collection_initialized(backend.backend_name, collection_name)

    # =========================================================================
    # Collection/Index Creation (Backend-Agnostic)
    # =========================================================================

    def create_collection_on_backend(
        self,
        backend: VectorStoreAdapter,
        collection_name: str,
        embedding_dim: int,
        config: RAGConfig,
    ) -> None:
        """Create a collection on ANY backend via the abstract VectorStoreAdapter API."""
        from ..vector_backends.base import IndexConfig, IndexType, MetricType

        abstract_schema = create_abstract_rag_schema(embedding_dim)
        index_config = IndexConfig(
            index_type=IndexType.__members__.get(
                _dense_index_type(backend, config), IndexType.AUTOINDEX
            ),
            metric_type=MetricType.COSINE if config.metric_type == "COSINE" else MetricType.L2,
            params=config.index_params,
        )
        backend.create_collection(collection_name, abstract_schema, index_config)

    def create_index_on_backend(
        self,
        backend: VectorStoreAdapter,
        collection_name: str,
        config: RAGConfig,
    ) -> None:
        """Create indexes on ANY backend via the abstract VectorStoreAdapter API."""
        from ..vector_backends.base import IndexConfig, IndexType, MetricType

        try:
            index_params = backend.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=_dense_index_type(backend, config),
                metric_type=config.metric_type,
                params=config.index_params,
            )
            backend.create_index(collection_name, "embedding", index_params)
            logger.info(f"Created index on {collection_name} ({backend.backend_name})")
        except Exception as e:
            error_msg = str(e).lower()
            if "invalid index type" in error_msg or "not support" in error_msg:
                logger.warning(
                    f"Index type '{config.index_type}' not supported on {backend.backend_name}, using AUTOINDEX..."
                )
                try:
                    fallback_config = IndexConfig(
                        index_type=IndexType.AUTOINDEX,
                        metric_type=MetricType.COSINE
                        if config.metric_type == "COSINE"
                        else MetricType.L2,
                        params={},
                    )
                    backend.create_index(collection_name, "embedding", fallback_config)
                    logger.info(f"Created AUTOINDEX on {collection_name} ({backend.backend_name})")
                except Exception as fallback_err:
                    logger.warning(
                        f"AUTOINDEX fallback also failed on {backend.backend_name}: {fallback_err}"
                    )
            else:
                logger.warning(
                    f"Index creation on {backend.backend_name} failed (may already exist): {e}"
                )

    # =========================================================================
    # Schema Migration & Index Verification
    # =========================================================================

    def needs_schema_migration(self, backend: VectorStoreAdapter, collection_name: str) -> bool:
        """Check if an existing collection has the old INT64 chunk_id schema."""
        try:
            info = backend.describe_collection(collection_name)
            for field in info.fields:
                name = (
                    field.get("name", "") if isinstance(field, dict) else getattr(field, "name", "")
                )
                dtype = (
                    field.get("dtype", "")
                    if isinstance(field, dict)
                    else str(getattr(field, "dtype", ""))
                )
                if name == "chunk_id" and "INT64" in str(dtype).upper():
                    logger.warning(
                        f"Schema migration needed for '{collection_name}': "
                        f"chunk_id is INT64 (old schema), expected VARCHAR. "
                        f"Collection will be dropped and recreated."
                    )
                    return True
            return False
        except Exception as e:
            logger.debug(f"Could not inspect schema for '{collection_name}': {e}")
            return False

    def ensure_index_exists(self, collection_name: str, backend: VectorStoreAdapter = None):
        """Ensure an index exists on the embedding field."""
        backend = backend or self._default_backend
        try:
            existing_indexes = backend.list_indexes(collection_name)
            if existing_indexes and "embedding" in existing_indexes:
                logger.debug(
                    f"Index already exists on 'embedding' field for collection '{collection_name}'"
                )
                return
        except Exception as e:
            logger.debug(f"Could not list indexes, will attempt creation: {e}")

        logger.info(f"Creating index on 'embedding' field for collection '{collection_name}'...")
        try:
            self.create_collection_index(collection_name)
        except Exception as e:
            error_msg = str(e).lower()
            if "already" in error_msg or "exist" in error_msg:
                logger.debug("Index already exists on 'embedding' field (race condition)")
            else:
                logger.error(f"Failed to create index: {e}")
                raise

    def mark_collection_initialized(self, backend_name: str, collection_name: str) -> None:
        """Mark a (backend, collection) pair as initialized with LRU eviction."""
        collection_key = (backend_name, collection_name)
        self._initialized_collections[collection_key] = time.monotonic()

        if len(self._initialized_collections) > self._MAX_INITIALIZED_COLLECTIONS:
            sorted_keys = sorted(
                self._initialized_collections.keys(),
                key=lambda k: self._initialized_collections[k],
            )
            evict_count = max(1, len(sorted_keys) // 10)
            for key in sorted_keys[:evict_count]:
                del self._initialized_collections[key]
            logger.debug(f"Evicted {evict_count} stale collection init entries")

    # =========================================================================
    # Embedding Dimension & Stats
    # =========================================================================

    def get_collection_embedding_dim(
        self, collection_name: str, backend: VectorStoreAdapter = None
    ) -> Optional[int]:
        """Get the embedding dimension from an existing vector collection."""
        backend = backend or self._default_backend

        def _field_value(field: Any, key: str) -> Optional[Any]:
            if isinstance(field, dict):
                return field.get(key)
            return getattr(field, key, None)

        def _extract_dim(params: Any) -> Optional[int]:
            if isinstance(params, dict):
                raw = params.get("dim") or params.get("dimension")
                if raw is not None:
                    try:
                        return int(raw)
                    except (TypeError, ValueError):
                        return None
                return None
            if isinstance(params, list):
                for item in params:
                    dim = _extract_dim(item)
                    if dim is not None:
                        return dim
            return None

        try:
            info = backend.describe_collection(collection_name)
            fields = []
            if isinstance(info, dict):
                fields = info.get("fields") or []
                if not fields:
                    schema = info.get("schema")
                    if isinstance(schema, dict):
                        fields = schema.get("fields") or []
                    else:
                        fields = getattr(schema, "fields", []) or []
            else:
                fields = getattr(info, "fields", []) or []

            for field in fields:
                name = _field_value(field, "name")
                if name != "embedding":
                    continue
                params = _field_value(field, "params")
                dim = _extract_dim(params)
                if dim is None:
                    params = _field_value(field, "type_params")
                    dim = _extract_dim(params)
                if dim is None:
                    dim = _extract_dim(field)
                return dim
        except Exception as exc:
            logger.debug(f"Could not read collection schema for {collection_name}: {exc}")
        return None

    def collection_has_documents(
        self, collection_name: str, backend: VectorStoreAdapter = None
    ) -> bool:
        """Check if a vector collection has any documents."""
        backend = backend or self._default_backend
        try:
            if not backend.has_collection(collection_name):
                return False
            stats = backend.get_collection_stats(collection_name)
            if isinstance(stats, dict):
                row_count = stats.get("row_count", 0)
            else:
                row_count = getattr(stats, "row_count", 0)
            return int(row_count) > 0
        except Exception as e:
            logger.debug(f"Could not check collection stats: {e}")
            return False

    def maybe_reload_embedding_manager(self, config: RAGConfig) -> None:
        """Reload embedding manager if the configured model changes.

        WARNING: Dimension mismatch is handled safely - we DO NOT auto-delete data.
        """
        collection_dim = self.get_collection_embedding_dim(config.collection_name)
        if collection_dim is not None and config.embedding_dim != collection_dim:
            has_data = self.collection_has_documents(config.collection_name)

            if has_data:
                error_msg = (
                    f"EMBEDDING DIMENSION MISMATCH DETECTED:\n"
                    f"  - Collection dimension: {collection_dim}\n"
                    f"  - Configured dimension: {config.embedding_dim}\n"
                    f"  - Collection has existing documents that would be DELETED.\n\n"
                    f"To resolve this:\n"
                    f"  1. Keep current model: Update config to use embedding_dim={collection_dim}\n"
                    f"  2. Migrate to new model: Export documents, clear collection, re-upload\n"
                    f"  3. Force rebuild (DATA LOSS): Call backend.force_rebuild_collection()\n"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                logger.warning(
                    "Embedding dimension mismatch detected on empty collection: "
                    f"collection_dim={collection_dim}, config_dim={config.embedding_dim}. "
                    "Rebuilding empty collection to match current config."
                )
                self.recreate_collection_for_config(config)

        from ..infrastructure.embedding.manager import _config_signature

        if getattr(self.embedding_manager, "signature", None) == _config_signature(config):
            return

        logger.info(
            "Reloading embedding manager for config change: "
            f"model={self.embedding_manager.model_name} -> {config.embedding_model}"
        )
        from ..infrastructure.embedding import reset_embedding_manager
        from ..retrieval.search import reset_reranker

        reset_embedding_manager()
        self.embedding_manager = get_embedding_manager(config)

        reset_reranker()

    def force_rebuild_collection(self, confirm: bool = False, hybrid_searcher=None) -> bool:
        """Force rebuild the collection, DELETING ALL EXISTING DOCUMENTS.

        Args:
            confirm: Must be True to proceed (safety check)
            hybrid_searcher: Optional hybrid searcher to clear indices

        Returns:
            True if rebuild succeeded, False otherwise
        """
        if not confirm:
            raise ValueError(
                "force_rebuild_collection requires confirm=True. "
                "This will DELETE ALL DOCUMENTS in the collection."
            )

        collection_name = rag_config.collection_name
        logger.warning(f"FORCE REBUILDING collection '{collection_name}' - ALL DATA WILL BE LOST")

        try:
            self.recreate_collection_for_config(rag_config)
            if rag_config.enable_hybrid_search and hybrid_searcher:
                with hybrid_searcher._session_lock:
                    hybrid_searcher._session_indices.clear()
                    hybrid_searcher._session_timestamps.clear()
                    hybrid_searcher._session_creation_times.clear()
                    hybrid_searcher._session_backends.clear()
            logger.info(f"Collection '{collection_name}' rebuilt successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild collection: {e}")
            return False

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _get_session_config(session_id: Optional[str] = None) -> RAGConfig:
        """Get a session-aware RAGConfig instance."""
        if session_id:
            return rag_config.with_session(session_id)
        return rag_config

    @staticmethod
    def create_vector_backend() -> VectorStoreAdapter:
        """Create the default vector database backend instance.

        Respects ``rag_config.vector_backend_config`` (populated from the
        ``RAG_VECTOR_BACKEND_CONFIG`` env var in Docker deployments).
        Falls back to the default ``dbs_dir/rag_milvus.db`` path for Milvus
        when no explicit config is provided.
        """
        import os
        from lasagna.providers import get_config_provider

        backend_name = getattr(rag_config, "vector_backend", "milvus")
        backend_cfg = rag_config.vector_backend_config or {}

        if backend_name == "milvus":
            db_path = backend_cfg.get("db_path")
            if not db_path:
                try:
                    dbs_dir = str(get_config_provider().db_dir)
                except RuntimeError:
                    dbs_dir = os.path.abspath("dbs")
                db_path = os.path.join(dbs_dir, "rag_milvus.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            logger.info(f"Initializing vector backend: Milvus at {db_path}")
            return get_backend(
                "milvus",
                db_path=db_path,
                **{k: v for k, v in backend_cfg.items() if k != "db_path"},
            )
        else:
            logger.info(f"Initializing vector backend: {backend_name}")
            return get_backend(backend_name, **backend_cfg)
