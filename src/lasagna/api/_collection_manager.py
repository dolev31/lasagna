"""Vector backend and collection lifecycle management.

Handles backend resolution (per-session switching), collection creation,
schema migration, index management, and embedding model reload.
"""

import time
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from lasagna.core.config import (
    LEGACY_COLLECTION_NAME,
    adopt_collection,
    is_derived_collection_name,
    rag_config,
    RAGConfig,
)
from lasagna.core.schema import (
    create_abstract_rag_schema,
)
from ..vector_backends import (
    get_backend,
    VectorStoreAdapter,
)
from ..infrastructure.embedding import get_embedding_manager, EmbeddingManager


class CollectionIdentityError(ValueError):
    """A collection holds vectors from another embedding model or size."""


def _stamp_store() -> Any:
    """The metadata database that holds the collection stamps, or None."""
    try:
        from lasagna.core.database import get_rag_database

        db = get_rag_database()
    except Exception as exc:  # noqa: BLE001 - a missing stamp store skips the check
        logger.debug(f"Collection stamps unavailable: {exc}")
        return None
    return db if hasattr(db, "get_collection_meta") else None


def _read_stamp(backend_name: str, collection_name: str) -> Optional[Dict[str, Any]]:
    db = _stamp_store()
    if db is None:
        return None
    try:
        return db.get_collection_meta(collection_name, vector_backend=backend_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read the stamp of collection {collection_name!r}: {exc}")
        return None


def _write_stamp(
    backend_name: str,
    collection_name: str,
    model: Optional[str],
    dim: int,
    *,
    replace: bool = False,
) -> None:
    """Stamp a collection. ``replace`` is for one just created: a stamp left by
    a collection of that name that was removed outside Lasagna is stale."""
    db = _stamp_store()
    if db is None:
        return
    try:
        from lasagna.sdk._version import __version__

        if replace:
            db.forget_collection_meta(collection_name, vector_backend=backend_name)
        db.stamp_collection(
            collection_name,
            model,
            int(dim),
            vector_backend=backend_name,
            lasagna_version=__version__,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not record which model collection {collection_name!r} holds: {exc}")


def _forget_stamp(backend_name: str, collection_name: str) -> None:
    db = _stamp_store()
    if db is None:
        return
    try:
        db.forget_collection_meta(collection_name, vector_backend=backend_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not remove the stamp of collection {collection_name!r}: {exc}")


def _profiles_using(model: Optional[str]) -> List[str]:
    """Built-in profiles that embed with ``model``."""
    try:
        from lasagna.core.session_config._profiles import ProfileMixin

        specs = ProfileMixin._PROFILE_SPECS
    except Exception:  # noqa: BLE001
        return []
    return [name for name, spec in specs.items() if model and spec.get("embedding_model") == model]


def _switch_back(model: Optional[str], session_id: str) -> str:
    """How to make a session embed with ``model`` again, as a command when one exists."""
    profiles = _profiles_using(model)
    if profiles:
        return f"lasagna configure --profile {profiles[0]} -s {session_id}"
    return f"set embedding_model to {model} for session {session_id!r}"


def _effective_dim(config: Any) -> int:
    dim = getattr(config, "effective_embedding_dim", None)
    return int(dim if dim is not None else config.embedding_dim)


def _model_mismatch_message(
    config: Any, collection_name: str, stamped_model: str, stamped_dim: Any
) -> str:
    session = getattr(config, "_session_id", None) or "default"
    model = getattr(config, "embedding_model", None)
    return (
        f"Collection {collection_name!r} holds documents embedded with {stamped_model} "
        f"({stamped_dim} dimensions), but this configuration embeds with {model}. "
        f"Vectors from two models cannot be compared, so nothing was changed. To keep "
        f"using these documents, switch back: {_switch_back(stamped_model, session)}. "
        f"To use {model}, upload into a new session, or set collection_name to a name "
        f"of its own (or remove the setting, which gives each model its own collection)."
    )


def _dimension_mismatch_message(
    config: Any, collection_name: str, stored_dim: int, stamped_model: Optional[str]
) -> str:
    session = getattr(config, "_session_id", None) or "default"
    wanted = _effective_dim(config)
    if stamped_model:
        back = _switch_back(stamped_model, session)
    else:
        back = f"use a model that embeds at {stored_dim} dimensions in session {session!r}"
    return (
        f"Collection {collection_name!r} holds documents embedded at {stored_dim} "
        f"dimensions, but this configuration embeds at {wanted}, so every search and "
        f"upload would fail. Nothing was changed. To keep using these documents, switch "
        f"back: {back}. To embed at {wanted}, upload into a new session, or set "
        f"collection_name to a name of its own (or remove the setting, which gives each "
        f"model its own collection)."
    )


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
        """Create the default configuration's collection, or adopt the legacy one."""
        self.adopt_legacy_collection(rag_config)
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
        embedding_dim = _effective_dim(config)

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
        """Drop and recreate the collection for the provided config.

        Rebuilding an empty collection loses nothing, so it is logged as
        information; only a drop that removes documents is a warning.
        """
        backend = backend or self._default_backend
        collection_name = config.collection_name
        if backend.has_collection(collection_name):
            if collection_is_empty(backend, collection_name):
                logger.info(f"Recreating empty vector collection: {collection_name}")
            else:
                logger.warning(f"Dropping vector collection for rebuild: {collection_name}")
            backend.drop_collection(collection_name)
            _forget_stamp(backend.backend_name, collection_name)
        collection_key = (backend.backend_name, collection_name)
        self._initialized_collections.pop(collection_key, None)
        self._checked_collections().pop(collection_key, None)
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
        _write_stamp(
            backend.backend_name,
            collection_name,
            getattr(config, "embedding_model", None),
            embedding_dim,
            replace=True,
        )

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
            # Runs on every search; the fields alone skip a count of the rows.
            fields = backend.describe_fields(collection_name) or []

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

    def collection_is_empty(self, collection_name: str, backend: VectorStoreAdapter = None) -> bool:
        """Whether the collection positively holds no rows: the test before a drop.

        Every session's documents live in this one collection, so unlike
        `collection_has_documents` a count that cannot be read is not "empty".
        """
        return collection_is_empty(backend or self._default_backend, collection_name)

    def maybe_reload_embedding_manager(self, config: RAGConfig) -> None:
        """Make the embedder current for ``config``, after checking its collection.

        A collection holds the vectors of one model at one size. Opening it
        with another model or size is refused with the ways out; nothing is
        dropped unless it is empty. The collection is created when missing, so
        a session's first upload under a new profile simply starts one.
        """
        self.adopt_legacy_collection(config)
        self.check_collection_identity(config)

        from ..infrastructure.embedding.manager import _config_signature

        if getattr(self.embedding_manager, "signature", None) != _config_signature(config):
            logger.info(
                "Reloading embedding manager for config change: "
                f"model={self.embedding_manager.model_name} -> {config.embedding_model}"
            )
            from ..infrastructure.embedding import reset_embedding_manager
            from ..retrieval.search import reset_reranker

            reset_embedding_manager()
            self.embedding_manager = get_embedding_manager(config)

            reset_reranker()

        if self._uses_default_backend(config):
            self.ensure_backend_collection(self._default_backend, config)

    # =========================================================================
    # Collection Identity
    # =========================================================================

    def _checked_collections(self) -> Dict[tuple, set]:
        """(backend, collection) -> the (model, size) pairs already checked here."""
        checked = getattr(self, "_identity_checked", None)
        if checked is None:
            checked = self._identity_checked = {}
        return checked

    def _uses_default_backend(self, config: Any) -> bool:
        name = getattr(config, "vector_backend", None)
        if name is None:
            return True
        return name == getattr(self, "_default_backend_name", name) and (
            getattr(config, "vector_backend_config", None) or {}
        ) == (getattr(self, "_default_backend_config", None) or {})

    def check_collection_identity(self, config: Any) -> None:
        """Refuse a collection whose vectors another model or size made.

        The stamp names the model; the schema gives the size. Either differing
        from the configuration raises :class:`CollectionIdentityError`. An
        empty collection at the wrong size is rebuilt instead, since it holds
        nothing to lose.
        """
        backend = self._default_backend
        backend_name = getattr(backend, "backend_name", "milvus")
        name = config.collection_name
        model = getattr(config, "embedding_model", None)
        wanted = _effective_dim(config)
        key = (backend_name, name)
        if (model, wanted) in self._checked_collections().get(key, set()):
            return

        stamp = _read_stamp(backend_name, name)
        stamped_model = (stamp or {}).get("embedding_model")
        if stamped_model and model and stamped_model != model:
            message = _model_mismatch_message(config, name, stamped_model, stamp["embedding_dim"])
            logger.error(message)
            raise CollectionIdentityError(message)

        stored = self.get_collection_embedding_dim(name)
        if stored is not None and stored != wanted:
            if not self.collection_is_empty(name):
                message = _dimension_mismatch_message(config, name, stored, stamped_model)
                logger.error(message)
                raise CollectionIdentityError(message)
            logger.info(
                f"Collection {name!r} is empty and sized {stored}; recreating it at {wanted}."
            )
            self.recreate_collection_for_config(config)
        elif stored is not None and stamp is None and not self.collection_is_empty(name):
            # Documents from before collections were stamped: the first model
            # to open them at their size is taken to be the one that made them.
            _write_stamp(backend_name, name, model, stored)

        self._checked_collections().setdefault(key, set()).add((model, wanted))

    def adopt_legacy_collection(self, config: Any) -> bool:
        """Let a model keep the documents an earlier version stored for it.

        Before 0.1.0 every model wrote to ``rag_documents_v2``. The first model
        to open it at the size it holds adopts it under that name, so upgrading
        does not hide anyone's documents. A model at another size gets its own
        collection and the old one is left alone (see :meth:`legacy_collections`).
        Returns whether ``config`` now uses the legacy collection.
        """
        name = getattr(config, "collection_name", None)
        if not is_derived_collection_name(name):
            return name == LEGACY_COLLECTION_NAME
        backend = self._default_backend
        backend_name = getattr(backend, "backend_name", "milvus")
        model = getattr(config, "embedding_model", None)
        dim = _effective_dim(config)
        store = (getattr(config, "_config", None) or {}).get("db_path")
        decided = getattr(self, "_legacy_decided", None)
        if decided is None:
            decided = self._legacy_decided = set()
        if (store, model, dim) in decided:
            return False
        try:
            stamp = _read_stamp(backend_name, LEGACY_COLLECTION_NAME)
            if stamp:
                if stamp.get("embedding_model") == model and int(stamp["embedding_dim"]) == dim:
                    adopt_collection(LEGACY_COLLECTION_NAME, model, dim, store)
                    return True
                return False
            if backend.has_collection(name) or not backend.has_collection(LEGACY_COLLECTION_NAME):
                return False
            if collection_is_empty(backend, LEGACY_COLLECTION_NAME):
                return False
            if self.get_collection_embedding_dim(LEGACY_COLLECTION_NAME) != dim:
                return False
            _write_stamp(backend_name, LEGACY_COLLECTION_NAME, model, dim)
            adopt_collection(LEGACY_COLLECTION_NAME, model, dim, store)
            logger.info(
                f"Using the existing collection {LEGACY_COLLECTION_NAME!r} for {model}: it "
                f"holds {dim}-dimension documents from an earlier version of Lasagna."
            )
            return True
        finally:
            decided.add((store, model, dim))

    def legacy_collections(self, backend: VectorStoreAdapter = None) -> List[Dict[str, Any]]:
        """Collections from before 0.1.0 that no configured model uses.

        ``rag_documents_v2`` held every model's vectors. When no model adopted
        it (its documents are at a size no configuration opened), it is left in
        place for its owner to switch back to or remove; ``status`` lists it.
        """
        return legacy_collections(backend or self._default_backend)

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


def legacy_collections(backend: VectorStoreAdapter) -> List[Dict[str, Any]]:
    """The pre-0.1.0 shared collection, when it holds documents no model adopted.

    Each entry gives the name, row count and embedding size, which is what a
    user needs to decide whether to switch back to the model that made it or
    remove it. Empty when it does not exist, is empty, or was adopted.
    """
    name = LEGACY_COLLECTION_NAME
    backend_name = getattr(backend, "backend_name", "milvus")
    try:
        if not backend.has_collection(name):
            return []
        stats = backend.get_collection_stats(name)
        rows = stats.get("row_count") if isinstance(stats, dict) else stats.row_count
        rows = int(rows or 0)
    except Exception as exc:  # noqa: BLE001 - a listing must not fail a status command
        logger.debug(f"Could not inspect {name!r}: {exc}")
        return []
    if rows == 0 or _read_stamp(backend_name, name):
        return []
    shim = type("_Shim", (), {"_default_backend": backend})()
    dim = CollectionManager.get_collection_embedding_dim(shim, name)
    return [{"name": name, "rows": rows, "embedding_dim": dim}]


def collection_is_empty(backend: VectorStoreAdapter, collection_name: str) -> bool:
    """Whether ``collection_name`` positively holds no rows on ``backend``.

    False when the count cannot be read: a drop decided on "could not tell"
    takes every session's documents with it.
    """
    try:
        if not backend.has_collection(collection_name):
            return True
        stats = backend.get_collection_stats(collection_name)
        rows = (
            stats.get("row_count") if isinstance(stats, dict) else getattr(stats, "row_count", None)
        )
        return rows is not None and int(rows) == 0
    except Exception as e:
        logger.warning(
            f"Could not count the rows of {collection_name!r}; treating it as not empty: {e}"
        )
        return False
