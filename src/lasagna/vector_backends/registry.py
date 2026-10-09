"""Backend Registry and Factory for Vector Store adapters.

This module provides a centralized registry for vector database backends,
enabling dynamic backend selection and configuration-driven instantiation.

Design Pattern: Registry + Factory
- Registry holds backend class registrations
- Factory creates instances based on configuration
- Singleton pattern for backend instances (configurable)

Usage:
    # Register a custom backend
    register_backend("my_backend", MyBackendClass)

    # Get backend by name
    backend = get_backend("milvus", db_path="/path/to/db")

    # Get default backend (from config)
    backend = get_backend()

    # List available backends
    backends = get_available_backends()
    # ["milvus", "pinecone", "qdrant", ...]
"""

import os
import threading
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from .base import VectorStoreAdapter


class BackendRegistry:
    """Registry for vector store backend classes.

    Thread-safe singleton registry that manages backend class registrations
    and instance creation.
    """

    _instance: Optional["BackendRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "BackendRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry."""
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._backends: Dict[str, Type[VectorStoreAdapter]] = {}
        self._instances: Dict[str, VectorStoreAdapter] = {}
        self._instance_lock = threading.Lock()

        # Register built-in backends
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in backend implementations.

        Each backend is registered if its required client library is available.
        This allows users to install only the backends they need.
        """
        # Milvus (available if pymilvus is installed)
        try:
            from .milvus_backend import MilvusBackend, MILVUS_AVAILABLE

            if MILVUS_AVAILABLE:
                self._backends["milvus"] = MilvusBackend
                logger.debug("Registered built-in backend: milvus")
        except ImportError:
            logger.debug("Milvus backend not available (pymilvus not installed)")

        # Pinecone (available if pinecone-client is installed)
        try:
            from .pinecone_backend import PineconeBackend, PINECONE_AVAILABLE

            if PINECONE_AVAILABLE:
                self._backends["pinecone"] = PineconeBackend
                logger.debug("Registered built-in backend: pinecone")
        except ImportError:
            logger.debug("Pinecone backend not available (pinecone-client not installed)")

        # Qdrant (available if qdrant-client is installed)
        try:
            from .qdrant_backend import QdrantBackend, QDRANT_AVAILABLE

            if QDRANT_AVAILABLE:
                self._backends["qdrant"] = QdrantBackend
                logger.debug("Registered built-in backend: qdrant")
        except ImportError:
            logger.debug("Qdrant backend not available (qdrant-client not installed)")

        # Weaviate (available if weaviate-client is installed)
        try:
            from .weaviate_backend import WeaviateBackend, WEAVIATE_AVAILABLE

            if WEAVIATE_AVAILABLE:
                self._backends["weaviate"] = WeaviateBackend
                logger.debug("Registered built-in backend: weaviate")
        except ImportError:
            logger.debug("Weaviate backend not available (weaviate-client not installed)")

        # PgVector (available if psycopg + pgvector are installed)
        try:
            from .pgvector_backend import PgVectorBackend, PGVECTOR_AVAILABLE

            if PGVECTOR_AVAILABLE:
                self._backends["pgvector"] = PgVectorBackend
                logger.debug("Registered built-in backend: pgvector")
        except ImportError:
            logger.debug("PgVector backend not available (psycopg/pgvector not installed)")

        # OpenSearch (available if opensearch-py is installed)
        try:
            from .opensearch_backend import OpenSearchBackend, OPENSEARCH_AVAILABLE

            if OPENSEARCH_AVAILABLE:
                self._backends["opensearch"] = OpenSearchBackend
                logger.debug("Registered built-in backend: opensearch")
        except ImportError:
            logger.debug("OpenSearch backend not available (opensearch-py not installed)")

    def register(
        self,
        name: str,
        backend_class: Type[VectorStoreAdapter],
        override: bool = False,
    ) -> None:
        """Register a backend class.

        Args:
            name: Unique name for the backend (e.g., "milvus", "pinecone")
            backend_class: Class that implements VectorStoreAdapter
            override: If True, replace existing registration

        Raises:
            ValueError: If name already registered and override=False
            TypeError: If backend_class doesn't implement VectorStoreAdapter
        """
        if not issubclass(backend_class, VectorStoreAdapter):
            raise TypeError(f"{backend_class} must be a subclass of VectorStoreAdapter")

        if name in self._backends and not override:
            raise ValueError(
                f"Backend '{name}' is already registered. Use override=True to replace it."
            )

        self._backends[name] = backend_class
        logger.info(f"Registered vector backend: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a backend.

        Args:
            name: Name of backend to unregister

        Returns:
            True if backend was unregistered, False if not found
        """
        if name in self._backends:
            del self._backends[name]
            # Also remove any cached instance
            with self._instance_lock:
                if name in self._instances:
                    del self._instances[name]
            return True
        return False

    def get_class(self, name: str) -> Optional[Type[VectorStoreAdapter]]:
        """Get backend class by name.

        Args:
            name: Backend name

        Returns:
            Backend class or None if not found
        """
        return self._backends.get(name)

    def list_backends(self) -> List[str]:
        """List all registered backend names.

        Returns:
            List of backend names
        """
        return list(self._backends.keys())

    def has_backend(self, name: str) -> bool:
        """Check if backend is registered.

        Args:
            name: Backend name

        Returns:
            True if backend is registered
        """
        return name in self._backends

    def create(
        self,
        name: str,
        singleton: bool = True,
        **kwargs,
    ) -> VectorStoreAdapter:
        """Create backend instance.

        Args:
            name: Backend name
            singleton: If True, return cached instance (default)
            **kwargs: Arguments passed to backend constructor

        Returns:
            Backend instance

        Raises:
            ValueError: If backend not found
        """
        backend_class = self._backends.get(name)
        if not backend_class:
            available = ", ".join(self._backends.keys())
            raise ValueError(f"Unknown backend '{name}'. Available: {available or 'none'}")

        if singleton:
            # Build cache key from name + sorted kwargs
            cache_key = self._build_cache_key(name, kwargs)

            with self._instance_lock:
                if cache_key not in self._instances:
                    instance = backend_class(**kwargs)
                    self._instances[cache_key] = instance
                    logger.debug(f"Created singleton backend instance: {name}")
                return self._instances[cache_key]
        else:
            return backend_class(**kwargs)

    def _build_cache_key(self, name: str, kwargs: Dict[str, Any]) -> str:
        """Build cache key for singleton instances.

        Uses a hash of serialized kwargs to ensure uniqueness even for
        complex objects that might have identical string representations.
        """
        import hashlib
        import json

        # Build a canonical representation
        def serialize_value(v: Any) -> Any:
            """Serialize values to JSON-compatible types."""
            if v is None:
                return None
            if isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (list, tuple)):
                return [serialize_value(x) for x in v]
            if isinstance(v, dict):
                return {k: serialize_value(val) for k, val in sorted(v.items())}
            # For other types, use repr + type name for uniqueness
            return f"{type(v).__name__}:{repr(v)}"

        try:
            sorted_kwargs = {
                k: serialize_value(v) for k, v in sorted(kwargs.items()) if v is not None
            }
            kwargs_json = json.dumps(sorted_kwargs, sort_keys=True)
            kwargs_hash = hashlib.md5(kwargs_json.encode()).hexdigest()[:12]
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            kwargs_hash = hashlib.md5(repr(sorted(kwargs.items())).encode()).hexdigest()[:12]

        return f"{name}:{kwargs_hash}"

    def clear_instances(self) -> None:
        """Clear all cached backend instances.

        Useful for testing or reconfiguration.
        """
        with self._instance_lock:
            # Disconnect all instances first
            for instance in self._instances.values():
                try:
                    instance.disconnect()
                except Exception as e:
                    logger.debug(f"Error disconnecting backend: {e}")
            self._instances.clear()
        logger.debug("Cleared all backend instances")


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


def get_registry() -> BackendRegistry:
    """Get the global backend registry.

    Returns:
        BackendRegistry singleton instance
    """
    return BackendRegistry()


def register_backend(
    name: str,
    backend_class: Type[VectorStoreAdapter],
    override: bool = False,
) -> None:
    """Register a backend class with the global registry.

    Args:
        name: Unique name for the backend
        backend_class: Class implementing VectorStoreAdapter
        override: If True, replace existing registration
    """
    get_registry().register(name, backend_class, override)


def get_available_backends() -> List[str]:
    """List all available backend names.

    Returns:
        List of registered backend names
    """
    return get_registry().list_backends()


def get_backend(
    name: Optional[str] = None,
    singleton: bool = True,
    pool_size: Optional[int] = None,
    **kwargs,
) -> VectorStoreAdapter:
    """Get or create a backend instance.

    This is the main entry point for obtaining vector store backends.
    If no name is provided, uses the default backend from configuration.

    When ``pool_size`` > 1 (or ``connection_pool_size`` > 1 in config),
    the returned backend is wrapped in a :class:`PooledVectorBackend` that
    distributes operations across multiple connections.

    Args:
        name: Backend name (default: from config or "milvus")
        singleton: Return cached instance if True (default)
        pool_size: Override connection pool size (default: from config)
        **kwargs: Backend-specific configuration

    Returns:
        VectorStoreAdapter instance (possibly pooled)

    Examples:
        # Get default Milvus backend
        backend = get_backend()

        # Get specific backend with config
        backend = get_backend("milvus", db_path="/path/to/db")

        # Create fresh (non-singleton) instance
        backend = get_backend("milvus", singleton=False)

        # Explicitly pooled backend
        backend = get_backend("milvus", pool_size=10)
    """
    registry = get_registry()

    # Determine backend name
    if name is None:
        name = _get_default_backend_name()

    # Apply sensible default configs when none provided
    if not kwargs:
        if name == "milvus":
            kwargs = _get_default_milvus_config()
        elif name == "qdrant":
            kwargs = _get_default_qdrant_config()
        elif name == "weaviate":
            kwargs = _get_default_weaviate_config()
        elif name == "pgvector":
            kwargs = _get_default_pgvector_config()
        elif name == "opensearch":
            kwargs = _get_default_opensearch_config()

    # Determine pool size from explicit arg or config
    effective_pool_size = pool_size
    if effective_pool_size is None:
        try:
            from ..core.config import rag_config

            effective_pool_size = getattr(rag_config, "connection_pool_size", 1)
        except Exception:
            effective_pool_size = 1

    # Wrap in PooledVectorBackend when pool_size > 1
    if effective_pool_size > 1:
        from .pool import PooledVectorBackend

        # Capture for lambda closure
        _backend_name = name
        _backend_kwargs = dict(kwargs)

        if singleton:
            cache_key = f"pooled:{name}:{effective_pool_size}"
            with registry._instance_lock:
                if cache_key not in registry._instances:
                    pooled = PooledVectorBackend(
                        factory=lambda: registry.create(
                            _backend_name, singleton=False, **_backend_kwargs
                        ),
                        pool_size=effective_pool_size,
                    )
                    registry._instances[cache_key] = pooled
                return registry._instances[cache_key]
        else:
            return PooledVectorBackend(
                factory=lambda: registry.create(_backend_name, singleton=False, **_backend_kwargs),
                pool_size=effective_pool_size,
            )

    return registry.create(name, singleton=singleton, **kwargs)


def _get_default_backend_name() -> str:
    """Get default backend name from configuration.

    Returns:
        Backend name (defaults to "milvus")
    """
    # Environment variable takes precedence over configuration.
    env_backend = os.environ.get("RAG_VECTOR_BACKEND")
    if env_backend:
        return env_backend.lower()

    # Try to get from RAG config
    try:
        from ..core.config import rag_config

        backend = getattr(rag_config, "vector_backend", None)
        if backend:
            return backend.lower()
    except ImportError:
        pass

    # Default to Milvus
    return "milvus"


def _get_default_milvus_config() -> Dict[str, Any]:
    """Get default Milvus configuration.

    Returns:
        Dict with db_path or uri for MilvusBackend

    This function ensures the database directory exists and returns
    the appropriate configuration for Milvus Lite or Distributed.

    Configuration priority:
    1. Environment variables (MILVUS_URI, MILVUS_TOKEN)
    2. Session config (if available)
    3. Default DBS_DIR path
    4. Fallback to current directory
    """
    # Check environment variables first (highest priority)
    # Use `or None` so empty strings from compose passthrough are ignored
    uri = os.environ.get("MILVUS_URI") or None
    if uri:
        token = os.environ.get("MILVUS_TOKEN") or None
        logger.debug(f"Using Milvus URI from environment: {uri}")
        return {"uri": uri, "token": token}

    # Try to get from session config
    # NOTE: Import is deferred to avoid circular imports during initialization
    # session_config -> vector_backends -> registry -> session_config would be circular
    try:
        # Use importlib to avoid module-level circular import
        import importlib

        session_config_module = importlib.import_module("lasagna.session_config")
        get_knowledge_config_manager = getattr(
            session_config_module, "get_knowledge_config_manager"
        )
        manager = get_knowledge_config_manager()
        backend_config = manager.get_vector_backend_config()
        config = backend_config.get("vector_backend_config", {})
        if config.get("uri"):
            logger.debug(f"Using Milvus URI from session config: {config['uri']}")
            return {"uri": config["uri"], "token": config.get("token")}
        if config.get("db_path"):
            db_path = config["db_path"]
            _ensure_db_directory(db_path)
            logger.debug(f"Using Milvus db_path from session config: {db_path}")
            return {"db_path": db_path}
    except Exception as e:
        # This is expected during early initialization before session config is ready
        logger.debug(f"Could not get session config (expected during init): {e}")

    # Use config provider's db_dir for Milvus Lite (default)
    try:
        from lasagna.providers import get_config_provider

        dbs_dir = str(get_config_provider().db_dir)
        os.makedirs(dbs_dir, exist_ok=True)
        db_path = os.path.join(dbs_dir, "rag_milvus.db")
        logger.debug(f"Using default Milvus Lite path: {db_path}")
        return {"db_path": db_path}
    except (RuntimeError, Exception):
        # Fallback to current directory
        db_path = os.path.abspath("milvus.db")
        logger.warning(f"db_dir not available, using fallback: {db_path}")
        return {"db_path": db_path}


def _get_default_qdrant_config() -> Dict[str, Any]:
    """Get default Qdrant configuration for local persistent storage.

    Returns:
        Dict with path for QdrantBackend (file-based, persistent)

    Configuration priority:
    1. Environment variables (QDRANT_URL, QDRANT_API_KEY)
    2. Default DBS_DIR path (persistent local storage)
    3. Fallback to current directory
    """
    # Check environment variables for remote Qdrant
    # Use `or None` so empty strings from compose passthrough are ignored
    url = os.environ.get("QDRANT_URL") or None
    if url:
        api_key = os.environ.get("QDRANT_API_KEY") or None
        logger.debug(f"Using Qdrant URL from environment: {url}")
        return {"url": url, "api_key": api_key}

    # Use config provider's db_dir for local persistent storage
    try:
        from lasagna.providers import get_config_provider

        dbs_dir = str(get_config_provider().db_dir)
        qdrant_path = os.path.join(dbs_dir, "qdrant_data")
        os.makedirs(qdrant_path, exist_ok=True)
        logger.debug(f"Using default Qdrant local path: {qdrant_path}")
        return {"path": qdrant_path}
    except (RuntimeError, Exception):
        qdrant_path = os.path.abspath("qdrant_data")
        os.makedirs(qdrant_path, exist_ok=True)
        logger.warning(f"db_dir not available, using fallback: {qdrant_path}")
        return {"path": qdrant_path}


def _get_default_weaviate_config() -> Dict[str, Any]:
    """Get default Weaviate configuration.

    Returns:
        Dict with configuration for WeaviateBackend

    Configuration priority:
    1. Environment variables (WEAVIATE_URL, WEAVIATE_API_KEY)
    2. Default to embedded mode (zero-config local)
    """
    # Check environment variables for remote Weaviate
    # Use `or None` so empty strings from compose passthrough are ignored
    url = os.environ.get("WEAVIATE_URL") or None
    if url:
        api_key = os.environ.get("WEAVIATE_API_KEY") or None
        logger.debug(f"Using Weaviate URL from environment: {url}")
        return {"url": url, "api_key": api_key}

    # Default to embedded mode for zero-config local usage
    logger.debug("Using Weaviate embedded mode (zero-config)")
    return {"embedded": True}


def _get_default_pgvector_config() -> Dict[str, Any]:
    """Get default PgVector configuration.

    Returns:
        Dict with dsn and pool settings for PgVectorBackend

    Configuration priority:
    1. Environment variables (PGVECTOR_DSN)
    2. RAG_DATABASE_DSN (reuse existing PostgreSQL connection)
    3. Default localhost DSN
    """
    # Use `or None` so empty strings from compose passthrough are ignored
    dsn = os.environ.get("PGVECTOR_DSN") or None
    if dsn:
        logger.debug("Using PgVector DSN from PGVECTOR_DSN environment variable")
        return {"dsn": dsn}

    dsn = os.environ.get("RAG_DATABASE_DSN") or None
    if dsn:
        logger.debug("Using PgVector DSN from RAG_DATABASE_DSN environment variable")
        return {"dsn": dsn}

    logger.debug("Using default PgVector DSN: localhost:5432/rag")
    return {"dsn": "postgresql://rag:rag@localhost:5432/rag"}


def _get_default_opensearch_config() -> Dict[str, Any]:
    """Get default OpenSearch configuration.

    Delegates to the backend's own config resolver to avoid duplication.
    The backend reads OPENSEARCH_HOSTS, OPENSEARCH_USER, OPENSEARCH_PASSWORD,
    OPENSEARCH_USE_SSL, OPENSEARCH_VERIFY_CERTS, OPENSEARCH_TIMEOUT env vars.
    """
    from .opensearch_backend import (
        _get_default_opensearch_config as _os_config,
    )

    config = _os_config()
    logger.debug(f"Using OpenSearch hosts: {config.get('hosts')}")
    return config


def _ensure_db_directory(db_path: str) -> None:
    """Ensure the directory for a database path exists.

    Args:
        db_path: Path to database file
    """
    directory = os.path.dirname(db_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created database directory: {directory}")


# =============================================================================
# Backend Auto-Discovery (for plugins)
# =============================================================================


def discover_backends(package_prefix: str = "lasagna_vector_") -> int:
    """Auto-discover and register backend plugins.

    Looks for installed packages with the given prefix that expose
    a `register_backend(registry)` function.

    Args:
        package_prefix: Prefix for plugin packages

    Returns:
        Number of backends discovered

    Example:
        A plugin package `lasagna_vector_pinecone` would:
        1. Have a `register_backend(registry)` function
        2. Call `registry.register("pinecone", PineconeBackend)`
    """
    import importlib
    import pkgutil

    registry = get_registry()
    count = 0

    for module_info in pkgutil.iter_modules():
        if module_info.name.startswith(package_prefix):
            try:
                module = importlib.import_module(module_info.name)
                if hasattr(module, "register_backend"):
                    module.register_backend(registry)
                    count += 1
                    logger.info(f"Discovered backend plugin: {module_info.name}")
            except Exception as e:
                logger.warning(f"Failed to load backend plugin {module_info.name}: {e}")

    return count
