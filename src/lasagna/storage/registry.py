"""Backend Registry and Factory for Storage adapters.

This module provides a centralized registry for file storage backends,
enabling dynamic backend selection and configuration-driven instantiation.

Mirrors the design of vector_backends/registry.py:
- Registry holds backend class registrations
- Factory creates instances based on configuration
- Singleton pattern for backend instances (configurable)
- Plugin auto-discovery via entry points

Usage::

    # Register a custom backend
    register_storage_backend("my_storage", MyStorageClass)

    # Get backend by name
    storage = get_storage_backend("s3", bucket="my-bucket")

    # Get default backend (from config)
    storage = get_storage_backend()

    # List available backends
    backends = get_available_storage_backends()
    # ["local", "s3", "fsspec"]
"""

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from .base import StorageBackend


class StorageRegistry:
    """Registry for file storage backend classes.

    Thread-safe singleton registry that manages backend class registrations
    and instance creation.
    """

    _instance: Optional["StorageRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StorageRegistry":
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
        self._backends: Dict[str, Type[StorageBackend]] = {}
        self._instances: Dict[str, StorageBackend] = {}
        self._instance_lock = threading.Lock()

        # Register built-in backends
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in backend implementations.

        Each backend is registered if its required library is available.
        Local is always available (zero deps).
        """
        # Local (always available — zero deps)
        try:
            from .local_backend import LocalStorageBackend, LOCAL_AVAILABLE

            if LOCAL_AVAILABLE:
                self._backends["local"] = LocalStorageBackend
                logger.debug("Registered built-in storage backend: local")
        except ImportError:
            logger.debug("Local storage backend not available (unexpected)")

        # S3-Compatible (available if boto3 is installed)
        try:
            from .s3_backend import S3StorageBackend, BOTO3_AVAILABLE

            if BOTO3_AVAILABLE:
                self._backends["s3"] = S3StorageBackend
                logger.debug("Registered built-in storage backend: s3")
        except ImportError:
            logger.debug("S3 storage backend not available (boto3 not installed)")

        # Fsspec Universal (available if fsspec is installed)
        try:
            from .fsspec_backend import FsspecStorageBackend, FSSPEC_AVAILABLE

            if FSSPEC_AVAILABLE:
                self._backends["fsspec"] = FsspecStorageBackend
                logger.debug("Registered built-in storage backend: fsspec")
        except ImportError:
            logger.debug("Fsspec storage backend not available (fsspec not installed)")

    def register(
        self,
        name: str,
        backend_class: Type[StorageBackend],
        override: bool = False,
    ) -> None:
        """Register a backend class.

        Args:
            name: Unique name for the backend (e.g., "local", "s3")
            backend_class: Class that implements StorageBackend
            override: If True, replace existing registration

        Raises:
            ValueError: If name already registered and override=False
            TypeError: If backend_class doesn't implement StorageBackend
        """
        if not issubclass(backend_class, StorageBackend):
            raise TypeError(f"{backend_class} must be a subclass of StorageBackend")

        if name in self._backends and not override:
            raise ValueError(
                f"Storage backend '{name}' is already registered. Use override=True to replace it."
            )

        self._backends[name] = backend_class
        logger.info(f"Registered storage backend: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a backend.

        Args:
            name: Name of backend to unregister

        Returns:
            True if backend was unregistered, False if not found
        """
        if name in self._backends:
            del self._backends[name]
            with self._instance_lock:
                # Remove any cached instances for this backend
                keys_to_remove = [k for k in self._instances if k.startswith(f"{name}:")]
                for k in keys_to_remove:
                    del self._instances[k]
            return True
        return False

    def get_class(self, name: str) -> Optional[Type[StorageBackend]]:
        """Get backend class by name."""
        return self._backends.get(name)

    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self._backends.keys())

    def has_backend(self, name: str) -> bool:
        """Check if backend is registered."""
        return name in self._backends

    def create(
        self,
        name: str,
        singleton: bool = True,
        **kwargs,
    ) -> StorageBackend:
        """Create backend instance.

        Args:
            name: Backend name
            singleton: If True, return cached instance (default)
            **kwargs: Arguments passed to backend constructor

        Returns:
            StorageBackend instance

        Raises:
            ValueError: If backend not found
        """
        backend_class = self._backends.get(name)
        if not backend_class:
            available = ", ".join(self._backends.keys())
            raise ValueError(f"Unknown storage backend '{name}'. Available: {available or 'none'}")

        if singleton:
            cache_key = self._build_cache_key(name, kwargs)

            with self._instance_lock:
                if cache_key not in self._instances:
                    instance = backend_class(**kwargs)
                    self._instances[cache_key] = instance
                    logger.debug(f"Created singleton storage backend instance: {name}")
                return self._instances[cache_key]
        else:
            return backend_class(**kwargs)

    def _build_cache_key(self, name: str, kwargs: Dict[str, Any]) -> str:
        """Build cache key for singleton instances."""

        def serialize_value(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (list, tuple)):
                return [serialize_value(x) for x in v]
            if isinstance(v, dict):
                return {k: serialize_value(val) for k, val in sorted(v.items())}
            return f"{type(v).__name__}:{repr(v)}"

        try:
            sorted_kwargs = {
                k: serialize_value(v) for k, v in sorted(kwargs.items()) if v is not None
            }
            kwargs_json = json.dumps(sorted_kwargs, sort_keys=True)
            kwargs_hash = hashlib.md5(kwargs_json.encode()).hexdigest()[:12]
        except (TypeError, ValueError):
            kwargs_hash = hashlib.md5(repr(sorted(kwargs.items())).encode()).hexdigest()[:12]

        return f"{name}:{kwargs_hash}"

    def clear_instances(self) -> None:
        """Clear all cached backend instances."""
        with self._instance_lock:
            self._instances.clear()
        logger.debug("Cleared all storage backend instances")


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


def get_storage_registry() -> StorageRegistry:
    """Get the global storage backend registry."""
    return StorageRegistry()


def register_storage_backend(
    name: str,
    backend_class: Type[StorageBackend],
    override: bool = False,
) -> None:
    """Register a backend class with the global registry.

    Args:
        name: Unique name for the backend
        backend_class: Class implementing StorageBackend
        override: If True, replace existing registration
    """
    get_storage_registry().register(name, backend_class, override)


def get_available_storage_backends() -> List[str]:
    """List all available storage backend names."""
    return get_storage_registry().list_backends()


def get_storage_backend(
    name: Optional[str] = None,
    singleton: bool = True,
    **kwargs,
) -> StorageBackend:
    """Get or create a storage backend instance.

    This is the main entry point for obtaining storage backends.
    If no name is provided, uses the default backend from configuration.

    Args:
        name: Backend name (default: from config or "local")
        singleton: Return cached instance if True (default)
        **kwargs: Backend-specific configuration

    Returns:
        StorageBackend instance

    Examples::

        # Get default local backend
        storage = get_storage_backend()

        # Get S3 backend with config
        storage = get_storage_backend("s3", bucket="my-docs")

        # Get fsspec backend for GCS
        storage = get_storage_backend("fsspec", protocol="gcs", base_path="my-bucket/rag")
    """
    registry = get_storage_registry()

    if name is None:
        name = _get_default_storage_backend_name()

    # Apply default configs when none provided
    if not kwargs:
        if name == "local":
            kwargs = _get_default_local_config()
        elif name == "s3":
            kwargs = _get_default_s3_config()
        elif name == "fsspec":
            kwargs = _get_default_fsspec_config()

    return registry.create(name, singleton=singleton, **kwargs)


def _get_default_storage_backend_name() -> str:
    """Get default storage backend name from configuration.

    Priority:
    1. RAG_STORAGE_BACKEND environment variable
    2. rag_config.storage_backend setting
    3. Default: "local"
    """
    env_backend = os.environ.get("RAG_STORAGE_BACKEND")
    if env_backend:
        return env_backend.lower()

    try:
        from ..core.config import rag_config

        backend = getattr(rag_config, "storage_backend", None)
        if backend:
            return backend.lower()
    except (ImportError, Exception):
        pass

    return "local"


def _get_default_local_config() -> Dict[str, Any]:
    """Get default local storage configuration."""
    # base_dir=None means lazy resolution from rag_config.upload_dir
    return {}


def _get_default_s3_config() -> Dict[str, Any]:
    """Get default S3 configuration from environment.

    Environment variables:
    - RAG_STORAGE_S3_BUCKET (required)
    - RAG_STORAGE_S3_PREFIX (default: "rag-documents")
    - RAG_STORAGE_S3_ENDPOINT_URL (for MinIO, R2, etc.)
    - RAG_STORAGE_S3_REGION (default: "us-east-1")
    - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (standard AWS SDK vars)
    """
    bucket = os.environ.get("RAG_STORAGE_S3_BUCKET")
    if not bucket:
        logger.warning("RAG_STORAGE_S3_BUCKET not set, S3 backend may fail")
        return {}

    config: Dict[str, Any] = {"bucket": bucket}

    prefix = os.environ.get("RAG_STORAGE_S3_PREFIX")
    if prefix:
        config["prefix"] = prefix

    endpoint_url = os.environ.get("RAG_STORAGE_S3_ENDPOINT_URL")
    if endpoint_url:
        config["endpoint_url"] = endpoint_url

    region = os.environ.get("RAG_STORAGE_S3_REGION")
    if region:
        config["region_name"] = region

    return config


def _get_default_fsspec_config() -> Dict[str, Any]:
    """Get default fsspec configuration from environment.

    Environment variables:
    - RAG_STORAGE_FSSPEC_PROTOCOL (required, e.g. "gcs", "az", "ftp")
    - RAG_STORAGE_FSSPEC_BASE_PATH (default: "")
    - RAG_STORAGE_FSSPEC_OPTIONS (JSON string of storage_options)
    """
    protocol = os.environ.get("RAG_STORAGE_FSSPEC_PROTOCOL")
    if not protocol:
        logger.warning("RAG_STORAGE_FSSPEC_PROTOCOL not set, fsspec backend may fail")
        return {}

    config: Dict[str, Any] = {"protocol": protocol}

    base_path = os.environ.get("RAG_STORAGE_FSSPEC_BASE_PATH")
    if base_path:
        config["base_path"] = base_path

    options_json = os.environ.get("RAG_STORAGE_FSSPEC_OPTIONS")
    if options_json:
        try:
            config["storage_options"] = json.loads(options_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid JSON in RAG_STORAGE_FSSPEC_OPTIONS, ignoring")

    return config


# =============================================================================
# Backend Auto-Discovery (for plugins)
# =============================================================================


def discover_storage_backends(package_prefix: str = "lasagna_storage_") -> int:
    """Auto-discover and register storage backend plugins.

    Looks for installed packages with the given prefix that expose
    a ``register_backend(registry)`` function.

    Args:
        package_prefix: Prefix for plugin packages

    Returns:
        Number of backends discovered

    Example:
        A plugin package ``lasagna_storage_azure`` would:
        1. Have a ``register_backend(registry)`` function
        2. Call ``registry.register("azure", AzureStorageBackend)``
    """
    import importlib
    import pkgutil

    registry = get_storage_registry()
    count = 0

    for module_info in pkgutil.iter_modules():
        if module_info.name.startswith(package_prefix):
            try:
                module = importlib.import_module(module_info.name)
                if hasattr(module, "register_backend"):
                    module.register_backend(registry)
                    count += 1
                    logger.info(f"Discovered storage backend plugin: {module_info.name}")
            except Exception as e:
                logger.warning(f"Failed to load storage backend plugin {module_info.name}: {e}")

    return count
