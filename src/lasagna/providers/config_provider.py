"""Configuration provider interface for RAG system.

This interface abstracts configuration sources, allowing the RAG module
to work with any configuration system (not just one framework’s config).

Example usage for external agents:

    from lasagna.providers import SimpleConfigProvider
    from pathlib import Path

    config = SimpleConfigProvider(
        db_dir=Path("/tmp/my_agent/dbs"),
        upload_dir=Path("/tmp/my_agent/uploads"),
        rag_settings={"chunk_size": 1000, "enable_reranking": True},
    )
"""

import copy
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigProvider(ABC):
    """Abstract interface for RAG configuration.

    Implementations provide paths, settings, and configuration values
    needed by the RAG system. This abstraction allows the RAG module
    to work with different configuration sources.

    Default: SimpleConfigProvider (auto-configured from env vars)
    Custom: Implement this ABC and pass to configure_providers()
    """

    @property
    @abstractmethod
    def db_dir(self) -> Path:
        """Directory for database files (SQLite metadata, citations, etc.).

        Returns:
            Path to database directory (will be created if missing)
        """
        ...

    @property
    @abstractmethod
    def upload_dir(self) -> Path:
        """Directory for uploaded document files.

        Returns:
            Path to upload directory (will be created if missing)
        """
        ...

    @property
    def data_dir(self) -> Path:
        """Directory for application state (session config, citations, etc.).

        Defaults to db_dir. Override to separate state from database files.
        """
        return self.db_dir

    @property
    @abstractmethod
    def package_root(self) -> Path:
        """Root directory of the package (for finding settings files).

        Used to locate settings.lasagna.toml and other config files.

        Returns:
            Path to package root directory
        """
        ...

    @abstractmethod
    def get_rag_settings(self) -> Dict[str, Any]:
        """Get RAG-specific settings dictionary.

        Returns settings for RAG configuration like:
        - embedding_model, embedding_dim
        - chunk_size, chunk_overlap
        - enable_reranking, reranker_model
        - enable_hybrid_search, hybrid_alpha
        - etc.

        Returns:
            Dictionary of RAG settings (can be empty for defaults)
        """
        ...

    @abstractmethod
    def get_llm_settings(self, key: str) -> Optional[Dict[str, Any]]:
        """Get LLM model settings by key path.

        Used to resolve model references like 'agent.action.model'
        to actual model configuration.

        Args:
            key: Settings path like 'agent.action.model' or 'rag.refinement_model'

        Returns:
            Model settings dict with 'platform', 'model_name', etc.
            None if key not found.
        """
        ...


class SimpleConfigProvider(ConfigProvider):
    """Simple in-memory configuration provider.

    Thread-safe: all state is immutable after construction; getters return
    deep copies to prevent external mutation.

    Use this for testing or when you want to provide config programmatically
    without relying on file-based configuration.

    Example:
        config = SimpleConfigProvider(
            db_dir=Path("/data/rag/dbs"),
            upload_dir=Path("/data/rag/uploads"),
            rag_settings={
                "chunk_size": 800,
                "enable_reranking": True,
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            llm_settings={
                "default": {
                    "platform": "openai",
                    "model_name": "gpt-6-luna",
                    "max_tokens": 4096,
                }
            },
        )
    """

    def __init__(
        self,
        db_dir: Path,
        upload_dir: Path,
        package_root: Optional[Path] = None,
        rag_settings: Optional[Dict[str, Any]] = None,
        llm_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize simple config provider.

        Args:
            db_dir: Directory for database files
            upload_dir: Directory for uploaded documents
            package_root: Optional package root (defaults to db_dir)
            rag_settings: Optional RAG settings dictionary
            llm_settings: Optional LLM settings keyed by reference name

        Raises:
            ValueError: If directories cannot be created (permissions, read-only FS)
        """
        self._db_dir = Path(db_dir)
        self._upload_dir = Path(upload_dir)
        self._package_root = Path(package_root) if package_root else self._db_dir
        self._rag_settings = rag_settings or {}
        self._llm_settings = llm_settings or {}

        # Ensure directories exist — fail fast with clear message
        for path, label in [(self._db_dir, "db_dir"), (self._upload_dir, "upload_dir")]:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"Cannot create {label} directory '{path}': {exc}") from exc

    def __repr__(self) -> str:
        return f"SimpleConfigProvider(db_dir={self._db_dir!r}, upload_dir={self._upload_dir!r})"

    @property
    def db_dir(self) -> Path:
        return self._db_dir

    @property
    def upload_dir(self) -> Path:
        return self._upload_dir

    @property
    def package_root(self) -> Path:
        return self._package_root

    def get_rag_settings(self) -> Dict[str, Any]:
        return copy.deepcopy(self._rag_settings)

    def get_llm_settings(self, key: str) -> Optional[Dict[str, Any]]:
        """Get LLM settings by key.

        Supports both flat keys ('default') and dotted paths ('agent.action.model').
        For dotted paths, returns the value at 'default' if path not found.
        """
        # Direct lookup
        if key in self._llm_settings:
            return copy.deepcopy(self._llm_settings[key])

        # Try last part of dotted path
        if "." in key:
            last_part = key.split(".")[-1]
            if last_part in self._llm_settings:
                return copy.deepcopy(self._llm_settings[last_part])

        # Fall back to default
        if "default" in self._llm_settings:
            return copy.deepcopy(self._llm_settings["default"])

        return None
