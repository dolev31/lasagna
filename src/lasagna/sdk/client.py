"""Knowledge Client - Primary entry point for the Knowledge SDK.

This client provides a unified interface for all knowledge operations,
following the same patterns as RAGManager in the main SDK.

Example:
    ```python
    from lasagna.sdk import KnowledgeClient, KnowledgeConfig

    # Simple usage with defaults
    client = KnowledgeClient()

    # Custom configuration
    client = KnowledgeClient(
        config=KnowledgeConfig(
            embedding_model="BAAI/bge-large-en-v1.5",
            enable_reranking=True,
        )
    )

    # Upload documents
    result = await client.upload("./document.pdf")
    print(f"Uploaded: {result.document_id}")

    # Search with citations
    results = await client.search("What is the revenue?")
    for result in results:
        print(f"[{result.score:.2f}] {result.chunk.content[:100]}...")

    # Get formatted context for LLM injection
    context = await client.get_context("quarterly performance")

    # Cleanup
    await client.close()
    ```
"""

from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path
import uuid
import asyncio

from loguru import logger

from .config import KnowledgeConfig
from .types import (
    Document,
    SearchResults,
    UploadResult,
    AgenticSearchResults,
    DocumentStatus,
)

if TYPE_CHECKING:
    from .managers.document_manager import DocumentManager
    from .managers.search_manager import SearchManager
    from .managers.config_manager import ConfigManager
    from .managers.citation_manager import CitationManager


class KnowledgeClient:
    """Unified client for knowledge base operations.

    Provides an async API for document management, search, and
    retrieval-augmented generation. Wraps the underlying RAGBackend
    with a clean, simple interface.

    Attributes:
        config: Current knowledge configuration
        session_id: Session identifier for multi-tenancy
        documents: DocumentManager for document operations
        citations: CitationManager for citation tracking

    Example:
        ```python
        # Create client with defaults
        client = KnowledgeClient()

        # With custom configuration
        client = KnowledgeClient(
            config=KnowledgeConfig(
                enable_reranking=True,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
            ),
            session_id="my-session",
        )

        # Use as context manager
        async with KnowledgeClient() as client:
            await client.upload("./doc.pdf")
            results = await client.search("query")
        ```
    """

    def __init__(
        self,
        config: Optional[KnowledgeConfig] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the Knowledge Client.

        Args:
            config: Knowledge configuration. Uses defaults if not provided.
            session_id: Session identifier for multi-tenant usage.
                Auto-generated if not provided.

        Raises:
            ValueError: If session_id is an invalid string like 'None'.
        """
        self._config = config or KnowledgeConfig()

        # Session ID validation
        if session_id is None:
            self._session_id = f"sdk_{uuid.uuid4().hex[:12]}"
        elif session_id in ("None", "null", "undefined", ""):
            raise ValueError(
                f"Invalid session_id: '{session_id}'. "
                "Must be a non-empty string or None (to auto-generate)."
            )
        else:
            self._session_id = session_id

        # Lazy-initialized backend
        self._backend = None
        #: Uploads still processing; close() waits for them.
        self._in_flight_documents: set = set()
        self._backend_lock = asyncio.Lock()

        # Lazy-initialized managers
        self._document_manager: Optional["DocumentManager"] = None
        self._search_manager: Optional["SearchManager"] = None
        self._config_manager: Optional["ConfigManager"] = None
        self._citation_manager: Optional["CitationManager"] = None

        # Focused documents for scoped searches
        self._focused_document_ids: Optional[List[str]] = None

        # Tracks which backend keys the bridge wrote to session overrides,
        # so stale keys can be removed when config reverts to defaults.
        self._bridged_runtime_keys: set = set()

        logger.debug(f"KnowledgeClient initialized: session={self._session_id}")

    # =========================================================================
    # Alternative Constructors
    # =========================================================================

    @classmethod
    def from_url(
        cls,
        url: str,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "KnowledgeClient":
        """Connect to a remote RAG Knowledge API server.

        Creates a lightweight client that delegates all operations to a
        REST API — no heavy dependencies required (only ``httpx``).

        Args:
            url: Base URL of the RAG Knowledge API (e.g. ``https://rag.myapp.com``)
            api_key: Optional Bearer token for authentication
            session_id: Session identifier. Auto-generated if not provided.

        Returns:
            KnowledgeClient connected to the remote server

        Example:
            ```python
            from lasagna.sdk import KnowledgeClient

            client = KnowledgeClient.from_url(
                "https://rag.myapp.com",
                api_key="sk-...",
            )
            await client.upload("./report.pdf")
            context = await client.get_context("quarterly revenue")
            await client.close()
            ```
        """
        from .remote_backend import RemoteRAGBackend

        # Validate session_id the same way __init__ does
        if session_id is not None and session_id in ("None", "null", "undefined", ""):
            raise ValueError(
                f"Invalid session_id: '{session_id}'. "
                "Must be a non-empty string or None (to auto-generate)."
            )

        instance = cls.__new__(cls)
        instance._config = KnowledgeConfig()
        instance._session_id = session_id or f"remote_{uuid.uuid4().hex[:8]}"
        instance._backend = RemoteRAGBackend(
            base_url=url,
            api_key=api_key,
            session_id=instance._session_id,
        )
        instance._backend_lock = asyncio.Lock()
        instance._document_manager = None
        instance._search_manager = None
        instance._config_manager = None
        instance._citation_manager = None
        instance._focused_document_ids = None
        instance._bridged_runtime_keys = set()

        logger.info(f"KnowledgeClient connected to remote: {url}")
        return instance

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def session_id(self) -> str:
        """Current session identifier."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Set the session ID with validation.

        Note: Changing session ID clears focused documents and re-applies
        bridged runtime config to the new session.
        """
        if value in ("None", "null", "undefined", ""):
            raise ValueError(f"Invalid session_id: '{value}'")
        self._session_id = value
        self._focused_document_ids = None
        # Reset tracked keys so the new session gets a fresh set of overrides
        self._bridged_runtime_keys = set()
        if self._backend is not None:
            self._apply_runtime_config()
        logger.info(f"Session ID changed to: {value}")

    @property
    def config(self) -> KnowledgeConfig:
        """Current configuration."""
        return self._config

    @property
    def documents(self) -> "DocumentManager":
        """Document management operations.

        Provides upload, list, delete, and chunk retrieval.

        Example:
            ```python
            # Upload
            result = await client.documents.upload("./doc.pdf")

            # List documents
            docs = await client.documents.list()

            # Delete
            await client.documents.delete("doc_123")
            ```
        """
        if self._document_manager is None:
            from .managers.document_manager import DocumentManager

            self._document_manager = DocumentManager(self)
        return self._document_manager

    @property
    def search(self) -> "SearchManager":
        """Search operations.

        Provides semantic search, context retrieval, and hybrid search.

        Example:
            ```python
            # Basic search
            results = await client.search.search("revenue")

            # Get context for LLM
            context = await client.search.get_context("key findings")
            ```
        """
        if self._search_manager is None:
            from .managers.search_manager import SearchManager

            self._search_manager = SearchManager(self)
        return self._search_manager

    @property
    def citations(self) -> "CitationManager":
        """Citation tracking operations.

        Provides citation retrieval and verification.

        Example:
            ```python
            citation = await client.citations.get("abc123")
            ```
        """
        if self._citation_manager is None:
            from .managers.citation_manager import CitationManager

            self._citation_manager = CitationManager(self)
        return self._citation_manager

    @property
    def focused_document_ids(self) -> Optional[List[str]]:
        """Currently focused document IDs, if any."""
        return self._focused_document_ids

    # =========================================================================
    # Backend Access (Internal)
    # =========================================================================

    async def _get_backend(self):
        """Get or initialize the RAG backend.

        Lazy initialization ensures backend is only created when needed.
        Thread-safe via asyncio.Lock.

        Config bridge integration:
        - Init-time config is applied to ``rag_config._env_overrides``
          BEFORE the backend singleton is created.
        - Runtime config is applied via ``SessionConfigProvider`` AFTER
          creation, scoped per-session.

        Returns:
            RAGBackend instance
        """
        if self._backend is None:
            async with self._backend_lock:
                if self._backend is None:
                    # Tier 1: Init-time config BEFORE backend singleton
                    self._apply_init_time_config()

                    from ..api.rag_backend import get_rag_backend

                    self._backend = get_rag_backend()

                    # Tier 2: Runtime config via session provider (per-session)
                    self._apply_runtime_config()

                    logger.debug("RAGBackend initialized with config bridge")
        return self._backend

    def _apply_init_time_config(self) -> None:
        """Apply init-time overrides to global rag_config BEFORE backend creation.

        Only non-default fields are written.  Previously-bridged keys that
        are no longer non-default are removed, preventing leakage across
        clients.  Init-time fields are inherently process-wide (backend is
        a singleton).
        """
        try:
            from ..core.config import rag_config
            from ._config_bridge import apply_init_overrides

            apply_init_overrides(rag_config._env_overrides, self._config)
            logger.info("Init-time config bridge applied")
        except Exception as e:
            logger.debug(f"Init-time config bridge skipped: {e}")

    @property
    def _is_remote(self) -> bool:
        """True when connected to a remote API (from_url path)."""
        if self._backend is None:
            return False
        return type(self._backend).__name__ == "RemoteRAGBackend"

    def _apply_runtime_config(self) -> None:
        """Apply runtime overrides via SessionConfigProvider (per-session safe).

        Only non-default fields are written, minimizing masking of
        profile/TOML settings.  Keys that were previously bridged but
        are no longer non-default are explicitly removed so that
        reverting a field to its SDK default clears the session override.

        Skipped for remote backends — the remote server manages its own config.
        """
        if self._is_remote:
            return
        from ..api._profile_guard import EmbeddingSizeConflict

        try:
            from ..providers import get_session_provider
            from ._config_bridge import get_runtime_overrides

            overrides = get_runtime_overrides(self._config)
            session_provider = get_session_provider()

            # Remove stale keys: previously bridged but no longer overridden
            new_keys = set(overrides.keys())
            stale_keys = self._bridged_runtime_keys - new_keys
            if stale_keys:
                session_provider.remove_config_overrides(
                    self._session_id,
                    stale_keys,
                )
                logger.debug(
                    f"Removed stale bridge overrides from session {self._session_id}: {stale_keys}"
                )

            # Write current overrides
            if overrides:
                session_provider.set_config_overrides(
                    self._session_id,
                    overrides,
                )

            self._bridged_runtime_keys = new_keys
            logger.debug(
                f"Runtime config applied to session {self._session_id}: {len(overrides)} keys"
            )
        except EmbeddingSizeConflict:
            # The store refused this client's embedding size. Skipping quietly
            # would run its searches on the session's other model instead.
            raise
        except Exception as e:
            logger.debug(f"Runtime config bridge skipped: {e}")

    # =========================================================================
    # Core Operations (Convenience Methods)
    # =========================================================================

    async def upload(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> UploadResult:
        """Upload and index a document.

        This is a convenience method that delegates to documents.upload().

        Args:
            file_path: Path to the document file (must exist)
            filename: Optional display name (defaults to original filename)
            metadata: Optional custom metadata
            document_id: Optional explicit document ID for benchmark evaluation

        Returns:
            UploadResult with document_id, chunk_count, status

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported

        Example:
            ```python
            result = await client.upload("./report.pdf")
            print(f"Uploaded: {result.document_id}, {result.chunk_count} chunks")
            ```
        """
        return await self.documents.upload(file_path, filename, metadata, document_id=document_id)

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
        routing_tier: Optional[str] = None,
    ) -> SearchResults:
        """Search for relevant document chunks.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold (0-1)
            document_ids: Optional filter by document IDs (overrides focus)
            routing_tier: Force a routing tier ("fast", "standard",
                "comprehensive"). If None, the router auto-selects.

        Returns:
            SearchResults containing ranked chunks with scores

        Example:
            ```python
            results = await client.search_documents("What is the revenue?")
            for result in results:
                print(f"[{result.score:.2f}] {result.chunk.content[:100]}")
            ```
        """
        return await self.search.search(
            query, top_k, min_score, document_ids, routing_tier=routing_tier
        )

    async def get_context(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5,
        routing_tier: Optional[str] = None,
    ) -> str:
        """Get formatted context string for LLM injection.

        This is the recommended method for RAG. Returns context with
        source citations ready to inject into LLM prompts.

        Args:
            query: Natural language search query
            document_ids: Optional filter by document IDs (overrides focus)
            max_chunks: Maximum chunks to include in context
            routing_tier: Force a routing tier ("fast", "standard",
                "comprehensive"). If None, the router auto-selects.

        Returns:
            Formatted context string with citations

        Example:
            ```python
            context = await client.get_context("What are the key findings?")
            # Use context in LLM prompt
            prompt = f"Based on the following context:\\n{context}\\n\\nAnswer: ..."
            ```
        """
        return await self.search.get_context(
            query, document_ids, max_chunks, routing_tier=routing_tier
        )

    async def list_documents(
        self,
        status: Optional[DocumentStatus] = None,
    ) -> List[Document]:
        """List all documents in the current session.

        Args:
            status: Optional filter by status

        Returns:
            List of Document objects

        Example:
            ```python
            docs = await client.list_documents()
            for doc in docs:
                print(f"{doc.filename}: {doc.chunk_count} chunks")
            ```
        """
        return await self.documents.list(status)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks.

        Args:
            document_id: ID of document to delete

        Returns:
            True if deleted, False if not found

        Example:
            ```python
            success = await client.delete_document("doc_abc123")
            ```
        """
        return await self.documents.delete(document_id)

    # =========================================================================
    # Document Focus (Scoped Searches)
    # =========================================================================

    def set_focused_documents(self, document_ids: List[str]) -> None:
        """Focus all searches on specific documents.

        When set, all search operations will be scoped to these documents.
        Use clear_focused_documents() to remove the restriction.

        Args:
            document_ids: List of document IDs to focus on

        Example:
            ```python
            # Focus on specific documents
            client.set_focused_documents(["doc_123", "doc_456"])

            # Now searches only look in these documents
            results = await client.search_documents("revenue")

            # Clear focus
            client.clear_focused_documents()
            ```
        """
        self._focused_document_ids = document_ids
        logger.info(f"Focused on {len(document_ids)} document(s): {document_ids}")

    def clear_focused_documents(self) -> None:
        """Clear focused document restriction.

        After calling this, searches will include all documents in the session.
        """
        self._focused_document_ids = None
        logger.info("Document focus cleared")

    # =========================================================================
    # Advanced Features
    # =========================================================================

    async def agentic_search(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> AgenticSearchResults:
        """Perform multi-hop agentic search with query refinement.

        Uses a ReAct-style agent to iteratively search and refine queries
        for complex information needs.

        Args:
            query: Initial search query
            max_iterations: Maximum search iterations (default from config)
            document_ids: Optional filter by document IDs

        Returns:
            AgenticSearchResults with hop data and final context

        Example:
            ```python
            results = await client.agentic_search(
                "What were the key factors affecting Q3 performance?"
            )
            print(f"Found context in {results.total_iterations} hops")
            print(results.final_context)
            ```
        """
        from .agentic import AgenticRAGWrapper

        wrapper = AgenticRAGWrapper(self)
        return await wrapper.search(
            query,
            max_iterations or self._config.max_agent_iterations,
            document_ids or self._focused_document_ids,
        )

    # =========================================================================
    # Configuration
    # =========================================================================

    async def set_config(self, **kwargs) -> None:
        """Update configuration options.

        Creates a new config with the specified updates.
        Runtime fields are re-applied to the session provider immediately.

        Note: Init-time fields (``vector_backend``, ``database_backend``,
        etc.) only take effect when creating a new ``KnowledgeClient``.
        Changing them after the backend is initialized logs a warning.

        Args:
            **kwargs: Configuration options to update

        Example:
            ```python
            await client.set_config(
                enable_reranking=True,
                enable_hyde=True,
                chunk_size=800,
            )
            ```
        """
        self._config = self._config.with_updates(**kwargs)

        if self._is_remote:
            logger.warning(
                "set_config() on a remote client only affects local SDK behavior "
                "(citations, presets). Search pipeline config is managed by the "
                "remote server."
            )
        elif self._backend is not None:
            from ._config_bridge import INIT_TIME_FIELDS

            init_changes = [k for k in kwargs if k in INIT_TIME_FIELDS]
            if init_changes:
                logger.warning(
                    f"Init-time fields changed after backend creation: {init_changes}. "
                    "Create a new KnowledgeClient for these to take effect."
                )
            self._apply_runtime_config()

        logger.info(f"Config updated: {list(kwargs.keys())}")

    def use_preset(self, preset: str) -> None:
        """Apply a configuration preset.

        Args:
            preset: One of 'speed', 'standard', 'balanced', 'max_quality'

        Raises:
            ValueError: If preset name not recognized

        Example:
            ```python
            client.use_preset("standard")     # SOTA features + lightweight model
            client.use_preset("balanced")      # SOTA features + best model
            client.use_preset("max_quality")   # All features enabled
            client.use_preset("speed")         # Pure vector search only
            ```
        """
        from .config import (
            config_for_speed,
            config_for_standard,
            config_for_balanced,
            config_for_max_quality,
        )

        presets = {
            "speed": config_for_speed,
            "standard": config_for_standard,
            "balanced": config_for_balanced,
            "max_quality": config_for_max_quality,
            # Backward compatibility
            "quality": config_for_max_quality,
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: '{preset}'. Use one of: {list(presets.keys())}")

        self._config = presets[preset]()

        if self._backend is not None:
            self._apply_runtime_config()

        logger.info(f"Applied preset: {preset}")

    def get_config_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary.

        Returns:
            Dictionary with all configuration values
        """
        return self._config.to_dict()

    # =========================================================================
    # Lifecycle policy
    # =========================================================================

    async def get_lifecycle_policy(self) -> Dict[str, Any]:
        """Return the currently active lifecycle garbage-collection policy.

        For remote backends, calls ``GET /api/v1/admin/gc/policy`` (requires
        ``RAG_ADMIN_TOKEN`` env var).  For local backends, reads the live
        policy from the backend instance directly.

        Returns:
            Dict with all policy fields (soft_delete_grace_hours, gc_cycle_hours, …)
        """
        backend = await self._get_backend()
        if hasattr(backend, "get_lifecycle_policy"):
            # RemoteRAGBackend — HTTP call
            return backend.get_lifecycle_policy()
        # Local backend — read from lifecycle manager
        import dataclasses

        mgr = getattr(backend, "_lifecycle_manager", None)
        if mgr is not None:
            return dataclasses.asdict(mgr._policy)
        from ..lifecycle.policy import LifecyclePolicy

        return dataclasses.asdict(LifecyclePolicy.load())

    async def set_lifecycle_policy(self, **updates: Any) -> Dict[str, Any]:
        """Update lifecycle policy fields at runtime.

        Changes are process-local and not persisted across restarts.  To make
        them permanent, update ``[rag.lifecycle]`` in ``settings.lasagna.toml``
        or the corresponding ``RAG_GC_*`` environment variables.

        Args:
            **updates: Policy fields to change.  Supported keys:
                soft_delete_grace_hours, document_ttl_days (None or 0 disables TTL),
                error_retention_days, session_idle_days, gc_cycle_hours,
                orphan_scan_hours, safety_threshold_percent, batch_size,
                reconcile_quota_on_startup.

        Returns:
            Dict with the full updated policy.

        Example:
            ```python
            # Disable TTL (both None and 0 are accepted)
            await client.set_lifecycle_policy(document_ttl_days=None)
            await client.set_lifecycle_policy(document_ttl_days=0)  # same effect

            # Change GC cadence
            await client.set_lifecycle_policy(gc_cycle_hours=12)
            ```
        """
        backend = await self._get_backend()
        if hasattr(backend, "set_lifecycle_policy"):
            # RemoteRAGBackend — HTTP PATCH call
            return backend.set_lifecycle_policy(updates)
        # Local backend — update lifecycle manager directly
        import dataclasses

        mgr = getattr(backend, "_lifecycle_manager", None)
        if mgr is None:
            raise RuntimeError("Lifecycle manager not available on this backend instance.")
        policy = mgr._policy
        # Normalize document_ttl_days: 0 and None both mean "disabled" → store as None.
        updated = dataclasses.replace(
            policy,
            **{
                k: (None if k == "document_ttl_days" and not v else v)
                for k, v in updates.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        mgr._policy = updated
        gc = getattr(mgr, "_gc", None)
        if gc is not None:
            gc._policy = updated
        scheduler = getattr(backend, "_lifecycle_scheduler", None)
        if scheduler is not None:
            scheduler.update_policy(updated)
        return dataclasses.asdict(updated)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self, processing_timeout: float = 600.0) -> None:
        """Close client and cleanup resources.

        Call this when done using the client to release connections. Documents
        this client uploaded that are still processing are waited for first
        (up to ``processing_timeout`` seconds): closing used to shut the vector
        store down under the worker, which then retried against a dead
        connection for minutes and left the document failed.
        """
        pending = list(self._in_flight_documents)
        if pending and self._document_manager is not None:
            logger.info(f"Waiting for {len(pending)} document(s) to finish processing")
            results = await asyncio.gather(
                *(
                    self._document_manager.wait_for_processing(d, timeout=processing_timeout)
                    for d in pending
                ),
                return_exceptions=True,
            )
            for doc_id, outcome in zip(pending, results):
                if isinstance(outcome, BaseException):
                    logger.warning(f"Document {doc_id} did not finish processing: {outcome}")

        # Clear cached managers
        self._document_manager = None
        self._search_manager = None
        self._config_manager = None
        self._citation_manager = None

        # Close remote backend if applicable
        if self._backend is not None and hasattr(self._backend, "close"):
            try:
                await self._backend.close()
            except Exception as e:
                logger.debug(f"Backend close error: {e}")

        self._backend = None

        logger.info("KnowledgeClient closed")

    async def delete_session(self, confirm: bool = False) -> Dict[str, Any]:
        """Delete all data associated with the current session.

        This is a comprehensive cleanup that removes:
        - All documents and their vector embeddings/chunks
        - All citations from the citation database
        - Search cache entries for this session
        - Session configuration state
        - BM25 hybrid search indices

        WARNING: This operation is irreversible. All data for this
        session will be permanently deleted.

        Args:
            confirm: Must be True to proceed (safety check)

        Returns:
            Dictionary with counts of deleted items:
            {
                "documents_deleted": int,
                "citations_deleted": int,
                "cache_invalidated": bool,
                "session_cleared": bool,
                "errors": list[str],  # partial failure messages, if any
            }

        Raises:
            ValueError: If confirm is not True
            KnowledgeError: If deletion operation fails

        Example:
            ```python
            # Delete entire session
            result = await client.delete_session(confirm=True)
            print(f"Deleted {result['documents_deleted']} documents")
            print(f"Deleted {result['citations_deleted']} citations")

            # Client remains usable with same session_id
            await client.upload("new_doc.pdf")  # Works - starts fresh
            ```
        """
        if not confirm:
            raise ValueError(
                "delete_session requires confirm=True. "
                "This will permanently delete ALL data for this session."
            )

        from .exceptions import KnowledgeError

        result = {
            "documents_deleted": 0,
            "citations_deleted": 0,
            "cache_invalidated": False,
            "session_cleared": False,
            "errors": [],
        }

        try:
            # 1. Delete all documents (includes vector DB cleanup)
            try:
                result["documents_deleted"] = await self.documents.delete_all(confirm=True)
            except Exception as e:
                logger.warning(f"Document deletion partial failure: {e}")
                result["errors"].append(f"documents: {e}")

            # 2. Clear all citations
            try:
                result["citations_deleted"] = await self.citations.clear(confirm=True)
            except Exception as e:
                logger.warning(f"Citation clearing partial failure: {e}")
                result["errors"].append(f"citations: {e}")

            # 3. Clear backend session state (cache, BM25, config)
            try:
                backend = await self._get_backend()
                backend_result = backend.clear_session_state(self._session_id)
                result["cache_invalidated"] = backend_result.get("cache_invalidated", False)
                result["session_cleared"] = backend_result.get("session_cleared", False)
            except Exception as e:
                logger.warning(f"Backend session cleanup partial failure: {e}")
                result["errors"].append(f"session_state: {e}")

            # 4. Clear client-side state
            self._focused_document_ids = None

            logger.info(
                f"Session {self._session_id} deleted: "
                f"{result['documents_deleted']} docs, "
                f"{result['citations_deleted']} citations"
            )

            return result

        except Exception as e:
            raise KnowledgeError(f"Session deletion failed: {e}") from e

    async def __aenter__(self) -> "KnowledgeClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"KnowledgeClient(session_id={self._session_id!r})"
