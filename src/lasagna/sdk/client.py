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
from lasagna.core.provenance import redact_url

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


#: What a client whose settings would leave its session's documents behind can do.
_SDK_REMEDY = (
    "Either use a client on another session for {new_model} and upload there, or reindex "
    "{them} in one step: await client.set_config(reindex=True) reindexes every document "
    "with {new_model} first and keeps the session as it is if one fails. To apply these "
    "settings anyway: await client.set_config(force=True); until each document is "
    "reindexed (client.documents.reindex(document_id)), searches return nothing from {them}."
)


def _changes(current: Dict[str, Any], overrides: Dict[str, Any], base: Any = None) -> bool:
    """Whether writing ``overrides`` over the session's ``current`` settings changes one.

    A key the session does not set has the process's value (``base``, a
    ``RAGConfig``), from the environment or the settings file.
    """
    for key, value in overrides.items():
        if key in current:
            have = current[key]
        else:
            have = getattr(base, key, _changes) if base is not None else _changes
        if isinstance(value, dict) and isinstance(have, dict):
            if _changes(have, value):
                return True
        elif have != value:
            return True
    return False


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
        #: Whether this client's settings were saved as its session's (first write).
        self._session_config_saved = False
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
        # The refusal of this client's settings (DocumentsLeftBehind), until
        # set_config resolves it or the settings change.
        self._settings_refused: Optional[Exception] = None

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
        session_token: Optional[str] = None,
    ) -> "KnowledgeClient":
        """Connect to a remote Lasagna API server.

        Creates a lightweight client that delegates all operations to a
        REST API — no heavy dependencies required (only ``httpx``).

        Args:
            url: Base URL of the Lasagna API (e.g. ``https://rag.myapp.com``)
            api_key: A key from the server's ``RAG_API_KEYS``. With
                ``RAG_REQUIRE_AUTH`` on, the client mints a session token with
                it (``POST /api/v1/sessions``) when the server first answers
                401, and its session becomes the token's. Also sent as
                ``Authorization: Bearer``.
            session_id: Session identifier. Auto-generated if not provided.
                With auth on, only an admin may name one, so leave it out.
            session_token: A token from ``POST /api/v1/sessions``, sent as
                ``X-Session-Token`` on every request.

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

        # Through __init__, so the remote client has every attribute a local
        # one has: built field by field, it lacked _in_flight_documents and
        # close() raised AttributeError.
        instance = cls(session_id=session_id or f"remote_{uuid.uuid4().hex[:8]}")

        def adopt(minted: str) -> None:
            instance._session_id = minted

        instance._backend = RemoteRAGBackend(
            base_url=url,
            api_key=api_key,
            session_id=instance._session_id,
            session_token=session_token,
            requested_session=session_id,
            on_session=adopt,
        )

        logger.info(f"KnowledgeClient connected to remote: {redact_url(url)}")
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
        self._session_config_saved = False
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
    def config_manager(self) -> "ConfigManager":
        """Profile and session configuration operations.

        The client reserved this manager and its documentation used it, but no
        property returned it, so ``client.config_manager`` raised
        AttributeError.

        Example:
            ```python
            await client.config_manager.set_profile("speed")
            profiles = await client.config_manager.list_profiles()
            ```
        """
        if self._config_manager is None:
            from .managers.config_manager import ConfigManager

            self._config_manager = ConfigManager(self)
        return self._config_manager

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
                    # Storage location first: it is fixed when the
                    # configuration first loads, which the next step triggers.
                    self._apply_storage_paths()

                    # Tier 1: Init-time config BEFORE backend singleton
                    self._apply_init_time_config()

                    from ..api.rag_backend import get_rag_backend

                    self._backend = get_rag_backend()

                    # Tier 2: Runtime config via session provider (per-session)
                    self._apply_runtime_config()

                    logger.debug("RAGBackend initialized with config bridge")
        elif self._settings_refused is not None:
            # Refused before: every operation says so until it is resolved,
            # rather than running on the session's settings instead.
            self._apply_runtime_config()
        return self._backend

    def _apply_storage_paths(self) -> None:
        """Put databases and uploads where ``db_path`` and ``upload_dir`` say.

        Where they live is process-wide and fixed when the configuration first
        loads. Before that, these two fields choose it. After it they cannot
        move it, and a warning names where the data is going instead. They used
        to be read by nothing, so ``config_for_standalone(db_path=...)`` wrote
        to the default data directory without a word.
        """
        cfg = self._config
        if cfg.db_path is None and cfg.upload_dir is None:
            return
        import warnings

        from .. import providers
        from ..core import config as core_config

        def _directory(path) -> Path:
            p = Path(path).expanduser()
            # A file path names its directory: the databases inside it have
            # fixed names (rag.db, rag_milvus.db).
            return p.parent if p.suffix and not p.is_dir() else p

        want_db = _directory(cfg.db_path) if cfg.db_path else None
        want_up = Path(cfg.upload_dir).expanduser() if cfg.upload_dir else None

        if core_config._rag_config is None and providers._config_provider is None:
            from ..core import paths

            providers.set_config_provider(
                providers.SimpleConfigProvider(
                    db_dir=want_db or paths.db_dir(),
                    upload_dir=want_up or paths.upload_dir(),
                    package_root=Path(providers.__file__).parent.parent,
                )
            )
            return

        effective = core_config.get_rag_config()
        db_dir = Path(effective.db_path).parent
        ignored = []
        if want_db is not None and want_db.resolve() != db_dir.resolve():
            ignored.append(f"db_path (databases are in {db_dir})")
        if want_up is not None and want_up.resolve() != Path(effective.upload_dir).resolve():
            ignored.append(f"upload_dir (uploads go to {effective.upload_dir})")
        if ignored:
            warnings.warn(
                f"KnowledgeConfig {' and '.join(ignored)} could not take effect: this "
                f"process loaded its configuration before the client was first used, "
                f"and storage cannot move after that. Set LASAGNA_DB_DIR and LASAGNA_UPLOAD_DIR "
                f"before importing lasagna, or make this the first client the process "
                f"uses.",
                UserWarning,
                stacklevel=3,
            )

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

    def _apply_runtime_config(self, *, force: bool = False, reindex: bool = False) -> None:
        """Apply runtime overrides via SessionConfigProvider (per-session safe).

        Only non-default fields are written, minimizing masking of
        profile/TOML settings.  Keys that were previously bridged but
        are no longer non-default are explicitly removed so that
        reverting a field to its SDK default clears the session override.
        Settings the session already has are not written again.

        Settings that would leave the session's documents in another model's
        collection are refused, as on every other surface
        (``DocumentsLeftBehind``): ``reindex`` reindexes every document with
        them first, ``force`` writes them anyway. They were written whatever
        the session held, so ``KnowledgeClient(session_id="r1",
        config=config_for_speed())`` hid r1's BGE-M3 documents from the CLI,
        REST and MCP, which refuse that change.

        Skipped for remote backends — the remote server manages its own config.

        Every failure raises (a refusal, a model that cannot load, a store
        that did not save them), and every later operation of the client
        tries again until they apply. A failure other than a refusal was
        logged and skipped: ``set_config(..., reindex=True)`` returned with
        nothing reindexed, and the client named a model the session did not
        search with.
        """
        if self._is_remote:
            return
        try:
            from ..providers import get_session_provider
            from ._config_bridge import get_runtime_overrides

            overrides = get_runtime_overrides(self._config)
            session_provider = get_session_provider()

            # Keys bridged before and no longer set: removed once the rest is
            # written, so a refused write leaves the session as it was.
            new_keys = set(overrides.keys())
            stale_keys = self._bridged_runtime_keys - new_keys

            # Write current overrides, unless the session has them already
            from ..core.config import get_rag_config

            read = getattr(session_provider, "get_resolved_config", None)
            current = (read(self._session_id) if read is not None else None) or {}
            changed = overrides and _changes(current, overrides, get_rag_config())
            if (overrides and (force or reindex or changed)) or stale_keys:
                self._write_settings(
                    session_provider, overrides, force=force, reindex=reindex, stale=stale_keys
                )
            if stale_keys:
                session_provider.remove_config_overrides(self._session_id, stale_keys)
                logger.debug(
                    f"Removed stale bridge overrides from session {self._session_id}: {stale_keys}"
                )

            self._bridged_runtime_keys = new_keys
            self._settings_refused = None
            logger.debug(
                f"Runtime config applied to session {self._session_id}: {len(overrides)} keys"
            )
        except Exception as failed:
            # Skipping it would run this client's searches on the session's
            # other settings while it reports its own.
            self._settings_refused = failed
            raise

    def _write_settings(
        self,
        session_provider,
        overrides: Dict[str, Any],
        *,
        force: bool,
        reindex: bool,
        stale: Optional[set] = None,
    ) -> None:
        """Write this client's settings to its session, or refuse them (see ``_apply_runtime_config``).

        ``stale``: keys this client set before and no longer does; the
        caller removes them after this write.
        """
        from ..core.session_config import get_knowledge_config_manager
        from ..core.session_config._data_models import _resolve

        sid = self._session_id
        mgr = get_knowledge_config_manager(session_id=sid)
        if reindex:
            from ..api._profile_switch import switch_profile

            # The session's own profile, with this client's settings as its
            # overrides: every document is reindexed with them first, and
            # the session changes only once all are.
            state = mgr.get_session_state(sid)
            kept = {k: v for k, v in state.session_overrides.items() if k not in (stale or ())}
            switch_profile(
                mgr,
                state.active_profile,
                session_id=sid,
                reindex=True,
                config={**kept, **overrides},
                backend=self._backend,
            )
            return
        if force:
            # As `lasagna configure --force` does: forced past the documents,
            # the models must still load, or every search fails after it.
            from ..api._profile_guard import documents_now
            from ..api._profile_switch import ProfileCannotLoad, unloadable_models

            problems = unloadable_models({**mgr.get_chosen_config(sid), **overrides})
            if problems:
                now = documents_now(sid, mgr.get_resolved_config(sid))
                raise ProfileCannotLoad(
                    f"These settings cannot be used here: {' '.join(problems)} Nothing was "
                    f"changed for session {sid!r}." + (f" {now}" if now else "")
                )
        else:
            # Before any write, in either session store: the in-memory one
            # would otherwise search another collection in this process only.
            state = mgr.get_session_state(sid)
            active = state.get_active_profile()
            kept = {k: v for k, v in state.session_overrides.items() if k not in (stale or ())}
            mgr._refuse_unsearchable(
                state,
                _resolve(_resolve(active.config if active else {}, kept), overrides),
                change="This client's settings",
                session_id=sid,
                remedy=_SDK_REMEDY,
            )
        if not overrides:
            return  # only keys to remove, which the caller does
        if getattr(session_provider, "_delegates_to_manager", False):
            # This process's own settings: every key may be set, but not
            # past the session's documents unless forced.
            mgr.patch_config(overrides, session_id=sid, trusted=True, force=force)
        else:
            session_provider.set_config_overrides(sid, overrides)

    def _remember_session_config(self) -> None:
        """Save this client's settings as its session's, on the session's first write.

        ``lasagna configure`` saves a session's profile to disk. With
        ``RAG_SESSION_BACKEND=memory`` a client's configuration lives in its
        process only (the in-memory session store, and the process-wide
        embedder), so after an SDK upload with ``config_for_speed()``,
        ``lasagna status`` showed BGE-M3 and the CLI, REST and MCP searched that
        corpus with a model that had not built it. A session that already has
        settings of its own keeps them. With the default store, applying the
        client's settings (``_apply_runtime_config``) has saved them already.
        """
        if self._session_config_saved or self._is_remote:
            return
        self._session_config_saved = True
        from ..api._profile_guard import EmbeddingSizeConflict

        try:
            from ..core.session_config import get_knowledge_config_manager
            from ._config_bridge import get_runtime_overrides

            overrides = get_runtime_overrides(self._config)
            if not overrides:
                return
            mgr = get_knowledge_config_manager(session_id=self._session_id)
            if mgr.get_chosen_config(self._session_id):
                return
            # This process's own settings, not a client's request.
            mgr.patch_config(
                overrides, session_id=self._session_id, trusted=True, remedy=_SDK_REMEDY
            )
        except EmbeddingSizeConflict:
            raise
        except Exception as e:
            logger.warning(
                f"This client's settings could not be saved for session {self._session_id!r}, "
                f"so the CLI, REST API and MCP server will search it with their defaults: {e}. "
                f"Run: lasagna configure --profile <name> -s {self._session_id}"
            )

    def _record_reindexed_model(self) -> None:
        """After a reindex, make the stored session name the model it was reindexed with.

        A client configured with another model than its session's moved the
        document to that model's collection, and the session kept the old
        model: ``lasagna search`` then printed "0 results" and ``status``
        named the old model. With the default store the client's settings are
        the session's already; this matters with ``RAG_SESSION_BACKEND=memory``.
        """
        if self._is_remote:
            return
        from ..core.config import rag_config
        from ..core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(session_id=self._session_id)
        here = rag_config.with_session(self._session_id)
        chosen = mgr.get_chosen_config(self._session_id)
        if chosen.get("embedding_model", rag_config.embedding_model) == here.embedding_model:
            return
        model = {"embedding_model": here.embedding_model, "embedding_dim": here.embedding_dim}
        if here.embedding_truncate_dim:
            model["embedding_truncate_dim"] = here.embedding_truncate_dim
        # The documents moved there: what the session searches with now.
        mgr.patch_config(model, session_id=self._session_id, force=True, trusted=True)

    # =========================================================================
    # Core Operations (Convenience Methods)
    # =========================================================================

    async def upload(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        allow_duplicates: bool = False,
    ) -> UploadResult:
        """Upload and index a document.

        This is a convenience method that delegates to documents.upload().

        Args:
            file_path: Path to the document file (must exist)
            filename: Optional display name (defaults to original filename)
            metadata: Not stored; a non-empty one raises ``ValueError``
            document_id: Optional explicit document ID for benchmark evaluation
            allow_duplicates: Index the file even when the session already
                holds the same content. By default that document is returned
                instead, with ``duplicate_of`` set.

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
        return await self.documents.upload(
            file_path,
            filename,
            metadata,
            document_id=document_id,
            allow_duplicates=allow_duplicates,
        )

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        routing_tier: Optional[str] = None,
    ) -> SearchResults:
        """Search for relevant document chunks.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            min_score: Drop results whose score is below this. A score is the
                reranker's when the profile reranks, else the retrieval score
                ((cosine similarity + 1) / 2 for dense search). Default: no threshold.
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

    async def set_config(self, *, reindex: bool = False, force: bool = False, **kwargs) -> None:
        """Update configuration options.

        Creates a new config with the specified updates.
        Runtime fields are re-applied to the session provider immediately.

        Note: Init-time fields (``vector_backend``, ``database_backend``,
        etc.) only take effect when creating a new ``KnowledgeClient``.
        Changing them after the backend is initialized logs a warning. The
        embedding fields are also session settings, and those apply to the
        session at once.

        Settings that would leave the session's documents in another model's
        collection raise ``DocumentsLeftBehind``, and nothing changes, as
        ``lasagna configure`` refuses them. ``reindex=True`` reindexes every
        document with the new settings first and changes the session only once
        all are (with no fields, it applies the client's current settings so);
        ``force=True`` applies them anyway, and the documents are found again
        once reindexed (``client.documents.reindex``).

        Args:
            **kwargs: Configuration options to update

        Example:
            ```python
            await client.set_config(
                enable_reranking=True,
                enable_hyde=True,
                chunk_size=800,
            )
            await client.set_config(embedding_model="BAAI/bge-small-en-v1.5",
                                    embedding_dim=384, reindex=True)
            ```
        """
        if reindex or force:
            await self._get_backend_for_settings()
        self._config = self._config.with_updates(**kwargs)
        # Raises when the session did not take them; until it does (a later
        # set_config(reindex=True) or force=True applies them), every
        # operation says so and get_config_dict() reports the session's.
        await self._apply_updated_config(kwargs, reindex=reindex, force=force)
        logger.info(f"Config updated: {list(kwargs.keys())}")

    async def _apply_updated_config(self, kwargs: Dict[str, Any], *, reindex: bool, force: bool):
        """Apply ``set_config``'s new settings to the session (see ``set_config``)."""
        if self._is_remote:
            logger.warning(
                "set_config() on a remote client only affects local SDK behavior "
                "(citations, presets). Search pipeline config is managed by the "
                "remote server."
            )
        elif self._backend is not None:
            from ._config_bridge import INIT_TIME_FIELDS

            from ._config_bridge import RUNTIME_FIELD_MAP

            init_changes = [
                k for k in kwargs if k in INIT_TIME_FIELDS and k not in RUNTIME_FIELD_MAP
            ]
            if init_changes:
                logger.warning(
                    f"Init-time fields changed after backend creation: {init_changes}. "
                    "Create a new KnowledgeClient for these to take effect."
                )
            if reindex:  # reindexes every document of the session
                await asyncio.to_thread(self._apply_runtime_config, reindex=True)
            else:
                self._apply_runtime_config(force=force)

    async def _get_backend_for_settings(self) -> None:
        """The backend, for a reindex or a forced write; a failure to apply the
        client's settings so far is what that call resolves, not an error."""
        if self._backend is None and not self._is_remote:
            try:
                await self._get_backend()
            except Exception:
                if self._backend is None:
                    raise  # the backend itself failed, not the settings

    def use_preset(self, preset: str) -> None:
        """Apply a configuration preset.

        Args:
            preset: One of 'speed', 'standard', 'balanced', 'max_quality'

        Raises:
            ValueError: If preset name not recognized

        Example:
            ```python
            client.use_preset("standard")     # hybrid search, reranking, agentic, BGE-M3
            client.use_preset("balanced")      # standard plus hierarchical chunking
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

        While the client's settings are not applied to its session (refused,
        or not saved), the settings it sets are reported as the session has
        them in force, not as asked.

        Returns:
            Dictionary with all configuration values
        """
        data = self._config.to_dict()
        if self._settings_refused is not None and self._backend is not None:
            data.update(self._settings_in_force(data))
        return data

    def _settings_in_force(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """The session's values, in force now, of the runtime fields this client sets."""
        from ..core.config import rag_config
        from ._config_bridge import RUNTIME_FIELD_MAP, get_set_fields

        try:
            config = rag_config.with_session(self._session_id)
        except Exception:  # noqa: BLE001 - the client's own values are all there is
            return {}
        in_force = {}
        for field_name in get_set_fields(self._config):
            key = RUNTIME_FIELD_MAP.get(field_name)
            if key is None or field_name not in data:
                continue
            try:
                in_force[field_name] = getattr(config, key)
            except Exception:  # noqa: BLE001 - a key the config does not resolve
                continue
        return in_force

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
                    self._document_manager.wait_for_processing(
                        d, timeout=processing_timeout, raise_on_failure=False
                    )
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
