"""Remote RAG backend — httpx-based client for remote API access.

Enables ``KnowledgeClient.from_url("https://rag.myapp.com")`` without
any heavy dependencies (no torch, no sentence-transformers).

This backend implements the same **synchronous** interface that
KnowledgeClient's SDK managers expect (``upload_document``, ``search``,
``get_documents``, etc.), delegating every call to the REST API via
httpx.  Async variants are available as ``_async_*`` methods.

Example:
    ```python
    from lasagna.sdk import KnowledgeClient

    # Connect to a remote RAG server (lightweight — only httpx required)
    client = KnowledgeClient.from_url("https://rag.myapp.com", api_key="sk-...")
    await client.upload("./report.pdf")
    context = await client.get_context("quarterly revenue")
    ```
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from lasagna.utils import degraded as _degraded
from lasagna.core.provenance import redact_url

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class RemoteDocument:
    """Minimal document representation from REST API response."""

    document_id: str
    session_id: str
    filename: str
    file_type: str = ""
    file_size: int = 0
    status: str = "ready"
    chunks_count: int = 0
    upload_date: str = ""
    error_message: Optional[str] = None
    chunk_count: int = 0  # Alias for SDK compatibility

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "RemoteDocument":
        doc = cls(
            document_id=data.get("document_id", ""),
            session_id=data.get("session_id", ""),
            filename=data.get("filename", ""),
            file_type=data.get("file_type", ""),
            file_size=data.get("file_size", 0),
            status=data.get("status", "ready"),
            chunks_count=data.get("chunks_count", 0),
            upload_date=data.get("upload_date", ""),
            error_message=data.get("error_message"),
        )
        doc.chunk_count = doc.chunks_count
        return doc


@dataclass
class RemoteUploadResult:
    """Upload result from remote API."""

    document_id: str
    filename: str
    file_type: str = ""
    file_size: int = 0
    status: str = "ready"
    chunk_count: int = 0

    @property
    def chunks_count(self) -> int:
        """Alias for SDK manager compatibility (expects chunks_count)."""
        return self.chunk_count


@dataclass
class RemoteChunk:
    """Chunk representation for search results."""

    chunk_id: str = ""
    content: str = ""
    document_id: str = ""
    chunk_index: int = 0
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemoteDocumentRef:
    """Minimal document reference attached to search results.

    Provides the same attributes that ``SearchManager`` reads from
    ``raw_result.document`` so the code path works for both local
    (full DocumentMetadata) and remote (this lightweight ref) backends.
    """

    document_id: str = ""
    filename: str = ""
    session_id: str = ""
    file_type: str = ""
    file_size: int = 0
    chunks_count: int = 0
    upload_date: str = ""


@dataclass
class RemoteSearchResult:
    """Search result from remote API."""

    chunk: RemoteChunk
    score: float = 0.0
    document_id: str = ""
    filename: str = ""
    document: Optional[RemoteDocumentRef] = None


if HTTPX_AVAILABLE:

    class _SessionAuth(httpx.Auth):
        """Send the session token, minting one with the API key when the server asks.

        With ``RAG_REQUIRE_AUTH`` on, the server takes no ``X-Session-ID`` as
        identity and answers 401 until a request carries an ``X-Session-Token``
        from ``POST /api/v1/sessions``, which needs an ``X-API-Key`` from
        ``RAG_API_KEYS``. The client never asked for one, so it got 401 on every
        call. On a 401 this mints a token once (for the named session, if one
        was given, which the server allows only to an admin), then sends the
        request again. A refused mint is what the caller sees, with the
        server's reason.
        """

        requires_request_body = True  # the request is sent again after minting
        requires_response_body = True  # the minted token is read from the reply

        def __init__(
            self,
            sessions_url: str,
            api_key: Optional[str],
            token: Optional[str],
            requested_session: Optional[str],
            on_session: Any,
        ) -> None:
            self._sessions_url = sessions_url
            self._api_key = api_key
            self.token = token
            self._requested_session = requested_session
            self._on_session = on_session

        def auth_flow(self, request):
            sent = self.token
            if sent:
                request.headers["X-Session-Token"] = sent
            response = yield request
            if response.status_code != 401 or not self._api_key:
                return
            if self.token and self.token != sent:
                # Another request minted one meanwhile.
                request.headers["X-Session-Token"] = self.token
                yield request
                return
            body = {"session_id": self._requested_session} if self._requested_session else {}
            minted = yield httpx.Request(
                "POST", self._sessions_url, headers={"X-API-Key": self._api_key}, json=body
            )
            if minted.status_code != 201:
                return
            data = minted.json()
            self.token = data["token"]
            self._on_session(data["session_id"])
            request.headers["X-Session-Token"] = self.token
            yield request


class RemoteRAGBackend:
    """httpx-based backend for remote RAG API access.

    Provides **both** sync and async interfaces.  The SDK managers
    (``DocumentManager``, ``SearchManager``) call the sync methods
    (``upload_document``, ``search``, ``get_documents``, etc.) which
    internally run the async httpx calls in a helper thread so they
    work correctly even when called from within a running event loop.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        session_id: str = "default",
        timeout: float = 60.0,
        session_token: Optional[str] = None,
        requested_session: Optional[str] = None,
        on_session: Any = None,
    ):
        """``api_key`` is a key from the server's ``RAG_API_KEYS``: with
        ``RAG_REQUIRE_AUTH`` on it mints a session token on the first 401.
        ``session_token`` is one already minted. ``requested_session`` is the
        session the caller named, if any; ``on_session(session_id)`` is told
        which session a minted token belongs to."""
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for remote backend. Install with: pip install httpx"
            )

        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._timeout = timeout
        self._on_session = on_session

        headers = {"X-Session-ID": session_id}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._auth = _SessionAuth(
            f"{self._base_url}{self._api('/sessions')}",
            api_key,
            session_token,
            requested_session,
            self._adopt_session,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            auth=self._auth,
        )
        self._sync_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        logger.debug(f"RemoteRAGBackend connected to {redact_url(self._base_url)}")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _api(self, path: str) -> str:
        """Build API path."""
        return f"/api/v1{path}"

    @property
    def session_id(self) -> str:
        """The session requests go to: a minted token's, once there is one."""
        return self._session_id

    def _adopt_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._client.headers["X-Session-ID"] = session_id
        if self._on_session is not None:
            self._on_session(session_id)

    @staticmethod
    def _parse_json(resp) -> Dict[str, Any]:
        """Parse JSON from an httpx response, raising a clear error on failure."""
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid JSON response from RAG API "
                f"(status={resp.status_code}, url={resp.url}): {exc}"
            ) from exc

    def _run_sync(self, coro):
        """Run an async coroutine synchronously.

        Safe to call from within a running event loop (e.g. from an
        ``async def`` SDK manager method) — spins up a private loop
        in a cached worker thread when necessary.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            if self._sync_pool is None:
                with self._pool_lock:
                    if self._sync_pool is None:
                        self._sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            return self._sync_pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    # =========================================================================
    # Async implementation (internal)
    # =========================================================================

    async def _async_upload_document(
        self,
        file_path: str,
        session_id: Optional[str] = None,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> RemoteUploadResult:
        """Upload a document via REST API (async).

        Raises:
            FileNotFoundError: If file_path does not exist.
            OSError: If file cannot be read (permissions, etc.).
            httpx.HTTPStatusError: If server returns an error response.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        fname = filename or path.name

        try:
            with open(path, "rb") as f:
                files = {"file": (fname, f)}
                data = {}
                if document_id is not None:
                    data["document_id"] = document_id
                resp = await self._client.post(
                    self._api("/documents"),
                    files=files,
                    data=data if data else None,
                )
        except OSError as exc:
            raise OSError(f"Cannot read file '{file_path}': {exc}") from exc

        resp.raise_for_status()
        data = self._parse_json(resp)

        return RemoteUploadResult(
            document_id=data.get("document_id", ""),
            filename=data.get("filename", fname),
            file_type=data.get("file_type", ""),
            file_size=data.get("file_size", 0),
            status=data.get("status", "ready"),
            chunk_count=data.get("chunks_count", 0),
        )

    async def _async_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        routing_tier: Optional[str] = None,
    ) -> List[RemoteSearchResult]:
        """Search via REST API (async)."""
        body: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "use_reranking": use_reranking,
        }
        if min_score is not None:
            body["min_score"] = min_score
        if document_ids:
            body["document_ids"] = document_ids
        if routing_tier is not None:
            body["routing_tier"] = routing_tier

        resp = await self._client.post(self._api("/search"), json=body)
        resp.raise_for_status()
        data = self._parse_json(resp)
        # The server's fallbacks, so SearchResults reports them as a local search does.
        _degraded.report(data.get("degraded_stages") or [])

        return self._parse_search_results(data)

    async def _async_search_with_citations(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ):
        """Search with citations via REST API (async)."""
        body: Dict[str, Any] = {"query": query, "top_k": top_k}
        if document_ids:
            body["document_ids"] = document_ids

        resp = await self._client.post(self._api("/search/citations"), json=body)
        resp.raise_for_status()
        data = self._parse_json(resp)
        _degraded.report(data.get("degraded_stages") or [])

        results = self._parse_search_results(data)
        context = data.get("context", "")
        citations = data.get("citations", [])

        return results, citations, context

    async def _async_get_documents(
        self,
        session_id: Optional[str] = None,
    ) -> List[RemoteDocument]:
        """List documents via REST API (async)."""
        resp = await self._client.get(self._api("/documents"))
        resp.raise_for_status()
        data = self._parse_json(resp)

        return [RemoteDocument.from_api(d) for d in data.get("documents", [])]

    async def _async_delete_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Delete a document via REST API (async)."""
        resp = await self._client.delete(self._api(f"/documents/{document_id}"))
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    # =========================================================================
    # Sync public interface (matches RAGBackend signatures for SDK managers)
    # =========================================================================

    def upload_document(
        self,
        file_path: str,
        session_id: Optional[str] = None,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> RemoteUploadResult:
        """Upload a document via REST API (sync).

        This is the sync interface called by ``DocumentManager``.
        """
        return self._run_sync(
            self._async_upload_document(file_path, session_id, filename, document_id)
        )

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        **kwargs,
    ) -> List[RemoteSearchResult]:
        """Search via REST API (sync).

        This is the sync interface called by ``SearchManager``.
        ``routing_tier`` is forwarded to the REST API.  Other extra
        kwargs (``use_hybrid``, ``hybrid_alpha``, ``trace_id``) are
        accepted for interface compatibility but not forwarded.
        """
        routing_tier = kwargs.pop("routing_tier", None)
        # Warn about genuinely unknown kwargs
        known_extra = {"use_hybrid", "hybrid_alpha", "trace_id"}
        unknown = set(kwargs) - known_extra
        if unknown:
            logger.warning(f"RemoteRAGBackend.search: unsupported kwargs ignored: {unknown}")
        if kwargs.get("use_hybrid") is not None or kwargs.get("hybrid_alpha") is not None:
            logger.debug("use_hybrid/hybrid_alpha not supported via remote API, ignored")

        return self._run_sync(
            self._async_search(
                query,
                session_id,
                top_k or 5,
                min_score,
                document_ids,
                use_hyde,
                use_reranking,
                routing_tier=routing_tier,
            )
        )

    def get_documents(self, session_id: Optional[str] = None) -> List[RemoteDocument]:
        """List documents via REST API (sync).

        This is the sync interface called by ``DocumentManager``.
        """
        return self._run_sync(self._async_get_documents(session_id))

    def get_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[RemoteDocument]:
        """Get a single document by ID (sync).

        Uses the dedicated status endpoint for O(1) lookup when available,
        falling back to the full document list otherwise.
        """
        return self._run_sync(self._async_get_document_status(document_id, session_id))

    def get_document_by_id(
        self, document_id: str, session_id: Optional[str] = None
    ) -> Optional[RemoteDocument]:
        """``get_document``, under the name the SDK's upload wait polls with.

        Without it ``client.documents.wait_for_processing`` failed on a remote
        client, so ``lasagna upload`` through a running server could not finish.
        """
        return self.get_document(document_id, session_id)

    async def _async_get_document_status(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[RemoteDocument]:
        """Get a single document status via REST ``GET /documents/{id}/status`` (async).

        O(1) primary-key lookup on the server side.
        """
        sid = session_id or self._session_id
        try:
            resp = await self._client.get(
                self._api(f"/documents/{document_id}/status"),
                params={"session_id": sid},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = self._parse_json(resp)
            return RemoteDocument(
                document_id=data.get("document_id", document_id),
                session_id=sid,
                filename=data.get("filename", ""),
                status=data.get("status", ""),
                chunks_count=data.get("chunks_count", 0),
                error_message=data.get("error_message"),
            )
        except Exception as exc:
            logger.debug(
                f"Status endpoint failed for {document_id}, falling back to document list: {exc}"
            )
            # Fallback to full list scan
            docs = await self._async_get_documents(session_id)
            for doc in docs:
                if doc.document_id == document_id:
                    return doc
            return None

    def delete_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Delete a document via REST API (sync).

        This is the sync interface called by ``DocumentManager``.
        """
        return self._run_sync(self._async_delete_document(document_id, session_id))

    def get_document_chunks(
        self,
        document_id: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> List[RemoteChunk]:
        """Get chunks for a document.

        Not yet supported via the remote REST API.  Returns an empty
        list rather than crashing so callers degrade gracefully.
        """
        logger.warning("get_document_chunks is not yet supported via RemoteRAGBackend")
        return []

    def clear_session_state(self, session_id: str) -> Dict[str, Any]:
        """Clear session state.

        Not yet supported via the remote REST API.
        """
        logger.warning("clear_session_state is not yet supported via RemoteRAGBackend")
        return {"cache_invalidated": False, "session_cleared": False}

    # =========================================================================
    # Async public interface (for direct async usage)
    # =========================================================================

    async def async_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
    ) -> List[RemoteSearchResult]:
        """Search via REST API (async public interface)."""
        return await self._async_search(
            query,
            session_id,
            top_k,
            min_score,
            document_ids,
            use_hyde,
            use_reranking,
        )

    async def async_search_with_citations(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ):
        """Search with citations via REST API (async public interface)."""
        return await self._async_search_with_citations(
            query,
            session_id,
            top_k,
            document_ids,
        )

    # =========================================================================
    # Document Lifecycle (soft-delete, restore, bulk-delete)
    # =========================================================================

    async def _async_soft_delete_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Soft-delete (archive) a document via REST API (async)."""
        sid = session_id or self._session_id
        resp = await self._client.post(
            self._api(f"/documents/{document_id}/archive"),
            params={"session_id": sid},
        )
        if resp.status_code == 404:
            raise ValueError(f"Document not found: {document_id}")
        resp.raise_for_status()
        return self._parse_json(resp)

    def soft_delete_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Soft-delete (archive) a document via REST API (sync).

        The document enters the GC grace window.  Use ``restore_document()``
        to recover it before the grace window expires.
        """
        return self._run_sync(self._async_soft_delete_document(document_id, session_id))

    async def _async_restore_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore a soft-deleted document via REST API (async)."""
        sid = session_id or self._session_id
        resp = await self._client.post(
            self._api(f"/documents/{document_id}/restore"),
            params={"session_id": sid},
        )
        if resp.status_code == 404:
            raise ValueError(f"Document not found: {document_id}")
        if resp.status_code == 409:
            raise Exception(self._parse_json(resp).get("detail", "Grace window expired"))
        resp.raise_for_status()
        return self._parse_json(resp)

    def restore_document(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore a soft-deleted document within the grace window (sync)."""
        return self._run_sync(self._async_restore_document(document_id, session_id))

    async def _async_bulk_delete_session(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bulk-delete all documents in a session via REST API (async)."""
        sid = session_id or self._session_id
        admin_token = os.environ.get("RAG_ADMIN_TOKEN", "")
        resp = await self._client.delete(
            self._api(f"/admin/sessions/{sid}"),
            headers={"X-Admin-Token": admin_token},
        )
        resp.raise_for_status()
        return self._parse_json(resp)

    def bulk_delete_session(
        self,
        session_id: Optional[str] = None,
        *,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Delete all documents in a session via REST API (sync).

        Only available when ``RAG_ADMIN_TOKEN`` is configured on the server.
        """
        if not confirm:
            raise ValueError("bulk_delete_session requires confirm=True")
        return self._run_sync(self._async_bulk_delete_session(session_id))

    # =========================================================================
    # Lifecycle policy (admin)
    # =========================================================================

    async def _async_get_lifecycle_policy(self) -> Dict[str, Any]:
        """Fetch the current lifecycle garbage-collection policy (async).

        Requires ``RAG_ADMIN_TOKEN`` to be configured on the server.
        """
        admin_token = os.environ.get("RAG_ADMIN_TOKEN", "")
        resp = await self._client.get(
            self._api("/admin/gc/policy"),
            headers={"X-Admin-Token": admin_token},
        )
        resp.raise_for_status()
        return self._parse_json(resp)

    def get_lifecycle_policy(self) -> Dict[str, Any]:
        """Fetch the current lifecycle garbage-collection policy (sync).

        Requires ``RAG_ADMIN_TOKEN`` to be set in environment.

        Returns:
            Dict with all policy fields (soft_delete_grace_hours, gc_cycle_hours, etc.)
        """
        return self._run_sync(self._async_get_lifecycle_policy())

    async def _async_set_lifecycle_policy(
        self,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update lifecycle policy fields at runtime via REST API (async).

        Changes take effect immediately but are not persisted across restarts.
        """
        admin_token = os.environ.get("RAG_ADMIN_TOKEN", "")
        resp = await self._client.patch(
            self._api("/admin/gc/policy"),
            json=updates,
            headers={"X-Admin-Token": admin_token},
        )
        resp.raise_for_status()
        return self._parse_json(resp)

    def set_lifecycle_policy(
        self,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update lifecycle policy fields at runtime (sync).

        Requires ``RAG_ADMIN_TOKEN`` to be set in environment.
        Changes are not persisted — the server reverts on restart.

        Args:
            updates: Dict of policy field names to new values.
                Allowed fields: soft_delete_grace_hours, document_ttl_days,
                error_retention_days, session_idle_days, gc_cycle_hours,
                orphan_scan_hours, safety_threshold_percent, batch_size,
                reconcile_quota_on_startup.

        Returns:
            Dict with the full updated policy.
        """
        return self._run_sync(self._async_set_lifecycle_policy(updates))

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the httpx client and thread pool."""
        await self._client.aclose()
        with self._pool_lock:
            if self._sync_pool is not None:
                self._sync_pool.shutdown(wait=False)
                self._sync_pool = None

    async def __aenter__(self) -> "RemoteRAGBackend":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def __del__(self) -> None:
        """Safety net: warn if not closed explicitly."""
        if hasattr(self, "_client") and self._client and not self._client.is_closed:
            import warnings

            warnings.warn(
                "RemoteRAGBackend was not closed. Use 'async with' or call close() explicitly.",
                ResourceWarning,
                stacklevel=2,
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _parse_search_results(data: Dict[str, Any]) -> List[RemoteSearchResult]:
        """Parse search results from API response JSON."""
        results = []
        for r in data.get("results", []):
            chunk_data = r.get("chunk", {})
            chunk = RemoteChunk(
                chunk_id=chunk_data.get("chunk_id", ""),
                content=chunk_data.get("content", ""),
                document_id=chunk_data.get("document_id", ""),
                chunk_index=chunk_data.get("chunk_index", 0),
                page_number=chunk_data.get("page_number"),
                metadata=chunk_data.get("metadata", {}),
            )
            doc_id = r.get("document_id", "")
            fname = r.get("filename", "")
            doc_ref = RemoteDocumentRef(
                document_id=doc_id,
                filename=fname,
            )
            results.append(
                RemoteSearchResult(
                    chunk=chunk,
                    score=r.get("score", 0.0),
                    document_id=doc_id,
                    filename=fname,
                    document=doc_ref,
                )
            )
        return results
