"""MCP Knowledge Client — SSE transport layer for the RAG Knowledge MCP Server.

Provides persistent connection management, automatic reconnection with
exponential backoff, periodic health checks, and optional query caching.

This is the client counterpart to the MCPKnowledgeServer in mcp.py.

Example:
    ```python
    from lasagna.sdk import MCPKnowledgeClient

    async with MCPKnowledgeClient() as client:
        result = await client.search("What is the revenue?", "session123")
        print(result)
    ```
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List

from loguru import logger


class MCPKnowledgeClient:
    """Client for communicating with the RAG Knowledge MCP Server.

    This class provides the SDK's interface to the MCP knowledge server,
    ensuring all RAG operations go through MCP rather than direct backend calls.

    The client uses FastMCP's SSE transport with persistent connection management,
    automatic reconnection, and optional query caching.

    Example:
        ```python
        client = MCPKnowledgeClient(url="http://localhost:8005/sse")
        await client.connect()

        # Search
        result = await client.search("What is the revenue?", "session123")

        # Upload
        result = await client.upload_document("/path/to/file.pdf", "session123")

        # Cleanup when done
        await client.disconnect()
        ```
    """

    DEFAULT_URL = "http://localhost:8005/sse"
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY_SECONDS = 1.0
    HEALTH_CHECK_INTERVAL = 60.0  # seconds
    QUERY_CACHE_TTL = 300.0  # 5 minutes
    QUERY_CACHE_MAX_SIZE = 100

    def __init__(
        self,
        url: Optional[str] = None,
        enable_query_cache: bool = False,
        cache_ttl: float = QUERY_CACHE_TTL,
    ):
        """Initialize the MCP Knowledge client.

        Args:
            url: MCP server SSE endpoint URL. Defaults to localhost:8005/sse
            enable_query_cache: Enable LRU caching for search queries
            cache_ttl: Cache entry TTL in seconds (default 300s)
        """
        self._url = url or self.DEFAULT_URL
        self._transport = None
        self._client = None
        self._client_context = None  # Persistent context manager
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._last_health_check = 0.0
        self._consecutive_failures = 0

        # Query cache configuration
        self._enable_query_cache = enable_query_cache
        self._cache_ttl = cache_ttl
        self._query_cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._cache_lock = asyncio.Lock()

    @property
    def url(self) -> str:
        """Get the configured MCP server URL."""
        return self._url

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to MCP server."""
        return self._connected and self._client_context is not None

    async def connect(self) -> None:
        """Establish persistent connection to MCP server.

        Uses FastMCP's SSE transport for communication.
        Connection is thread-safe and idempotent.
        """
        async with self._connection_lock:
            if self._connected and self._client_context is not None:
                return

            try:
                from fastmcp import Client as FastMCPClient
                from fastmcp.client.transports import SSETransport

                self._transport = SSETransport(url=self._url)
                self._client = FastMCPClient(self._transport)

                # Enter the client context manager and keep it open
                self._client_context = await self._client.__aenter__()
                self._connected = True
                self._consecutive_failures = 0
                self._last_health_check = time.monotonic()
                logger.info(f"MCPKnowledgeClient connected to {self._url}")
            except ImportError as e:
                raise ImportError(
                    "fastmcp package is required for MCP client. Install with: pip install fastmcp"
                ) from e
            except Exception as e:
                self._connected = False
                self._client_context = None
                raise ConnectionError(f"Failed to connect to MCP server at {self._url}: {e}") from e

    async def __aenter__(self) -> "MCPKnowledgeClient":
        """Async context manager entry - connect to MCP server.

        Example:
            ```python
            async with MCPKnowledgeClient() as client:
                result = await client.search("query", "session123")
            # Automatically disconnected on exit
            ```
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - disconnect from MCP server."""
        await self.disconnect()

    async def disconnect(self) -> None:
        """Disconnect from MCP server and cleanup resources."""
        async with self._connection_lock:
            if self._client_context is not None:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Error during disconnect: {e}")
            self._client_context = None
            self._client = None
            self._transport = None
            self._connected = False
            logger.info("MCPKnowledgeClient disconnected")

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to MCP server with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise
        """
        async with self._connection_lock:
            # Cleanup existing connection state
            if self._client_context is not None:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:
                    pass
            self._client_context = None
            self._connected = False

        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            delay = self.RECONNECT_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.info(
                f"MCP reconnection attempt {attempt}/{self.MAX_RECONNECT_ATTEMPTS} in {delay}s..."
            )
            await asyncio.sleep(delay)

            try:
                await self.connect()
                logger.info("MCP reconnection successful")
                return True
            except Exception as e:
                logger.warning(f"MCP reconnection attempt {attempt} failed: {e}")

        logger.error(f"MCP reconnection failed after {self.MAX_RECONNECT_ATTEMPTS} attempts")
        return False

    async def _ensure_connected(self) -> None:
        """Ensure client is connected, reconnecting if necessary."""
        if not self._connected or self._client_context is None:
            await self.connect()

    def _get_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate cache key for a tool call."""
        # Only cache read operations (not writes like upload, delete, set_focused_documents)
        cacheable_tools = (
            "search",
            "search_knowledge",
            "search_with_context",
            "list_documents",
            "get_focused_documents",
            "list_profiles",
            "get_active_profile",
            "get_profile_stats",
        )
        if tool_name not in cacheable_tools:
            return ""
        return f"{tool_name}:{json.dumps(args, sort_keys=True)}"

    async def _get_cached(self, cache_key: str) -> Optional[str]:
        """Get cached result if valid."""
        if not self._enable_query_cache or not cache_key:
            return None

        async with self._cache_lock:
            if cache_key in self._query_cache:
                result, timestamp = self._query_cache[cache_key]
                if time.monotonic() - timestamp < self._cache_ttl:
                    logger.debug(f"MCP cache hit for: {cache_key[:50]}...")
                    return result
                else:
                    del self._query_cache[cache_key]
        return None

    async def _set_cached(self, cache_key: str, result: str) -> None:
        """Store result in cache with LRU eviction."""
        if not self._enable_query_cache or not cache_key:
            return

        async with self._cache_lock:
            # LRU eviction if cache is full
            if len(self._query_cache) >= self.QUERY_CACHE_MAX_SIZE:
                oldest_key = min(self._query_cache, key=lambda k: self._query_cache[k][1])
                del self._query_cache[oldest_key]

            self._query_cache[cache_key] = (result, time.monotonic())

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        logger.debug("MCP query cache cleared")

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result.

        Features:
        - Persistent connection (no per-call connection overhead)
        - Automatic reconnection on failure
        - Periodic health checks to detect stale connections
        - Optional query caching for read operations

        Args:
            tool_name: Name of the MCP tool to call
            args: Arguments to pass to the tool

        Returns:
            Tool result as a string

        Raises:
            ConnectionError: If cannot connect to MCP server
            RuntimeError: If tool call fails after retries
        """
        # Check cache first
        cache_key = self._get_cache_key(tool_name, args)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        await self._ensure_connected()

        # Periodic health check - proactively detect stale connections
        # Skip health check for the health check tool itself to avoid recursion
        if tool_name != "get_system_status":
            try:
                current_time = time.monotonic()
                if current_time - self._last_health_check >= self.HEALTH_CHECK_INTERVAL:
                    # Perform lightweight health check
                    await self._client.call_tool("get_system_status", {})
                    self._last_health_check = current_time
                    self._consecutive_failures = 0
                    logger.debug("MCP health check passed")
            except Exception as health_error:
                logger.warning(
                    f"MCP health check failed: {health_error}. Attempting reconnection..."
                )
                if not await self._reconnect():
                    raise ConnectionError(
                        f"MCP server unhealthy and reconnection failed: {health_error}"
                    )

        try:
            result = await self._client.call_tool(tool_name, args)
            # Extract text content from result
            if result.content:
                text_result = (
                    result.content[0].text
                    if hasattr(result.content[0], "text")
                    else str(result.content[0])
                )
            else:
                text_result = ""

            # Reset failure counter on success
            self._consecutive_failures = 0

            # Cache successful read results
            await self._set_cached(cache_key, text_result)

            return text_result

        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                f"MCP tool call '{tool_name}' failed (attempt {self._consecutive_failures}): {e}"
            )

            # Try reconnection if we've had consecutive failures
            if self._consecutive_failures >= 2:
                if await self._reconnect():
                    # Retry the call after successful reconnection
                    try:
                        result = await self._client.call_tool(tool_name, args)
                        if result.content:
                            return (
                                result.content[0].text
                                if hasattr(result.content[0], "text")
                                else str(result.content[0])
                            )
                        return ""
                    except Exception as retry_error:
                        logger.error(
                            f"MCP tool call '{tool_name}' failed after reconnection: {retry_error}"
                        )
                        raise RuntimeError(f"MCP tool call failed: {retry_error}") from retry_error

            raise RuntimeError(f"MCP tool call failed: {e}") from e

    # =========================================================================
    # Knowledge Base Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        allow_broaden: bool = False,
        trace_id: Optional[str] = None,
    ) -> str:
        """Search the knowledge base using SOTA semantic retrieval.

        Retrieval behaviour (hybrid search, reranking, HyDE, routing) is
        driven by the session configuration.

        Args:
            query: Natural-language search query
            session_id: Session identifier
            top_k: Optional result count override (default: session config)
            document_ids: Optional list of document IDs to filter
            allow_broaden: If True, search outside focused documents
            trace_id: Optional trace ID for observability

        Returns:
            Formatted context with citations and ``<search_metadata>`` block
        """
        args: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
        }
        if top_k is not None:
            args["top_k"] = top_k
        if document_ids:
            args["document_ids"] = ",".join(document_ids)
        if allow_broaden:
            args["allow_broaden"] = True
        if trace_id:
            args["trace_id"] = trace_id

        return await self.call_tool("search", args)

    # Deprecated aliases — kept for one release cycle

    async def search_knowledge(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
        use_reranking: bool = True,
        use_hyde: bool = False,
    ) -> str:
        """Deprecated: use :meth:`search` instead."""
        import warnings

        warnings.warn(
            "search_knowledge() is deprecated; use search()", DeprecationWarning, stacklevel=2
        )
        return await self.search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_ids,
        )

    async def search_with_context(
        self,
        query: str,
        session_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> str:
        """Deprecated: use :meth:`search` instead."""
        import warnings

        warnings.warn(
            "search_with_context() is deprecated; use search()", DeprecationWarning, stacklevel=2
        )
        return await self.search(
            query=query,
            session_id=session_id,
            document_ids=document_ids,
        )

    async def upload_document(
        self,
        file_path: str,
        session_id: str,
        filename: Optional[str] = None,
        wait: bool = True,
    ) -> str:
        """Upload and index a document into the knowledge base.

        Args:
            file_path: Absolute path to the file on disk
            session_id: Session identifier
            filename: Optional display name
            wait: If True (default), block until processing completes.
                If False, return immediately with status "processing".

        Returns:
            JSON string with document_id, filename, status, chunks_count,
            file_size_kb.
        """
        args: dict = {
            "file_path": file_path,
            "session_id": session_id,
            "wait": wait,
        }
        if filename:
            args["filename"] = filename

        return await self.call_tool("upload_document", args)

    async def get_document_status(
        self,
        document_id: str,
        session_id: str,
    ) -> str:
        """Get the processing status of a document (lightweight O(1) lookup).

        Args:
            document_id: The document ID to query
            session_id: Session identifier

        Returns:
            JSON string with document_id, status, chunks_count, error_message.
        """
        return await self.call_tool(
            "get_document_status",
            {
                "document_id": document_id,
                "session_id": session_id,
            },
        )

    async def list_documents(self, session_id: str) -> str:
        """List all documents in the knowledge base for a session.

        Args:
            session_id: Session identifier

        Returns:
            Human-readable list of documents
        """
        return await self.call_tool("list_documents", {"session_id": session_id})

    async def delete_document(self, document_id: str, session_id: str) -> str:
        """Delete a document and its chunks from the knowledge base.

        Args:
            document_id: The document ID to delete
            session_id: Session identifier

        Returns:
            Confirmation message
        """
        return await self.call_tool(
            "delete_document",
            {
                "document_id": document_id,
                "session_id": session_id,
            },
        )

    async def get_document_info(self, document_id: str, session_id: str) -> str:
        """Get detailed information about a specific document.

        Args:
            document_id: The document ID to query
            session_id: Session identifier

        Returns:
            Document information in JSON format
        """
        return await self.call_tool(
            "get_document_info",
            {
                "document_id": document_id,
                "session_id": session_id,
            },
        )

    async def get_document_chunks(
        self,
        document_id: str,
        session_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> str:
        """Get chunks from a specific document.

        Args:
            document_id: The document ID to query
            session_id: Session identifier
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            List of chunks in JSON format
        """
        return await self.call_tool(
            "get_document_chunks",
            {
                "document_id": document_id,
                "session_id": session_id,
                "offset": offset,
                "limit": limit,
            },
        )

    async def get_system_status(self) -> str:
        """Get RAG system health and statistics.

        Returns:
            System status in JSON format
        """
        return await self.call_tool("get_system_status", {})

    async def clear_session(self, session_id: str, confirm: bool = False) -> str:
        """Delete all documents in a session.

        Args:
            session_id: Session identifier
            confirm: Must be True to confirm deletion

        Returns:
            Confirmation message
        """
        return await self.call_tool(
            "clear_session",
            {
                "session_id": session_id,
                "confirm": confirm,
            },
        )

    async def reindex_document(self, document_id: str, session_id: str) -> str:
        """Reindex an existing document.

        Args:
            document_id: The document ID to reindex
            session_id: Session identifier

        Returns:
            Reindex confirmation
        """
        return await self.call_tool(
            "reindex_document",
            {
                "document_id": document_id,
                "session_id": session_id,
            },
        )

    # =========================================================================
    # Profile Management (via MCP)
    # =========================================================================

    async def list_profiles(self, include_builtin: bool = True) -> str:
        """List available RAG configuration profiles.

        Args:
            include_builtin: Whether to include built-in profiles

        Returns:
            List of profiles in JSON format
        """
        return await self.call_tool(
            "list_profiles",
            {
                "include_builtin": include_builtin,
            },
        )

    async def get_active_profile(self, session_id: str) -> str:
        """Get the active RAG profile for a session.

        Args:
            session_id: Session identifier

        Returns:
            Active profile information
        """
        return await self.call_tool(
            "get_active_profile",
            {
                "session_id": session_id,
            },
        )

    async def set_active_profile(self, session_id: str, profile_name: str) -> str:
        """Set the active RAG profile for a session.

        Args:
            session_id: Session identifier
            profile_name: Name of the profile to activate (e.g., "default", "deep_search")

        Returns:
            Confirmation message
        """
        return await self.call_tool(
            "set_active_profile",
            {
                "session_id": session_id,
                "profile_name": profile_name,
            },
        )

    # =========================================================================
    # RAG Configuration (via MCP)
    # =========================================================================

    async def get_config(self, session_id: str) -> str:
        """Get the current RAG configuration for a session.

        Returns the active profile and all settings including chunking,
        embedding, reranking, and search parameters.

        Args:
            session_id: Session identifier

        Returns:
            JSON string with full configuration
        """
        return await self.call_tool(
            "get_rag_config",
            {
                "session_id": session_id,
            },
        )

    async def patch_config(self, session_id: str, config_patch: Dict[str, Any]) -> str:
        """Apply configuration overrides to the current session.

        Merges the provided settings with the active profile's config.
        These overrides persist for the session duration.

        Args:
            session_id: Session identifier
            config_patch: Dictionary with config keys to override

        Returns:
            JSON string with updated configuration
        """
        return await self.call_tool(
            "patch_rag_config",
            {
                "session_id": session_id,
                "config_patch": json.dumps(config_patch),
            },
        )

    async def save_profile(
        self,
        session_id: str,
        profile_name: str,
        config: Dict[str, Any],
        set_active: bool = True,
    ) -> str:
        """Save a custom RAG profile.

        Creates or updates a named profile with the provided configuration.
        Saved profiles persist across sessions.

        Args:
            session_id: Session identifier
            profile_name: Name for the profile (cannot be 'default')
            config: Profile configuration dictionary
            set_active: Whether to activate this profile (default True)

        Returns:
            Confirmation message
        """
        return await self.call_tool(
            "save_rag_profile",
            {
                "session_id": session_id,
                "profile_name": profile_name,
                "config": json.dumps(config),
                "set_active": set_active,
            },
        )

    async def delete_profile(self, session_id: str, profile_name: str) -> str:
        """Delete a custom RAG profile.

        Built-in profiles (default, minimal, deep_search) cannot be deleted.

        Args:
            session_id: Session identifier
            profile_name: Name of the profile to delete

        Returns:
            Confirmation message
        """
        return await self.call_tool(
            "delete_rag_profile",
            {
                "session_id": session_id,
                "profile_name": profile_name,
            },
        )

    async def reset_config(self, session_id: str, scope: str = "active") -> str:
        """Reset RAG configuration to defaults.

        Args:
            session_id: Session identifier
            scope: Reset scope - 'active' (clear overrides) or 'all' (full reset)

        Returns:
            JSON string with reset configuration
        """
        return await self.call_tool(
            "reset_rag_config",
            {
                "session_id": session_id,
                "scope": scope,
            },
        )

    # =========================================================================
    # Document Focus (via MCP)
    # =========================================================================

    async def set_focused_documents(
        self,
        session_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> str:
        """Set focused document IDs for a session via MCP.

        When focused documents are set, all RAG searches will be scoped to
        only these documents.

        Args:
            session_id: Session identifier
            document_ids: List of document IDs to focus on, or None/empty to clear

        Returns:
            Confirmation message
        """
        doc_ids_str = ",".join(document_ids) if document_ids else ""
        return await self.call_tool(
            "set_focused_documents",
            {
                "session_id": session_id,
                "document_ids": doc_ids_str,
            },
        )

    async def get_focused_documents(self, session_id: str) -> str:
        """Get the currently focused document IDs for a session.

        Args:
            session_id: Session identifier

        Returns:
            JSON object with focused document IDs
        """
        return await self.call_tool(
            "get_focused_documents",
            {
                "session_id": session_id,
            },
        )

    # ------------------------------------------------------------------
    # Document awareness
    # ------------------------------------------------------------------

    async def get_document_context(
        self,
        session_id: str,
        rag_search_count: int = 0,
    ) -> str:
        """Get document awareness context for agent prompts.

        Returns adaptive guidance about available documents based on how
        many searches have already been performed this turn.

        Args:
            session_id: Session identifier
            rag_search_count: Number of RAG searches done this turn

        Returns:
            Formatted context string, or empty string if no documents
        """
        return await self.call_tool(
            "get_document_context",
            {
                "session_id": session_id,
                "rag_search_count": rag_search_count,
            },
        )

    # ------------------------------------------------------------------
    # Metadata parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def parse_search_metadata(response: str) -> Dict[str, Any]:
        """Extract structured citation metadata from a search response.

        The ``search`` tool appends a ``<search_metadata>`` JSON block after
        the formatted context.  This helper parses it out so callers can
        access citations and chunks without a separate call.

        Args:
            response: Raw string returned by ``search``

        Returns:
            Dict with ``citations`` and ``chunks`` lists (empty dict on
            parse failure or missing metadata).
        """
        tag = "<search_metadata>"
        end_tag = "</search_metadata>"
        start = response.find(tag)
        if start == -1:
            return {}
        start += len(tag)
        end = response.find(end_tag, start)
        if end == -1:
            return {}
        try:
            return json.loads(response[start:end])
        except (json.JSONDecodeError, TypeError):
            return {}
