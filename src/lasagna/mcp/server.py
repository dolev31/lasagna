"""Lasagna MCP Server.

Exposes the RAG knowledge base as MCP tools so any agent can upload documents,
search them, list documents, and manage configuration.

Run it the way an MCP client launches it (stdio, the default)::

    lasagna-mcp

or as a network server::

    lasagna-mcp --transport sse --host 127.0.0.1 --port 8005

Tool sets (``--tools`` or ``LASAGNA_MCP_TOOLS``):

- ``safe`` (default): every tool except the six that delete data or reset
  configuration (see ``DESTRUCTIVE_TOOLS``).
- ``all``: every tool.

Endpoints (SSE mode):
- /        : Welcome page with server info and links
- /docs    : Interactive API documentation (test tools directly)
- /sse     : MCP SSE transport endpoint (for MCP clients)
- /health  : Health check endpoint

The tool catalogue is in ``README.md`` next to this file.
"""

import os
import sys
import functools
import inspect
import re
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from loguru import logger

from lasagna.api.rest.auth_mode import (
    UnsafeBindError,
    require_auth_or_refuse_to_start,
    warn_once_if_unauthenticated,
)
from lasagna.core._text import LOG_LEVELS
from lasagna.core.local_server import StoreHeld as _StoreHeld
from lasagna.core.model_downloads import ModelNotDownloaded, fix_for_session
from lasagna.mcp import transport_guard
from lasagna.utils.monitoring import hash_text


def get_rag_backend():
    """The shared backend, imported on first use so `lasagna-mcp --help` stays light."""
    from lasagna.api.rag_backend import get_rag_backend as _get_rag_backend

    return _get_rag_backend()


# Configure comprehensive logging for the MCP server
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>lasagna.mcp</cyan> | "
    "<level>{message}</level>"
)

# When True, this module owns logging config (standalone MCP server).
# When False (imported by another app), skip logger.remove() to avoid
# destroying the host app's handlers.  main() sets this to True.
_owns_logging = False

#: The level main() chose: --log-level, else LASAGNA_LOG_LEVEL or RAG_LOG_LEVEL,
#: else WARNING on stdio and INFO on SSE. A client reads a stdio server's stderr
#: as its log, and one search wrote 144 DEBUG lines there in ANSI color.
_pinned_level: Optional[str] = None


def _configure_rag_logging(level: str) -> None:
    if not _owns_logging:
        return
    level = _pinned_level or level
    from lasagna.api.rest.limits import apply_log_format

    # LASAGNA_LOG_FORMAT=json: one JSON object per line, as the REST API does.
    if apply_log_format(level):
        return
    logger.remove()
    # Color only on a terminal: a client launching the server reads a pipe.
    # diagnose=False: a failed search's traceback printed every frame's local
    # variables, the query text among them six times, into the client's log.
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=level,
        colorize=sys.stderr.isatty(),
        backtrace=False,
        diagnose=False,
    )


# No-op at import time (_owns_logging is False).
# main() enables logging ownership before calling _configure_rag_logging().
_configure_rag_logging("DEBUG")

# Server configuration - can be overridden via command line args
_server_host = "127.0.0.1"
_server_port = 8005

# Create the MCP server instance (host/port set at runtime before run()).
# FastMCP configures the stdlib root logger in its constructor; an application
# that merely imports this module must not have its logging rewired, so the
# root logger is put back as it was.
_root_logger = logging.getLogger()
_root_state = (list(_root_logger.handlers), _root_logger.level)
mcp = FastMCP("lasagna", host=_server_host, port=_server_port)
_root_logger.handlers[:] = _root_state[0]
_root_logger.setLevel(_root_state[1])

#: Tools that delete documents, profiles or collections, or throw away a
#: session's configuration. The default ``safe`` set does not register them:
#: an agent handed every tool will, sooner or later, call one of these to
#: "clean up", and nothing it can do with the rest is irreversible.
DESTRUCTIVE_TOOLS = frozenset(
    {
        "clear_session",
        "bulk_delete_session",
        "rebuild_collection",
        "delete_document",
        "delete_rag_profile",
        "reset_rag_config",
        # Server-wide, not the caller's session: a grace period of 0 or a
        # document TTL makes the next GC delete other sessions' documents.
        "set_lifecycle_policy",
    }
)

#: Accepted values of ``--tools`` / ``LASAGNA_MCP_TOOLS``.
TOOL_SETS = ("safe", "all")
TOOLS_ENV = "LASAGNA_MCP_TOOLS"


def resolve_tool_set(value: Optional[str] = None) -> str:
    """The tool set to serve: ``value``, else ``LASAGNA_MCP_TOOLS``, else ``safe``."""
    raw = os.environ.get(TOOLS_ENV, "") if value is None else value
    name = (raw or "").strip().lower() or "safe"
    if name not in TOOL_SETS:
        raise ValueError(
            f"Unknown MCP tool set {raw!r}. Use --tools safe (the default, no tools "
            f"that delete data) or --tools all, or set {TOOLS_ENV} to one of them."
        )
    return name


class _LasagnaMCP(FastMCP):
    """FastMCP that reports Lasagna's version as ``serverInfo.version``.

    The MCP SDK reported its own ("1.30.0").
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from lasagna import __version__

        self._mcp_server.version = __version__

    def listed_tool_count(self) -> int:
        """How many tools ``tools/list`` returns."""
        return len(self._tool_manager._tools)

    def sse_app(self, *args, **kwargs):
        """FastMCP's SSE app, refusing a foreign ``Host`` with the setting that admits it.

        FastMCP's own refusal says only "Invalid Host header", which leaves an
        operator who exposed the server on purpose guessing. The same list is
        checked first, on every path but ``/health``, which answers any Host
        for probes, as on REST.
        """
        app = super().sse_app(*args, **kwargs)
        security = self.settings.transport_security
        if security is None or not security.enable_dns_rebinding_protection:
            return app
        from starlette.datastructures import Headers

        from lasagna.api.rest.auth_mode import foreign_host_message, host_is_allowed

        patterns = list(security.allowed_hosts)

        async def checked(scope, receive, send):
            path, root = scope.get("path", ""), scope.get("root_path", "")
            if root and path.startswith(root):
                path = path[len(root) :]
            if scope["type"] == "http" and path != "/health":
                host = Headers(scope=scope).get("host")
                if not host_is_allowed(host, patterns):
                    refusal = JSONResponse({"error": foreign_host_message(host)}, status_code=421)
                    await refusal(scope, receive, send)
                    return
            await app(scope, receive, send)

        return checked


#: Keys a tool's JSON error reply may carry; any other key makes it a report.
_ERROR_ONLY_KEYS = frozenset({"error", "hint", "detail"})


def _tool_failure(reply: Any) -> Optional[str]:
    """The message when a tool's reply follows this module's error convention."""
    if not isinstance(reply, str):
        return None
    if reply.startswith(("Search failed:", "Failed to ")):
        return reply
    if reply.lstrip().startswith("{"):
        try:
            data = json.loads(reply)
        except ValueError:
            return None
        if isinstance(data, dict) and data.get("error") and set(data) <= _ERROR_ONLY_KEYS:
            return reply
    return None


def _flag_failures(fn):
    """Register ``fn`` so a failure reaches the client as an MCP tool error.

    Tools return failures as text ("Search failed: ...") or ``{"error": ...}``,
    which went out with isError=False: an agent read "EMBEDDING DIMENSION
    MISMATCH" or "No LLMProvider configured" as retrieved knowledge. Only the
    registered copy raises; the module function keeps returning the text, so
    Python callers see what they always did.

    A synchronous tool is registered as a coroutine that runs it in a worker
    thread. FastMCP calls a sync tool on the event loop itself, so one 40 s
    upload froze every other client, and in the embedded deployment the REST
    API's own health check with it. REST routes already run blocking work in a
    thread pool.
    """
    from mcp.server.fastmcp.exceptions import ToolError

    def checked(reply):
        message = _tool_failure(reply)
        if message is not None:
            raise ToolError(message)
        return reply

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def flagged(*args, **kwargs):
            return checked(await fn(*args, **kwargs))

    else:
        import anyio.to_thread

        @functools.wraps(fn)
        async def flagged(*args, **kwargs):
            call = functools.partial(fn, *args, **kwargs)
            return checked(await anyio.to_thread.run_sync(call))

    return flagged


_register_tool = mcp.tool


def _tool(*args, **kwargs):
    register = _register_tool(*args, **kwargs)

    def decorator(fn):
        register(_flag_failures(fn))
        return fn

    return decorator


mcp.tool = _tool

# Shared backend factory — both MCP and REST API use the same RAGBackend
# singleton via get_rag_backend(). Mutable state uses thread-safe locking.
import threading as _threading  # noqa: E402

_eager_init = True  # Default to eager initialization
_last_log_level = None
_rebuild_lock = _threading.Lock()


def _get_rebuild_signature() -> str | None:
    """Read the last rebuild signature from shared state (Redis or in-memory)."""
    try:
        from lasagna.infrastructure.session_store import get_session_store

        store = get_session_store()
        state = store.load("__mcp_rebuild_signature")
        if state:
            return state.get("signature")
    except Exception:
        pass
    return _local_rebuild_signature


def _set_rebuild_signature(sig: str) -> None:
    """Write the rebuild signature to shared state (Redis or in-memory)."""
    global _local_rebuild_signature
    _local_rebuild_signature = sig
    try:
        from lasagna.infrastructure.session_store import get_session_store

        store = get_session_store()
        store.save("__mcp_rebuild_signature", {"signature": sig})
    except Exception:
        pass


_local_rebuild_signature: str | None = None


def _initialize_backend():
    """Initialize the RAG backend with logging.

    Uses the shared get_rag_backend() factory so both MCP and REST API
    share the same singleton instance.

    Returns the initialized backend instance.
    """
    start_time = time.time()

    from lasagna.core.config import rag_config

    _configure_rag_logging(rag_config.log_level)
    logger.info("Initializing RAG backend...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"RAG config initialized: upload_dir={rag_config.upload_dir}, db_path={rag_config.db_path}"
    )

    # Log enabled features
    features = []
    if rag_config.enable_hybrid_search:
        features.append("hybrid_search")
    if rag_config.enable_reranking:
        features.append("reranking")
    if rag_config.enable_hyde:
        features.append("hyde")
    if rag_config.enable_mmr:
        features.append("mmr")
    if rag_config.enable_ocr:
        features.append("ocr")
    logger.info(f"Enabled features: {features or ['basic']}")

    _refuse_a_held_store()
    backend = get_rag_backend()

    elapsed = time.time() - start_time
    logger.success(f"RAG backend initialized in {elapsed:.2f}s")
    return backend


def _refuse_a_held_store() -> None:
    """Raise at once, with the fix, when another process holds the local store.

    Milvus Lite opens a store in one process. While `lasagna serve` held it, a
    tool call loaded models for about 7 s and then failed with "Another
    process is using the local store".
    """
    from lasagna.core import local_server

    if not local_server.local_store_held():
        return
    running = local_server.running_server()
    raise local_server.StoreHeld(
        f"lasagna serve at {running} holds the local store, and this MCP server cannot "
        f"open it. Register the server's own MCP endpoint by URL instead of "
        f"lasagna-mcp: {running}/mcp/sse (start it with `lasagna serve --mcp`)."
        if running
        else local_server.held_store_message("this MCP server")
    )


def _get_backend():
    """Return the RAG backend singleton via the shared factory.

    All features are controlled via configuration in settings.lasagna.toml.
    """
    _refuse_a_held_store()
    return get_rag_backend()


def _refresh_log_level(session_id: Optional[str] = None, config: Optional[Any] = None) -> None:
    """Refresh MCP log level using session-aware config (no restart required)."""
    global _last_log_level
    try:
        if config is None:
            backend = _get_backend()
            if session_id:
                config = backend.get_session_config(session_id)
            else:
                from lasagna.core.config import rag_config

                config = rag_config
        level = str(getattr(config, "log_level", "INFO")).upper()
        if level != _last_log_level:
            _configure_rag_logging(level)
            _last_log_level = level
            logger.debug(f"Refreshed RAG log level to {level}")
    except Exception as e:
        logger.debug(f"Failed to refresh RAG log level: {e}")


def _get_config():
    """Return the RAG configuration singleton."""
    from lasagna.core.config import rag_config

    return rag_config


#: Argument names that carry a credential. `session_token` is a signed JWT valid
#: for seven days by default and `admin_token` authorises instance-wide
#: destruction, so neither may reach a log file -- and docker-compose ships the
#: MCP server's stdout straight to the Docker log driver.
_CREDENTIAL_ARG_NAMES = frozenset({"session_token", "admin_token", "api_key", "password"})


def _redact_credentials(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace credential values with a fixed marker.

    Presence is worth logging -- "was a token supplied?" is a real debugging
    question -- so the key stays and only the value goes.
    """
    return {k: ("<redacted>" if k in _CREDENTIAL_ARG_NAMES and v else v) for k, v in params.items()}


def _log_tool_call(tool_name: str, params: Dict[str, Any]):
    """Log incoming tool call with parameters."""
    # Mask sensitive data
    safe_params = _redact_credentials(params)
    if "file_path" in safe_params:
        safe_params["file_path"] = f".../{os.path.basename(safe_params['file_path'])}"
    if "query" in safe_params:
        query_value = safe_params.get("query") or ""
        safe_params["query"] = {
            "hash": hash_text(query_value),
            "len": len(query_value),
        }
    logger.info(f"TOOL CALL: {tool_name} | params={json.dumps(safe_params, default=str)}")


def _log_tool_result(tool_name: str, success: bool, result_summary: str, duration_ms: float):
    """Log tool call result."""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"TOOL RESULT: {tool_name} | {status} | {result_summary} | {duration_ms:.1f}ms")


def _llm_configured() -> bool:
    """Whether an LLM provider can be built from this process's environment."""
    from lasagna.providers import LLMProviderNotConfigured, get_llm_provider

    try:
        get_llm_provider()
    except LLMProviderNotConfigured:
        return False
    return True


def _log_tool_failure(what: str, exc: BaseException) -> None:
    """Log a tool's failure at ERROR, with its traceback.

    Except a store another process holds (a running `lasagna serve`): the
    tool's reply already says so and how to fix it, and an ERROR line with a
    traceback on every call buried that sentence, so it is logged at DEBUG.
    """
    try:
        from lasagna.api.rag_backend import _is_data_dir_locked

        held = _is_data_dir_locked(exc)
    except Exception:  # the backend module itself failed to import
        held = False
    if held or isinstance(exc, (_StoreHeld, ModelNotDownloaded)):
        logger.debug("{}: {}", what, exc)
    else:
        logger.opt(exception=exc).error("{}: {}", what, exc)


def _truncate_text(value: Optional[str], max_len: int = 200) -> str:
    if not value:
        return ""
    return value if len(value) <= max_len else f"{value[:max_len]}..."


def _log_rag_settings(tool_name: str, config) -> None:
    # Skip building the large snapshot dict unless DEBUG logging is active.
    # _last_log_level is maintained by _refresh_log_level().
    if _last_log_level and _last_log_level not in ("DEBUG", "TRACE"):
        return
    snapshot = {
        "collection_name": config.collection_name,
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "fallback_model": config.embedding_model_fallback,
            "dim": config.embedding_dim,
            "normalize": config.normalize_embeddings,
            "batch_size": config.batch_size,
            "use_gpu": config.use_gpu,
            "metric_type": config.metric_type,
            "index_type": config.index_type,
            "index_params": config.index_params,
            "search_params": config.search_params,
        },
        "chunking": {
            "strategy": config.chunking_strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "semantic_threshold": config.semantic_threshold,
            "min_chunk_size": config.min_chunk_size,
            "max_chunk_size": config.max_chunk_size,
            "enable_hierarchical": config.enable_hierarchical,
            "parent_chunk_size": config.parent_chunk_size,
            "child_chunk_size": config.child_chunk_size,
            "auto_merge_threshold": config.auto_merge_threshold,
            "preserve_tables": config.preserve_tables,
            "preserve_headers": config.preserve_headers,
            "max_content_length": config.max_content_length,
            "window_size": config.window_size,
            "sentence_overlap": config.sentence_overlap,
        },
        "hybrid_search": {
            "enabled": config.enable_hybrid_search,
            "alpha": config.hybrid_alpha,
            "sparse_method": config.sparse_method,
            "default_top_k": config.default_top_k,
            "similarity_threshold": config.similarity_threshold,
        },
        "reranking": {
            "enabled": config.enable_reranking,
            "model": config.reranker_model,
            "top_k": config.rerank_top_k,
            "batch_size": config.reranker_batch_size,
            "enable_mmr": config.enable_mmr,
            "mmr_lambda": config.mmr_lambda,
        },
        "hyde": {
            "enabled": config.enable_hyde,
            "model": config.hyde_model,
            "query_expansion": config.enable_query_expansion,
            "max_query_variations": config.max_query_variations,
            "query_decomposition": config.enable_query_decomposition,
        },
        "citations": {
            "format": config.citation_format,
            "id_length": config.citation_id_length,
            "verify": config.verify_citations,
            "track_db": config.track_citations_db,
            "show_page_numbers": config.show_page_numbers,
            "citation_db_path": config.citation_db_path,
        },
        "storage": {
            "max_file_size": config.max_file_size,
            "supported_extensions": config.supported_extensions,
            "upload_dir": config.upload_dir,
            "db_path": config.db_path,
        },
        "ocr": {
            "enabled": config.enable_ocr,
            "engine": config.ocr_engine,
            "languages": config.ocr_languages,
            "confidence_threshold": config.ocr_confidence_threshold,
        },
        "processing": {
            "extract_images": config.extract_images,
            "preserve_formatting": config.preserve_formatting,
            "async_processing": config.async_processing,
        },
        "cache": {
            "embedding_cache_enabled": config.enable_embedding_cache,
            "embedding_cache_size": config.embedding_cache_size,
            "search_cache_enabled": config.enable_search_cache,
            "search_cache_ttl": config.search_cache_ttl,
        },
        "agentic_rag": {
            "enabled": config.use_agentic_rag,
            "final_search": config.use_final_search,
            "max_iterations": config.max_agent_iterations,
            "agent_temperature": config.agent_temperature,
            "max_query_length": config.max_query_length,
            "hyde_strategy": config.agentic_hyde_strategy,
            "chunk_refinement_method": config.chunk_refinement_method,
            "refinement_model": config.refinement_model,
            "auto_refine_avg_length_threshold": config.auto_refine_avg_length_threshold,
            "auto_refine_avg_score_threshold": config.auto_refine_avg_score_threshold,
            "auto_refine_chunk_count_threshold": config.auto_refine_chunk_count_threshold,
            "chunk_cite_retrieval": config.chunk_cite_retrieval,
        },
        "document_awareness": {
            "enabled": config.document_awareness_enabled,
            "enrichment_strategy": config.enrichment_strategy,
            "first_tokens_count": config.first_tokens_count,
            "include_filename": config.include_filename,
            "include_upload_date": config.include_upload_date,
            "include_file_type": config.include_file_type,
        },
        "summarization": {
            "summary_model": config.summary_model,
            "summary_max_tokens": config.summary_max_tokens,
            "summary_prompt": _truncate_text(config.summary_prompt),
            "summary_prompt_len": len(config.summary_prompt or ""),
            "hybrid_llm_threshold_kb": config.hybrid_llm_threshold_kb,
            "max_documents_in_context": config.max_documents_in_context,
        },
        "snippets": {
            "cache_snippets": config.cache_snippets,
            "snippet_cache_ttl": config.snippet_cache_ttl,
        },
    }
    logger.debug(f"{tool_name} settings: {json.dumps(snapshot, default=str)}")


# ---------------------------------------------------------------------------
# Citation Monitoring & Validation
# ---------------------------------------------------------------------------

# Keywords that suggest late-document content (experiments, results, etc.)
_LATE_DOCUMENT_KEYWORDS = [
    "experiment",
    "table",
    "figure",
    "appendix",
    "references",
    "ablation",
    "results",
    "evaluation",
    "conclusion",
    "discussion",
    "methodology",
    "4.",
    "5.",
    "6.",
    "7.",
    "section 4",
    "section 5",
    "section 6",
]


def _log_citation_diagnostics(
    results: List[Any], citations: List[Any], query: str, session_id: str
):
    """Log detailed citation diagnostics for monitoring and debugging.

    Logs:
    - Per-chunk info: doc_id, filename, chunk_id, page_number, score, text preview
    - Summary stats: unique pages, min/max page, missing page %
    - Suspicious patterns that might indicate page metadata issues

    Args:
        results: List of SearchResult objects
        citations: List of Citation objects
        query: The search query
        session_id: Session identifier
    """
    if not results or not citations:
        return

    # Collect per-chunk diagnostics
    chunk_diagnostics = []
    pages_found = []
    missing_page_count = 0
    paged_chunks = 0

    for i, (result, citation) in enumerate(zip(results, citations)):
        # Extract info from result and citation
        chunk = getattr(result, "chunk", None)
        doc = getattr(result, "document", None)

        doc_id = getattr(doc, "document_id", citation.document_id) if doc else citation.document_id
        filename = (
            getattr(doc, "filename", citation.document_name) if doc else citation.document_name
        )
        chunk_id = getattr(chunk, "chunk_id", citation.chunk_id) if chunk else citation.chunk_id
        chunk_index = (
            getattr(chunk, "chunk_index", citation.chunk_index) if chunk else citation.chunk_index
        )

        # Get page number from multiple sources
        page_number = citation.page_number
        if page_number is None and chunk:
            page_number = getattr(chunk, "page_number", None)
            if page_number is None and hasattr(chunk, "metadata"):
                page_number = chunk.metadata.get("page_number")

        score = getattr(result, "score", citation.score)
        content = citation.content if hasattr(citation, "content") else ""

        # Create text preview: first 120 chars + last 60 chars
        if len(content) > 200:
            preview = f"{content[:120]}...{content[-60:]}"
        else:
            preview = content
        # Clean preview for logging
        preview = preview.replace("\n", " ").replace("\r", "")[:200]

        # Track pages. Only a PDF has them: a Markdown or text file always
        # "missed" its page, and every search on one warned that its citations
        # looked suspicious.
        if page_number is not None:
            pages_found.append(page_number)
        elif str(filename or "").lower().endswith(".pdf"):
            missing_page_count += 1
            paged_chunks += 1
        if page_number is not None:
            paged_chunks += 1

        chunk_diagnostics.append(
            {
                "index": i + 1,
                "doc_id": doc_id[:16] if doc_id else "N/A",
                "filename": filename,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "page": page_number,
                "score": round(score, 3) if score else 0,
                "preview": preview,
            }
        )

    # Log per-chunk diagnostics at DEBUG level
    logger.debug(f"Citation diagnostics for query: '{query[:50]}...'")
    for diag in chunk_diagnostics:
        logger.debug(
            f"  [{diag['index']}] {diag['filename']} | "
            f"chunk={diag['chunk_index']} | page={diag['page']} | "
            f"score={diag['score']} | preview: {diag['preview'][:80]}..."
        )

    # Calculate summary stats
    total_chunks = len(chunk_diagnostics)
    unique_pages = set(pages_found)
    min_page = min(pages_found) if pages_found else None
    max_page = max(pages_found) if pages_found else None
    missing_pct = (missing_page_count / total_chunks * 100) if total_chunks > 0 else 0

    # Log summary
    logger.debug(
        f"Citation summary: {total_chunks} chunks | "
        f"unique_pages={len(unique_pages)} ({sorted(unique_pages) if unique_pages else 'none'}) | "
        f"range=[{min_page}-{max_page}] | missing_page={missing_pct:.0f}%"
    )

    # Check for suspicious patterns, among the chunks that can have a page
    if not paged_chunks:
        return
    _check_suspicious_citations(
        chunk_diagnostics,
        pages_found,
        unique_pages,
        missing_page_count,
        paged_chunks,
        query,
        session_id,
    )


def _check_suspicious_citations(
    chunk_diagnostics: List[Dict],
    pages_found: List[int],
    unique_pages: set,
    missing_page_count: int,
    total_chunks: int,
    query: str,
    session_id: str = "",
):
    """Check for suspicious citation patterns and emit warnings.

    Suspicious patterns:
    1. All chunks have same page but content suggests late-document material
    2. High percentage of missing page metadata
    3. All pages are 0 or 1 (likely default values)
    """
    warnings = []

    # Check 1: All same page with late-document keywords in content
    if len(unique_pages) == 1 and pages_found:
        single_page = list(unique_pages)[0]
        # Check if any chunk content contains late-document keywords
        late_content_chunks = []
        for diag in chunk_diagnostics:
            preview_lower = diag["preview"].lower()
            for keyword in _LATE_DOCUMENT_KEYWORDS:
                if keyword in preview_lower:
                    late_content_chunks.append((diag["index"], keyword))
                    break

        if late_content_chunks and single_page <= 2:
            warnings.append(
                f"SUSPICIOUS: All {total_chunks} chunks report page {single_page}, "
                f"but content contains late-document keywords: "
                f"{[(idx, kw) for idx, kw in late_content_chunks[:3]]}"
            )

    # Check 2: High percentage of missing page metadata
    if total_chunks > 0:
        missing_pct = missing_page_count / total_chunks * 100
        if missing_pct > 50:
            warnings.append(
                f"SUSPICIOUS: {missing_pct:.0f}% of chunks ({missing_page_count}/{total_chunks}) "
                f"have no page metadata"
            )

    # Check 3: All pages are 0 or default values
    if pages_found and all(p in (0, 1) for p in pages_found) and len(pages_found) > 2:
        warnings.append(
            f"SUSPICIOUS: All {len(pages_found)} chunks have page=0 or 1, "
            f"likely default values not actual page numbers"
        )

    # Emit warnings
    for warning in warnings:
        logger.warning(f"CITATION_CHECK | session={session_id} | {warning}")

    return warnings


def _validate_page_mapping(results: List[Any], session_id: str, backend: Any) -> Dict[str, Any]:
    """Optional validation: verify chunk page numbers against source document.

    Only runs if RAG_VALIDATE_PAGE_MAP=1 environment variable is set.
    This is expensive and should only be used for debugging.

    Args:
        results: Search results with chunks
        session_id: Session identifier
        backend: RAG backend instance

    Returns:
        Validation report dict
    """
    if not os.environ.get("RAG_VALIDATE_PAGE_MAP"):
        return {}

    logger.info("Running page mapping validation (RAG_VALIDATE_PAGE_MAP=1)")
    validation_results = []

    for result in results:
        chunk = getattr(result, "chunk", None)
        if not chunk:
            continue

        claimed_page = getattr(chunk, "page_number", None)
        if claimed_page is None and hasattr(chunk, "metadata"):
            claimed_page = chunk.metadata.get("page_number")

        if claimed_page is None:
            validation_results.append(
                {
                    "chunk_id": getattr(chunk, "chunk_id", "unknown"),
                    "status": "no_page_claimed",
                    "claimed_page": None,
                }
            )
            continue

        # Get a short anchor text from chunk content (30-50 chars)
        content = getattr(chunk, "content", "")
        # Skip leading whitespace and get a clean anchor
        anchor = " ".join(content.split()[:8])  # First 8 words as anchor

        if len(anchor) < 20:
            validation_results.append(
                {
                    "chunk_id": getattr(chunk, "chunk_id", "unknown"),
                    "status": "anchor_too_short",
                    "claimed_page": claimed_page,
                }
            )
            continue

        # Try to verify against stored document (this would require page-level text storage)
        # For now, we just log what we would check
        validation_results.append(
            {
                "chunk_id": getattr(chunk, "chunk_id", "unknown"),
                "status": "would_validate",
                "claimed_page": claimed_page,
                "anchor_preview": anchor[:50],
            }
        )

    # Log validation summary
    statuses = [v["status"] for v in validation_results]
    logger.info(
        f"Page validation summary: {len(validation_results)} chunks | "
        f"would_validate={statuses.count('would_validate')} | "
        f"no_page={statuses.count('no_page_claimed')} | "
        f"short_anchor={statuses.count('anchor_too_short')}"
    )

    return {"validated": len(validation_results), "results": validation_results}


# ---------------------------------------------------------------------------
# Shared search preamble
# ---------------------------------------------------------------------------


def _prepare_search(
    tool_name: str,
    query: str,
    session_id: str,
    document_ids: Optional[str],
    allow_broaden: bool,
    user_id: str = "__anonymous__",
) -> Tuple[Any, Any, Optional[List[str]]]:
    """Common preamble for the ``search`` tool (and deprecated aliases).

    Performs:
    1. RAG-enabled check
    2. Backend + session config init
    3. Document ID parsing
    4. Focused-document enforcement

    Returns:
        (backend, config, doc_id_filter)

    Raises:
        _RAGDisabledError: if RAG is disabled for the session
    """
    # Check if RAG is enabled for this session
    config_manager = None
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        config_manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        if not config_manager.is_rag_enabled(session_id):
            logger.info(f"RAG is disabled for session {session_id}, blocking {tool_name}")
            raise _RAGDisabledError()
    except _RAGDisabledError:
        raise
    except Exception as e:
        logger.warning(f"Could not check rag_enabled state: {e}")

    backend = _get_backend()
    config = backend.get_session_config(session_id)
    _refresh_log_level(session_id, config)
    _log_rag_settings(tool_name, config)

    # Parse optional comma-separated document_ids filter
    doc_id_filter: Optional[List[str]] = None
    if document_ids:
        doc_id_filter = [d.strip() for d in document_ids.split(",") if d.strip()]
        logger.debug(f"Filtering by document_ids: {doc_id_filter}")

    # Enforce focused document IDs from session config
    if config_manager is not None:
        try:
            focused_ids = config_manager.get_focused_document_ids(session_id)
            if focused_ids:
                if allow_broaden:
                    logger.warning(
                        f"ENFORCEMENT: allow_broaden=True for session {session_id}, "
                        f"bypassing focused-document restriction ({len(focused_ids)} focused docs)"
                    )
                else:
                    logger.info(
                        f"ENFORCEMENT: Overriding document_ids with focused_document_ids: {focused_ids}"
                    )
                    doc_id_filter = focused_ids
        except Exception as e:
            logger.error(f"Failed to enforce focused_document_ids for session {session_id}: {e}")
            raise

    return backend, config, doc_id_filter


class _RAGDisabledError(Exception):
    """Sentinel for RAG-disabled check in _prepare_search."""

    pass


_RAG_DISABLED_MSG = (
    "Knowledge base search is currently disabled. Enable 'Use Knowledge Base' to search documents."
)

_NO_RESULTS_MSG = (
    "No relevant information found in the knowledge base for this query. "
    "Try rephrasing your search or check that documents have been uploaded."
)


#: The REST API's bound for the same argument (`SearchRequest.top_k`).
_MAX_TOP_K = 100


def _store_citations(
    user_id: str,
    session_id: str,
    citations: list,
    chunks: list,
    backend_tag: str = "",
) -> None:
    """Store citation results in session state for frontend retrieval.

    Failures are logged but never propagated — citation storage is
    best-effort and must not break the search response.
    """
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        config_manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        tag = f"[CITATIONS-MCP{'-' + backend_tag if backend_tag else ''}]"
        logger.info(f"{tag} Storing {len(citations)} citations for session {session_id}")
        config_manager.set_last_search_results(
            citations=citations,
            chunks=chunks,
            session_id=session_id,
        )
        logger.info(f"{tag} Successfully stored citations for session {session_id}")
    except Exception as store_err:
        tag = f"[CITATIONS-MCP{'-' + backend_tag if backend_tag else ''}]"
        logger.error(
            f"{tag} Failed to store citations: {store_err}. "
            f"Frontend citation panel will not display results for this query."
        )


def _calling_tool_name(default: str = "this tool") -> str:
    """The name of the tool function that reached the refusal.

    Read from the stack rather than passed by each of the ~30 call sites,
    because a hand-written name is one rename away from being wrong and this
    string is the only thing telling the caller which tool they must fix. Only
    evaluated on the failure path.
    """
    import sys

    try:
        # 0 is this helper, 1 is _resolve_session, 2 is the tool.
        return sys._getframe(2).f_code.co_name or default
    except (ValueError, AttributeError):  # pragma: no cover - platform-dependent
        return default


def _resolve_session(
    session_id: str = "",
    session_token: Optional[str] = None,
    *,
    tool: Optional[str] = None,
    admin_verified: bool = False,
) -> Tuple[str, str]:
    """Resolve (session_id, user_id) from a session token or explicit ID.

    When ``session_token`` is provided and valid, both session_id and user_id
    are extracted from it.  Otherwise falls back to the explicit ``session_id``
    with a **deterministic** ``anon_<hash>`` user_id derived from the session_id.

    Determinism matters: MCP clients that send ``session_id`` without a token
    need the same user_id across calls so that profile ownership checks and
    config scoping remain consistent within a session.

    This is also the single chokepoint where the MCP surface decides whether an
    asserted ``session_id`` counts as identity — see `transport_guard`. Every
    tool that takes a session routes through here, which is why the guard lives
    at this seam rather than on 30 individual decorators.
    """
    if session_token:
        try:
            from lasagna.api.rest.session_token import decode_session_token

            result = decode_session_token(session_token)
            if result:
                return result  # (session_id, user_id)
        except Exception:
            logger.debug("Failed to decode session_token in MCP _resolve_session")
        # A credential was presented and did not verify. Falling through to the
        # header-equivalent path would silently downgrade the caller to whatever
        # session_id they typed, which is how the REST side leaked before.
        raise transport_guard.MCPAuthError(
            f"{tool or _calling_tool_name()}: the supplied session_token is invalid or expired."
        )
    if not session_id:
        # The documented way to point an MCP client at one corpus.
        session_id = os.environ.get("RAG_SESSION_ID", "").strip()
    if not session_id:
        raise ValueError(
            "No session named: pass session_id (or session_token), or start the server "
            "with one: lasagna serve --mcp -s <session>, or RAG_SESSION_ID=<session> in "
            "lasagna-mcp's environment."
        )

    if transport_guard.enforcement_active():
        transport_guard.guard_identity(
            session_id,
            from_token=False,
            tool=tool or _calling_tool_name(),
            admin_verified=admin_verified,
        )

    # Deterministic: same session_id always yields same user_id
    import hashlib

    user_hash = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return session_id, f"anon_{user_hash}"


def _profile_owner(user_id: str) -> str:
    """Who owns the profiles this caller saves, and whose it may see.

    Only a verified token names a tenant. Otherwise the surface is single-user
    (stdio, or loopback with auth off), where saved profiles were always meant
    to be shared across sessions, and the per-session ``anon_`` id kept a
    profile saved in one session out of every other.
    """
    if transport_guard.enforcement_active():
        return user_id
    from lasagna.core.session_config._data_models import ANONYMOUS

    return ANONYMOUS


# ---------------------------------------------------------------------------
# Tool: search (unified)
# ---------------------------------------------------------------------------


def _reports_degraded(tool):
    """Collect the stages a search tool's backend calls fell back on (see `_append_search_metadata`)."""

    @functools.wraps(tool)
    async def collecting(*args, **kwargs):
        from lasagna.utils import degraded

        with degraded.collecting():
            return await tool(*args, **kwargs)

    return collecting


@mcp.tool()
@_reports_degraded
async def search(
    query: str,
    session_id: str = "",
    top_k: Optional[int] = None,
    document_ids: Optional[str] = None,
    allow_broaden: bool = False,
    trace_id: Optional[str] = None,
    session_token: Optional[str] = None,
) -> str:
    """Search the knowledge base with the session's retrieval pipeline.

    Returns formatted context with citations ready for LLM injection,
    plus a trailing ``<search_metadata>`` JSON block with structured
    citation data.

    Retrieval behavior (hybrid search, reranking, HyDE, adaptive routing)
    is governed by the session configuration: callers only specify *what*
    to search and optional scope constraints.

    :param query: Natural-language search query.
    :param session_id: Session identifier; only documents belonging to this
        session are searched.
    :param top_k: Optional override for the number of results to return.
        When ``None`` (default), uses the session config value.
    :param document_ids: Optional comma-separated list of document IDs to
        restrict the search to specific documents (e.g. "doc_abc,doc_def").
    :param allow_broaden: If True, allows searching outside focused documents
        when focused scope is set for the session (explicit broadening).
    :returns: Formatted context with citations, plus appended search_metadata JSON.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "search",
        {
            "query": query,
            "session_id": session_id,
            "top_k": top_k,
            "document_ids": document_ids,
            "allow_broaden": allow_broaden,
            "trace_id": trace_id,
        },
    )

    # REST refuses these with a 422; here an empty query returned a knowledge
    # block and a negative top_k "no results", both read as answers.
    if not query or not query.strip():
        return "Search failed: the query is empty."
    if top_k is not None and not 1 <= top_k <= _MAX_TOP_K:
        return f"Search failed: top_k must be between 1 and {_MAX_TOP_K}, got {top_k}."

    try:
        backend, config, doc_id_filter = _prepare_search(
            "search",
            query,
            session_id,
            document_ids,
            allow_broaden,
            user_id,
        )

        # Resolve top_k: caller override → session config
        effective_top_k = top_k if top_k is not None else config.rerank_top_k

        # Without an LLM the agentic loop cannot run, and every search failed:
        # run the session's standard pipeline instead and say so, as REST does
        # for a query it routes to the agentic tier.
        agentic = bool(config.use_agentic_rag)
        if agentic and not _llm_configured():
            from lasagna.utils import degraded

            agentic = False
            degraded.record("agentic", "no LLM is configured; ran the standard pipeline")

        if agentic:
            from lasagna.agentic.agentic_rag import AgenticRAG

            agentic_rag = AgenticRAG.create(
                max_iterations=config.max_agent_iterations,
                enable_final_search=config.use_final_search,
                session_config=config,
            )
            rag_result = await agentic_rag.retrieve(
                question=query,
                session_id=session_id,
                document_ids=doc_id_filter,
                top_k=effective_top_k,
                trace_id=trace_id,
            )

            duration_ms = (time.time() - start_time) * 1000

            if not rag_result.formatted_context:
                _log_tool_result("search", True, "No results", duration_ms)
                return _NO_RESULTS_MSG

            _store_citations(
                user_id, session_id, rag_result.citations, rag_result.chunks, "AGENTIC"
            )

            _log_tool_result("search", True, f"{len(rag_result.citations)} chunks", duration_ms)
            return _append_search_metadata(
                rag_result.formatted_context,
                rag_result.citations or [],
                rag_result.chunks or [],
            )

        is_v2 = hasattr(backend, "search_with_citations")

        if is_v2:
            results, citations, formatted_context = backend.search_with_citations(
                query=query,
                session_id=session_id,
                top_k=effective_top_k,
                document_ids=doc_id_filter,
                trace_id=trace_id,
            )

            duration_ms = (time.time() - start_time) * 1000

            if not results:
                _log_tool_result("search", True, "No results", duration_ms)
                return _NO_RESULTS_MSG

            # Log citation diagnostics for monitoring
            _log_citation_diagnostics(results, citations, query, session_id)

            # Optional page validation (only if RAG_VALIDATE_PAGE_MAP=1)
            _validate_page_mapping(results, session_id, backend)

            # Build serializable citation and chunk lists
            chunks_list = [
                {
                    "chunk_id": str(r.chunk.chunk_id),
                    "content": r.chunk.content,
                    "chunk_index": r.chunk.chunk_index,
                }
                for r in results
            ]
            citations_as_dicts = [c.to_dict() if hasattr(c, "to_dict") else c for c in citations]

            _store_citations(user_id, session_id, citations_as_dicts, chunks_list, "V2")

            _log_tool_result("search", True, f"{len(results)} chunks", duration_ms)
            return _append_search_metadata(formatted_context, citations_as_dicts, chunks_list)

        else:
            # V1 fallback
            v1_top_k = top_k if top_k is not None else config.default_top_k
            results = await backend.async_search(
                query=query,
                session_id=session_id,
                top_k=v1_top_k,
                min_score=None,  # the same results as the CLI, REST and SDK
                document_ids=doc_id_filter,
                trace_id=trace_id,
            )

            duration_ms = (time.time() - start_time) * 1000

            if not results:
                _log_tool_result("search", True, "No results", duration_ms)
                return _NO_RESULTS_MSG

            # Build formatted context for V1
            sources = "\n".join([f"[{i + 1}] {r.document.filename}" for i, r in enumerate(results)])
            content = "\n\n".join(
                [f"[Source {i + 1}]\n{r.chunk.content}" for i, r in enumerate(results)]
            )

            formatted = f"""<knowledge_base>
The following information was retrieved from the user's uploaded documents.

SOURCES:
{sources}

CONTENT:
{content}
</knowledge_base>"""

            # Build citation/chunk metadata for V1 results
            v1_citations = []
            v1_chunks = []
            for i, r in enumerate(results, 1):
                v1_citations.append(
                    {
                        "index": i,
                        "document_id": str(r.document.document_id),
                        "document_name": r.document.filename,
                        "chunk_id": str(r.chunk.chunk_id),
                        "content_preview": r.chunk.content[:200] if r.chunk.content else "",
                        "score": r.score,
                        "chunk_index": r.chunk.chunk_index,
                        "page_number": getattr(r.chunk, "page_number", None),
                    }
                )
                v1_chunks.append(
                    {
                        "chunk_id": str(r.chunk.chunk_id),
                        "content": r.chunk.content,
                        "chunk_index": r.chunk.chunk_index,
                    }
                )

            _store_citations(user_id, session_id, v1_citations, v1_chunks, "V1")

            _log_tool_result("search", True, f"{len(results)} chunks", duration_ms)
            return _append_search_metadata(formatted, v1_citations, v1_chunks)

    except _RAGDisabledError:
        return _RAG_DISABLED_MSG
    except ModelNotDownloaded as e:
        _log_tool_result("search", False, str(e), (time.time() - start_time) * 1000)
        return f"Search failed: {e} {fix_for_session(session_id, e.model, surface='mcp')}"
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("search failed", e)
        _log_tool_result("search", False, str(e), duration_ms)
        return f"Search failed: {e}"


def _append_search_metadata(formatted_context: str, citations: list, chunks: list) -> str:
    """Append machine-parseable citation metadata after the formatted context.

    The formatted_context (human/LLM-readable) is returned unchanged.
    A <search_metadata> JSON block is appended so callers can extract
    citation data without a separate API call. It names the stages that fell
    back (``degraded_stages``) when any did, as the REST API does.
    """
    from lasagna.utils import degraded

    payload = {"citations": citations or [], "chunks": chunks or []}
    if degraded.current():
        payload["degraded_stages"] = degraded.current()
    metadata = json.dumps(payload, ensure_ascii=False)
    return f"{formatted_context}\n<search_metadata>{metadata}</search_metadata>"


# ---------------------------------------------------------------------------
# Tool: get_document_context
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_document_context(
    session_id: str = "",
    rag_search_count: int = 0,
    session_token: Optional[str] = None,
) -> str:
    """Get document awareness context for the current session.

    Returns information about available documents and adaptive guidance
    based on how many searches have been performed this turn. Helps agents
    decide when and how to use RAG search tools.

    :param session_id: Session identifier.
    :param rag_search_count: Number of RAG searches done this turn (affects guidance).
    :returns: Formatted context string describing available documents,
        or empty string if no documents exist.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "get_document_context",
        {
            "session_id": session_id,
            "rag_search_count": rag_search_count,
        },
    )

    try:
        from lasagna.agentic.document_awareness import get_document_awareness_context

        result = await get_document_awareness_context(
            session_id,
            rag_search_count=rag_search_count,
        )
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result(
            "get_document_context", True, f"{'has context' if result else 'no docs'}", duration_ms
        )
        return result or ""

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_document_context failed", e)
        _log_tool_result("get_document_context", False, str(e), duration_ms)
        # "" is also the answer for "no documents"; a failure must not read as that.
        return f"Failed to get document context: {e}"


# ---------------------------------------------------------------------------
# Tool: list_documents
# ---------------------------------------------------------------------------


@mcp.tool()
def list_documents(session_id: str = "", session_token: Optional[str] = None) -> str:
    """List all documents in the knowledge base for a session.

    Returns document metadata including processing status, file type, size,
    and number of indexed chunks.

    :param session_id: Session identifier.
    :returns: Human-readable list of documents with their status.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("list_documents", {"session_id": session_id})

    try:
        backend = _get_backend()
        logger.debug(f"Calling backend.get_documents for session_id={session_id}")
        documents = backend.get_documents(session_id)
        logger.debug(f"get_documents returned {len(documents) if documents else 0} documents")
        duration_ms = (time.time() - start_time) * 1000

        if not documents:
            _log_tool_result("list_documents", True, "No documents found", duration_ms)
            return "No documents found for this session."

        lines = [f"Documents in session '{session_id}' ({len(documents)} total):"]
        for doc in documents:
            lines.append(
                f"  - {doc.filename} [{doc.document_id}]"
                f" | status={doc.status.value}"
                f" | type={doc.file_type}"
                f" | size={doc.file_size // 1024} KB"
                f" | chunks={doc.chunks_count}"
            )

        _log_tool_result("list_documents", True, f"{len(documents)} documents", duration_ms)
        return "\n".join(lines)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("list_documents failed", e)
        _log_tool_result("list_documents", False, str(e), duration_ms)
        return f"Failed to list documents: {e}"


# ---------------------------------------------------------------------------
# Tool: upload_document
# ---------------------------------------------------------------------------


@mcp.tool()
def upload_document(
    file_path: str,
    session_id: str = "",
    filename: Optional[str] = None,
    wait: bool = True,
    session_token: Optional[str] = None,
) -> str:
    """Upload and index a document into the knowledge base.

    The document is read from disk, its text is extracted (PDF, DOCX, TXT, MD,
    CSV, JSON, HTML, XML supported), split into overlapping chunks, and each
    chunk is embedded and stored in the vector database for fast retrieval.

    When ``wait=true`` (default), the tool blocks until processing is complete
    and returns the final status.  When ``wait=false``, the tool returns
    immediately with ``status: "processing"``; use ``get_document_status``
    to poll for completion.

    :param file_path: Absolute path to the file on disk.
    :param session_id: Session identifier; the document will only be visible
        within this session.
    :param filename: Optional display name to override the basename of
        file_path.
    :param wait: If true (default), block until processing completes.
        If false, return immediately after dispatch.
    :returns: JSON object with document_id, filename, status, chunks_count,
        and file_size_kb.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "upload_document",
        {
            "file_path": file_path,
            "session_id": session_id,
            "filename": filename,
            "wait": wait,
        },
    )

    # Outside the try: a refusal must surface as a tool error, not a
    # summarisable message (see transport_guard.MCPAuthError).
    resolved_path = transport_guard.guard_server_path(file_path)
    try:
        # Validate the resolved path exists
        if not os.path.isfile(resolved_path):
            return json.dumps({"error": f"file not found at '{file_path}'"})

        backend = _get_backend()
        doc = backend.upload_document(resolved_path, session_id, filename)

        status_value = doc.status.value if hasattr(doc.status, "value") else str(doc.status)

        # If processing was dispatched async and caller wants to wait,
        # poll until the document reaches a terminal state.
        if wait and status_value == "processing":
            poll_interval = 2.0
            timeout = 300.0
            elapsed = time.time() - start_time
            while elapsed < timeout:
                time.sleep(poll_interval)
                updated = backend.get_document_by_id(doc.document_id, session_id)
                if updated is None:
                    break
                updated_status = (
                    updated.status.value
                    if hasattr(updated.status, "value")
                    else str(updated.status)
                )
                if updated_status not in ("processing", "uploading", "indexing"):
                    doc = updated
                    status_value = updated_status
                    break
                elapsed = time.time() - start_time
            else:
                # Timeout — return current status (still processing)
                updated = backend.get_document_by_id(doc.document_id, session_id)
                if updated is not None:
                    doc = updated
                    status_value = (
                        updated.status.value
                        if hasattr(updated.status, "value")
                        else str(updated.status)
                    )

        duration_ms = (time.time() - start_time) * 1000

        report = {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "status": status_value,
            "chunks_count": doc.chunks_count,
            "file_size_kb": doc.file_size // 1024,
        }
        duplicate_of = getattr(doc, "duplicate_of", None)
        if isinstance(duplicate_of, str) and duplicate_of:
            report["duplicate_of"] = duplicate_of
            report["note"] = "The session already holds this content, so it was not uploaded again."
        if status_value == "error":
            # A failed ingest went back as an ordinary report with isError=False.
            _log_tool_result("upload_document", False, f"status={status_value}", duration_ms)
            reason = getattr(doc, "error_message", None) or "no reason was recorded"
            return json.dumps(
                {"error": f"'{doc.filename}' could not be indexed: {reason}", "detail": report}
            )
        result = json.dumps(report)

        _log_tool_result(
            "upload_document",
            True,
            f"doc_id={doc.document_id}, status={status_value}, chunks={doc.chunks_count}",
            duration_ms,
        )
        return result

    except ValueError as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"upload_document validation error: {e}")
        _log_tool_result("upload_document", False, f"Validation: {e}", duration_ms)
        return json.dumps({"error": str(e)})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("upload_document unexpected error", e)
        _log_tool_result("upload_document", False, f"Error: {e}", duration_ms)
        return json.dumps({"error": f"Unexpected error: {e}"})


# ---------------------------------------------------------------------------
# Tool: delete_document
# ---------------------------------------------------------------------------


@mcp.tool()
def delete_document(
    document_id: str, session_id: str = "", session_token: Optional[str] = None
) -> str:
    """Delete a document and all its indexed chunks from the knowledge base.

    The document must belong to the provided session_id. Both the vector index
    entries (Milvus) and the metadata record (SQLite) are removed, and the
    original file is deleted from disk.

    :param document_id: The document ID to delete (e.g. "doc_abc123").
    :param session_id: Session identifier; used to verify ownership.
    :returns: Confirmation or error message.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("delete_document", {"document_id": document_id, "session_id": session_id})

    try:
        backend = _get_backend()
        success = backend.delete_document(document_id, session_id)
        duration_ms = (time.time() - start_time) * 1000

        if success:
            _log_tool_result("delete_document", True, f"Deleted {document_id}", duration_ms)
            return f"Document '{document_id}' deleted successfully."
        else:
            _log_tool_result("delete_document", False, "Not found or unauthorized", duration_ms)
            return (
                f"Failed to delete document: '{document_id}' was not found or does not "
                f"belong to session '{session_id}'."
            )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("delete_document failed", e)
        _log_tool_result("delete_document", False, str(e), duration_ms)
        return f"Failed to delete document: {e}"


# ---------------------------------------------------------------------------
# Tool: get_document_status
# ---------------------------------------------------------------------------


@mcp.tool()
def get_document_status(
    document_id: str, session_id: str = "", session_token: Optional[str] = None
) -> str:
    """Get the processing status of a document (lightweight O(1) lookup).

    Use this to poll for completion after an async upload
    (``upload_document`` with ``wait=false``).

    :param document_id: The document ID returned by ``upload_document``.
    :param session_id: Session identifier; used to verify ownership.
    :returns: JSON object with document_id, status, chunks_count, and
        error_message (if any).
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "get_document_status",
        {
            "document_id": document_id,
            "session_id": session_id,
        },
    )

    try:
        backend = _get_backend()
        doc = backend.get_document_by_id(document_id, session_id)
        duration_ms = (time.time() - start_time) * 1000

        if not doc:
            _log_tool_result("get_document_status", False, "Not found", duration_ms)
            return json.dumps(
                {"error": f"Document '{document_id}' not found in session '{session_id}'."}
            )

        status_value = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
        result = json.dumps(
            {
                "document_id": doc.document_id,
                "status": status_value,
                "chunks_count": doc.chunks_count,
                "error_message": doc.error_message,
            }
        )

        _log_tool_result("get_document_status", True, f"status={status_value}", duration_ms)
        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_document_status failed", e)
        _log_tool_result("get_document_status", False, str(e), duration_ms)
        return json.dumps({"error": f"Failed to get document status: {e}"})


# ---------------------------------------------------------------------------
# Tool: get_document_info
# ---------------------------------------------------------------------------


@mcp.tool()
def get_document_info(
    document_id: str, session_id: str = "", session_token: Optional[str] = None
) -> str:
    """Get detailed information about a specific document.

    Returns comprehensive metadata including processing statistics,
    file information, and chunk details.

    :param document_id: The document ID to query.
    :param session_id: Session identifier; used to verify ownership.
    :returns: Detailed document information in JSON format.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("get_document_info", {"document_id": document_id, "session_id": session_id})

    try:
        backend = _get_backend()
        documents = backend.get_documents(session_id)
        duration_ms = (time.time() - start_time) * 1000

        # Find the specific document
        doc = next((d for d in documents if d.document_id == document_id), None)

        if not doc:
            _log_tool_result("get_document_info", False, "Not found", duration_ms)
            return (
                f"Failed to get document info: '{document_id}' was not found in "
                f"session '{session_id}'."
            )

        info = {
            "document_id": doc.document_id,
            "session_id": doc.session_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size,
            "file_size_kb": doc.file_size // 1024,
            "status": doc.status.value,
            "chunks_count": doc.chunks_count,
            "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
            "processing_time_seconds": doc.processing_time,
            "text_length": doc.text_length,
            "error_message": doc.error_message,
        }

        _log_tool_result("get_document_info", True, f"Found {document_id}", duration_ms)
        return json.dumps(info, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_document_info failed", e)
        _log_tool_result("get_document_info", False, str(e), duration_ms)
        return f"Failed to get document info: {e}"


# ---------------------------------------------------------------------------
# Tool: get_document_chunks
# ---------------------------------------------------------------------------


#: The REST API's bound for a page of chunks.
_MAX_CHUNK_PAGE = 500


@mcp.tool()
def get_document_chunks(
    document_id: str,
    session_id: str = "",
    offset: int = 0,
    limit: int = 10,
    session_token: Optional[str] = None,
) -> str:
    """Get chunks from a specific document.

    Retrieves the indexed text chunks for a document with pagination support.
    Useful for inspecting how a document was split.

    :param document_id: The document ID to query.
    :param session_id: Session identifier; used to verify ownership.
    :param offset: Number of chunks to skip (for pagination).
    :param limit: Maximum number of chunks to return.
    :returns: List of chunks with their content and metadata.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "get_document_chunks",
        {"document_id": document_id, "session_id": session_id, "offset": offset, "limit": limit},
    )

    if offset < 0 or not 1 <= limit <= _MAX_CHUNK_PAGE:
        return (
            f"Failed to get document chunks: offset must be >= 0 and limit between 1 "
            f"and {_MAX_CHUNK_PAGE} (got offset={offset}, limit={limit})."
        )

    try:
        backend = _get_backend()
        if not any(d.document_id == document_id for d in backend.get_documents(session_id)):
            return (
                f"Failed to get document chunks: '{document_id}' was not found in "
                f"session '{session_id}'."
            )

        # Use the session-aware backend via public API (BYOVDB support).
        # We fetch all chunks (up to 10k) so that total_chunks in the response
        # is accurate and client-side pagination works correctly.
        results = backend.get_document_chunks(
            document_id=document_id,
            session_id=session_id,
            limit=10_000,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Sort by chunk_index and apply pagination
        sorted_results = sorted(results, key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        paginated = sorted_results[offset : offset + limit]

        chunks_info = []
        for chunk in paginated:
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", 0)
            content = chunk.get("content", "")
            chunks_info.append(
                {
                    "chunk_index": chunk_idx,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "content_length": len(content),
                }
            )

        result = {
            "document_id": document_id,
            "total_chunks": len(sorted_results),
            "showing": (
                f"{offset + 1}-{offset + len(paginated)} of {len(sorted_results)}"
                if paginated
                else f"none of {len(sorted_results)}"
            ),
            "chunks": chunks_info,
        }

        _log_tool_result(
            "get_document_chunks",
            True,
            f"{len(paginated)}/{len(sorted_results)} chunks",
            duration_ms,
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_document_chunks failed", e)
        _log_tool_result("get_document_chunks", False, str(e), duration_ms)
        return f"Failed to get document chunks: {e}"


# ---------------------------------------------------------------------------
# Tool: get_system_status
# ---------------------------------------------------------------------------


@mcp.tool()
def get_system_status(
    session_id: str = "",
    session_token: Optional[str] = None,
    admin_token: Optional[str] = None,
) -> str:
    """Get RAG system health and statistics.

    Returns system status including the vector store connection, embedding
    model info, collection statistics, and configuration summary. On a
    networked transport with auth on it needs a session token like the other
    tools, and the database and upload paths and the row count of the
    collection, which every session on the model shares, need the admin token.

    :param session_id: Session whose configuration is reported; the server's
        session (RAG_SESSION_ID) when omitted.
    :param session_token: Required on a networked transport with RAG_REQUIRE_AUTH.
    :param admin_token: With auth on, adds the paths and the shared row count.
    :returns: System status in JSON format.
    """
    start_time = time.time()
    _log_tool_call("get_system_status", {"session_id": session_id})

    if transport_guard.enforcement_active():
        # It answered with no token, and named the database and upload paths
        # and every tenant's row count to anyone who could reach the server.
        operator = transport_guard.admin_token_matches(admin_token)
        if admin_token and not operator:
            raise transport_guard.MCPAuthError(
                "get_system_status: the admin_token is not this server's RAG_ADMIN_TOKEN. "
                "Leave it out to see the status your session token may see."
            )
        session_id, _user_id = _resolve_session(session_id, session_token, admin_verified=operator)
    else:
        # stdio, or loopback with auth off: one user, who runs the server.
        session_id = session_id or os.environ.get("RAG_SESSION_ID", "").strip() or "default"
        operator = True

    try:
        from lasagna.knowledge_evaluation_sdk.stages import (
            CHUNKER_DISPATCH_KEYS,
            effective_chunking_strategy,
        )

        backend = _get_backend()
        # The session's own configuration: the global one named the default
        # collection, and a session with 76 chunks reported row_count 0.
        try:
            config = backend.get_session_config(session_id)
        except Exception:
            config = _get_config()
        # The chunker that runs: structural settings decide it, so the presets
        # that name `structure_aware` or `hierarchical` were reported as not
        # chunking semantically while their sections were.
        chunking = effective_chunking_strategy(
            {k: getattr(config, k, None) for k in CHUNKER_DISPATCH_KEYS}
        )

        # Get collection stats (using session-aware backend - BYOVDB support)
        collection_name = config.collection_name
        vb = backend.get_vector_backend()
        has_collection = vb.has_collection(collection_name)

        # Check if using V2 backend
        is_v2 = hasattr(backend, "embedding_manager")

        # The component states readiness and GET /api/v1/health report: this
        # said "healthy" while /health/ready answered 503.
        from lasagna.api.rest.routes.health import component_checks_blocking, summarize

        checks = component_checks_blocking(backend)
        overall, failing, degraded_components = summarize(checks)

        stats = {
            "status": overall,
            "failing": failing,
            "degraded": degraded_components,
            "checks": {name: check.model_dump() for name, check in checks.items()},
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend_version": "V2" if is_v2 else "V1",
            "components": {
                "milvus": {
                    "connected": checks["vector_backend"].status == "up",
                    "collection_name": collection_name,
                    "collection_exists": has_collection,
                },
                "embedding_model": {
                    "loaded": bool(
                        is_v2
                        and getattr(backend.embedding_manager, "is_loaded", False)
                        and backend.embedding_manager.model_name == config.embedding_model
                    ),
                    "model_name": config.embedding_model,
                    "embedding_dim": config.embedding_dim,
                },
                "sqlite": {"db_path": config.db_path, "exists": os.path.exists(config.db_path)},
            },
            "configuration": {
                "chunking_strategy": chunking,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "default_top_k": config.default_top_k,
                "similarity_threshold": config.similarity_threshold,
                "metric_type": config.metric_type,
                "max_file_size_mb": config.max_file_size // (1024 * 1024),
                "supported_extensions": config.supported_extensions,
                "use_agentic_rag": config.use_agentic_rag,
                "chunk_refinement_method": config.chunk_refinement_method,
            },
            "features": {
                "hybrid_search": config.enable_hybrid_search,
                "reranking": config.enable_reranking,
                "mmr_diversity": config.enable_mmr,
                "hyde_expansion": config.enable_hyde,
                "ocr_enabled": config.enable_ocr,
                "semantic_chunking": chunking in ("semantic", "structure_aware"),
            },
            "storage": {
                "upload_dir": config.upload_dir,
                "upload_dir_exists": os.path.exists(config.upload_dir),
            },
        }

        # Add embedding cache stats if available (V2 only)
        if is_v2:
            cache_stats = backend.embedding_manager.get_cache_stats()
            if cache_stats:
                stats["components"]["embedding_cache"] = cache_stats

        # Try to get collection row count (using session-aware backend - BYOVDB support)
        if has_collection and operator:
            try:
                collection_stats = vb.get_collection_stats(collection_name)
                stats["components"]["milvus"]["row_count"] = collection_stats.get("row_count", 0)
            except Exception as e:
                logger.warning(f"Could not get collection stats: {e}")
                stats["components"]["milvus"]["row_count"] = "unknown"
        if not operator:
            # Where the server keeps its data, and how much the other
            # tenants of the collection hold, are the operator's to see.
            stats["components"].pop("sqlite", None)
            stats.pop("storage", None)

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("get_system_status", True, "Status retrieved", duration_ms)
        return json.dumps(stats, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_system_status failed", e)
        _log_tool_result("get_system_status", False, str(e), duration_ms)
        return json.dumps(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Tool: rebuild_collection
# ---------------------------------------------------------------------------


@mcp.tool()
def rebuild_collection(
    session_id: str = "",
    session_token: Optional[str] = None,
    admin_token: Optional[str] = None,
    confirm: bool = False,
) -> str:
    """Drop and recreate the vector collection for the session's embedding model.

    Use when embedding dimensions or index settings change and there are
    no documents to reindex.

    **This is not scoped to your session.** Each embedding model has its own
    collection, and every session on that model shares it (sessions are a
    filter within it), so a drop takes every session's documents on that model
    with it. The dimension check below means it only drops when the configured
    embedding dimension no longer matches the collection, but a caller can
    produce that mismatch themselves with one prior ``patch_rag_config``. Hence
    the admin credential and the ``confirm`` flag that the strictly less
    destructive ``bulk_delete_session`` has always had.

    :param admin_token: Required on a networked transport with
        ``RAG_REQUIRE_AUTH`` set.
    :param confirm: Must be ``true`` to proceed. Safety guard.
    """
    transport_guard.guard_admin(admin_token, tool="rebuild_collection")
    session_id, _user_id = _resolve_session(
        session_id, session_token, admin_verified=bool(admin_token)
    )
    start_time = time.time()
    _log_tool_call("rebuild_collection", {"session_id": session_id, "confirm": confirm})

    if not confirm:
        return json.dumps(
            {
                "error": (
                    "confirm must be true. This drops the collection for this "
                    "session's embedding model, which every session on that "
                    "model shares, so it removes every session's documents "
                    f"there, not only {session_id!r}'s."
                )
            }
        )

    try:
        with _rebuild_lock:
            backend = _get_backend()
            config = backend.get_session_config(session_id)
            collection_dim = backend.get_collection_embedding_dim(config.collection_name)
            if collection_dim is None or collection_dim != config.embedding_dim:
                backend.recreate_collection_for_config(config)
            else:
                backend.ensure_collection_for_config(config)

            # Checks the collection for this session's model. Not the shared
            # default: other sessions' requests read that at the same time.
            backend.embedder_for(config)
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("rebuild_collection", True, "Collection rebuilt", duration_ms)
        return (
            "Collection rebuilt successfully.\n"
            f"  collection : {config.collection_name}\n"
            f"  embedding_dim : {config.embedding_dim}"
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("rebuild_collection failed", e)
        _log_tool_result("rebuild_collection", False, str(e), duration_ms)
        return f"Failed to rebuild collection: {e}"


# ---------------------------------------------------------------------------
# Tool: reindex_document
# ---------------------------------------------------------------------------


@mcp.tool()
def reindex_document(
    document_id: str,
    session_id: str = "",
    force_rebuild: bool = False,
    session_token: Optional[str] = None,
) -> str:
    """Reindex an existing document with the session's current settings.

    Use it after changing the embedding model or the chunking. The document
    keeps answering searches from its old chunks until the new ones are
    stored, and a reindex that fails leaves it as it was.

    :param document_id: The document ID to reindex.
    :param session_id: Session identifier; used to verify ownership.
    :param force_rebuild: If true, recreate the collection at the configured
        embedding dimension when it differs, but only while the collection is
        empty. A collection holding documents is never dropped from here:
        every session's documents live in it.
    :returns: Confirmation or error message.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "reindex_document",
        {"document_id": document_id, "session_id": session_id, "force_rebuild": force_rebuild},
    )

    try:
        backend = _get_backend()

        # Get document metadata
        documents = backend.get_documents(session_id)
        doc = next((d for d in documents if d.document_id == document_id), None)

        if not doc:
            duration_ms = (time.time() - start_time) * 1000
            _log_tool_result("reindex_document", False, "Not found", duration_ms)
            return (
                f"Failed to reindex document: '{document_id}' was not found in "
                f"session '{session_id}'."
            )

        config = backend.get_session_config(session_id)

        if force_rebuild:
            from lasagna.api._collection_manager import collection_is_empty

            collection_dim = backend.get_collection_embedding_dim(config.collection_name)
            rebuild_signature = f"{config.collection_name}:{config.embedding_dim}"
            with _rebuild_lock:
                if _get_rebuild_signature() != rebuild_signature:
                    if collection_dim is None or collection_dim != config.embedding_dim:
                        # A session-scoped tool dropping the shared collection let
                        # any tenant, after one patch of its own embedding_dim,
                        # delete every other tenant's vectors with no admin token.
                        if not collection_is_empty(
                            backend.get_vector_backend(session_id), config.collection_name
                        ):
                            duration_ms = (time.time() - start_time) * 1000
                            _log_tool_result(
                                "reindex_document", False, "Collection not empty", duration_ms
                            )
                            stored = (
                                f"{collection_dim} dimensions"
                                if collection_dim is not None
                                else "a size that could not be read"
                            )
                            return (
                                f"Failed to reindex document: the collection "
                                f"{config.collection_name!r} is embedded at {stored} and "
                                f"holds documents, possibly other sessions'. Rebuilding it "
                                f"for {config.embedding_dim} would delete all of them, so "
                                f"reindex_document never does. Use a profile at the stored "
                                f"size, or have an operator run rebuild_collection (admin "
                                f"token, confirm=true)."
                            )
                        logger.info(
                            "Rebuild requested: "
                            f"collection_dim={collection_dim}, "
                            f"config_dim={config.embedding_dim}"
                        )
                        backend.recreate_collection_for_config(config)
                    else:
                        backend.ensure_collection_for_config(config)
                    _set_rebuild_signature(rebuild_signature)

        # The document keeps its chunks, and answers searches from them, until
        # the new ones are stored; a reindex that fails leaves it as it was. It
        # used to delete the chunks first and leave the row "ready" with no
        # vectors when processing failed, and it wrote the new chunks to the
        # current collection without recording it. The shared default
        # embedder is left alone: other sessions' requests read it.
        updated_doc = backend.reindex_document(document_id, session_id)

        duration_ms = (time.time() - start_time) * 1000

        if updated_doc:
            _log_tool_result(
                "reindex_document",
                True,
                f"Reindexed with {updated_doc.chunks_count} chunks",
                duration_ms,
            )
            return (
                f"Document '{document_id}' reindexed successfully.\n"
                f"  New chunks: {updated_doc.chunks_count}\n"
                f"  Status: {updated_doc.status.value}"
            )
        else:
            _log_tool_result("reindex_document", True, "Reindexed", duration_ms)
            return f"Document '{document_id}' reindexed successfully."

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("reindex_document failed", e)
        _log_tool_result("reindex_document", False, str(e), duration_ms)
        return f"Failed to reindex document: {e}"


# ---------------------------------------------------------------------------
# Tool: clear_session
# ---------------------------------------------------------------------------


@mcp.tool()
def clear_session(
    session_id: str = "", confirm: bool = False, session_token: Optional[str] = None
) -> str:
    """Delete all documents in a session.

    WARNING: This will permanently delete all documents and their indexed
    chunks for the specified session.

    :param session_id: Session identifier; all documents in this session will be deleted.
    :param confirm: Must be set to True to confirm deletion.
    :returns: Confirmation or error message.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("clear_session", {"session_id": session_id, "confirm": confirm})

    if not confirm:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("clear_session", False, "Not confirmed", duration_ms)
        return json.dumps(
            {
                "error": (
                    "confirm must be true. Nothing was deleted; this deletes every "
                    f"document in session {session_id!r}."
                )
            }
        )

    try:
        backend = _get_backend()
        documents = backend.get_documents(session_id)

        if not documents:
            duration_ms = (time.time() - start_time) * 1000
            _log_tool_result("clear_session", True, "No documents to delete", duration_ms)
            return f"No documents found in session '{session_id}'."

        deleted_count = 0
        failed_count = 0

        for doc in documents:
            try:
                success = backend.delete_document(doc.document_id, session_id)
                if success:
                    deleted_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {doc.document_id}: {e}")
                failed_count += 1

        duration_ms = (time.time() - start_time) * 1000

        result = f"Session '{session_id}' cleared: {deleted_count} documents deleted"
        if failed_count > 0:
            result += f", {failed_count} failed"
            _log_tool_result(
                "clear_session",
                True,
                f"Partial: {deleted_count} deleted, {failed_count} failed",
                duration_ms,
            )
        else:
            _log_tool_result(
                "clear_session", True, f"{deleted_count} documents deleted", duration_ms
            )

        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("clear_session failed", e)
        _log_tool_result("clear_session", False, str(e), duration_ms)
        return f"Failed to clear session: {e}"


# ---------------------------------------------------------------------------
# Tool: get_lifecycle_policy / set_lifecycle_policy
# ---------------------------------------------------------------------------

_LIFECYCLE_POLICY_KEYS = {
    "soft_delete_grace_hours",
    "document_ttl_days",
    "error_retention_days",
    "session_idle_days",
    "gc_cycle_hours",
    "orphan_scan_hours",
    "safety_threshold_percent",
    "batch_size",
    "reconcile_quota_on_startup",
}


@mcp.tool()
def get_lifecycle_policy(session_id: str = "default", admin_token: Optional[str] = None) -> str:
    """Return the currently active lifecycle garbage-collection policy.

    The policy governs:
    - ``soft_delete_grace_hours``: grace window before GC permanently removes soft-deleted docs
    - ``document_ttl_days``: auto-expiry TTL (null or 0 = disabled)
    - ``error_retention_days``: how long ERROR docs survive before soft-deletion
    - ``gc_cycle_hours``: interval between full GC cycles
    - ``orphan_scan_hours``: interval between orphan detection scans
    - ``safety_threshold_percent``: GC abort threshold (fraction of all docs)
    - ``batch_size``: documents per GC batch
    - ``reconcile_quota_on_startup``: rebuild quota counters at startup

    :param session_id: Unused; policy is global, not per-session.
    :param admin_token: Required on a networked transport with
        ``RAG_REQUIRE_AUTH`` set, matching ``GET /gc/policy``.
    :returns: JSON object with all policy fields and their current values.
    """
    transport_guard.guard_admin(admin_token, tool="get_lifecycle_policy")
    start_time = time.time()
    _log_tool_call("get_lifecycle_policy", {"session_id": session_id})
    try:
        import dataclasses

        backend = _get_backend()
        mgr = getattr(backend, "_lifecycle_manager", None)
        if mgr is not None:
            policy = mgr._policy
        else:
            from lasagna.lifecycle.policy import LifecyclePolicy

            policy = LifecyclePolicy.load()
        result = dataclasses.asdict(policy)
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("get_lifecycle_policy", True, f"keys={list(result.keys())}", duration_ms)
        return json.dumps({"policy": result})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_lifecycle_policy failed", e)
        _log_tool_result("get_lifecycle_policy", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def set_lifecycle_policy(
    policy_updates: str, session_id: str = "", admin_token: Optional[str] = None
) -> str:
    """Update lifecycle policy fields at runtime.

    Changes take effect immediately for the current process but are **not
    persisted**: the server reverts to its configured defaults on restart.
    To make changes permanent, update ``[rag.lifecycle]`` in
    ``settings.lasagna.toml`` or set the corresponding ``RAG_GC_*`` env vars.

    Unknown field names are rejected with an error.

    :param session_id: Unused; policy is global, not per-session.
    :param policy_updates: JSON string mapping field names to new values.
        Example: '{"gc_cycle_hours": 12, "soft_delete_grace_hours": 48}'
    :param admin_token: Required on a networked transport with
        ``RAG_REQUIRE_AUTH`` set, matching ``PATCH /gc/policy``.
    :returns: JSON object with the full updated policy.
    """
    transport_guard.guard_admin(admin_token, tool="set_lifecycle_policy")
    start_time = time.time()
    _log_tool_call("set_lifecycle_policy", {"session_id": session_id})
    try:
        updates = json.loads(policy_updates)
        if not isinstance(updates, dict):
            return json.dumps({"error": "policy_updates must be a JSON object"})

        unknown = set(updates.keys()) - _LIFECYCLE_POLICY_KEYS
        if unknown:
            return json.dumps(
                {
                    "error": (
                        f"Unknown lifecycle policy fields: {sorted(unknown)}. "
                        f"Allowed: {sorted(_LIFECYCLE_POLICY_KEYS)}"
                    )
                }
            )

        # Validate types and ranges
        from lasagna.api.rest.routes.gc_admin import validate_policy_updates

        validation_error = validate_policy_updates(updates)
        if validation_error:
            return json.dumps({"error": f"Validation error: {validation_error}"})

        import dataclasses

        backend = _get_backend()
        mgr = getattr(backend, "_lifecycle_manager", None)
        if mgr is None:
            return json.dumps(
                {"error": "Lifecycle manager not initialized on this backend instance"}
            )

        policy = mgr._policy
        # Allow explicit null or 0 for document_ttl_days (both mean TTL disabled → None).
        # For all other fields, skip None so omitted fields aren't overwritten.
        updated = dataclasses.replace(
            policy,
            **{
                k: (None if k == "document_ttl_days" and not v else v)
                for k, v in updates.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        mgr._policy = updated
        # Propagate to the GC instance
        gc = getattr(mgr, "_gc", None)
        if gc is not None:
            gc._policy = updated
        # Propagate cadence changes to the scheduler (reschedules APScheduler jobs)
        scheduler = getattr(backend, "_lifecycle_scheduler", None)
        if scheduler is not None:
            scheduler.update_policy(updated)

        result = dataclasses.asdict(updated)
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result(
            "set_lifecycle_policy", True, f"updated={list(updates.keys())}", duration_ms
        )
        return json.dumps({"policy": result})

    except json.JSONDecodeError as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("set_lifecycle_policy", False, f"Invalid JSON: {e}", duration_ms)
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("set_lifecycle_policy failed", e)
        _log_tool_result("set_lifecycle_policy", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool: list_profiles
# ---------------------------------------------------------------------------


@mcp.tool()
def list_profiles(
    include_builtin: bool = True, session_id: str = "", session_token: Optional[str] = None
) -> str:
    """List available RAG configuration profiles.

    Profiles control RAG behavior like chunk size, embedding model,
    reranking settings, and more. Users can create custom profiles
    to tune retrieval for different use cases.

    :param include_builtin: Whether to include built-in profiles (default True).
    :param session_id: Session identifier.
    :returns: JSON array of profile objects with name and description.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("list_profiles", {"include_builtin": include_builtin})

    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)

        # Use the public list_profiles API which handles user-scoped filtering
        profiles = manager.list_profiles(
            include_builtin=include_builtin,
            session_id=session_id,
            user_id=_profile_owner(user_id),
        )

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("list_profiles", True, f"{len(profiles)} profiles", duration_ms)
        return json.dumps(profiles, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("list_profiles failed", e)
        _log_tool_result("list_profiles", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool: get_active_profile
# ---------------------------------------------------------------------------


@mcp.tool()
def get_active_profile(session_id: str = "", session_token: Optional[str] = None) -> str:
    """Get the active RAG profile for a session.

    Returns the currently active profile name and its configuration
    for the specified session.

    :param session_id: Session identifier.
    :returns: JSON object with active profile name and resolved config.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("get_active_profile", {"session_id": session_id})

    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        state = manager.get_session_state(session_id)
        active_profile = state.get_active_profile()

        result = {
            "session_id": session_id,
            "active_profile_name": state.active_profile,
            "profile_description": active_profile.description if active_profile else None,
            "has_session_overrides": bool(state.session_overrides),
        }

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("get_active_profile", True, f"active={state.active_profile}", duration_ms)
        return json.dumps(result, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_active_profile failed", e)
        _log_tool_result("get_active_profile", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool: set_active_profile
# ---------------------------------------------------------------------------


@mcp.tool()
def set_active_profile(
    profile_name: str,
    session_id: str = "",
    session_token: Optional[str] = None,
    force: bool = False,
    reindex: bool = False,
) -> str:
    """Set the active RAG profile for a session.

    Switches the session to use the specified profile for all subsequent
    RAG operations. This affects embedding model, chunking strategy,
    reranking settings, and more.

    :param session_id: Session identifier.
    :param profile_name: Name of the profile to activate: a preset (speed, standard,
        balanced, max_quality, default) or one saved with save_rag_profile.
    :param force: Switch even when the session's documents were embedded with
        another model and would not be searched; reindex them afterwards with
        reindex_document.
    :param reindex: Reindex the session's documents with the new profile first,
        and switch only when every one is reindexed; if one fails, the session
        keeps its profile.
    :returns: Confirmation message or error.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("set_active_profile", {"session_id": session_id, "profile_name": profile_name})

    try:
        from lasagna.api._profile_switch import summary, switch_profile
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        report = switch_profile(
            manager,
            profile_name,
            session_id=session_id,
            user_id=_profile_owner(user_id),
            force=force,
            reindex=reindex,
        )

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("set_active_profile", True, f"activated={profile_name}", duration_ms)
        done = summary(report)
        return f"Profile '{profile_name}' activated for session '{session_id}'." + (
            f" {done}" if done else ""
        )

    except ValueError as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("set_active_profile", False, f"Validation: {e}", duration_ms)
        return f"Failed to set profile: {e}"
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("set_active_profile failed", e)
        _log_tool_result("set_active_profile", False, str(e), duration_ms)
        return f"Failed to set profile: {e}"


# ---------------------------------------------------------------------------
# Tool: get_profile_stats
# ---------------------------------------------------------------------------


@mcp.tool()
def get_profile_stats(session_id: str = "", session_token: Optional[str] = None) -> str:
    """Get statistics about profiles and sessions.

    Returns counts and limits for profiles and sessions, useful for
    monitoring and debugging.

    :param session_id: Session identifier.
    :returns: JSON object with profile and session statistics.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("get_profile_stats", {})

    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        stats = manager.get_profile_stats()

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("get_profile_stats", True, "Stats retrieved", duration_ms)
        return json.dumps(stats, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_profile_stats failed", e)
        _log_tool_result("get_profile_stats", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool: set_focused_documents
# ---------------------------------------------------------------------------


@mcp.tool()
def set_focused_documents(
    session_id: str = "", document_ids: str = "", session_token: Optional[str] = None
) -> str:
    """Set focused document IDs for a session.

    When focused documents are set, all RAG searches will be scoped to
    only these documents. This is the server-side equivalent of the
    UI's "/" document picker feature.

    :param session_id: Session identifier.
    :param document_ids: Comma-separated list of document IDs to focus on,
                        or empty string to clear focus.
    :returns: Confirmation message.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call(
        "set_focused_documents",
        {
            "session_id": session_id,
            "document_ids": document_ids[:100] + "..." if len(document_ids) > 100 else document_ids,
        },
    )

    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)

        # ENFORCEMENT: Check if RAG is enabled for this session
        if not manager.is_rag_enabled(session_id):
            logger.info(f"RAG is disabled for session {session_id}, blocking set_focused_documents")
            return "Cannot set focused documents: Knowledge base is currently disabled. Enable 'Use Knowledge Base' first."

        # Parse document IDs from comma-separated string
        doc_id_list = []
        if document_ids and document_ids.strip():
            doc_id_list = [d.strip() for d in document_ids.split(",") if d.strip()]

        if doc_id_list:
            # An unknown id was accepted with "Focused on 2 document(s)", and
            # every later search in the session then quietly returned nothing.
            known = {d.document_id for d in _get_backend().get_documents(session_id)}
            unknown = [d for d in doc_id_list if d not in known]
            if unknown:
                return (
                    f"Failed to set focused documents: not in session '{session_id}': "
                    f"{', '.join(unknown)}. Focus was not changed."
                )

        manager.set_focused_document_ids(doc_id_list, session_id)

        duration_ms = (time.time() - start_time) * 1000

        if doc_id_list:
            msg = f"Focused on {len(doc_id_list)} document(s) for session '{session_id}'."
            _log_tool_result(
                "set_focused_documents", True, f"focused={len(doc_id_list)}", duration_ms
            )
        else:
            msg = f"Document focus cleared for session '{session_id}'."
            _log_tool_result("set_focused_documents", True, "cleared", duration_ms)

        return msg

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("set_focused_documents failed", e)
        _log_tool_result("set_focused_documents", False, str(e), duration_ms)
        return f"Failed to set focused documents: {e}"


# ---------------------------------------------------------------------------
# Tool: get_focused_documents
# ---------------------------------------------------------------------------


@mcp.tool()
def get_focused_documents(session_id: str = "", session_token: Optional[str] = None) -> str:
    """Get the currently focused document IDs for a session.

    :param session_id: Session identifier.
    :returns: JSON object with focused document IDs.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("get_focused_documents", {"session_id": session_id})

    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        doc_ids = manager.get_focused_document_ids(session_id)

        result = {
            "session_id": session_id,
            "focused_document_ids": doc_ids,
            "count": len(doc_ids),
            "is_focused": len(doc_ids) > 0,
        }

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("get_focused_documents", True, f"count={len(doc_ids)}", duration_ms)
        return json.dumps(result, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_focused_documents failed", e)
        _log_tool_result("get_focused_documents", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# RAG Configuration Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_rag_config(session_id: str = "", session_token: Optional[str] = None) -> str:
    """Get the current RAG configuration for a session.

    Returns the active profile and all configuration settings including
    chunking, embedding, reranking, and search parameters.

    :param session_id: Session identifier.
    :returns: JSON object with full configuration.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        config = manager.get_resolved_config(session_id)

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result(
            "get_rag_config", True, f"profile={config.get('active_profile')}", duration_ms
        )
        return json.dumps(config, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_rag_config failed", e)
        _log_tool_result("get_rag_config", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def patch_rag_config(
    config_patch: str,
    session_id: str = "",
    session_token: Optional[str] = None,
    force: bool = False,
) -> str:
    """Apply configuration overrides to the current session.

    Merges the provided settings with the active profile's config.
    These overrides persist for the session duration. Only retrieval and
    generation settings: the vector store, collection, paths, URLs and
    credentials are the server's, and a patch naming one is refused.

    :param session_id: Session identifier.
    :param config_patch: JSON string with config keys to override.
        Example: '{"rerank_top_k": 10, "enable_reranking": false}'
    :param force: Apply a change of embedding model even when the session's
        documents would not be searched; reindex them afterwards with
        reindex_document.
    :returns: JSON object naming each setting the patch changed, with its value
        before and after (get_rag_config shows the whole configuration).
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        patch = json.loads(config_patch)
        if not isinstance(patch, dict):
            raise ValueError("config_patch must be a JSON object")

        # Reject lifecycle policy fields — these belong to set_lifecycle_policy
        _LIFECYCLE_KEYS = {
            "soft_delete_grace_hours",
            "document_ttl_days",
            "error_retention_days",
            "session_idle_days",
            "gc_cycle_hours",
            "orphan_scan_hours",
            "safety_threshold_percent",
            "batch_size",
            "reconcile_quota_on_startup",
        }
        misrouted = set(patch.keys()) & _LIFECYCLE_KEYS
        if misrouted:
            return json.dumps(
                {
                    "error": (
                        f"Lifecycle policy fields are not accepted here: {sorted(misrouted)}. "
                        "Use the set_lifecycle_policy tool instead."
                    )
                }
            )

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        if force:
            # A forced change of model must still leave a model that loads,
            # or every later search fails.
            from lasagna.api._profile_switch import unloadable_models

            problems = unloadable_models({**manager.get_resolved_config(session_id), **patch})
            if problems:
                raise ValueError(
                    f"{' '.join(problems)} Nothing was changed for session {session_id!r}."
                )
        before = manager.get_resolved_config(session_id)
        manager.patch_config(patch, session_id, force=force)
        after = manager.get_resolved_config(session_id)
        # What changed, not the whole configuration: the reply was every
        # profile of the session, a few hundred lines for one setting.
        changed = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in patch
            if before.get(key) != after.get(key)
        }
        result = {
            "session_id": session_id,
            "active_profile": manager.get_active_profile(session_id=session_id),
            "changed": changed,
            "unchanged": sorted(key for key in patch if key not in changed),
        }

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("patch_rag_config", True, f"keys={list(patch.keys())}", duration_ms)
        return json.dumps(result, indent=2, default=str)

    except json.JSONDecodeError as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("patch_rag_config", False, f"Invalid JSON: {e}", duration_ms)
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("patch_rag_config failed", e)
        _log_tool_result("patch_rag_config", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def save_rag_profile(
    profile_name: str,
    config: str,
    session_id: str = "",
    set_active: bool = True,
    session_token: Optional[str] = None,
) -> str:
    """Save a custom RAG profile.

    Creates or updates a named profile with the provided configuration.
    Saved profiles persist across sessions and can be shared.

    :param session_id: Session identifier.
    :param profile_name: Name for the profile (cannot be 'default').
    :param config: JSON string with profile configuration.
    :param set_active: Whether to activate this profile (default True).
    :returns: Confirmation message.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        config_dict = json.loads(config)
        if not isinstance(config_dict, dict):
            raise ValueError("config must be a JSON object")

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        _result = manager.save_profile(
            profile_name, config_dict, session_id, set_active, user_id=_profile_owner(user_id)
        )

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("save_rag_profile", True, f"profile={profile_name}", duration_ms)
        return f"Profile '{profile_name}' saved successfully."

    except json.JSONDecodeError as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("save_rag_profile", False, f"Invalid JSON: {e}", duration_ms)
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("save_rag_profile failed", e)
        _log_tool_result("save_rag_profile", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def delete_rag_profile(
    profile_name: str, session_id: str = "", session_token: Optional[str] = None
) -> str:
    """Delete a custom RAG profile.

    Built-in profiles (default, speed, standard, balanced, max_quality) cannot be deleted.

    :param session_id: Session identifier.
    :param profile_name: Name of the profile to delete.
    :returns: Confirmation message.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        owner = _profile_owner(user_id)
        if not any(
            p["name"] == profile_name
            for p in manager.list_profiles(session_id=session_id, user_id=owner)
        ):
            # Deleting nothing answered "deleted successfully".
            return json.dumps({"error": f"Profile {profile_name!r} not found."})
        _result = manager.delete_profile(profile_name, session_id, user_id=owner)

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("delete_rag_profile", True, f"profile={profile_name}", duration_ms)
        return f"Profile '{profile_name}' deleted successfully."

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("delete_rag_profile failed", e)
        _log_tool_result("delete_rag_profile", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def reset_rag_config(
    session_id: str = "", scope: str = "active", session_token: Optional[str] = None
) -> str:
    """Reset RAG configuration to defaults.

    :param session_id: Session identifier.
    :param scope: Reset scope - 'active' (clear overrides) or 'all' (full reset).
    :returns: JSON object with reset configuration.
    """
    session_id, user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        if scope not in ("active", "all"):
            raise ValueError("scope must be 'active' or 'all'")

        manager = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        reset = manager.reset_config(scope, session_id)
        result = manager.get_config(session_id, user_id=_profile_owner(user_id))
        if "config" in reset:
            result["config"] = reset["config"]

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("reset_rag_config", True, f"scope={scope}", duration_ms)
        return json.dumps(result, indent=2)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("reset_rag_config failed", e)
        _log_tool_result("reset_rag_config", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Lifecycle tools (soft-delete, restore, bulk-delete, GC status)
# ---------------------------------------------------------------------------


@mcp.tool()
def archive_document(
    document_id: str,
    session_id: str = "",
    session_token: Optional[str] = None,
) -> str:
    """Soft-delete (archive) a document.

    The document is hidden from search results but is NOT immediately deleted.
    Vectors and the stored file remain until the garbage collector runs after
    the grace window (default 24 h).

    Use ``restore_document`` to recover the document within the grace window.
    ``delete_document`` (served with ``--tools all``) deletes immediately and
    irreversibly.

    :param document_id: ID of the document to archive.
    :param session_id: Session that owns the document.
    :returns: Updated document metadata as JSON.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("archive_document", {"document_id": document_id, "session_id": session_id})

    try:
        backend = _get_backend()
        doc_dict = backend.soft_delete_document(document_id, session_id)
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("archive_document", True, f"archived doc_id={document_id}", duration_ms)
        return json.dumps(doc_dict, indent=2, default=str)
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("archive_document failed", e)
        _log_tool_result("archive_document", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def restore_document(
    document_id: str,
    session_id: str = "",
    session_token: Optional[str] = None,
) -> str:
    """Restore a soft-deleted (archived) document to READY status.

    Only possible within the grace window after archiving.  If the grace
    window has expired and the GC has already hard-deleted the document,
    this call will return an error.

    :param document_id: ID of the archived document to restore.
    :param session_id: Session that owns the document.
    :returns: Updated document metadata as JSON, or an error message.
    """
    session_id, _user_id = _resolve_session(session_id, session_token)
    start_time = time.time()
    _log_tool_call("restore_document", {"document_id": document_id, "session_id": session_id})

    try:
        backend = _get_backend()
        doc_dict = backend.restore_document(document_id, session_id)
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("restore_document", True, f"restored doc_id={document_id}", duration_ms)
        return json.dumps(doc_dict, indent=2, default=str)
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("restore_document failed", e)
        _log_tool_result("restore_document", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def bulk_delete_session(
    session_id: str = "",
    confirm: bool = False,
    session_token: Optional[str] = None,
    admin_token: Optional[str] = None,
) -> str:
    """Delete ALL documents in a session at once.

    This is a destructive, irreversible operation.  Use with care.
    The ``confirm`` flag must be set to ``true`` as a safety guard.

    :param session_id: Session to wipe.
    :param confirm: Must be ``true`` to proceed.  Safety guard.
    :param admin_token: Required on a networked transport with
        ``RAG_REQUIRE_AUTH`` set, matching ``DELETE /api/v1/sessions/{id}``.
    :returns: JSON with counts: requested, deleted, bytes_freed, errors.
    """
    transport_guard.guard_admin(admin_token, tool="bulk_delete_session")
    session_id, _user_id = _resolve_session(
        session_id, session_token, admin_verified=bool(admin_token)
    )
    start_time = time.time()
    _log_tool_call("bulk_delete_session", {"session_id": session_id, "confirm": confirm})

    if not confirm:
        return json.dumps({"error": "confirm must be true to proceed with bulk session delete."})

    try:
        backend = _get_backend()
        result = backend.bulk_delete_session(session_id, confirm=True)
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result(
            "bulk_delete_session",
            True,
            f"deleted {result.deleted}/{result.requested} docs",
            duration_ms,
        )
        return json.dumps(
            {
                "session_id": session_id,
                "requested": result.requested,
                "deleted": result.deleted,
                "bytes_freed": result.bytes_freed,
                "chunks_freed": result.chunks_freed,
                "errors": result.errors,
            },
            indent=2,
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("bulk_delete_session failed", e)
        _log_tool_result("bulk_delete_session", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_gc_status(
    session_id: str = "", session_token: Optional[str] = None, admin_token: Optional[str] = None
) -> str:
    """Get the last garbage-collection run report.

    Returns a summary of the most recent GC cycle including per-phase
    results (docs scanned, docs cleaned, bytes freed, errors) and whether
    the run was aborted due to the safety threshold.

    The audit log is **server-wide**: ``session_id`` scopes the auth check, not
    the rows returned, which is why this takes the admin credential.

    :param session_id: Session identifier (used for auth resolution only).
    :param admin_token: Required on a networked transport with
        ``RAG_REQUIRE_AUTH`` set: the report covers every session.
    :returns: JSON GC report or error message.
    """
    transport_guard.guard_admin(admin_token, tool="get_gc_status")
    session_id, _user_id = _resolve_session(
        session_id, session_token, admin_verified=bool(admin_token)
    )
    start_time = time.time()
    _log_tool_call("get_gc_status", {"session_id": session_id})

    try:
        backend = get_rag_backend()
        # Use the backend facade — avoids accessing private DB connection attrs.
        rows = backend.get_gc_audit_log(limit=20)

        duration_ms = (time.time() - start_time) * 1000
        _log_tool_result("get_gc_status", True, f"{len(rows)} audit entries", duration_ms)
        return json.dumps({"gc_audit_log": rows}, indent=2, default=str)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_tool_failure("get_gc_status failed", e)
        _log_tool_result("get_gc_status", False, str(e), duration_ms)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Custom HTTP Routes (Welcome page, Docs, Health check)
# ---------------------------------------------------------------------------


def _get_welcome_html(host: str, port: int) -> str:
    """Generate welcome page HTML."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lasagna - MCP Server</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: #888;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }}
        .links {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .link-card {{
            background: rgba(0,212,255,0.1);
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 8px;
            padding: 20px;
            text-decoration: none;
            color: #fff;
            transition: all 0.3s ease;
        }}
        .link-card:hover {{
            background: rgba(0,212,255,0.2);
            transform: translateY(-2px);
        }}
        .link-card h3 {{
            color: #00d4ff;
            margin-bottom: 8px;
        }}
        .link-card p {{
            color: #aaa;
            font-size: 0.9rem;
        }}
        .tools-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
        }}
        .tool {{
            background: rgba(123,44,191,0.1);
            border: 1px solid rgba(123,44,191,0.3);
            border-radius: 8px;
            padding: 15px;
        }}
        .tool h4 {{
            color: #b388ff;
            margin-bottom: 5px;
        }}
        .tool p {{
            color: #999;
            font-size: 0.85rem;
        }}
        .status {{
            display: inline-block;
            padding: 4px 12px;
            background: #00c853;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #fff;
            margin-left: 10px;
        }}
        code {{
            background: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Lasagna <span class="status">Running</span></h1>
        <p class="subtitle">MCP Server for Document Management & Semantic Search</p>

        <div class="links">
            <a href="/docs" class="link-card">
                <h3>📚 Interactive Docs</h3>
                <p>Test tools directly in your browser</p>
            </a>
            <a href="/health" class="link-card">
                <h3>💚 Health Check</h3>
                <p>Server status and diagnostics</p>
            </a>
            <a href="/sse" class="link-card">
                <h3>🔌 MCP Endpoint</h3>
                <p>SSE transport for MCP clients</p>
            </a>
        </div>

        <div class="card">
            <h2>Retrieval features</h2>
            <div class="tools-grid">
                <div class="tool">
                    <h4>Hybrid Search</h4>
                    <p>Combines vector similarity with BM25 keyword matching</p>
                </div>
                <div class="tool">
                    <h4>Cross-Encoder Reranking</h4>
                    <p>Re-scores results for better precision</p>
                </div>
                <div class="tool">
                    <h4>MMR Diversity</h4>
                    <p>Reduces redundancy in search results</p>
                </div>
                <div class="tool">
                    <h4>Semantic Chunking</h4>
                    <p>Splits documents at semantic boundaries</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🛠 Available Tools</h2>
            <div class="tools-grid">
                <div class="tool">
                    <h4>search</h4>
                    <p>Search with citations, formatted for an LLM prompt</p>
                </div>
                <div class="tool">
                    <h4>list_documents</h4>
                    <p>List all documents in a session with metadata</p>
                </div>
                <div class="tool">
                    <h4>upload_document</h4>
                    <p>Upload and index a document (PDF, DOCX, TXT, etc.)</p>
                </div>
                <div class="tool">
                    <h4>delete_document (--tools all)</h4>
                    <p>Delete a document and all its indexed chunks</p>
                </div>
                <div class="tool">
                    <h4>get_document_info</h4>
                    <p>Get detailed metadata about a specific document</p>
                </div>
                <div class="tool">
                    <h4>get_document_chunks</h4>
                    <p>Retrieve indexed chunks with pagination</p>
                </div>
                <div class="tool">
                    <h4>get_system_status</h4>
                    <p>RAG system health, config, and enabled features</p>
                </div>
                <div class="tool">
                    <h4>reindex_document</h4>
                    <p>Re-process and re-embed a document</p>
                </div>
                <div class="tool">
                    <h4>clear_session (--tools all)</h4>
                    <p>Delete all documents in a session</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔗 Connection Info</h2>
            <p style="margin-bottom: 10px;">MCP clients can connect using:</p>
            <code>http://{host}:{port}/sse</code>
            <p style="margin-top: 15px; color: #888;">
                Lasagna MCP Server.
                Uses Milvus for vector storage and sentence-transformers for embeddings.
            </p>
        </div>
    </div>
</body>
</html>
"""


def _get_docs_html(host: str, port: int) -> str:
    """Generate interactive documentation page HTML."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lasagna - API Docs</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{
            color: #00d4ff;
            margin-bottom: 5px;
        }}
        .subtitle {{ color: #888; margin-bottom: 30px; }}
        .back-link {{
            color: #00d4ff;
            text-decoration: none;
            margin-bottom: 20px;
            display: inline-block;
        }}
        .tool-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .tool-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
            cursor: pointer;
        }}
        .tool-header h3 {{
            color: #b388ff;
        }}
        .tool-header .method {{
            background: #00c853;
            color: #fff;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
        }}
        .tool-desc {{
            color: #aaa;
            margin-bottom: 15px;
        }}
        .params {{
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .params h4 {{
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }}
        .param {{
            display: grid;
            grid-template-columns: 150px 100px 1fr;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .param:last-child {{ border-bottom: none; }}
        .param-name {{ color: #fff; font-weight: 500; }}
        .param-type {{ color: #ffd54f; font-size: 0.85rem; }}
        .param-desc {{ color: #999; font-size: 0.85rem; }}
        .param-required {{ color: #ff5252; font-size: 0.75rem; }}
        .test-form {{
            background: rgba(0,212,255,0.1);
            border-radius: 8px;
            padding: 15px;
        }}
        .test-form h4 {{
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .form-group {{
            margin-bottom: 12px;
        }}
        .form-group label {{
            display: block;
            color: #aaa;
            margin-bottom: 5px;
            font-size: 0.9rem;
        }}
        .form-group input, .form-group textarea {{
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-family: inherit;
        }}
        .form-group textarea {{
            min-height: 80px;
            resize: vertical;
        }}
        button {{
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            transition: opacity 0.3s;
        }}
        button:hover {{ opacity: 0.9; }}
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .result {{
            margin-top: 15px;
            background: rgba(0,0,0,0.4);
            border-radius: 8px;
            padding: 15px;
            display: none;
        }}
        .result.show {{ display: block; }}
        .result h4 {{
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .result pre {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .collapsible {{ display: none; }}
        .collapsible.open {{ display: block; }}
        .toggle-icon {{
            color: #00d4ff;
            font-size: 1.2rem;
            transition: transform 0.3s;
        }}
        .tool-header.open .toggle-icon {{
            transform: rotate(90deg);
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back to Home</a>
        <h1>📚 API Documentation</h1>
        <p class="subtitle">Interactive testing for Lasagna tools</p>

        <!-- search -->
        <div class="tool-section">
            <div class="tool-header" onclick="toggleTool(this)">
                <h3>search</h3>
                <div><span class="method">TOOL</span> <span class="toggle-icon">▶</span></div>
            </div>
            <div class="collapsible">
                <p class="tool-desc">Search with citations. Retrieval behavior is driven by session config.</p>
                <div class="params">
                    <h4>Parameters</h4>
                    <div class="param">
                        <span class="param-name">query <span class="param-required">*</span></span>
                        <span class="param-type">string</span>
                        <span class="param-desc">Natural-language search query</span>
                    </div>
                    <div class="param">
                        <span class="param-name">session_id <span class="param-required">*</span></span>
                        <span class="param-type">string</span>
                        <span class="param-desc">Session identifier</span>
                    </div>
                    <div class="param">
                        <span class="param-name">top_k</span>
                        <span class="param-type">integer</span>
                        <span class="param-desc">Optional result count override (default: session config)</span>
                    </div>
                    <div class="param">
                        <span class="param-name">document_ids</span>
                        <span class="param-type">string</span>
                        <span class="param-desc">Comma-separated doc IDs to filter</span>
                    </div>
                </div>
                <div class="test-form">
                    <h4>Try it out</h4>
                    <div class="form-group">
                        <label>query *</label>
                        <input type="text" id="search_query" placeholder="What is machine learning?">
                    </div>
                    <div class="form-group">
                        <label>session_id *</label>
                        <input type="text" id="search_session" placeholder="test-session-123">
                    </div>
                    <div class="form-group">
                        <label>top_k</label>
                        <input type="number" id="search_topk" value="5">
                    </div>
                    <button onclick="testSearch()">Execute</button>
                    <div class="result" id="search_result">
                        <h4>Response</h4>
                        <pre id="search_response"></pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- list_documents -->
        <div class="tool-section">
            <div class="tool-header" onclick="toggleTool(this)">
                <h3>list_documents</h3>
                <div><span class="method">TOOL</span> <span class="toggle-icon">▶</span></div>
            </div>
            <div class="collapsible">
                <p class="tool-desc">List all documents in the knowledge base for a session.</p>
                <div class="params">
                    <h4>Parameters</h4>
                    <div class="param">
                        <span class="param-name">session_id <span class="param-required">*</span></span>
                        <span class="param-type">string</span>
                        <span class="param-desc">Session identifier</span>
                    </div>
                </div>
                <div class="test-form">
                    <h4>Try it out</h4>
                    <div class="form-group">
                        <label>session_id *</label>
                        <input type="text" id="list_session" placeholder="test-session-123">
                    </div>
                    <button onclick="testList()">Execute</button>
                    <div class="result" id="list_result">
                        <h4>Response</h4>
                        <pre id="list_response"></pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- get_system_status -->
        <div class="tool-section">
            <div class="tool-header" onclick="toggleTool(this)">
                <h3>get_system_status</h3>
                <div><span class="method">TOOL</span> <span class="toggle-icon">▶</span></div>
            </div>
            <div class="collapsible">
                <p class="tool-desc">Get RAG system health and statistics. No parameters required.</p>
                <div class="test-form">
                    <h4>Try it out</h4>
                    <button onclick="testStatus()">Execute</button>
                    <div class="result" id="status_result">
                        <h4>Response</h4>
                        <pre id="status_response"></pre>
                    </div>
                </div>
            </div>
        </div>

        <!-- get_document_info -->
        <div class="tool-section">
            <div class="tool-header" onclick="toggleTool(this)">
                <h3>get_document_info</h3>
                <div><span class="method">TOOL</span> <span class="toggle-icon">▶</span></div>
            </div>
            <div class="collapsible">
                <p class="tool-desc">Get detailed information about a specific document.</p>
                <div class="params">
                    <h4>Parameters</h4>
                    <div class="param">
                        <span class="param-name">document_id <span class="param-required">*</span></span>
                        <span class="param-type">string</span>
                        <span class="param-desc">The document ID to query</span>
                    </div>
                    <div class="param">
                        <span class="param-name">session_id <span class="param-required">*</span></span>
                        <span class="param-type">string</span>
                        <span class="param-desc">Session identifier</span>
                    </div>
                </div>
                <div class="test-form">
                    <h4>Try it out</h4>
                    <div class="form-group">
                        <label>document_id *</label>
                        <input type="text" id="info_docid" placeholder="doc_abc123">
                    </div>
                    <div class="form-group">
                        <label>session_id *</label>
                        <input type="text" id="info_session" placeholder="test-session-123">
                    </div>
                    <button onclick="testInfo()">Execute</button>
                    <div class="result" id="info_result">
                        <h4>Response</h4>
                        <pre id="info_response"></pre>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        const API_BASE = 'http://{host}:{port}';

        function toggleTool(header) {{
            header.classList.toggle('open');
            const content = header.nextElementSibling;
            content.classList.toggle('open');
        }}

        async function callTool(toolName, args) {{
            try {{
                const response = await fetch(API_BASE + '/api/call', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ tool: toolName, arguments: args }})
                }});
                return await response.json();
            }} catch (e) {{
                return {{ error: e.message }};
            }}
        }}

        async function testSearch() {{
            const args = {{
                query: document.getElementById('search_query').value,
                session_id: document.getElementById('search_session').value,
            }};
            const topK = parseInt(document.getElementById('search_topk').value);
            if (topK) args.top_k = topK;
            const result = await callTool('search', args);
            document.getElementById('search_response').textContent = JSON.stringify(result, null, 2);
            document.getElementById('search_result').classList.add('show');
        }}

        async function testList() {{
            const result = await callTool('list_documents', {{
                session_id: document.getElementById('list_session').value
            }});
            document.getElementById('list_response').textContent = JSON.stringify(result, null, 2);
            document.getElementById('list_result').classList.add('show');
        }}

        async function testStatus() {{
            const result = await callTool('get_system_status', {{}});
            document.getElementById('status_response').textContent = JSON.stringify(result, null, 2);
            document.getElementById('status_result').classList.add('show');
        }}

        async function testInfo() {{
            const result = await callTool('get_document_info', {{
                document_id: document.getElementById('info_docid').value,
                session_id: document.getElementById('info_session').value
            }});
            document.getElementById('info_response').textContent = JSON.stringify(result, null, 2);
            document.getElementById('info_result').classList.add('show');
        }}
    </script>
</body>
</html>
"""


def _docs_api_enabled() -> bool:
    """Whether the unauthenticated /api/call tester is switched on.

    Read at request time rather than registration time so that embedding the
    server in another app (``api/rest/app.py`` mounts it on 0.0.0.0) cannot
    capture a stale value, and so tests can toggle it.
    """
    return os.environ.get("LASAGNA_ENABLE_DOCS_API", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _drain_body(request: Request, limit: int = 1 << 20) -> None:
    """Read and drop a request's body before refusing it.

    Answering before the body is read lets the server close the connection with
    the body still arriving, and the kernel then resets it: the client reads
    "connection reset" instead of the refusal and its reason. Capped, so a
    refusal never buffers an arbitrary upload.
    """
    seen = 0
    async for chunk in request.stream():
        seen += len(chunk)
        if seen > limit:
            break


def _register_custom_routes(mcp_instance: FastMCP, host: str, port: int):
    """Register custom HTTP routes for welcome page, docs, and health check."""

    @mcp_instance.custom_route("/", methods=["GET"])
    async def welcome_page(request: Request) -> HTMLResponse:
        """Welcome page with server info."""
        return HTMLResponse(_get_welcome_html(host, port))

    @mcp_instance.custom_route("/docs", methods=["GET"])
    async def docs_page(request: Request) -> HTMLResponse:
        """Interactive API documentation."""
        return HTMLResponse(_get_docs_html(host, port))

    @mcp_instance.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        """Liveness: the process is up. No dependency is checked here.

        The get_system_status tool reports each component, as the REST API's
        /api/v1/health does; "healthy" here claimed what nothing had checked.
        """
        try:
            status = {
                "status": "alive",
                "server": "rag_knowledge_mcp",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoints": {
                    "welcome": f"http://{host}:{port}/",
                    "docs": f"http://{host}:{port}/docs",
                    "sse": f"http://{host}:{port}/sse",
                    "health": f"http://{host}:{port}/health",
                },
            }
            return JSONResponse(status)
        except Exception as e:
            return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=500)

    @mcp_instance.custom_route("/api/call", methods=["POST"])
    async def call_tool_api(request: Request) -> JSONResponse:
        """Invoke any tool over plain HTTP, for the in-browser /docs tester.

        **Unauthenticated, and off unless explicitly enabled.** It reaches every
        registered tool with arbitrary arguments, including `delete_document`,
        `bulk_delete_session` and `reset_rag_config`.

        It used to be registered unconditionally while the MCP README
        recommended `--host 0.0.0.0` as the "production" command -- so following
        the documentation exposed unauthenticated document deletion on every
        interface. The endpoint exists only to power the `/docs` page's "try it"
        button, so requiring an opt-in costs nothing real.
        """
        if not _docs_api_enabled():
            await _drain_body(request)
            return JSONResponse(
                {
                    "error": (
                        "The /docs tester endpoint is disabled. It can invoke any "
                        "tool without authentication, so it must be enabled "
                        "deliberately: pass --enable-docs-api, or set "
                        "LASAGNA_ENABLE_DOCS_API=1. Do not enable it on an "
                        "interface reachable from an untrusted network."
                    )
                },
                status_code=404,
            )
        # Off unless enabled is not enough: a page the operator visits could
        # drive it with a text/plain POST (no preflight) or by DNS rebinding.
        # Ran `clear_session` from a foreign origin before this.
        from mcp.server.transport_security import TransportSecurityMiddleware

        refused = await TransportSecurityMiddleware(
            transport_guard.loopback_browser_guard()
        ).validate_request(request, is_post=True)
        if refused is not None:
            await _drain_body(request)
            return refused
        try:
            body = await request.json()
            tool_name = body.get("tool")
            arguments = body.get("arguments", {})

            if not tool_name:
                return JSONResponse({"error": "Missing 'tool' parameter"}, status_code=400)

            # Names only, at DEBUG: `arguments` carries the query of a search
            # tool, a session_token and the admin_token, and the result is
            # the text the tool returned; logged at INFO, they landed in the
            # container log.
            logger.debug(
                f"API call: tool={tool_name}, argument names="
                f"{sorted(arguments) if isinstance(arguments, dict) else []}"
            )

            # Call the tool directly
            result = await mcp_instance.call_tool(tool_name, arguments)
            logger.debug(f"Tool result type: {type(result).__name__}")

            # Handle different result types - extract text from ContentBlocks
            if isinstance(result, list):
                # It's a list of ContentBlocks
                result_text = ""
                for block in result:
                    logger.debug(f"Block type: {type(block)}, attrs: {dir(block)}")
                    if hasattr(block, "text"):
                        result_text += block.text
                    elif isinstance(block, dict) and "text" in block:
                        result_text += block["text"]
                    else:
                        # Try to convert to string
                        result_text += str(block)
                logger.debug(f"Tool {tool_name} returned {len(result_text)} characters")
                return JSONResponse({"result": result_text})
            elif isinstance(result, dict):
                return JSONResponse({"result": result})
            elif isinstance(result, str):
                return JSONResponse({"result": result})
            else:
                return JSONResponse({"result": str(result)})

        except Exception as e:
            logger.exception(f"API call failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


#: Every tool, in the order clients list them.
_TOOLS = (
    # Search
    search,
    list_documents,
    upload_document,
    delete_document,
    get_document_status,
    get_document_info,
    get_document_chunks,
    get_system_status,
    rebuild_collection,
    reindex_document,
    clear_session,
    get_document_context,
    # Focus / scoping
    set_focused_documents,
    get_focused_documents,
    # Profile management
    list_profiles,
    get_active_profile,
    set_active_profile,
    get_profile_stats,
    # Runtime config
    get_rag_config,
    patch_rag_config,
    save_rag_profile,
    delete_rag_profile,
    reset_rag_config,
    # Lifecycle (soft-delete, restore, bulk-delete, GC status, policy management)
    archive_document,
    restore_document,
    bulk_delete_session,
    get_gc_status,
    get_lifecycle_policy,
    set_lifecycle_policy,
)


#: A reST field line in a tool docstring: ``:param name: text`` or ``:returns: text``.
_DOC_FIELD = re.compile(r"^:(param|returns?|raises?)(?:\s+(\w+))?\s*:\s*(.*)$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s")


def _split_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """A tool's description as plain text, and each parameter's own text.

    Clients showed the docstring as written: reST ``:param:`` fields and
    double backticks, about 26 KB across the tools with ``--tools all``. The
    prose becomes the description (paragraphs unwrapped, lists kept), the
    ``:returns:`` field its last line, and each ``:param:`` the description of
    that parameter in the input schema.
    """
    text = inspect.cleandoc(doc or "")
    prose: List[str] = []
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        match = _DOC_FIELD.match(line.strip())
        if match:
            kind, name, rest = match.groups()
            current = name if kind == "param" else "_returns" if kind.startswith("return") else None
            if current:
                fields[current] = rest.strip()
            continue
        if current and line[:1].isspace() and line.strip():
            fields[current] += " " + line.strip()
            continue
        current = None
        prose.append(line)

    paragraphs = []
    for block in "\n".join(prose).split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if any(_LIST_ITEM.match(line) for line in lines):
            paragraphs.append("\n".join(lines))
        else:
            paragraphs.append(" ".join(lines))
    returns = fields.pop("_returns", None)
    if returns:
        paragraphs.append(f"Returns: {returns}")

    def plain(value: str) -> str:
        return value.replace("``", "`")

    return plain("\n\n".join(paragraphs)), {k: plain(v) for k, v in fields.items()}


#: Said of every tool's ``session_id``: none is required, and `_resolve_session`
#: is how each tool picks the session when the call names none.
_SESSION_DEFAULT = (
    "Optional: without it, the server's session (RAG_SESSION_ID, or "
    "lasagna serve --mcp -s <session>)."
)


def _describe_parameters(server: FastMCP, tool_name: str, parameters: Dict[str, str]) -> None:
    """Put each parameter's text in the tool's input schema, where clients read it."""
    tool = server._tool_manager.get_tool(tool_name)
    properties = (tool.parameters or {}).get("properties", {}) if tool else {}
    if "session_id" in properties:
        text = parameters.get("session_id", "Session identifier.")
        if not text.startswith("Unused"):
            parameters = {**parameters, "session_id": f"{text} {_SESSION_DEFAULT}"}
    for name, text in parameters.items():
        if name in properties:
            properties[name]["description"] = text


def create_mcp_server(
    host: str = "0.0.0.0",
    port: int = 8005,
    transport_security: Any = None,
    tools: Optional[str] = None,
) -> FastMCP:
    """Create a configured MCP server instance without starting it.

    Can be used standalone (via ``main()``) or mounted inside a FastAPI app
    for single-process deployments (e.g. ARM64 with Milvus Lite).
    ``transport_security`` is FastMCP's Host/Origin check for the SSE
    endpoints; FastMCP turns it on by itself only for a loopback ``host``.

    ``tools`` is ``"safe"`` or ``"all"``; ``None`` reads ``LASAGNA_MCP_TOOLS``
    and falls back to ``"safe"``. The safe set leaves out ``DESTRUCTIVE_TOOLS``
    entirely, so ``call_tool`` cannot reach them either.
    """
    tool_set = resolve_tool_set(tools)
    # FastMCP sets up the stdlib root logger (a rich handler at INFO) in its
    # constructor, so "Processing request of type CallToolRequest" and every
    # library's INFO line went to a stdio client's stderr, and into the log of
    # the REST app it was mounted in. The root logger is put back as it was.
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    server = _LasagnaMCP("lasagna", host=host, port=port, transport_security=transport_security)
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    for fn in _TOOLS:
        if tool_set == "safe" and fn.__name__ in DESTRUCTIVE_TOOLS:
            continue
        description, parameters = _split_docstring(fn.__doc__)
        server.add_tool(_flag_failures(fn), description=description)
        _describe_parameters(server, fn.__name__, parameters)

    # Custom HTTP routes (welcome, docs, health, api/call)
    _register_custom_routes(server, host, port)

    # Drift guard: warn if a tool registered via @mcp.tool() is not in _TOOLS.
    # NOTE: Accessing private API (_tool_manager._tools) because the public
    # get_tools() is async but this factory is sync. Verified against
    # fastmcp 2.x. If this breaks on upgrade, make this factory async
    # or remove the drift guard.
    _module_tools = set(mcp._tool_manager._tools.keys())
    _missing = _module_tools - {fn.__name__ for fn in _TOOLS}
    if _missing:
        logger.warning(
            f"create_mcp_server() is missing {len(_missing)} tool(s) "
            f"registered via @mcp.tool(): {sorted(_missing)}"
        )

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _apply_path_flags(db_dir: Optional[str], upload_dir: Optional[str]) -> bool:
    """Make ``--db-dir`` and ``--upload-dir`` win over the environment.

    Both names are written: ``LASAGNA_`` wins over ``RAG_`` when both are set,
    so writing only the ``RAG_`` name let an ambient ``LASAGNA_DB_DIR``
    override the flag. Returns whether anything was set.
    """
    if db_dir:
        os.environ["LASAGNA_DB_DIR"] = os.environ["RAG_DB_DIR"] = db_dir
    if upload_dir:
        os.environ["LASAGNA_UPLOAD_DIR"] = os.environ["RAG_UPLOAD_DIR"] = upload_dir
    return bool(db_dir or upload_dir)


def main():
    """Run the Lasagna MCP server."""
    global mcp, _eager_init, _owns_logging, _pinned_level
    import argparse

    from lasagna._quiet import quiet_third_party_output

    quiet_third_party_output()

    # Enable logging ownership — we're the main process. Warnings only until
    # the arguments say otherwise.
    logger.enable("lasagna")
    if not _owns_logging:
        _owns_logging = True
        _configure_rag_logging("WARNING")

    parser = argparse.ArgumentParser(
        prog="lasagna-mcp",
        description=(
            "Lasagna MCP server: search and manage your documents from an MCP client. "
            "A tool acts on the session its session_id argument names, else on "
            "RAG_SESSION_ID (set it in the client's env block, e.g. "
            '"env": {"RAG_SESSION_ID": "my-corpus"}).'
        ),
    )
    from lasagna.sdk._version import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"lasagna-mcp {__version__}",
        help="Print the installed version and exit.",
    )
    # stdio by default: that is how Claude Code, Claude Desktop and Cursor
    # launch a server, and it needs no port, no bind decision and no auth.
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="stdio (default) for a client that launches this process; sse to serve on a port",
    )
    parser.add_argument("--host", default=None, help="Host to bind (sse only, default 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (sse only, default 8005)"
    )
    parser.add_argument(
        "--tools",
        default=None,
        choices=TOOL_SETS,
        help=(
            f"safe (default) leaves out the {len(DESTRUCTIVE_TOOLS)} tools that delete data "
            f"or reset configuration; all serves every tool. Env: {TOOLS_ENV}"
        ),
    )
    parser.add_argument(
        "--lazy-init",
        action="store_true",
        help="Defer RAG backend init to the first tool call (the default on stdio)",
    )
    parser.add_argument(
        "--eager-init",
        action="store_true",
        help="Load the backend before serving (the default on sse)",
    )
    parser.add_argument(
        "--enable-docs-api",
        action="store_true",
        help=(
            "Enable the unauthenticated /api/call tester behind the /docs page. "
            "It can invoke any tool, including deletion. Localhost only."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.upper,
        choices=list(LOG_LEVELS),
        help="Log level on stderr (default: WARNING on stdio, INFO on sse; or LASAGNA_LOG_LEVEL "
        "or RAG_LOG_LEVEL)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Override settings TOML path (same as RAG_SETTINGS_FILE env var)",
    )
    parser.add_argument(
        "--db-dir",
        default=None,
        help="Database directory (default: LASAGNA_DB_DIR or RAG_DB_DIR, else ~/.lasagna/dbs)",
    )
    parser.add_argument(
        "--upload-dir",
        default=None,
        help="Upload directory (default: LASAGNA_UPLOAD_DIR or RAG_UPLOAD_DIR, else ~/.lasagna/uploads)",
    )
    parser.add_argument(
        "--session-backend",
        # Persistent by default: `lasagna configure --profile ...` saves the
        # profile to disk, and an in-memory backend in this new process never
        # saw it, so search ran the defaults (agentic, LLM-only) while every
        # tool reported the saved profile as active. The default is applied
        # below without a reset; a default here made every start reset.
        default=None,
        choices=["memory", "persistent"],
        help=(
            "Session backend (default: RAG_SESSION_BACKEND, else persistent, so profiles "
            "saved by the CLI apply)"
        ),
    )
    args = parser.parse_args()

    # The default transport used to be sse, so `lasagna-mcp --port 8005` is in
    # scripts and notes. Serving stdio instead would sit waiting on stdin and
    # look like a hang; say what changed.
    if args.transport == "stdio" and (args.host is not None or args.port is not None):
        parser.error(
            "--host and --port apply only to --transport sse; the default transport "
            "is stdio. Add --transport sse to serve on a port."
        )
    args.host = args.host or "127.0.0.1"
    args.port = args.port or 8005
    try:
        tool_set = resolve_tool_set(args.tools)
    except ValueError as exc:
        parser.error(str(exc))

    # Apply overrides — env vars feed auto-config in the providers layer.
    # NOTE: Module-level imports may have already triggered auto-config with
    # default paths, so we must reset providers + config after setting env vars.
    #
    # Only a flag needs that. The default session backend is an environment
    # default read when the providers configure themselves; a reset on every
    # start imported the backend module, and with it torch, before the first
    # message: about 7 s before an MCP client got its tool list.
    _needs_reset = False
    if _apply_path_flags(args.db_dir, args.upload_dir):
        _needs_reset = True
    if args.session_backend:
        os.environ["RAG_SESSION_BACKEND"] = args.session_backend
        _needs_reset = True
    else:
        os.environ.setdefault("RAG_SESSION_BACKEND", "persistent")
    if args.config_file:
        os.environ["RAG_SETTINGS_FILE"] = args.config_file
        _needs_reset = True

    # Re-initialize providers and config with the CLI-provided values. The
    # backend exists only once its module is imported, so there is nothing to
    # reset (and nothing worth importing torch for) before that.
    if _needs_reset:
        from lasagna.providers import reset_providers
        from lasagna.core.config import reset_rag_config

        reset_providers()
        reset_rag_config()
        if "lasagna.api.rag_backend" in sys.modules:
            sys.modules["lasagna.api.rag_backend"].reset_instance()

    _pinned_level = (
        args.log_level
        or os.environ.get("LASAGNA_LOG_LEVEL")
        or os.environ.get("RAG_LOG_LEVEL")
        or ("WARNING" if args.transport == "stdio" else "INFO")
    ).upper()
    _configure_rag_logging(_pinned_level)

    # Stdio defaults to lazy. An MCP client sends `initialize` as soon as the
    # process starts and gives up long before a cold model download finishes,
    # so loading the backend before `mcp.run()` means the handshake never
    # completes -- and with no client-side progress it reads as a hang, not a
    # download. SSE is a long-lived server where warming up first is the right
    # trade, so it stays eager.
    if args.eager_init and args.lazy_init:
        parser.error("--eager-init and --lazy-init are mutually exclusive")

    if args.enable_docs_api:
        os.environ["LASAGNA_ENABLE_DOCS_API"] = "1"
    # Binding a tool-invocation endpoint to every interface is the combination
    # the README used to recommend as "production". Refuse it rather than warn:
    # a warning in a server's startup log is not read by the person who will be
    # affected by it.
    if (
        _docs_api_enabled()
        and args.transport == "sse"
        and args.host
        not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
    ):
        parser.error(
            f"--enable-docs-api exposes an unauthenticated endpoint that can "
            f"invoke any tool, and --host {args.host} would publish it on every "
            f"interface. Bind to 127.0.0.1, or drop --enable-docs-api."
        )
    # Before anything can serve a tool. stdio means the client launched this
    # process and owns its pipes, so the process boundary is the credential;
    # anything else is reachable by someone who did not.
    transport_guard.set_transport(args.transport)
    if args.transport != "stdio":
        try:
            require_auth_or_refuse_to_start(args.host, surface="MCP server")
        except UnsafeBindError as exc:
            parser.error(str(exc))
        warn_once_if_unauthenticated(args.host)

    if args.eager_init:
        _eager_init = True
    elif args.lazy_init:
        _eager_init = False
    else:
        _eager_init = args.transport != "stdio"

    mcp = create_mcp_server(
        host=args.host,
        port=args.port,
        tools=tool_set,
        transport_security=transport_guard.http_transport_security(),
    )

    logger.info("=" * 60)
    logger.info("Lasagna MCP Server")
    logger.info("=" * 60)
    logger.info(f"Transport: {args.transport}")
    if args.transport == "sse":
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Welcome:  http://{args.host}:{args.port}/")
        logger.info(f"Docs:     http://{args.host}:{args.port}/docs")
        logger.info(f"Health:   http://{args.host}:{args.port}/health")
        logger.info(f"SSE:      http://{args.host}:{args.port}/sse")
    # What tools/list returns: it said 25 while a client was shown 23.
    if tool_set == "safe":
        logger.info(
            f"Tools:    {mcp.listed_tool_count()} (safe set; --tools all adds "
            f"{', '.join(sorted(DESTRUCTIVE_TOOLS))})"
        )
    else:
        logger.info(f"Tools:    {mcp.listed_tool_count()} (all)")
    logger.info("=" * 60)

    if _eager_init:
        logger.info("Initializing RAG backend at startup (use --lazy-init to defer)...")
        try:
            _initialize_backend()
        except _StoreHeld as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
    else:
        from lasagna.core.setup_text import profile_download_size

        logger.info(
            "RAG backend will be initialized on the first tool call. A session whose "
            "embedding model is not downloaded is refused with the fix rather than "
            "downloaded inside a tool call (the speed profile, `lasagna configure "
            f"--profile speed -s <session>`, uses a {profile_download_size('speed')} model)."
        )

    from lasagna.core.model_downloads import refuse_downloads_in_requests

    with refuse_downloads_in_requests():
        if args.transport == "stdio":
            logger.info("Starting in STDIO mode...")
            mcp.run(transport="stdio")
        else:
            logger.info(f"Starting SSE server on {args.host}:{args.port}...")
            mcp.run(transport="sse")


if __name__ == "__main__":
    main()
