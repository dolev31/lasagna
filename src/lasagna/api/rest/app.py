"""FastAPI application factory for the Lasagna REST API.

Creates the FastAPI app with:
- Versioned API routes (``/api/v1/``)
- CORS middleware
- Request-ID middleware (the id is on every log line of the request)
- A request body size limit, enforced before the body is read
- Request timing middleware
- Lifespan management (backend init/shutdown)
- Prometheus ``/metrics`` endpoint (when enabled)
- Optional embedded MCP server (``RAG_EMBED_MCP=true``)
- Optional configurator UI (``RAG_CONFIGURATOR=true``)

Usage::

    # Programmatic
    from lasagna.api.rest import create_app
    app = create_app()

    # CLI
    uvicorn lasagna.api.rest.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import math
import os
import re
import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Paths exempt from rate limiting and timing metrics.
#
# Infra endpoints only. `/mcp` and `/configurator` used to be here, and both
# carry destructive operations: the embedded MCP surface includes
# `bulk_delete_session` and `clear_session`, and the configurator exposes
# profile deletion and a reindex that deletes every document in the session and
# re-ingests. Exempting them meant the most expensive and least authenticated
# routes in the app were also the only unthrottled ones.
#
# The reasoning that (correctly) kept `/api/v1/sessions` out of this list
# applies at least as strongly to both.
_EXEMPT_PREFIXES = (
    "/health",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    # NOTE: /api/v1/sessions is intentionally NOT exempt — session creation
    # must be rate-limited to prevent DoS via unlimited token generation.
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise the RAG backend on startup, clean up on shutdown."""
    # Serving: turn on the package logs the library keeps off by default.
    logger.enable("lasagna")
    from lasagna.api.rest.limits import apply_log_format

    apply_log_format()
    logger.info("RAG REST API starting: initializing backend...")
    try:
        from lasagna.api.rag_backend import get_rag_backend

        _backend = get_rag_backend()
        logger.info("RAG backend ready")
    except Exception as exc:
        from lasagna.api.rag_backend import _is_data_dir_locked

        if _is_data_dir_locked(exc):
            # Another process holds the store. Each request answers 503 with
            # the sentence that says so and how to fix it, and retries, so a
            # lock held for a moment (a `lasagna status` run) costs nothing.
            logger.debug(f"RAG backend not initialized: {exc}")
        else:
            logger.error(f"RAG backend init failed (will retry on first request): {exc}")

    # Log service configuration summary
    try:
        from lasagna.core.config import rag_config

        logger.info("=" * 60)
        logger.info("Lasagna")
        logger.info(f"  Database:       {rag_config.database_backend}")
        if rag_config.database_backend == "postgresql":
            dsn = rag_config.database_dsn
            safe_dsn = dsn.split("@")[-1] if "@" in dsn else dsn
            logger.info(f"  PostgreSQL:     {safe_dsn}")
        logger.info(f"  Vector backend: {rag_config.vector_backend}")
        logger.info(f"  Cache backend:  {rag_config.cache_backend}")
        if rag_config.cache_backend == "redis":
            from lasagna.core.provenance import redact_url

            logger.info(f"  Redis:          {redact_url(rag_config.redis_url)}")
        logger.info(
            f"  Embedding:      {rag_config.embedding_model} (dim={rag_config.embedding_dim})"
        )
        # Not an address: the app cannot see where it was bound, and a fixed
        # "http://0.0.0.0:8000" was wrong for every other bind. `lasagna serve`
        # prints the real one.
        logger.info("  Health:         /api/v1/health")
        logger.info("=" * 60)
    except Exception as exc:
        logger.debug(f"Could not log config summary: {exc}")

    # A request never downloads a model: one that is not on disk answers 503
    # with the fix (lasagna.core.model_downloads).
    from lasagna.core.model_downloads import refuse_downloads_in_requests

    with refuse_downloads_in_requests():
        yield  # app is running

    # `lasagna serve` recorded itself for other commands; uvicorn re-raises the
    # stop signal after shutdown, so the command's own cleanup may never run.
    from lasagna.core import local_server

    local_server.withdraw()
    logger.info("RAG REST API shutting down: releasing resources...")
    try:
        from lasagna.api.rag_backend import close_backend

        close_backend()
    except Exception as exc:
        logger.warning(f"RAG backend cleanup on shutdown failed: {exc}")


# ---------------------------------------------------------------------------
# Middleware helpers
# ---------------------------------------------------------------------------


def _one_response(app):
    """Pass on one response per request, dropping any that follows it.

    The MCP SDK's SSE endpoint returns an empty response after its event stream
    has ended. Behind this app's HTTP middleware that second response raised
    "Unexpected message: http.response.start" with a 40-line traceback each
    time an MCP client disconnected from ``lasagna serve --mcp``.
    """

    async def wrapped(scope, receive, send):
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return
        started = discarding = False

        async def guarded(message):
            nonlocal started, discarding
            if message["type"] == "http.response.start":
                if started:
                    discarding = True
                    return
                started = True
            if not discarding:
                await send(message)

        await app(scope, receive, guarded)

    return wrapped


#: A client-supplied X-Request-ID is kept only if it looks like an id; anything
#: else would be copied into every log line of the request.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


async def _request_id_middleware(request: Request, call_next):
    """Attach a request ID to the request, its response, and every log line.

    ``logger.contextualize`` puts it in ``record["extra"]``, so a JSON log line
    (``LASAGNA_LOG_FORMAT=json``) carries it and one request's lines can be
    pulled out of a busy server's log.
    """
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    with logger.contextualize(request_id=request_id):
        response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def _timing_middleware(request: Request, call_next):
    """Record request duration in the ``X-Response-Time`` header."""
    # Skip timing for exempt paths (SSE connections would skew metrics)
    path = request.url.path
    if path in ("/", "/metrics") or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
        return await call_next(request)

    start = perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (perf_counter() - start) * 1000
    response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

    # Use route template (e.g. "/api/v1/documents/{document_id}") instead
    # of the actual path to avoid Prometheus label cardinality explosion. A
    # request no route matched used to be labelled with its raw path, so a
    # scanner probing random URLs created one time series per URL.
    route = getattr(request.scope.get("route"), "path", None) or "unmatched"

    from lasagna.utils.monitoring import METRICS

    METRICS.histogram(
        "rag.http.request_duration_ms",
        elapsed_ms,
        tags={
            "method": request.method,
            "path": route,
            # Without it a dashboard could not tell fast errors from fast
            # successes: a 503 storm read as a latency improvement.
            "status": str(response.status_code),
        },
    )

    # Feed latency to rate limiter load estimator
    from lasagna.core.rate_limiter import get_rate_limiter

    get_rate_limiter().record_request_latency(elapsed_ms)

    return response


def _extract_session_id(request: Request) -> str:
    """Extract session_id from request, checking token first.

    Decodes the session token (if present) **once** and caches the result
    on ``request.state`` so that downstream FastAPI dependencies
    (``get_session_id``, ``get_user_id``) can reuse it without a second
    HMAC verification.
    """
    token = request.headers.get("X-Session-Token")
    if token:
        from lasagna.api.rest.session_token import decode_session_claims

        claims = decode_session_claims(token)
        if claims:
            # Cache decoded (session_id, user_id) for later dependencies
            request.state.decoded_session_token = (claims["sid"], claims["sub"])
            request.state.session_principal = claims.get("prn")
            return claims["sid"]

    from lasagna.api.rest.session_token import extract_session_id

    try:
        return extract_session_id(
            session_token=None,  # already handled above
            session_id_header=request.headers.get("X-Session-ID"),
            session_id_query=request.query_params.get("session_id"),
        )
    except ValueError:
        # Invalid session_id (too long) — fall back to "default" for
        # rate-limiter purposes; the endpoint dependency will reject it.
        return "default"


def _rate_limit_key(request: Request) -> str:
    """The bucket a request is counted against.

    With auth on, only a verified token names a session. Keying on the
    asserted ``X-Session-ID`` let an unauthenticated client exhaust a chosen
    session's bucket (and, through the shared "default" bucket, block every
    new sign-up); an unverified request is counted against its client address
    instead. With auth off the server is loopback-only and the asserted id is
    as good as any.

    A token minted with an API key is counted against that key's principal,
    not its session: one key can mint any number of sessions, so a per-session
    bucket gave each new session a fresh allowance.
    """
    from lasagna.api.rest.auth_mode import require_auth

    session_id = _extract_session_id(request)
    if not require_auth():
        return session_id
    if getattr(request.state, "decoded_session_token", None):
        principal = getattr(request.state, "session_principal", None)
        return f"prn:{principal}" if principal else f"sid:{session_id}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


async def _rate_limit_middleware(request: Request, call_next):
    """Enforce adaptive per-tenant rate limiting.

    Extracts session_id from ``X-Session-ID`` header or ``session_id``
    query parameter.  Returns 429 with ``Retry-After`` when rate-limited.
    Health and docs endpoints are exempt.
    """
    # Skip rate limiting for health probes, docs, and infra endpoints
    path = request.url.path
    if path in ("/", "/metrics") or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
        return await call_next(request)

    from lasagna.core.rate_limiter import get_rate_limiter

    rl = get_rate_limiter()

    if not rl.is_enabled:
        return await call_next(request)

    route = getattr(request.scope.get("route"), "path", path)
    allowed, retry_after = rl.try_acquire(_rate_limit_key(request), route)

    if not allowed:
        from lasagna.utils.monitoring import METRICS

        METRICS.counter("rag.http.rate_limited", 1, tags={"path": route})
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry later."},
            # Rounded up: int() truncated 0.4 s to "0" while still refusing.
            headers={"Retry-After": str(max(1, math.ceil(retry_after or 1)))},
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _package_version() -> str:
    from lasagna.sdk._version import __version__

    return __version__


def _require_a_shared_signing_secret() -> None:
    """Refuse to build an authenticating app whose tokens cannot be verified.

    With ``RAG_JWT_SECRET`` unset, `session_token._get_secret` generates one per
    *process*. The shipped k8s manifests run ``replicas: 2`` behind one Service
    and gunicorn adds four workers on top, so a token minted by one worker fails
    on the next -- intermittent 401s that look like flakiness rather than the
    missing configuration they are.

    Checked by asking for the secret rather than by re-reading the environment,
    so this cannot drift from what signing actually does.
    """
    from lasagna.api.rest.session_token import MissingSigningSecret, _get_secret

    try:
        _get_secret()
    except MissingSigningSecret as exc:
        raise ValueError(str(exc)) from exc


def create_app(
    title: str = "Lasagna API",
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: OpenAPI title.
        cors_origins: Allowed CORS origins.  Defaults to ``["*"]`` for
            development; restrict in production.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title=title,
        description=(
            "REST API for Lasagna. "
            "Upload documents, search them, retrieve citations, "
            "and manage configuration profiles."
        ),
        version=_package_version(),
        lifespan=_lifespan,
    )

    # -- CORS ---------------------------------------------------------------
    # Explicit origins from RAG_CORS_ORIGINS (comma-separated), then the
    # `cors_origins` parameter, then **no cross-origin access**.
    #
    # The default used to be `["*"]` with `allow_methods=["*"]` and
    # `allow_headers=["*"]`. Because this API authenticates with a custom header
    # rather than a cookie, `allow_credentials=False` bought nothing: the
    # preflight for `X-Session-ID` succeeded, so any web page a developer
    # visited could issue `DELETE /api/v1/documents/{id}` against their local
    # instance and the browser would carry it out.
    #
    # An empty list denies cross-origin requests and leaves same-origin ones
    # untouched, so the bundled configurator UI -- served from this very app --
    # is unaffected. A separate frontend dev server is exactly the case that
    # should name its origin.
    #
    # WHAT THIS DOES NOT DO. CORS governs whether the browser hands the
    # RESPONSE to the calling page; it does not stop the request executing. Only
    # requests that need a preflight are actually blocked, and a "simple"
    # request needs none: `POST /api/v1/documents` is multipart/form-data and
    # `POST /api/v1/documents/{id}/archive` has no body, so both still reach the
    # server cross-origin and take effect, with the attacker simply unable to
    # read the reply. The DELETE named above genuinely is blocked, because
    # `DELETE` always preflights.
    #
    # What closes that is RAG_REQUIRE_AUTH, not this setting. With it on,
    # identity must arrive in `X-Session-Token` -- a custom header, which forces
    # a preflight, which this denies -- and `?session_id=` no longer works
    # because an unverified session id is refused outright, `default` included.
    # With auth off, a cross-origin blind write remains possible, which is one
    # more reason the off state is only defensible on loopback. SECURITY.md
    # says so explicitly.
    # Fail here rather than per-request. The shipped Dockerfile runs gunicorn
    # against this app directly and never goes through `lasagna serve`, so
    # `require_auth_or_refuse_to_start` does not run in the deployment that
    # needs it most -- this constructor is the one chokepoint every way of
    # serving the app passes through.
    _require_a_shared_signing_secret()

    # Invalid limits or log format fail here, at startup, naming the setting.
    from lasagna.api.rest import limits

    limits.validate()

    _cors_env = os.environ.get("RAG_CORS_ORIGINS", "")
    _cors = (
        [o.strip() for o in _cors_env.split(",") if o.strip()]
        if _cors_env
        else list(cors_origins or [])
    )

    if _cors == ["*"]:
        from lasagna.api.rest.auth_mode import require_auth

        if require_auth():
            raise ValueError(
                "RAG_CORS_ORIGINS='*' with RAG_REQUIRE_AUTH enabled: a wildcard "
                "origin lets any site drive authenticated requests from a "
                "browser. Name the origins you actually serve."
            )
        logger.warning(
            "CORS allow_origins=['*']: any web page can call this API from a "
            "browser. Restrict via RAG_CORS_ORIGINS before exposing it."
        )
    elif not _cors:
        logger.debug("CORS: cross-origin requests denied (set RAG_CORS_ORIGINS to allow)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors,
        allow_credentials=(bool(_cors) and _cors != ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Custom middleware (outermost = first executed) ----------------------
    # Order: request-id → body limit → rate-limit → timing → handler
    # Rate limiting runs before timing so rejected requests don't skew latency.
    # The body limit runs before anything reads the body: by the time a route
    # with an UploadFile parameter runs, Starlette has already spooled the
    # whole multipart body to disk.
    app.middleware("http")(_timing_middleware)
    app.middleware("http")(_rate_limit_middleware)
    app.add_middleware(limits.BodySizeLimitMiddleware)
    app.middleware("http")(_request_id_middleware)

    # -- Routes -------------------------------------------------------------
    from .routes import router as api_router

    app.include_router(api_router)

    # -- Configurator API routes (mounted outside /api/v1/ to match frontend) -
    from .routes.knowledge_config import router as knowledge_config_router
    from .routes.reindex import router as reindex_router

    app.include_router(knowledge_config_router)
    app.include_router(reindex_router)

    # -- Prometheus /metrics endpoint (optional) ----------------------------
    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus scrape endpoint.

        Only returns real metrics when ``enable_prometheus = true``.
        Supports multi-process mode (gunicorn) via PROMETHEUS_MULTIPROC_DIR.
        """
        from lasagna.utils.monitoring import METRICS

        if METRICS.is_prometheus_enabled:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

            prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
            if prom_dir:
                from prometheus_client import CollectorRegistry, multiprocess

                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                return Response(
                    content=generate_latest(registry),
                    media_type=CONTENT_TYPE_LATEST,
                )
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Prometheus metrics not enabled. Set enable_prometheus = true in settings."
            },
        )

    # -- Embedded MCP Server (for single-process deployments) ----------------
    # When RAG_EMBED_MCP=true, mount the MCP SSE app inside FastAPI so both
    # share the same process.  Required for ARM64/Milvus Lite where only one
    # process can access the embedded database.
    _mcp_embedded = os.environ.get("RAG_EMBED_MCP", "").lower() in ("true", "1", "yes")
    if _mcp_embedded:
        try:
            from lasagna.mcp import transport_guard
            from lasagna.mcp.server import create_mcp_server

            # Mounted here, the tools are reachable by anything that can reach
            # this app -- not by a client that launched them over its own pipes.
            # `transport_guard` already defaults to the strict direction; this
            # says so at the site rather than relying on the default holding.
            transport_guard.set_transport("http")

            # With auth off this app is loopback-only, and loopback is what a
            # web page can reach by DNS rebinding: refuse a foreign Host or
            # Origin, as the standalone /sse does. With auth on, every tool
            # needs a token such a page does not have.
            from lasagna.api.rest.auth_mode import require_auth

            guard = None if require_auth() else transport_guard.loopback_browser_guard()

            # port is unused when mounted as sub-app; routing handled by FastAPI
            _mcp = create_mcp_server(host="0.0.0.0", port=8000, transport_security=guard)
            app.mount("/mcp", _one_response(_mcp.sse_app()))
            logger.info("MCP server embedded at /mcp/sse")
        except Exception as exc:
            _mcp_embedded = False
            logger.warning(f"Failed to embed MCP server: {exc}")

    # -- Embedded Configurator UI (opt-in) ------------------------------------
    # When RAG_CONFIGURATOR=true, mount the configurator frontend and API
    # routes inside the main FastAPI app.
    _configurator_enabled = os.environ.get("RAG_CONFIGURATOR", "").lower() in ("true", "1", "yes")
    if _configurator_enabled:
        try:
            from lasagna.configurator import mount_configurator

            mount_configurator(app)
            logger.info("Configurator UI mounted at /configurator")
        except Exception as exc:
            _configurator_enabled = False
            logger.warning(f"Failed to mount configurator: {exc}")

    # -- Root ---------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Lasagna",
            "version": _package_version(),
            "docs": "/docs",
            "api": "/api/v1",
            "mcp": "/mcp/sse" if _mcp_embedded else None,
            "configurator": "/configurator" if _configurator_enabled else None,
        }

    return app


# Module-level app for ``uvicorn lasagna.api.rest.app:app``
app = create_app()
