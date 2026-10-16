"""Health, liveness, and readiness probes.

One model, three answers:

- ``GET /api/v1/health/live``: liveness. The process is up: 200, and no
  dependency is checked.
- ``GET /api/v1/health/ready``: readiness. Can this instance serve? 503 naming
  every failing component in ``failing``; a degraded one (Redis down, which
  the search cache, rate limits and uploads fall back from) does not fail it.
- ``GET /api/v1/health``: detailed status, the same component states as
  readiness (``healthy``, ``degraded`` or ``unhealthy``, with ``failing`` and
  ``degraded``), always with 200 once the backend has loaded. The MCP
  ``get_system_status`` tool reports the same states.
- ``GET /api/v1/stats``: system statistics.

They contradicted each other: ``/health`` answered 200 ``"unhealthy"`` with
``failing: []``, readiness 503 with the embedding model failing, and MCP said
"healthy".
"""

from __future__ import annotations

import asyncio
import os
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from ..auth_mode import require_auth
from ..dependencies import get_backend, get_session_id
from ..models import (
    HealthCheckDetail,
    HealthResponse,
    LivenessResponse,
    StatsResponse,
)

router = APIRouter(tags=["health"])


def _served_embedder(backend):
    """The embedder a request without a session uses: that session's model.

    The session is `lasagna serve -s` (``RAG_SESSION_ID``), else ``default``.
    The backend's own manager is the settings file's model, which a session
    on a profile does not search with.
    """
    manager = backend.embedding_manager
    get_config = getattr(backend, "get_session_config", None)
    if get_config is None:
        return manager
    config = get_config(os.environ.get("RAG_SESSION_ID", "").strip() or "default")
    if config.embedding_model == manager.model_name:
        return manager
    from lasagna.infrastructure.embedding.manager import shared_embedding_manager

    return shared_embedding_manager(config)


def _embedding_check(backend) -> HealthCheckDetail:
    """The served session's embedding model: loaded, loadable from disk, or not loadable.

    It said "up" from the model's name alone, while the model was not
    downloaded and the server offline, and every search answered 500.
    """
    from lasagna.core.model_downloads import refusal

    try:
        t0 = perf_counter()
        manager = _served_embedder(backend)
        model_name = manager.model_name
        dim = manager.dimension
        problem = None if getattr(manager, "is_loaded", True) else refusal(model_name)
        ms = (perf_counter() - t0) * 1000
        if problem is not None:
            return HealthCheckDetail(
                status="down",
                latency_ms=round(ms, 2),
                detail=f"{problem} Sessions on another profile still work. For the "
                f"default: lasagna configure --profile speed, or hf download {model_name}.",
            )
        loaded = "" if getattr(manager, "is_loaded", True) else ", loads on first use"
        return HealthCheckDetail(
            status="up", latency_ms=round(ms, 2), detail=f"{model_name} (dim={dim}{loaded})"
        )
    except Exception as exc:
        return HealthCheckDetail(status="down", detail=str(exc))


@router.get("/health/live", response_model=LivenessResponse)
async def liveness():
    """Liveness probe: is this process alive?

    Always returns 200 unless the process is deadlocked.
    Used by Kubernetes to decide whether to restart the container.
    """
    return LivenessResponse(status="alive")


#: Seconds each dependency has to answer a readiness ping.
PING_TIMEOUT_S = 2.0

_ping_pool = None


def _ping_executor():
    """Threads of their own for readiness pings.

    A ping that hangs keeps its thread until the dependency answers. On the
    default executor, which the search path also uses, a probe every few
    seconds against a hung Redis would fill it with stuck pings and stall
    searches; here at most these four threads are stuck, and later pings
    time out in the queue instead.
    """
    global _ping_pool
    if _ping_pool is None:
        from concurrent.futures import ThreadPoolExecutor

        _ping_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lasagna-ping")
    return _ping_pool


async def _ping(name: str, call: Callable[[], Optional[str]]) -> Tuple[str, HealthCheckDetail]:
    """Run one blocking ping in a thread, bounded by ``PING_TIMEOUT_S``.

    A dependency that hangs (a Redis behind a dropped connection, a vector
    store mid-failover) used to hang the probe with it, so the orchestrator's
    own timeout fired and reported nothing about which one. The thread cannot
    be cancelled and finishes on its own; the probe answers on time.
    """
    t0 = perf_counter()
    try:
        loop = asyncio.get_running_loop()
        detail = await asyncio.wait_for(
            loop.run_in_executor(_ping_executor(), call), PING_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        return name, HealthCheckDetail(
            status="down", detail=f"no answer within {PING_TIMEOUT_S:g} s"
        )
    except Exception as exc:
        return name, HealthCheckDetail(status="down", detail=f"{type(exc).__name__}: {exc}")
    ms = (perf_counter() - t0) * 1000
    return name, HealthCheckDetail(status="up", latency_ms=round(ms, 2), detail=detail)


def _dependency_pings(backend) -> Dict[str, Callable[[], Optional[str]]]:
    """A real round trip to each external service this instance is configured to use."""
    from lasagna.core.config import rag_config
    from lasagna.core.provenance import redact_url

    def vector_store():
        vb = backend.get_vector_backend()
        vb.has_collection(rag_config.collection_name)
        return vb.backend_name

    pings: Dict[str, Callable[[], Optional[str]]] = {"vector_backend": vector_store}

    if rag_config.cache_backend == "redis" or getattr(rag_config, "celery_enabled", False):
        url = rag_config.redis_url

        def redis_ping():
            import redis

            client = redis.Redis.from_url(
                url, socket_connect_timeout=PING_TIMEOUT_S, socket_timeout=PING_TIMEOUT_S
            )
            try:
                client.ping()
            finally:
                client.close()
            return redact_url(url)

        pings["redis"] = redis_ping

    def database():
        from lasagna.core.database import get_rag_database

        db = get_rag_database()
        db.get_documents("__health_check__")
        return getattr(db, "database_backend", "sqlite")

    # SQLite or Postgres, whichever holds the metadata.
    pings["database"] = database
    return pings


async def component_checks(backend) -> Dict[str, HealthCheckDetail]:
    """Every component's state, as readiness, ``/health`` and MCP status report it.

    The embedding model is checked as loaded or loadable; the vector store,
    the metadata database and (when used) Redis are pinged, each within
    ``PING_TIMEOUT_S``. Redis is the one with a fallback, so it is degraded
    when down, not failing.
    """
    checks = {"embedding_model": _embedding_check(backend)}
    pinged = await asyncio.gather(
        *(_ping(name, call) for name, call in _dependency_pings(backend).items())
    )
    checks.update(dict(pinged))
    redis = checks.get("redis")
    if redis is not None and redis.status == "down":
        checks["redis"] = HealthCheckDetail(
            status="degraded",
            latency_ms=redis.latency_ms,
            detail=f"{redis.detail}. Without it the search cache and rate limits are kept "
            f"per process, and uploads are processed here instead of by a Celery worker.",
        )
    return checks


def summarize(checks: Dict[str, HealthCheckDetail]) -> Tuple[str, List[str], List[str]]:
    """(status, failing, degraded): ``unhealthy`` with any failing, else ``degraded`` or ``healthy``."""
    failing = sorted(name for name, check in checks.items() if check.status == "down")
    degraded = sorted(name for name, check in checks.items() if check.status == "degraded")
    status = "unhealthy" if failing else ("degraded" if degraded else "healthy")
    return status, failing, degraded


def component_checks_blocking(backend) -> Dict[str, HealthCheckDetail]:
    """`component_checks` for a synchronous caller (the MCP status tool), in a thread of its own."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lasagna-health") as pool:
        return pool.submit(lambda: asyncio.run(component_checks(backend))).result()


def _health_response(checks: Dict[str, HealthCheckDetail]) -> HealthResponse:
    status, failing, degraded = summarize(checks)
    return HealthResponse(
        status=status,
        checks=checks,
        require_auth=require_auth(),
        failing=failing,
        degraded=degraded,
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "A component is failing"}},
)
async def readiness(backend=Depends(get_backend)):
    """Readiness probe: can this instance handle requests?

    Checks the served session's embedding model and pings the vector store,
    the metadata database (Postgres or SQLite) and, when this instance uses
    it, Redis, each with a 2 s timeout. Returns 503 naming every failing
    component in ``failing``, so a load balancer stops routing here and an
    operator sees which one. Redis down is ``degraded``: answered with 200.
    """
    resp = _health_response(await component_checks(backend))
    if resp.failing:
        return JSONResponse(status_code=503, content=resp.model_dump())
    return resp


@router.get("/health", response_model=HealthResponse)
async def full_health(backend=Depends(get_backend)):
    """Detailed status: the component states readiness checks, always with 200.

    For dashboards and monitoring; ``/health/ready`` is the probe that
    answers 503 when a component is failing.
    """
    return _health_response(await component_checks(backend))


# With auth on, a verified caller only: it names the collection, the model and
# the feature flags. The probes above stay open for orchestrators.
@router.get("/stats", response_model=StatsResponse, dependencies=[Depends(get_session_id)])
async def system_stats(backend=Depends(get_backend)):
    """System statistics: embedding model, features, cache stats."""
    try:
        raw = backend.get_stats()
        return StatsResponse(
            embedding_model=raw.get("embedding_model"),
            embedding_dim=raw.get("embedding_dim"),
            collection_name=raw.get("collection_name"),
            vector_backend=raw.get("vector_backend"),
            features=raw.get("features", {}),
            cache=raw.get("cache", {}),
        )
    except Exception as exc:
        logger.warning(f"Failed to collect stats: {exc}")
        return StatsResponse()
