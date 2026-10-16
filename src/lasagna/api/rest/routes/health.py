"""Health, liveness, and readiness probes.

- ``GET /api/v1/health``       — full health check (all dependencies)
- ``GET /api/v1/health/live``  — liveness (is the process alive?)
- ``GET /api/v1/health/ready`` — readiness (can it handle requests?)
- ``GET /api/v1/stats``        — system statistics
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Callable, Dict, Optional, Tuple

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


def _embedding_check(backend) -> HealthCheckDetail:
    """The default embedding model: loaded, loadable from disk, or not loadable.

    It said "up" from the model's name alone, while the model was not
    downloaded and the server offline, and every search answered 500.
    """
    from lasagna.core.model_downloads import refusal

    try:
        t0 = perf_counter()
        manager = backend.embedding_manager
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


def _run_health_checks(backend) -> Tuple[Dict[str, HealthCheckDetail], str]:
    """The checks behind ``GET /health``: (checks, overall status).

    Readiness pings the same dependencies with timeouts; see `readiness`.
    """
    checks: dict[str, HealthCheckDetail] = {}
    overall = "healthy"

    checks["embedding_model"] = _embedding_check(backend)
    if checks["embedding_model"].status != "up":
        overall = "unhealthy"

    # Check vector backend
    try:
        t0 = perf_counter()
        vb = backend.get_vector_backend()
        backend_name = vb.backend_name
        ms = (perf_counter() - t0) * 1000
        checks["vector_backend"] = HealthCheckDetail(
            status="up",
            latency_ms=round(ms, 2),
            detail=backend_name,
        )
    except Exception as exc:
        checks["vector_backend"] = HealthCheckDetail(status="down", detail=str(exc))
        overall = "unhealthy"

    # Check metadata database
    try:
        from lasagna.core.database import get_rag_database

        t0 = perf_counter()
        db = get_rag_database()
        db.get_documents("__health_check__")
        ms = (perf_counter() - t0) * 1000
        db_detail = getattr(db, "database_backend", "sqlite")
        checks["database"] = HealthCheckDetail(
            status="up",
            latency_ms=round(ms, 2),
            detail=db_detail,
        )
    except Exception as exc:
        checks["database"] = HealthCheckDetail(status="down", detail=str(exc))
        overall = "degraded" if overall == "healthy" else overall

    return checks, overall


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


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "A dependency is down"}},
)
async def readiness(backend=Depends(get_backend)):
    """Readiness probe: can this instance handle requests?

    Pings the vector store, the metadata database (Postgres or SQLite) and,
    when this instance uses it, Redis, each with a 2 s timeout. Returns 503
    naming every dependency that did not answer in ``failing``, so a load
    balancer stops routing here and an operator sees which one.
    """
    checks = {"embedding_model": _embedding_check(backend)}
    pinged = await asyncio.gather(
        *(_ping(name, call) for name, call in _dependency_pings(backend).items())
    )
    checks.update(dict(pinged))
    failing = sorted(name for name, check in checks.items() if check.status != "up")
    resp = HealthResponse(
        status="healthy" if not failing else "unhealthy",
        checks=checks,
        require_auth=require_auth(),
        failing=failing,
    )

    if failing:
        return JSONResponse(
            status_code=503,
            content=resp.model_dump(),
        )
    return resp


@router.get("/health", response_model=HealthResponse)
async def full_health(backend=Depends(get_backend)):
    """Full health check: reports dependency status but always returns 200.

    Use this for dashboards and monitoring; use ``/health/ready`` for
    orchestrator probes that need to remove unhealthy instances.
    """
    checks, overall = _run_health_checks(backend)
    return HealthResponse(status=overall, checks=checks, require_auth=require_auth())


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
