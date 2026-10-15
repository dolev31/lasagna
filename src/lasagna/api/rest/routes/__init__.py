"""REST API route modules.

Each module defines an ``APIRouter`` registered on the ``/api/v1/`` prefix.
"""

from fastapi import APIRouter

from ..models import ErrorResponse
from . import health, documents, search, config, citations, sessions, pipelines, gc_admin

#: Errors any route can return, from the middleware and the shared
#: dependencies, in the one error body every route uses: ``{"detail": "..."}``.
#: A route documents its own further codes. 422 keeps FastAPI's validation
#: schema, whose ``detail`` is a list.
COMMON_ERRORS = {
    401: {
        "model": ErrorResponse,
        "description": "RAG_REQUIRE_AUTH is on and the credential is missing or invalid",
    },
    413: {
        "model": ErrorResponse,
        "description": "The request body is over the limit for this route",
    },
    429: {"model": ErrorResponse, "description": "Rate limit or quota exceeded; see Retry-After"},
    500: {
        "model": ErrorResponse,
        "description": "Internal error; the server log has the request id",
    },
    503: {
        "model": ErrorResponse,
        "description": "The backend is not ready, or a dependency is down",
    },
}

router = APIRouter(prefix="/api/v1", responses=COMMON_ERRORS)

router.include_router(health.router)
router.include_router(documents.router)
router.include_router(search.router)
router.include_router(config.router)
router.include_router(citations.router)
router.include_router(sessions.router)
router.include_router(pipelines.router)
router.include_router(gc_admin.router)

__all__ = ["router"]
