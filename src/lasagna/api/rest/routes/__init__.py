"""REST API route modules.

Each module defines an ``APIRouter`` registered on the ``/api/v1/`` prefix.
"""

from fastapi import APIRouter

from . import health, documents, search, config, citations, sessions, pipelines, gc_admin

router = APIRouter(prefix="/api/v1")

router.include_router(health.router)
router.include_router(documents.router)
router.include_router(search.router)
router.include_router(config.router)
router.include_router(citations.router)
router.include_router(sessions.router)
router.include_router(pipelines.router)
router.include_router(gc_admin.router)

__all__ = ["router"]
