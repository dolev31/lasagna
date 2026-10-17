"""Reindex endpoint for the configurator UI.

Mounted at ``/api/knowledge/reindex`` (not under ``/api/v1/``).
Triggers re-processing of all documents in a session with current config.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from lasagna.api.rag_backend import RAGBackend
from ..dependencies import get_backend, get_session_id
from ..models import ReindexRequest, ReindexResponse

router = APIRouter(prefix="/api/knowledge", tags=["configurator"])


def _reindex_sync(backend: RAGBackend, session_id: str):
    """Reindex every document of a session, one at a time.

    A document whose reindex fails keeps its previous index and is named in
    ``errors`` with the reason; the others are still reindexed.
    """
    docs = backend.get_documents(session_id)
    if not docs:
        return {
            "status": "completed",
            "documents_reindexed": 0,
            "message": "No documents to reindex",
        }

    reindexed = 0
    errors = []
    for doc in docs:
        try:
            backend.reindex_document(doc.document_id, session_id)
            reindexed += 1
        except Exception as exc:
            logger.warning(f"Reindex failed for {doc.document_id}: {exc}")
            errors.append(f"{doc.document_id}: {exc}")

    return {
        "status": "completed" if not errors else "partial",
        "documents_reindexed": reindexed,
        "message": (
            f"{len(errors)} of {len(docs)} document(s) could not be reindexed and keep their "
            "previous index."
            if errors
            else None
        ),
        "errors": errors if errors else None,
    }


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_documents(
    body: ReindexRequest,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Re-index all documents in a session with current settings.

    Useful after config changes (e.g. switching embedding model or
    chunking strategy) that require documents to be re-processed.

    The ``session_id`` is extracted from the authenticated session
    (header/token/query), NOT from the request body, to prevent
    cross-tenant document access.
    """
    try:
        result = await asyncio.to_thread(_reindex_sync, backend, session_id)
        return result
    except Exception as exc:
        logger.error(f"Reindex failed for session={session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
