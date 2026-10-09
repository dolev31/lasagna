"""Reindex endpoint for the configurator UI.

Mounted at ``/api/knowledge/reindex`` (not under ``/api/v1/``).
Triggers re-processing of all documents in a session with current config.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from lasagna.api.rag_backend import RAGBackend
from ..dependencies import get_backend, get_session_id
from ..models import ReindexRequest, ReindexResponse

router = APIRouter(prefix="/api/knowledge", tags=["configurator"])


def _reindex_sync(backend: RAGBackend, session_id: str):
    """Run blocking reindex in a thread-safe synchronous context."""
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
            file_path = getattr(doc, "file_path", None) or getattr(doc, "original_path", None)
            if not file_path:
                errors.append(f"{doc.document_id}: source file path not available")
                continue

            # Resolve the storage backend that holds this document's file
            from lasagna.storage import get_storage_backend

            stored_backend_name = getattr(doc, "storage_backend_name", None) or "local"
            try:
                storage = get_storage_backend(stored_backend_name)
            except Exception:
                storage = get_storage_backend()

            if not storage.exists(file_path):
                errors.append(f"{doc.document_id}: source file no longer available in storage")
                continue

            # Copy to temp to avoid self-copy truncation (upload writes
            # back to the same storage path when document_id is preserved).
            local_path = storage.ensure_local(file_path)
            tmp_dir = tempfile.mkdtemp(prefix="rag_reindex_")
            try:
                safe_name = os.path.basename(doc.filename) if doc.filename else "reindex"
                tmp_path = os.path.join(tmp_dir, safe_name)
                shutil.copy2(local_path, tmp_path)

                backend.delete_document(doc.document_id, session_id, keep_file=True)
                backend.upload_document(
                    file_path=tmp_path,
                    session_id=session_id,
                    filename=doc.filename,
                    document_id=doc.document_id,
                )
                reindexed += 1
            finally:
                # Clean up ensure_local temp file (cloud backends download
                # to a temp file; local backends return the stored path).
                try:
                    if local_path != file_path and os.path.isfile(local_path):
                        os.unlink(local_path)
                except OSError:
                    pass
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning(f"Reindex failed for {doc.document_id}: {exc}")
            errors.append(f"{doc.document_id}: {exc}")

    return {
        "status": "completed" if not errors else "partial",
        "documents_reindexed": reindexed,
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
