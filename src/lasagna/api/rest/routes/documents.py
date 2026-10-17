"""Document management endpoints.

- ``POST   /api/v1/documents``                  — upload document
- ``POST   /api/v1/documents/batch``            — upload multiple documents
- ``GET    /api/v1/documents``                  — list documents
- ``GET    /api/v1/documents/{id}``             — get document details
- ``DELETE /api/v1/documents/{id}``             — hard-delete document (irreversible)
- ``POST   /api/v1/documents/{id}/archive``     — soft-delete (recoverable, grace window)
- ``POST   /api/v1/documents/{id}/restore``     — restore a soft-deleted document
- ``GET    /api/v1/documents/{id}/chunks``      — get document chunks
- ``GET    /api/v1/documents/{id}/status``      — poll processing status
- ``POST   /api/v1/documents/{id}/reindex``     — re-process with current settings
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from lasagna.api.rag_backend import RAGBackend
from lasagna.core.quota import QuotaExceededError
from lasagna.infrastructure.tasks.document_tasks import ProcessingAtCapacity
from ..dependencies import get_backend, get_session_id
from ..models import (
    BatchUploadItemResponse,
    BatchUploadResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
    ErrorResponse,
    UploadResponse,
    ChunkResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def _doc_to_response(doc) -> DocumentResponse:
    """Convert internal Document/DocumentMetadata to API response."""
    return DocumentResponse(
        document_id=doc.document_id,
        session_id=doc.session_id,
        filename=doc.filename,
        file_type=getattr(doc, "file_type", ""),
        file_size=getattr(doc, "file_size", 0),
        status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
        chunks_count=getattr(doc, "chunks_count", 0),
        upload_date=doc.upload_date,
        error_message=getattr(doc, "error_message", None),
    )


def _reject_foreign_document_id(backend: RAGBackend, document_id: str, session_id: str) -> None:
    """Refuse a client-chosen ``document_id`` that belongs to another session.

    ``document_id`` is the global primary key of ``rag_documents``. The upload
    route accepts it as a form field so a caller can re-upload the same logical
    document idempotently, and it was passed through with no check at all: a
    caller who supplied someone else's id overwrote that row and rewrote its
    ``session_id``. Verified end to end against a running server -- the victim's
    document listing dropped to zero and the attacker owned the row.

    Uses the session-agnostic lookup deliberately. Asking "does this id exist in
    MY session" cannot distinguish an id that is free from one that is taken by
    someone else, which is the only distinction that matters here.

    404 rather than 403: telling an unauthorised caller that an id exists turns
    this into an enumeration oracle over other tenants' document ids.
    """
    existing = backend._get_document_metadata(document_id)
    if existing is None:
        return
    if getattr(existing, "session_id", None) != session_id:
        logger.warning(
            f"Rejected upload: session {session_id!r} supplied document_id "
            f"{document_id!r}, which belongs to another session."
        )
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )


def _refuse_when_the_embedder_cannot_load(backend: RAGBackend, session_id: str) -> None:
    """503, with the reason readiness gives, when the session's embedding model
    cannot load here (not downloaded, and offline or downloads refused).

    The upload answered 201 and the document failed in the worker, while
    ``/health/ready`` already reported the model down.
    """
    from .health import embedding_problem

    problem = embedding_problem(backend, session_id)
    if problem is not None:
        raise HTTPException(status_code=503, detail=problem)


@router.post(
    "",
    response_model=UploadResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "The file could not be accepted"},
        404: {"model": ErrorResponse, "description": "document_id belongs to another session"},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    allow_duplicates: bool = Form(False),
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Upload and index a document.

    Accepts any supported file type (PDF, DOCX, TXT, MD, CSV, JSON,
    HTML, XML).  The document is chunked, embedded, and indexed into
    the vector store. A file whose content the session already holds is
    not uploaded again: the response names that document in
    ``duplicate_of``. ``allow_duplicates=true`` indexes it anyway.
    """
    _refuse_when_the_embedder_cannot_load(backend, session_id)
    # Save uploaded file to a temp location.
    # Sanitize filename to prevent path-traversal attacks.
    safe_name = os.path.basename(file.filename) if file.filename else "upload"
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, safe_name)
    try:
        # Stream-copy with size validation to prevent disk exhaustion
        # before the pipeline's own max_file_size check.
        from lasagna.core.config import rag_config

        max_upload_bytes = rag_config.max_file_size
        copied = 0
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(8192):
                copied += len(chunk)
                if copied > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (>{max_upload_bytes} bytes)",
                    )
                f.write(chunk)

        upload_kwargs: Dict[str, Any] = dict(
            file_path=tmp_path,
            session_id=session_id,
            filename=file.filename,
            # A full processing pool is a 503, not processing in this request.
            refuse_when_busy=True,
            allow_duplicates=allow_duplicates,
        )
        if document_id is not None:
            # `document_id` is the GLOBAL primary key of rag_documents, and this
            # form field let the caller choose it. Uploading with another
            # session's document_id overwrote that row and rewrote its
            # session_id: verified against a running server, the victim's
            # listing went to zero documents and the attacker owned the row.
            #
            # It exists as an idempotency key -- re-uploading the same logical
            # document without creating a duplicate -- which only requires that
            # a caller may reuse an id they already own.
            _reject_foreign_document_id(backend, document_id, session_id)
            upload_kwargs["document_id"] = document_id
        doc = await run_in_threadpool(
            backend.upload_document,
            **upload_kwargs,
        )
        return UploadResponse(
            document_id=doc.document_id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
            chunks_count=doc.chunks_count,
            duplicate_of=getattr(doc, "duplicate_of", None),
        )
    except HTTPException:
        raise
    except ProcessingAtCapacity as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"})
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        from lasagna.api._document_pipeline import DocumentBusy, DocumentRemoved
        from lasagna.core.model_downloads import ModelNotDownloaded, fix_for_session

        if isinstance(exc, DocumentBusy):
            # A replacement of a document that is still processing.
            raise HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, DocumentRemoved):
            # A replacement of a document deleted while it was processed.
            raise HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ModelNotDownloaded):
            raise HTTPException(
                status_code=503, detail=f"{exc} {fix_for_session(session_id, exc.model)}"
            )
        logger.error(f"Upload failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse, "description": "More files than RAG_MAX_BATCH_FILES"}},
)
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    allow_duplicates: bool = Form(False),
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Upload multiple documents in a single request.

    Files are processed concurrently up to the configured
    ``max_concurrent_uploads`` limit.  Each file goes through the same
    pipeline as the single-upload endpoint (text extraction, chunking,
    embedding, vector indexing). Without Celery, a file that finds every
    processing thread busy waits for one to free up (at most
    ``processing_timeout`` seconds), so a batch larger than the pool is
    processed in full; the response returns once every file is accepted.

    A batch may hold at most ``RAG_MAX_BATCH_FILES`` files (default 20) and
    ``RAG_MAX_BATCH_BYTES`` in total (default 200 MiB, refused with 413 before
    the body is read). A file whose content the session already holds, in
    the batch or before it, is not uploaded again: its result names that
    document in ``duplicate_of``. ``allow_duplicates=true`` indexes it anyway.
    """
    from lasagna.api.rest.limits import max_batch_files
    from lasagna.core.config import rag_config

    _refuse_when_the_embedder_cannot_load(backend, session_id)
    limit = max_batch_files()
    if len(files) > limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{len(files)} files in one batch; the limit is {limit}. Send "
                f"them in several batches, or raise RAG_MAX_BATCH_FILES."
            ),
        )

    max_concurrent = getattr(rag_config, "max_concurrent_uploads", 5)
    max_upload_bytes = rag_config.max_file_size
    wait_for_thread = float(rag_config._config.get("processing_timeout", 300))
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_one(upload_file: UploadFile) -> BatchUploadItemResponse:
        async with semaphore:
            safe_name = os.path.basename(upload_file.filename) if upload_file.filename else "upload"
            tmp_dir = tempfile.mkdtemp(prefix="rag_batch_")
            tmp_path = os.path.join(tmp_dir, safe_name)
            try:
                # Stream-copy with size validation
                copied = 0
                with open(tmp_path, "wb") as f:
                    while chunk := await upload_file.read(8192):
                        copied += len(chunk)
                        if copied > max_upload_bytes:
                            return BatchUploadItemResponse(
                                document_id="",
                                filename=upload_file.filename or safe_name,
                                status="error",
                                error=f"File too large (>{max_upload_bytes} bytes)",
                            )
                        f.write(chunk)

                doc = await run_in_threadpool(
                    backend.upload_document,
                    file_path=tmp_path,
                    session_id=session_id,
                    filename=upload_file.filename,
                    refuse_when_busy=True,
                    wait_when_busy=wait_for_thread,
                    allow_duplicates=allow_duplicates,
                )
                return BatchUploadItemResponse(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    file_type=getattr(doc, "file_type", ""),
                    file_size=getattr(doc, "file_size", 0),
                    status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
                    chunks_count=getattr(doc, "chunks_count", 0),
                    duplicate_of=getattr(doc, "duplicate_of", None),
                )
            except QuotaExceededError as exc:
                return BatchUploadItemResponse(
                    document_id="",
                    filename=upload_file.filename or safe_name,
                    status="error",
                    error=f"Quota exceeded: {exc}",
                )
            except Exception as exc:
                logger.error(f"Batch upload failed for {safe_name}: {exc}")
                return BatchUploadItemResponse(
                    document_id="",
                    filename=upload_file.filename or safe_name,
                    status="error",
                    error=str(exc),
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    item_results = await asyncio.gather(*[_process_one(f) for f in files])

    succeeded = sum(1 for r in item_results if r.error is None)
    return BatchUploadResponse(
        results=list(item_results),
        total=len(item_results),
        succeeded=succeeded,
        failed=len(item_results) - succeeded,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
    exclude_ids: Optional[str] = Query(None, description="Comma-separated document IDs to exclude"),
):
    """List all documents in the session."""
    docs = await run_in_threadpool(backend.get_documents, session_id)
    if exclude_ids:
        excluded: Set[str] = {x for x in exclude_ids.split(",") if x}
        docs = [d for d in docs if d.document_id not in excluded]
    return DocumentListResponse(
        documents=[_doc_to_response(d) for d in docs],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Get a single document's metadata."""
    doc = await run_in_threadpool(backend.get_document_by_id, document_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return _doc_to_response(doc)


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Delete a document and all its chunks/vectors/citations."""
    success = await run_in_threadpool(backend.delete_document, document_id, session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DeleteResponse(
        success=True,
        document_id=document_id,
        message=f"Document {document_id} deleted",
    )


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: str,
    session_id: str = Depends(get_session_id),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    backend: RAGBackend = Depends(get_backend),
):
    """Get chunks from a document (paginated)."""
    from lasagna.api._document_pipeline import ChunkStoreUnavailable

    try:
        chunks = await run_in_threadpool(
            backend.get_document_chunks,
            document_id=document_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
    except ChunkStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return [
        ChunkResponse(
            chunk_id=c.get("chunk_id", ""),
            content=c.get("content", ""),
            document_id=c.get("document_id", document_id),
            chunk_index=c.get("chunk_index", 0),
            page_number=c.get("page_number"),
            metadata=c.get("metadata", {}),
        )
        for c in chunks
    ]


@router.post("/{document_id}/archive", response_model=DocumentResponse)
async def archive_document(
    document_id: str,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Soft-delete (archive) a document.

    The document is marked as ``soft_deleted`` and enters the GC grace window.
    Vectors and the stored file are **not** removed immediately: the GC
    hard-deletes them after the configured grace period (default: 24 h).

    Use ``POST /{document_id}/restore`` to recover the document before the
    grace window expires.

    Unlike ``DELETE /{document_id}``, this operation is **reversible**.
    """
    try:
        doc_dict = await run_in_threadpool(backend.soft_delete_document, document_id, session_id)
        if not doc_dict:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        # Map dict → response model
        return DocumentResponse(
            document_id=doc_dict.get("document_id", document_id),
            session_id=doc_dict.get("session_id", session_id),
            filename=doc_dict.get("filename", ""),
            file_type=doc_dict.get("file_type", ""),
            file_size=doc_dict.get("file_size", 0),
            status=doc_dict.get("status", "soft_deleted"),
            chunks_count=doc_dict.get("chunks_count", 0),
            upload_date=doc_dict.get("upload_date"),
            error_message=doc_dict.get("error_message"),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Archive failed for {document_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{document_id}/restore", response_model=DocumentResponse)
async def restore_document(
    document_id: str,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Restore a soft-deleted document to READY status.

    Only possible within the soft-delete grace window.  If the grace window
    has expired (and the GC has already hard-deleted the document), this
    endpoint returns ``409 Conflict``.
    """
    try:
        doc_dict = await run_in_threadpool(backend.restore_document, document_id, session_id)
        if not doc_dict:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return DocumentResponse(
            document_id=doc_dict.get("document_id", document_id),
            session_id=doc_dict.get("session_id", session_id),
            filename=doc_dict.get("filename", ""),
            file_type=doc_dict.get("file_type", ""),
            file_size=doc_dict.get("file_size", 0),
            status=doc_dict.get("status", "ready"),
            chunks_count=doc_dict.get("chunks_count", 0),
            upload_date=doc_dict.get("upload_date"),
            error_message=doc_dict.get("error_message"),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        # DocumentExpiredError — grace window has passed
        from lasagna.lifecycle import DocumentExpiredError

        if isinstance(exc, DocumentExpiredError):
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            )
        logger.error(f"Restore failed for {document_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Poll the processing status of a document.

    Useful after async uploads via Celery to check when processing is
    complete.  Returns status: ``processing``, ``ready``, or ``error``.
    """
    doc = await run_in_threadpool(backend.get_document_by_id, document_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    status = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
    return DocumentStatusResponse(
        document_id=document_id,
        status=status,
        chunks_count=getattr(doc, "chunks_count", 0),
        error_message=getattr(doc, "error_message", None),
    )


@router.post(
    "/{document_id}/reindex",
    response_model=UploadResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No such document in this session"},
        409: {"model": ErrorResponse, "description": "The document is processing or archived"},
        410: {"model": ErrorResponse, "description": "The document's source file is gone"},
    },
)
async def reindex_document(
    document_id: str,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Re-index an existing document with the session's current settings.

    Useful after config changes (e.g. switching to a different embedding
    model or chunking strategy). The document keeps answering searches from
    its old chunks until the new ones are stored, and a reindex that fails
    leaves it as it was.
    """
    try:
        new_doc = await run_in_threadpool(backend.reindex_document, document_id, session_id)
    except Exception as exc:
        raise reindex_failure(exc, document_id, session_id) from exc
    return UploadResponse(
        document_id=new_doc.document_id,
        filename=new_doc.filename,
        file_type=new_doc.file_type,
        file_size=new_doc.file_size,
        status=new_doc.status.value if hasattr(new_doc.status, "value") else str(new_doc.status),
        chunks_count=new_doc.chunks_count,
    )


def reindex_failure(exc: Exception, document_id: str, session_id: str) -> HTTPException:
    """The HTTP answer for a reindex that did not happen; the document is unchanged."""
    from lasagna.api._document_pipeline import DocumentBusy, DocumentNotFound, SourceFileMissing
    from lasagna.core.model_downloads import ModelNotDownloaded, fix_for_session

    from lasagna.api._document_pipeline import DocumentRemoved

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, DocumentRemoved):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DocumentNotFound):
        return HTTPException(status_code=404, detail=f"Document {document_id} not found")
    if isinstance(exc, SourceFileMissing):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, DocumentBusy):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ModelNotDownloaded):
        return HTTPException(
            status_code=503,
            detail=f"{exc} {fix_for_session(session_id, exc.model)} "
            f"Document {document_id} is unchanged.",
        )
    if isinstance(exc, QuotaExceededError):
        return HTTPException(status_code=429, detail=f"{exc} Document {document_id} is unchanged.")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=f"{exc} Document {document_id} is unchanged.")
    logger.error(f"Reindex failed for {document_id}: {exc}")
    return HTTPException(
        status_code=500,
        detail=(
            f"Reindex of {document_id} failed with an internal error; the document is "
            "unchanged. The server log has the cause."
        ),
    )
