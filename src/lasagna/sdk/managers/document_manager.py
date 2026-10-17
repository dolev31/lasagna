"""Document Manager - Document lifecycle operations.

Provides upload, listing, deletion, and chunk retrieval operations.
Wraps the underlying RAGBackend document operations.

Example:
    ```python
    # Upload a document
    result = await client.documents.upload("./report.pdf")

    # List all documents
    docs = await client.documents.list()

    # Get document info
    doc = await client.documents.get("doc_abc123")

    # Delete a document
    await client.documents.delete("doc_abc123")
    ```
"""

from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path
import asyncio
import functools
from datetime import datetime, timezone

from loguru import logger

from ..types import (
    Document,
    DocumentChunk,
    UploadResult,
    DocumentStatus,
)
from ..exceptions import (
    DocumentError,
    DocumentNotFoundError,
    DocumentProcessingError,
    UploadError,
)

if TYPE_CHECKING:
    from ..client import KnowledgeClient


def _resolve_status(raw_status) -> DocumentStatus:
    """Resolve status from both enum (local backend) and string (remote backend)."""
    status_str = raw_status.value if hasattr(raw_status, "value") else raw_status
    return DocumentStatus(status_str)


def _refuse_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """Refuse metadata an upload would drop.

    Accepted and dropped: no surface stores per-document metadata (the REST
    upload route takes none), so ``upload(metadata={"department": ...})``
    returned success and the metadata was gone.
    """
    if metadata:
        raise ValueError(
            f"upload() does not store metadata ({', '.join(sorted(map(str, metadata)))} "
            f"given): a document keeps its file, filename and id only. Keep the metadata "
            f"in your own records under the document_id that upload() returns, and call "
            f"upload() without it."
        )


class DocumentManager:
    """Manages document upload, listing, and deletion.

    This manager provides a clean API for document lifecycle operations.
    All operations are scoped to the client's session.

    Example:
        ```python
        # Upload a document
        result = await client.documents.upload("./report.pdf")
        print(f"Uploaded: {result.document_id}")

        # List documents
        docs = await client.documents.list()
        for doc in docs:
            print(f"  {doc.filename}: {doc.chunk_count} chunks")

        # Get specific document
        doc = await client.documents.get("doc_abc123")

        # Get document chunks
        chunks = await client.documents.get_chunks("doc_abc123")

        # Delete document
        success = await client.documents.delete("doc_abc123")
        ```
    """

    def __init__(self, client: "KnowledgeClient"):
        """Initialize document manager.

        Args:
            client: Parent KnowledgeClient instance
        """
        self._client = client

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a synchronous backend call in the default executor.

        Prevents blocking the event loop when called from async context,
        enabling true parallel execution in upload_batch() via asyncio.gather().
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(fn, *args, **kwargs),
        )

    # =========================================================================
    # Upload Operations
    # =========================================================================

    async def upload(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        allow_duplicates: bool = False,
    ) -> UploadResult:
        """Upload and index a document.

        A file whose content the session already holds is not uploaded again:
        the result names that document, with ``duplicate_of`` set and a
        message that says so. ``allow_duplicates=True`` indexes it anyway.

        The document will be processed through the full pipeline:
        1. Text extraction (with OCR if enabled for scanned PDFs)
        2. Semantic chunking
        3. Embedding generation
        4. Vector indexing

        Args:
            file_path: Path to the document file
            filename: Optional custom display name
            metadata: Not stored: a document keeps its file, filename and id
                only, on every surface. A non-empty one raises ``ValueError``
                rather than being dropped.
            document_id: Optional explicit document ID. When provided,
                used instead of a generated UUID. Useful for benchmark
                evaluation where corpus doc IDs must match qrels. An id the
                session already holds replaces that document: the new copy
                is indexed before the old one is removed.
            allow_duplicates: Index the file even when the session already
                holds the same content.

        Returns:
            UploadResult with document_id, chunk_count, status

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported, or ``metadata`` is given

        Example:
            ```python
            result = await client.documents.upload(
                "./quarterly_report.pdf",
                filename="Q3 2024 Report",
            )
            print(f"Uploaded: {result.document_id} ({result.chunk_count} chunks)")
            ```
        """
        _refuse_metadata(metadata)
        # Resolve path
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Determine filename
        display_name = filename or path.name

        # Get backend
        backend = await self._client._get_backend()
        session_id = self._client.session_id
        # So the CLI, REST and MCP read this session with the model that builds it.
        self._client._remember_session_config()

        logger.info(f"Uploading document: {display_name} (session={session_id})")
        start_time = datetime.now(timezone.utc)

        try:
            # Upload through backend (run in executor to avoid blocking the
            # event loop — enables true parallel uploads via upload_batch)
            doc = await self._run_sync(
                backend.upload_document,
                file_path=str(path),
                session_id=session_id,
                filename=display_name,
                document_id=document_id,
                # Only when asked: a backend written against the older
                # signature still takes every default upload.
                **({"allow_duplicates": True} if allow_duplicates else {}),
            )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            status = _resolve_status(doc.status)
            duplicate_of = getattr(doc, "duplicate_of", None)
            if duplicate_of:
                return UploadResult(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    status=status,
                    chunk_count=doc.chunks_count,
                    processing_time_ms=0.0,
                    duplicate_of=duplicate_of,
                    message=(
                        f"{display_name} is already in session {session_id!r} as "
                        f"{duplicate_of} (same content), so it was not uploaded again. "
                        f"Pass allow_duplicates=True to index it again."
                    ),
                )
            queued = str(getattr(status, "value", status)).lower() in ("processing", "pending")
            if queued:
                # close() waits for these, so shutting down never pulls the
                # vector store out from under a worker that is still writing.
                in_flight = getattr(self._client, "_in_flight_documents", None)
                if in_flight is not None:
                    in_flight.add(doc.document_id)
            return UploadResult(
                document_id=doc.document_id,
                filename=doc.filename,
                status=status,
                chunk_count=doc.chunks_count,
                processing_time_ms=processing_time,
                message=(
                    "Document queued for processing; await "
                    "client.documents.wait_for_processing(document_id) before searching it."
                    if queued
                    else "Document uploaded and indexed successfully."
                ),
            )

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise UploadError(
                f"Upload failed for {display_name}: {e}",
                file_path=str(path),
                stage="processing",
            ) from e

    async def upload_batch(
        self,
        file_paths: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5,
    ) -> List[UploadResult]:
        """Upload multiple documents with optional parallelism.

        Args:
            file_paths: List of file paths to upload
            metadata: Not stored; a non-empty one raises ``ValueError`` (see ``upload``)
            max_concurrent: Maximum concurrent uploads (default 5, use 1 for sequential)

        Returns:
            List of UploadResult, one per file (order preserved)

        Example:
            ```python
            # Parallel upload (default, up to 5 concurrent)
            results = await client.documents.upload_batch([
                "./doc1.pdf",
                "./doc2.pdf",
                "./doc3.pdf",
            ])

            # Sequential upload
            results = await client.documents.upload_batch(
                ["./doc1.pdf", "./doc2.pdf"],
                max_concurrent=1
            )

            for result in results:
                print(f"{result.filename}: {result.status}")
            ```
        """
        _refuse_metadata(metadata)
        if not file_paths:
            return []

        if max_concurrent <= 1:
            # Sequential upload (original behavior)
            results = []
            for path in file_paths:
                result = await self.upload(path, metadata=metadata)
                results.append(result)
            return results

        # Parallel upload with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def upload_with_limit(path: Union[str, Path]) -> UploadResult:
            async with semaphore:
                return await self.upload(path, metadata=metadata)

        return await asyncio.gather(*[upload_with_limit(p) for p in file_paths])

    async def wait_for_processing(
        self,
        document_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
        raise_on_failure: bool = True,
    ) -> Document:
        """Poll until a document reaches a terminal status (ready or error).

        Useful after dispatching an async upload to wait for completion.

        Args:
            document_id: The document ID to wait for.
            timeout: Maximum seconds to wait (default 300).
            poll_interval: Seconds between polls (default 2).
            raise_on_failure: Raise `DocumentProcessingError` when processing
                failed (the default). False returns the failed document, with
                the reason in ``error_message``.

        Returns:
            Document object with final status.

        Raises:
            DocumentProcessingError: If processing failed, naming the reason
                and, when there is a known one, the fix.
            DocumentNotFoundError: If the document disappears.
            TimeoutError: If timeout is exceeded.
        """
        try:
            doc = await self._wait_for_processing(document_id, timeout, poll_interval)
        finally:
            in_flight = getattr(self._client, "_in_flight_documents", None)
            if in_flight is not None:
                in_flight.discard(document_id)
        if raise_on_failure and doc.status == DocumentStatus.FAILED:
            # Returning it quietly let the README's quickstart go on to search,
            # which failed with a traceback about the embedding model.
            reason = doc.error_message or "no reason was recorded"
            raise DocumentProcessingError(
                f"{doc.filename} ({document_id}) could not be indexed: {reason}",
                document_id=document_id,
                filename=doc.filename,
                reason=reason,
            )
        return doc

    async def _wait_for_processing(
        self, document_id: str, timeout: float, poll_interval: float
    ) -> Document:
        import time as _time

        backend = await self._client._get_backend()
        session_id = self._client.session_id
        start = _time.monotonic()
        terminal_statuses = {"ready", "error", "failed"}

        while (_time.monotonic() - start) < timeout:
            doc = await self._run_sync(backend.get_document_by_id, document_id, session_id)
            if doc is None:
                raise DocumentNotFoundError(
                    f"Document {document_id} not found",
                    document_id=document_id,
                )
            status_str = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
            if status_str in terminal_statuses:
                # Field names deliberately mirror list_documents/get_document
                # below: the backend row calls this `upload_date`, the public
                # Document dataclass calls it `created_at`. This branch used to
                # pass `upload_date=` and omit the required `session_id`, so
                # every call raised TypeError the moment a document reached a
                # terminal status — i.e. wait_for_processing() never once
                # returned successfully.
                return Document(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    session_id=doc.session_id,
                    status=_resolve_status(doc.status),
                    chunk_count=doc.chunks_count,
                    file_size=doc.file_size,
                    created_at=doc.upload_date,
                    error_message=getattr(doc, "error_message", None),
                )
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Document {document_id} did not complete processing within {timeout}s")

    # =========================================================================
    # List Operations
    # =========================================================================

    async def list(
        self,
        status: Optional[DocumentStatus] = None,
    ) -> List[Document]:
        """List all documents in the current session.

        Args:
            status: Optional filter by document status

        Returns:
            List of Document objects

        Example:
            ```python
            # List all documents
            docs = await client.documents.list()

            # List only ready documents
            ready_docs = await client.documents.list(status=DocumentStatus.READY)
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        # Get documents from backend
        raw_docs = await self._run_sync(backend.get_documents, session_id)

        documents = []
        for doc in raw_docs:
            doc_status = _resolve_status(doc.status)

            # Filter by status if specified
            if status is not None and doc_status != status:
                continue

            documents.append(
                Document(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    session_id=doc.session_id,
                    status=doc_status,
                    file_path=getattr(doc, "file_path", None),
                    file_type=doc.file_type,
                    file_size=doc.file_size,
                    chunk_count=doc.chunks_count,
                    created_at=doc.upload_date,
                    error_message=getattr(doc, "error_message", None),
                    preview_text=getattr(doc, "preview_text", None),
                )
            )

        return documents

    async def get(self, document_id: str) -> Optional[Document]:
        """Get a specific document by ID.

        Args:
            document_id: Document ID to retrieve

        Returns:
            Document if found, None otherwise

        Example:
            ```python
            doc = await client.documents.get("doc_abc123")
            if doc:
                print(f"Found: {doc.filename} ({doc.chunk_count} chunks)")
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        # Get document from backend
        raw_doc = await self._run_sync(backend.get_document_by_id, document_id, session_id)
        if raw_doc is None:
            return None

        return Document(
            document_id=raw_doc.document_id,
            filename=raw_doc.filename,
            session_id=raw_doc.session_id,
            status=_resolve_status(raw_doc.status),
            file_path=getattr(raw_doc, "file_path", None),
            file_type=raw_doc.file_type,
            file_size=raw_doc.file_size,
            chunk_count=raw_doc.chunks_count,
            created_at=raw_doc.upload_date,
            error_message=getattr(raw_doc, "error_message", None),
            preview_text=getattr(raw_doc, "preview_text", None),
        )

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete(self, document_id: str) -> bool:
        """Delete a document and its chunks.

        Removes the document from the vector database and metadata store.

        Args:
            document_id: ID of document to delete

        Returns:
            True if deleted successfully

        Raises:
            DocumentNotFoundError: If document does not exist
            DocumentError: If delete operation fails

        Example:
            ```python
            try:
                await client.documents.delete("doc_abc123")
                print("Document deleted")
            except DocumentNotFoundError:
                print("Document not found")
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        logger.info(f"Deleting document: {document_id}")

        try:
            success = await self._run_sync(backend.delete_document, document_id, session_id)
            if success:
                logger.info(f"Document deleted: {document_id}")
                return True
            else:
                raise DocumentNotFoundError(
                    f"Document not found: {document_id}",
                    document_id=document_id,
                )
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise DocumentError(f"Delete failed for {document_id}: {e}") from e

    async def delete_all(self, confirm: bool = False) -> int:
        """Delete all documents in the session.

        WARNING: This is a destructive operation.

        Args:
            confirm: Must be True to proceed (safety check)

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If confirm is not True
            DocumentError: If any deletion fails (after attempting all)

        Example:
            ```python
            # This will fail (safety check)
            count = await client.documents.delete_all()

            # This will work
            count = await client.documents.delete_all(confirm=True)
            ```
        """
        if not confirm:
            raise ValueError(
                "delete_all requires confirm=True. This will delete all documents in the session."
            )

        backend = await self._client._get_backend()
        session_id = self._client.session_id

        # Get all documents
        docs = await self.list()
        count = 0
        errors = []

        for doc in docs:
            try:
                if await self._run_sync(backend.delete_document, doc.document_id, session_id):
                    count += 1
            except Exception as e:
                logger.error(f"Failed to delete {doc.document_id}: {e}")
                errors.append((doc.document_id, str(e)))

        logger.info(f"Deleted {count} documents from session {session_id}")

        if errors:
            error_msg = "; ".join([f"{doc_id}: {err}" for doc_id, err in errors])
            raise DocumentError(f"Failed to delete {len(errors)} document(s): {error_msg}")

        return count

    # =========================================================================
    # Lifecycle Operations (soft-delete, restore, bulk-delete)
    # =========================================================================

    async def archive(self, document_id: str) -> Document:
        """Soft-delete (archive) a document.

        The document is hidden from search results and enters the GC grace
        window.  Use ``restore()`` to recover it before the window expires.
        Unlike ``delete()``, this operation is **reversible**.

        Args:
            document_id: ID of the document to archive.

        Returns:
            Updated Document with status ``soft_deleted``.

        Raises:
            DocumentNotFoundError: If the document does not exist.
            DocumentError: If archiving fails.

        Example:
            ```python
            doc = await client.documents.archive("doc_abc123")
            print(doc.status)  # DocumentStatus.SOFT_DELETED
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        logger.info(f"Archiving document: {document_id}")
        try:
            doc_dict = await self._run_sync(backend.soft_delete_document, document_id, session_id)
            if not doc_dict:
                raise DocumentNotFoundError(
                    f"Document not found: {document_id}",
                    document_id=document_id,
                )
            return Document(
                document_id=doc_dict.get("document_id", document_id),
                filename=doc_dict.get("filename", ""),
                session_id=doc_dict.get("session_id", session_id),
                status=_resolve_status(doc_dict.get("status", "soft_deleted")),
                file_type=doc_dict.get("file_type", ""),
                file_size=doc_dict.get("file_size", 0),
                chunk_count=doc_dict.get("chunks_count", 0),
                error_message=doc_dict.get("error_message"),
            )
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Archive failed for {document_id}: {e}")
            raise DocumentError(f"Archive failed for {document_id}: {e}") from e

    async def restore(self, document_id: str) -> Document:
        """Restore a soft-deleted document to READY status.

        Only possible within the GC grace window.  If the window has
        expired, raises ``DocumentError``.

        Args:
            document_id: ID of the archived document to restore.

        Returns:
            Updated Document with status ``ready``.

        Raises:
            DocumentNotFoundError: If the document does not exist.
            DocumentError: If the grace window has expired or restoration fails.

        Example:
            ```python
            doc = await client.documents.restore("doc_abc123")
            print(doc.status)  # DocumentStatus.READY
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        logger.info(f"Restoring document: {document_id}")
        try:
            doc_dict = await self._run_sync(backend.restore_document, document_id, session_id)
            if not doc_dict:
                raise DocumentNotFoundError(
                    f"Document not found: {document_id}",
                    document_id=document_id,
                )
            return Document(
                document_id=doc_dict.get("document_id", document_id),
                filename=doc_dict.get("filename", ""),
                session_id=doc_dict.get("session_id", session_id),
                status=_resolve_status(doc_dict.get("status", "ready")),
                file_type=doc_dict.get("file_type", ""),
                file_size=doc_dict.get("file_size", 0),
                chunk_count=doc_dict.get("chunks_count", 0),
                error_message=doc_dict.get("error_message"),
            )
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Restore failed for {document_id}: {e}")
            raise DocumentError(f"Restore failed for {document_id}: {e}") from e

    async def bulk_delete(self, document_ids: List[str]) -> int:
        """Delete a specific list of documents.

        Args:
            document_ids: IDs of documents to delete.

        Returns:
            Number of documents successfully deleted.

        Raises:
            DocumentError: If one or more deletions fail.

        Example:
            ```python
            deleted = await client.documents.bulk_delete(["doc_1", "doc_2"])
            print(f"Deleted {deleted} documents")
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        count = 0
        errors = []
        for doc_id in document_ids:
            try:
                if await self._run_sync(backend.delete_document, doc_id, session_id):
                    count += 1
            except Exception as e:
                logger.error(f"Failed to delete {doc_id}: {e}")
                errors.append((doc_id, str(e)))

        if errors:
            error_msg = "; ".join([f"{doc_id}: {err}" for doc_id, err in errors])
            raise DocumentError(f"Failed to delete {len(errors)} document(s): {error_msg}")
        return count

    async def delete_session(self, *, confirm: bool = False) -> Dict[str, Any]:
        """Delete ALL documents in the session using the efficient bulk path.

        This calls ``bulk_delete_session()`` on the backend which uses a
        single vector-level delete (or partition drop) rather than looping.
        Much faster than ``delete_all()`` for sessions with many documents.

        Args:
            confirm: Must be ``True`` — safety guard.

        Returns:
            Dict with counts: ``requested``, ``deleted``, ``bytes_freed``,
            ``chunks_freed``, ``errors``.

        Raises:
            ValueError: If ``confirm`` is not ``True``.
            DocumentError: If the operation fails.

        Example:
            ```python
            result = await client.documents.delete_session(confirm=True)
            print(f"Deleted {result['deleted']} documents, freed {result['bytes_freed']} bytes")
            ```
        """
        if not confirm:
            raise ValueError(
                "delete_session requires confirm=True. "
                "This will delete ALL documents in the session."
            )
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        logger.info(f"Bulk-deleting session: {session_id}")
        try:
            result = await self._run_sync(backend.bulk_delete_session, session_id, confirm=True)
            # result may be a BulkDeleteResult dataclass (local backend) or a dict
            # (remote backend that JSON-serialised the response).
            if hasattr(result, "deleted"):
                deleted = result.deleted
                requested = result.requested
                bytes_freed = result.bytes_freed
                chunks_freed = result.chunks_freed
                errors = result.errors
            else:
                deleted = result.get("deleted", 0)
                requested = result.get("requested", 0)
                bytes_freed = result.get("bytes_freed", 0)
                chunks_freed = result.get("chunks_freed", 0)
                errors = result.get("errors", 0)
            logger.info(f"Session deleted: {deleted}/{requested} docs, {bytes_freed} bytes freed")
            return {
                "session_id": session_id,
                "requested": requested,
                "deleted": deleted,
                "bytes_freed": bytes_freed,
                "chunks_freed": chunks_freed,
                "errors": errors,
            }
        except Exception as e:
            logger.error(f"delete_session failed: {e}")
            raise DocumentError(f"delete_session failed: {e}") from e

    # =========================================================================
    # Chunk Operations
    # =========================================================================

    async def get_chunks(
        self,
        document_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> List[DocumentChunk]:
        """Get chunks from a specific document.

        Args:
            document_id: Document ID to get chunks from
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            List of DocumentChunk objects

        Example:
            ```python
            chunks = await client.documents.get_chunks("doc_abc123")
            for chunk in chunks:
                print(f"Chunk {chunk.chunk_index}: {chunk.content[:100]}...")
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        # Get chunks from backend
        raw_chunks = await self._run_sync(
            backend.get_document_chunks,
            document_id=document_id,
            session_id=session_id,
            offset=offset,
            limit=limit,
        )

        chunks = []
        for chunk in raw_chunks:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(chunk.chunk_id),
                    document_id=chunk.document_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    section_header=chunk.metadata.get("section_header"),
                    line_start=chunk.metadata.get("line_start"),
                    line_end=chunk.metadata.get("line_end"),
                    metadata=chunk.metadata,
                )
            )

        return chunks

    # =========================================================================
    # Reindex Operations
    # =========================================================================

    async def reindex(self, document_id: str) -> UploadResult:
        """Reindex an existing document with the session's current settings.

        Useful when:
        - Configuration has changed (chunk size, etc.)
        - The embedding model has changed: the document moves to the new
          model's collection

        The document keeps answering searches from its old chunks until the
        new ones are stored, and a reindex that fails leaves it as it was.

        Args:
            document_id: Document ID to reindex

        Returns:
            UploadResult with the new chunk count

        Raises:
            DocumentNotFoundError: If the document does not exist
            DocumentError: If the reindex failed; the document is unchanged

        Example:
            ```python
            # After changing chunk settings
            await client.set_config(chunk_size=1000)
            result = await client.documents.reindex("doc_abc123")
            ```
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id

        logger.info(f"Reindexing document: {document_id}")
        start_time = datetime.now(timezone.utc)

        if await self._run_sync(backend.get_document_by_id, document_id, session_id) is None:
            raise DocumentNotFoundError(
                f"Document not found: {document_id}", document_id=document_id
            )
        try:
            doc = await self._run_sync(backend.reindex_document, document_id, session_id)
        except Exception as e:
            # It returned an UploadResult with status FAILED, which a caller
            # that did not read the status took for a reindexed document.
            logger.error(f"Reindex failed for {document_id}: {e}")
            raise DocumentError(
                f"Reindex of {document_id} failed, and the document is unchanged: {e}"
            ) from e

        try:
            self._client._record_reindexed_model()
        except Exception as e:  # noqa: BLE001 - the reindex itself succeeded
            logger.warning(
                f"Document {document_id} was reindexed, but session {session_id!r} could not "
                f"be set to the model it was reindexed with: {e}. Run: lasagna configure "
                f"--profile <name> -s {session_id}"
            )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return UploadResult(
            document_id=doc.document_id,
            filename=doc.filename,
            status=_resolve_status(doc.status),
            chunk_count=doc.chunks_count,
            processing_time_ms=processing_time,
            message="Document reindexed successfully.",
        )
