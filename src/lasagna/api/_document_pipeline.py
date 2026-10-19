"""Document upload, processing, and management pipeline.

Handles document upload, text extraction (with OCR fallback), chunking,
embedding generation, vector indexing, and document deletion.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import uuid
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..processing.document_processor import DocumentProcessor

from loguru import logger
from lasagna.core.provenance import redact_url

from lasagna.core.config import rag_config, RAGConfig
from lasagna.core.schema import (
    Document,
    DocumentStatus,
    DocumentMetadata,
)
from ..vector_backends import (
    get_backend,
    FilterExpression,
)
from ..processing.chunker.semantic import chunk_document, SemanticChunker
from ..processing.docling_processor import is_docling_available, DoclingProcessor
from ..infrastructure.embedding import EmbeddingManager
from ..infrastructure.cache import get_search_cache
from ..retrieval.search.hybrid import HybridSearcher
from lasagna.utils.monitoring import METRICS
from lasagna.utils.tracing import TRACER

from ._validation import (
    is_valid_session_id,
    sanitize_filename,
    is_safe_document_id,
)
from lasagna.core.tenant import get_tenant_manager
from lasagna.core.quota import get_quota_manager
from lasagna.infrastructure.tasks.document_tasks import (
    ProcessingAtCapacity,
    _failure_reason,
    dispatch_processing,
)

if TYPE_CHECKING:
    from ._collection_manager import CollectionManager
    from lasagna.storage.base import StorageBackend


def is_dim_mismatch_error(exc: BaseException) -> bool:
    """Whether an insert failed because the vectors do not fit the collection's size.

    Milvus Lite 3 says "vector field 'embedding' expected dim 1024, got 384".
    Matching only the older wordings, the first upload of a session at a size
    other than the server's default never rebuilt the empty collection made at
    startup: the document failed, and was retried twice, 30 s apart, to fail
    the same way.
    """
    message = str(exc).lower()
    return (
        "divide the dim" in message
        or "expected dim" in message
        or ("dimension" in message and "length" in message)
    )


class ChunkStoreUnavailable(RuntimeError):
    """The vector store could not be read; REST answers 503."""


class DocumentNotFound(LookupError):
    """No document with this id in this session; REST answers 404."""


class SourceFileMissing(FileNotFoundError):
    """A document's stored file is gone, so it cannot be processed again; REST answers 410."""


class DocumentBusy(RuntimeError):
    """The document is being processed or is archived, so it is not replaced now; REST 409."""


class ReplacementNotRecorded(RuntimeError):
    """A reindex removed the old chunks and could not record the new copy, whose chunks it removed.

    The document has no chunks until it is reindexed again; a profile switch
    reindexes it back with the session's settings.
    """


class DocumentRemoved(DocumentNotFound):
    """The document was deleted while it was being processed; REST answers 404 with why.

    Its new chunks are removed before this is raised. The reindex used to
    finish anyway: its chunks stayed in the vector store with no row, and the
    request failed with a 500 once it read the row back.
    """


def _store_lock(name: str):
    """The metadata store's lock ``name``, held across every process that shares the store.

    These were ``threading`` locks, while the image runs four gunicorn workers,
    Kubernetes two to ten API pods, and MCP and Celery processes of their own:
    two identical uploads on two workers were both indexed, and a delete on
    one pod during a reindex on another left the reindexed chunks in the
    vector store with no row.
    """
    from lasagna.core.store_locks import store_lock

    return store_lock(name)


def _document_lock(document_id: str):
    """Replacing, retrying and deleting a document hold its lock.

    So two reindexes of one document do not each retire the other's chunks,
    and a delete during a reindex is not undone by the reindex finishing.
    """
    return _store_lock(f"document:{document_id}")


def _content_lock(session_id: str, collection: str, digest: str):
    """Identical uploads to one session's collection hold it from their
    duplicate check until the first one's row is saved."""
    return _store_lock(f"content:{session_id}:{collection}:{digest}")


def _digest(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def _file_digest(path: str) -> Optional[str]:
    """SHA-256 of a local file, or None when it cannot be read."""
    try:
        with open(path, "rb") as f:
            return _digest(f)
    except OSError:
        return None


#: Whether the Docling fallback has been logged in this process (once is enough).
_docling_fallback_said = False


class DocumentPipeline:
    """Manages document upload, processing, and lifecycle.

    Handles:
    - Document upload with file validation and storage
    - Text extraction with OCR fallback
    - Semantic chunking and embedding generation
    - Vector database indexing
    - Document metadata management (SQLite)
    - Document deletion with full cleanup
    """

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_manager: EmbeddingManager,
        chunker: SemanticChunker,
        hybrid_searcher: HybridSearcher,
        collection_manager: "CollectionManager",
        storage: Optional["StorageBackend"] = None,
    ):
        self.document_processor = document_processor
        self.embedding_manager = embedding_manager
        self.chunker = chunker
        self.hybrid_searcher = hybrid_searcher
        self._collection_manager = collection_manager
        self._storage = storage
        self._rebuild_lock = threading.Lock()
        self._rebuilding = False

    @property
    def storage(self) -> "StorageBackend":
        """Lazily resolve the storage backend."""
        if self._storage is None:
            from lasagna.storage import get_storage_backend

            self._storage = get_storage_backend()
        return self._storage

    # =========================================================================
    # Upload
    # =========================================================================

    def upload_document(
        self,
        file_path: str,
        session_id: str,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
        _chunker_override: Any = None,
        _document_processor_override: Any = None,
        refuse_when_busy: bool = False,
        wait_when_busy: float = 0.0,
        allow_duplicates: bool = False,
    ) -> Document:
        """Upload a document and run it through extraction, chunking and indexing.

        A file whose content is already in the session, ready or still
        processing, is not uploaded again: the existing document comes back
        with ``duplicate_of`` set. Rerunning the README's quickstart uploaded
        its paper once per run, and search returned the same passage once per
        copy. A file whose only copies in the session's collection failed is
        uploaded again under the newest failed copy's id, which it replaces:
        each rerun of a failing upload used to add another failed document.
        ``allow_duplicates`` indexes it again anyway. An explicit
        ``document_id`` or a component override is never deduplicated: the
        caller then decides what the document is (a reindex, a benchmark
        corpus, a chunker under test).

        Args:
            file_path: Path to the uploaded file
            session_id: Session ID
            filename: Original filename
            document_id: Optional explicit document ID. When provided, used
                instead of generating a random UUID. Useful for benchmark
                evaluation where corpus doc IDs must match qrels. An id the
                session already holds replaces that document
                (``_replace_document``, processed in this thread); one held
                by another session raises ValueError.
            allow_duplicates: Index the file even when the session already
                holds the same content.

        Returns:
            Document metadata
        """
        from lasagna.api._profile_guard import refuse_during_switch
        from lasagna.core.session_config._session_state import session_lock

        refuse_during_switch(session_id, "The upload")
        retired: Optional[str] = None
        admission = contextlib.ExitStack()
        # First, before any document's lock: held until this upload's row is
        # saved, so a profile switch that moves the session's documents sees
        # it or runs before it. One that started after this upload's settings
        # were read left it in the old model's collection.
        session = contextlib.ExitStack()
        session.enter_context(session_lock(session_id))
        admission.callback(session.close)
        try:
            refuse_during_switch(session_id, "The upload")
        except BaseException:
            admission.close()
            raise
        if document_id is not None:
            from lasagna.core.database import get_rag_database

            # Held from the check until this upload's row is saved: two
            # uploads under one new id both found it free, and the second's
            # row replaced the first's, session included.
            admission.enter_context(_document_lock(document_id))
            held = get_rag_database().get_document(document_id)
            if held is not None and held.get("session_id") != session_id:
                admission.close()
                raise ValueError(
                    f"document_id {document_id!r} belongs to another session; choose another id."
                )
            if held is not None:
                # The id names a document this session holds: its new copy
                # replaces it (chunks, file and quota), in this thread. The old
                # copy's chunks used to stay: twice in Milvus, past the new
                # count where the store upserts, and the quota counted both.
                # The session's lock is held for the settings this reads and
                # no longer: a replacement runs for as long as the file takes
                # to index, and every other write to the session waited for it.
                # A profile switch that starts now reindexes this document
                # after the replacement, under its lock.
                try:
                    config = self._get_session_config(session_id)
                finally:
                    session.close()
                with admission:
                    return self._replace_document(
                        held,
                        config=config,
                        source_path=file_path,
                        filename=filename,
                        counts_as_upload=True,
                        _chunker_override=_chunker_override,
                        _document_processor_override=_document_processor_override,
                    )

        if not (
            allow_duplicates
            or document_id is not None
            or _chunker_override is not None
            or _document_processor_override is not None
        ):
            collection = getattr(self._get_session_config(session_id), "collection_name", "")
            digest = _file_digest(file_path)
            if digest is not None:
                # Held until this upload's row is saved, where an identical
                # upload waiting here then finds it. Both used to pass the
                # check at once and both were indexed.
                admission.enter_context(_content_lock(session_id, collection, digest))
            existing = self._find_same_content(file_path, session_id, collection, digest)
            if existing is not None:
                admission.close()
                logger.info(
                    f"{filename or os.path.basename(file_path)} has the same content as "
                    f"{existing.document_id} in session {session_id}; not uploaded again"
                )
                return existing.model_copy(update={"duplicate_of": existing.document_id})
            failed = self._find_failed_copy(file_path, session_id, collection, digest)
            if failed is not None:
                # Retried in place: the failed copy (row, file, any chunks it
                # left) goes, and this upload takes its id.
                admission.enter_context(_document_lock(failed))
                retired = document_id = failed
                logger.info(
                    f"{filename or os.path.basename(file_path)} has the same content as "
                    f"{failed}, whose processing failed; processing it again under that id"
                )

        with admission:
            document, config, stored_file_path, reservation_id, file_size_for_quota = (
                self._admit_upload(
                    file_path, session_id, filename, document_id, retire_failed=retired
                )
            )
        quota_mgr = get_quota_manager()
        document_id, filename = document.document_id, document.filename
        file_type, file_size = document.file_type, document.file_size

        logger.info(f"Document uploaded: {document_id} ({filename})")

        # --- Celery async path: dispatch and return immediately ---
        try:
            dispatched = dispatch_processing(
                document_id,
                session_id,
                stored_file_path,
                reservation_id=reservation_id,
                reserved_file_size=file_size_for_quota,
                refuse_when_busy=refuse_when_busy,
                wait_when_busy=wait_when_busy,
            )
        except ProcessingAtCapacity:
            # Refused before any processing: leave nothing behind, so the
            # caller's retry is a clean upload rather than a duplicate.
            self._forget_unprocessed_upload(document_id, stored_file_path)
            quota_mgr.release_upload_reservation(
                session_id, file_size_for_quota, reservation_id=reservation_id
            )
            raise
        if dispatched:
            METRICS.counter("rag.document.uploads", 1)
            return document  # status=PROCESSING; client polls for completion

        # --- Synchronous path (default / Celery not enabled) ---

        with TRACER.span(
            "document_process",
            attributes={
                "document_id": document_id,
                "filename": filename or "",
                "file_type": file_type or "",
                "file_size": file_size,
            },
        ) as _span:
            try:
                self.process_document(
                    document,
                    stored_file_path,
                    config=config,
                    _chunker_override=_chunker_override,
                    _document_processor_override=_document_processor_override,
                )
            except Exception as e:
                logger.bind(document_failure=document_id).error(
                    f"Failed to process document {document_id}: {e}"
                )
                METRICS.counter("lasagna_ingests", 1, tags={"outcome": "error"})
                try:
                    self._update_document_status(
                        document_id,
                        DocumentStatus.ERROR,
                        error_message=_failure_reason(e, session_id),
                    )
                except Exception as status_exc:
                    logger.warning(f"Failed to update error status for {document_id}: {status_exc}")
                finally:
                    quota_mgr.release_upload_reservation(
                        session_id, file_size_for_quota, reservation_id=reservation_id
                    )
                raise

        # Refresh document object from database to get updated status and chunk count
        updated_doc = self._get_document_metadata(document_id)
        if updated_doc:
            document.status = updated_doc.status
            document.chunks_count = updated_doc.chunks_count
            logger.debug(
                f"Document refreshed after processing: "
                f"status={document.status.value}, chunks={document.chunks_count}"
            )

        # Invalidate search cache when new document is added
        if config.enable_search_cache:
            try:
                search_cache = get_search_cache(config)
                invalidated = search_cache.invalidate_session(session_id)
                if invalidated > 0:
                    logger.debug(
                        f"Search cache invalidated after upload: {invalidated} entries for session {session_id[:8]}..."
                    )
            except Exception as e:
                logger.warning(f"Failed to invalidate search cache after upload: {e}")

        METRICS.counter("rag.document.uploads", 1)

        # Finalize reservation with actual chunk count
        quota_mgr.finalize_upload_reservation(
            session_id=session_id,
            chunks=document.chunks_count,
        )

        return document

    def _admit_upload(
        self,
        file_path: str,
        session_id: str,
        filename: Optional[str],
        document_id: Optional[str],
        *,
        retire_failed: Optional[str] = None,
    ) -> tuple:
        """Check an upload, reserve its quota, store its file and save its row.

        Returns the document, the session's config, the stored file's URI,
        the quota reservation and the size it reserved. Anything that refuses
        the upload raises before a file or a row is left behind.
        ``retire_failed``: the failed copy this upload retries under its id,
        deleted once nothing can refuse the upload.
        """
        config = self._get_session_config(session_id)
        # Before anything is stored: a model set without its size, whose size
        # cannot be read, would make a collection of the wrong size.
        unknown = getattr(config, "embedding_size_unknown", None)
        problem = unknown() if callable(unknown) else None
        if isinstance(problem, str) and problem:
            raise ValueError(problem)

        # Quota enforcement — fast-fail before expensive operations (embedding
        # manager reload, file I/O).  The authoritative atomic reservation
        # happens later via check_and_reserve_upload().
        quota_mgr = get_quota_manager()
        file_size_for_quota = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        quota_mgr.check_upload(session_id, file_size_bytes=file_size_for_quota)

        embedder = self.use_embedder_for(config)
        logger.debug(
            f"Document load using embedding model: {embedder.model_name} (dim={embedder.dimension})"
        )

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > config.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {config.max_file_size})")

        if filename is None:
            filename = os.path.basename(file_path)

        file_type = self.document_processor.detect_file_type(file_path)
        if file_type not in config.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_type}")

        document_id = document_id or f"doc_{uuid.uuid4().hex[:16]}"

        document = Document(
            document_id=document_id,
            session_id=session_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            status=DocumentStatus.PROCESSING,
            chunks_count=0,
            upload_date=datetime.now(timezone.utc),
        )

        # --- Reserve quota BEFORE any I/O (file storage + metadata write) ---
        # This prevents orphaned files/metadata when a concurrent upload
        # exhausts the quota between validation and storage.
        reservation_id = quota_mgr.check_and_reserve_upload(
            session_id, file_size_bytes=file_size_for_quota
        )

        if retire_failed is not None:
            # Only now that the upload is admitted: deleted before the checks,
            # a retry the quota or the file type refused lost the failed copy
            # too, and the session kept neither.
            try:
                self._delete_document(retire_failed, session_id, keep_file=False)
            except Exception:
                quota_mgr.release_upload_reservation(
                    session_id, file_size_for_quota, reservation_id=reservation_id
                )
                raise

        try:
            stored_file_path = self._store_file(file_path, document_id, filename)
        except Exception:
            quota_mgr.release_upload_reservation(
                session_id, file_size_for_quota, reservation_id=reservation_id
            )
            raise

        try:
            upload_backend = self._collection_manager.resolve_backend(session_id)
            self._save_document_metadata(
                document, stored_file_path, vector_backend_name=upload_backend.backend_name
            )
        except Exception:
            # Best-effort cleanup of already-stored file
            try:
                self.storage.delete(stored_file_path)
            except Exception as cleanup_exc:
                logger.warning(
                    f"Failed to clean up stored file after metadata error: {cleanup_exc}"
                )
            quota_mgr.release_upload_reservation(
                session_id, file_size_for_quota, reservation_id=reservation_id
            )
            raise

        return document, config, stored_file_path, reservation_id, file_size_for_quota

    def _forget_unprocessed_upload(self, document_id: str, stored_file_path: str) -> None:
        """Remove the metadata row and stored file of an upload that was refused."""
        from lasagna.core.database import get_rag_database

        try:
            get_rag_database().delete_document(document_id)
        except Exception as exc:
            logger.warning(f"Could not remove the record of refused upload {document_id}: {exc}")
        try:
            self.storage.delete(stored_file_path)
        except Exception as exc:
            logger.warning(f"Could not remove the file of refused upload {document_id}: {exc}")

    def _find_same_content(
        self,
        file_path: str,
        session_id: str,
        collection_name: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> Optional[Document]:
        """The session's document with the same bytes as ``file_path``, if any.

        Only documents of the same size are hashed, so an upload to a session
        of other files reads nothing but its own file. A document counts when
        it is in ``collection_name`` (the session's current collection: after a
        model switch the old model's copy is not searchable with the new one)
        and is ready, or still within the time processing may take. A failed
        document, or one stuck processing past that time, does not count:
        uploading again is how it is retried.
        """
        from lasagna.lifecycle.gc import longest_processing

        unfinished = {
            DocumentStatus.UPLOADING.value,
            DocumentStatus.PROCESSING.value,
            DocumentStatus.INDEXING.value,
        }
        still_running_since = datetime.now(timezone.utc) - longest_processing()

        def counts(row: Dict[str, Any]) -> bool:
            if row.get("status") == DocumentStatus.READY.value:
                return True
            if row.get("status") not in unfinished:
                return False
            try:
                started = datetime.fromisoformat(str(row.get("upload_date")))
            except ValueError:
                return True
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return started >= still_running_since

        for row in self._same_content(file_path, session_id, collection_name, digest, counts):
            return self._get_document_metadata(row["document_id"])
        return None

    def _find_failed_copy(
        self,
        file_path: str,
        session_id: str,
        collection_name: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> Optional[str]:
        """The id of the newest failed document in the session with these bytes, if any.

        In ``collection_name`` only: after a model switch the failed copy
        belongs to the old model's collection.
        """

        def failed(row: Dict[str, Any]) -> bool:
            return row.get("status") == DocumentStatus.ERROR.value

        for row in self._same_content(file_path, session_id, collection_name, digest, failed):
            return row["document_id"]  # rows come newest first
        return None

    def _same_content(self, file_path, session_id, collection_name, digest, counts):
        """The session's rows that ``counts`` and whose stored file has ``file_path``'s bytes."""
        from lasagna.core.database import get_rag_database

        try:
            size = os.path.getsize(file_path)
        except OSError:
            return

        def candidate(row: Dict[str, Any]) -> bool:
            if row.get("file_size") != size or not row.get("file_path"):
                return False
            if collection_name and row.get("collection_name") not in (None, "", collection_name):
                return False
            return counts(row)

        candidates = [r for r in get_rag_database().get_documents(session_id) if candidate(r)]
        if not candidates:
            return
        wanted = digest or _file_digest(file_path)
        for row in candidates:
            try:
                with self.storage.retrieve(row["file_path"]) as stored:
                    same = _digest(stored) == wanted
            except Exception as exc:  # noqa: BLE001 - a missing file is no duplicate
                logger.debug(f"Could not read {row['document_id']} to compare: {exc}")
                continue
            if same:
                yield row

    # =========================================================================
    # Reindex and replacement
    # =========================================================================

    def reindex_document(
        self, document_id: str, session_id: str, *, config: Optional[RAGConfig] = None
    ) -> Document:
        """Process a stored document again with the session's current settings.

        ``config`` replaces those settings (``RAGConfig.with_session_settings``):
        a profile switch reindexes into the new profile before it switches.

        The document keeps answering searches from its old chunks until the new
        ones are stored; then the old chunks go and its row points at the new
        ones, in the collection of the session's current embedding model. Every
        reindex path used to delete the document first, so a model that was not
        downloaded, a quota, a file type the settings no longer accept or an
        extraction error lost it.

        Raises:
            DocumentNotFound: no such document in this session.
            SourceFileMissing: its stored file is gone.
            DocumentBusy: it is still being processed, or is archived.
            ModelNotDownloaded, QuotaExceededError, ValueError: the new copy
                cannot be made; the document is unchanged.
        """
        from lasagna.api._profile_guard import refuse_during_switch

        # During a profile switch that moves the session's documents (another
        # thread's or process's): this reindex would put the document back
        # where the switch had just moved it from. Checked again under the
        # lock, which the switch takes for each document it moves.
        refuse_during_switch(session_id, "The reindex")
        # The row is read under the lock: read before it, a second reindex
        # waiting for the first replaced the copy the first had already
        # retired, and after a model switch both copies stayed.
        with _document_lock(document_id):
            refuse_during_switch(session_id, "The reindex")
            return self._reindex_held(self._owned_row(document_id, session_id), config)

    def _reindex_held(self, row: Dict[str, Any], config: Optional[RAGConfig] = None) -> Document:
        document_id = row["document_id"]
        stored = row.get("file_path") or ""
        storage = self._storage_of(row)
        if not stored or not storage.exists(stored):
            raise SourceFileMissing(
                f"The source file of document {document_id} is no longer in storage, so it "
                "cannot be reindexed. Upload the file again to index it (with "
                f"document_id={document_id} to keep the id)."
            )
        if storage.backend_name == self.storage.backend_name:
            return self._replace_document(row, config=config)
        # Stored on another storage backend than the current one: the new copy
        # is stored on the current one, as an upload would be.
        local = storage.ensure_local(stored)
        try:
            return self._replace_document(
                row, source_path=local, filename=row.get("filename"), config=config
            )
        finally:
            if local != stored and os.path.isfile(local):
                os.unlink(local)

    def _owned_row(self, document_id: str, session_id: str) -> Dict[str, Any]:
        from lasagna.core.database import get_rag_database

        row = None
        if is_safe_document_id(document_id) and is_valid_session_id(session_id):
            row = get_rag_database().get_document(document_id)
        if not row or row.get("session_id") != session_id:
            raise DocumentNotFound(f"Document {document_id} not found in session {session_id}")
        return row

    def _storage_of(self, row: Dict[str, Any]) -> "StorageBackend":
        """The storage backend that holds ``row``'s file."""
        name = row.get("storage_backend_name") or "local"
        if self._storage is not None and self._storage.backend_name == name:
            return self._storage
        from lasagna.storage import get_storage_backend

        try:
            return get_storage_backend(name)
        except Exception:
            return self.storage

    def _replace_document(self, row: Dict[str, Any], **kwargs: Any) -> Document:
        """`_replace_held` with the document's lock held and its row read again under it.

        ``row`` may be stale: read before the lock, while another reindex of
        the document held it. That one replaced the copy ``row`` describes, so
        replacing it again left the other's chunks in place (twice after a
        model switch: the stale row named the old collection, which no longer
        held any chunk to retire).
        """
        document_id, session_id = row["document_id"], row["session_id"]
        with _document_lock(document_id):
            try:
                current = self._owned_row(document_id, session_id)
            except DocumentNotFound:
                raise DocumentRemoved(
                    f"Document {document_id} was deleted before it could be replaced."
                ) from None
            return self._replace_held(current, **kwargs)

    def _replace_held(
        self,
        row: Dict[str, Any],
        *,
        source_path: Optional[str] = None,
        filename: Optional[str] = None,
        counts_as_upload: bool = False,
        _chunker_override: Any = None,
        _document_processor_override: Any = None,
        config: Optional[RAGConfig] = None,
    ) -> Document:
        """Index a new copy of the document in ``row``, then retire the old copy.

        ``source_path`` is a local file stored as the new copy (an upload under
        the document's id); without it the stored file is processed again.
        Every check that can refuse runs before anything is written, and the old
        copy (row, chunks, file, quota) changes only once the new chunks are in.
        Processed in the calling thread, never queued: the old copy has to stay
        in place until this copy is known to be complete.
        """
        document_id, session_id = row["document_id"], row["session_id"]
        status = row.get("status")
        if status in (
            DocumentStatus.PROCESSING.value,
            DocumentStatus.UPLOADING.value,
            DocumentStatus.INDEXING.value,
        ):
            raise DocumentBusy(
                f"Document {document_id} is still being processed. Wait until its status "
                f"(GET /api/v1/documents/{document_id}/status) is ready or error, then retry."
            )
        if status == DocumentStatus.SOFT_DELETED.value:
            raise DocumentBusy(
                f"Document {document_id} is archived. Restore it first "
                f"(POST /api/v1/documents/{document_id}/restore)."
            )

        config = config or self._get_session_config(session_id)
        unknown = getattr(config, "embedding_size_unknown", None)
        problem = unknown() if callable(unknown) else None
        if isinstance(problem, str) and problem:
            raise ValueError(problem)
        # The model is loadable and its collection open before anything changes.
        self.use_embedder_for(config)

        if source_path is not None:
            if not os.path.exists(source_path):
                raise ValueError(f"File not found: {source_path}")
            filename = filename or os.path.basename(source_path)
            file_size = os.path.getsize(source_path)
            file_type = self.document_processor.detect_file_type(source_path)
        else:
            filename = row.get("filename")
            file_size = int(row.get("file_size") or 0)
            file_type = row.get("file_type")
        if file_size > config.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {config.max_file_size})")
        if file_type not in config.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_type}")

        old_size = int(row.get("file_size") or 0)
        quota_mgr = get_quota_manager()
        quota_mgr.check_replace(
            session_id,
            old_file_size=old_size,
            new_file_size=file_size,
            counts_as_upload=counts_as_upload,
        )

        with _document_lock(document_id):
            new_file = None
            if source_path is not None:
                # Its own folder: storage.delete removes a file's whole folder,
                # and the old copy's is removed once this one is in.
                new_file = self._store_file(
                    source_path, f"{document_id}__r{uuid.uuid4().hex[:8]}", filename
                )
            document = Document(
                document_id=document_id,
                session_id=session_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                status=DocumentStatus.PROCESSING,
                chunks_count=0,
                upload_date=(
                    datetime.now(timezone.utc)
                    if new_file
                    else datetime.fromisoformat(row["upload_date"])
                    if isinstance(row.get("upload_date"), str)
                    else row.get("upload_date") or datetime.now(timezone.utc)
                ),
            )
            try:
                self.process_document(
                    document,
                    new_file or row["file_path"],
                    config=config,
                    _chunker_override=_chunker_override,
                    _document_processor_override=_document_processor_override,
                    replacing=row,
                )
            except Exception:
                if new_file:
                    try:
                        self.storage.delete(new_file)
                    except Exception as exc:
                        logger.warning(f"Could not remove the unused new file {new_file}: {exc}")
                raise

            if new_file and row.get("file_path") and row["file_path"] != new_file:
                try:
                    self._storage_of(row).delete(row["file_path"])
                except Exception as exc:
                    logger.warning(
                        f"Could not remove the replaced file of {document_id} "
                        f"({row['file_path']}): {exc}"
                    )
            replaced = self._get_document_metadata(document_id)
        if replaced is None:
            raise DocumentRemoved(
                f"Document {document_id} was deleted while it was being reindexed."
            )
        quota_mgr.record_replace(
            session_id,
            old_file_size=old_size,
            new_file_size=file_size,
            old_chunks=int(row.get("chunks_count") or 0),
            new_chunks=replaced.chunks_count,
            counts_as_upload=counts_as_upload,
        )
        self._invalidate_search_cache(session_id)
        logger.info(
            f"Document {document_id} replaced: {row.get('chunks_count') or 0} chunks in "
            f"{row.get('collection_name')!r} became "
            f"{replaced.chunks_count} in {config.collection_name!r}"
        )
        return replaced

    def _replaced_chunks(
        self,
        replaced: Dict[str, Any],
        config: RAGConfig,
        backend: Any,
        partition_key: Optional[str],
    ) -> tuple:
        """Where the replaced copy's chunks are, and their ids.

        Read before the new copy is inserted, so the ids are the old copy's
        only; a failure here leaves everything as it was.
        """
        old_backend_name = replaced.get("vector_backend_name") or backend.backend_name
        old_collection = replaced.get("collection_name") or config.collection_name
        old_backend = (
            backend if old_backend_name == backend.backend_name else get_backend(old_backend_name)
        )
        if not old_backend.is_connected():
            old_backend.connect()
        if not old_backend.has_collection(old_collection):
            return old_backend, old_collection, []
        rows = old_backend.iter_all(
            collection_name=old_collection,
            filter_expr=FilterExpression.eq("document_id", replaced["document_id"]),
            output_fields=["chunk_id"],
            partition_key=partition_key,
        )
        ids = [str(r.get("chunk_id") or r.get("id")) for r in rows]
        return old_backend, old_collection, [i for i in ids if i and i != "None"]

    @staticmethod
    def _delete_chunk_ids(
        backend: Any, collection: str, ids: List[str], partition_key: Optional[str]
    ) -> None:
        for start in range(0, len(ids), 1000):  # FilterExpression.in_list's limit
            backend.delete(
                collection_name=collection,
                filter_expr=FilterExpression.in_list("chunk_id", ids[start : start + 1000]),
                partition_key=partition_key,
            )

    def _discard_new_chunks(
        self,
        document: Document,
        backend: Any,
        collection: str,
        ids: List[str],
        partition_key: Optional[str],
        *,
        indexed_here: bool,
        quiet: bool = False,
    ) -> None:
        """Remove chunks this processing inserted, best effort and logged.

        Ids of their own, so the document's other chunks (the copy a reindex
        replaces) are untouched. What cannot be removed here the orphan scan
        finds: a document's chunks with no row, or chunks no row counts.
        ``quiet``: the insert itself failed, and may have stored none of them.
        """
        try:
            self._delete_chunk_ids(backend, collection, ids, partition_key)
        except Exception as exc:
            (logger.debug if quiet else logger.warning)(
                f"Could not remove the {len(ids)} new chunk(s) of {document.document_id} from "
                f"{collection!r} ({type(exc).__name__}: {exc}); the weekly orphan scan, "
                "or POST /api/v1/admin/gc/repair?dry_run=false, removes them."
            )
        if indexed_here:
            try:
                self.hybrid_searcher.remove_document(document.document_id, document.session_id)
            except Exception as exc:
                logger.debug(f"Keyword index kept {document.document_id}: {exc}")

    def _retire_replaced_chunks(
        self,
        document: Document,
        old: tuple,
        config: RAGConfig,
        backend: Any,
        new_ids: List[str],
        partition_key: Optional[str],
    ) -> None:
        """Remove the replaced copy's chunks, now that the new ones are stored.

        The new chunks have ids of their own, so deleting the old ids never
        touches them. If the old ones cannot be removed, the new ones are, and
        the document stays the old copy.
        """
        old_backend, old_collection, old_ids = old
        try:
            self._delete_chunk_ids(old_backend, old_collection, old_ids, partition_key)
        except Exception as exc:
            try:
                self._delete_chunk_ids(backend, config.collection_name, new_ids, partition_key)
            except Exception as cleanup_exc:
                logger.warning(
                    f"Could not remove the new chunks of {document.document_id} either: "
                    f"{cleanup_exc}"
                )
            raise RuntimeError(
                f"The previous chunks of {document.document_id} in {old_collection!r} could "
                f"not be removed ({type(exc).__name__}: {exc}), so the new copy was not kept "
                "and the document is unchanged."
            ) from exc

    def _store_file(self, source_path: str, document_id: str, filename: str) -> str:
        """Store uploaded file via the pluggable storage backend.

        SECURITY: Filename is sanitized to prevent path traversal attacks.
        """
        if not is_safe_document_id(document_id):
            raise ValueError(f"Invalid document_id: {document_id!r}")

        safe_filename = sanitize_filename(filename)

        with open(source_path, "rb") as f:
            info = self.storage.store(f, document_id, safe_filename)

        logger.info(f"File stored: {redact_url(info.uri)}")
        return info.uri

    def _save_document_metadata(
        self, document: Document, file_path: str, vector_backend_name: str = None
    ):
        """Save document metadata to SQLite."""
        from lasagna.core.database import get_rag_database
        from lasagna.lifecycle.policy import LifecyclePolicy
        from datetime import timedelta

        if vector_backend_name is None:
            vector_backend_name = self._collection_manager._default_backend_name

        # Compute expires_at from LifecyclePolicy.document_ttl_days (None = never expire)
        expires_at = None
        try:
            policy = LifecyclePolicy.from_env()
            if policy.document_ttl_days is not None:
                from datetime import datetime, timezone

                expires_at = (
                    datetime.now(timezone.utc) + timedelta(days=policy.document_ttl_days)
                ).isoformat()
        except Exception:
            pass  # TTL is optional; don't fail the upload if policy is unavailable

        rag_db = get_rag_database()
        # A plain insert: an id taken meanwhile is refused, never overwritten.
        rag_db.save_document(
            {
                "document_id": document.document_id,
                "session_id": document.session_id,
                "filename": document.filename,
                "file_size": document.file_size,
                "file_type": document.file_type,
                "status": document.status.value,
                "chunks_count": document.chunks_count,
                "upload_date": document.upload_date.isoformat(),
                "vector_backend_name": vector_backend_name,
                "storage_backend_name": self.storage.backend_name,
                "expires_at": expires_at,
            },
            file_path,
            # Each embedding model has its own collection; deleting the
            # document later must find the one its vectors went to.
            collection_name=self._get_session_config(document.session_id).collection_name,
            replace=False,
        )

    # =========================================================================
    # Processing
    # =========================================================================

    def use_embedder_for(self, config: RAGConfig) -> EmbeddingManager:
        """The embedder for ``config``, with the collection it writes to ready.

        The upload route checks this before it stores a file, and
        ``process_document`` resolves it again where it indexes, which may be
        another process (the Celery worker): there the worker had embedded with
        whatever model it had loaded and written to a collection nobody had
        created.

        The embedder is returned, not stored on the pipeline, which every
        request shares: two sessions on different models uploading at once
        each read the other's model from the shared attribute, and one
        collection received the other model's vectors.
        """
        return self._collection_manager.embedder_for(config)

    def process_document(
        self,
        document: Document,
        file_path: str,
        config: Optional[RAGConfig] = None,
        _chunker_override: Any = None,
        _document_processor_override: Any = None,
        *,
        replacing: Optional[Dict[str, Any]] = None,
        clear_previous_attempt: bool = False,
    ):
        """Extract, chunk, embed and index an uploaded document.

        ``_chunker_override`` is a ``BaseChunker`` instance that replaces the
        built-in chunking strategies for this document, which is what makes a
        chunking experiment possible at all.

        ``replacing`` is the stored row of the copy this one replaces (see
        ``_replace_document``). The new chunks go in under ids of their own;
        only then are the old ones removed and the row pointed at the new
        copy, so a failure before that leaves the old copy as it was.

        A failure after the chunks are inserted removes them again (best
        effort, logged; the orphan scan is the backstop): they used to stay,
        and search served the chunks of a document that had failed.
        ``clear_previous_attempt`` first removes chunks an earlier attempt at
        this document left (a retry, a Celery message delivered again after its
        worker died): Milvus ``insert`` kept both copies of each id.
        """
        start_time = _time.monotonic()
        logger.info(f"Processing document: {document.document_id}")
        config = config or rag_config
        embedder = self.use_embedder_for(config)
        # A new copy's chunk ids never collide with the old copy's: Milvus
        # `insert` keeps both rows of one id, and a backend that upserts would
        # keep any old chunk past the new count.
        chunk_prefix = (
            f"{document.document_id}_r{uuid.uuid4().hex[:8]}" if replacing else document.document_id
        )

        # Ensure we have a local path for processors (Docling, OCR, etc.)
        # For local storage this is zero-copy; for cloud storage it downloads to temp.
        local_path = self.storage.ensure_local(file_path)
        is_temp = local_path != self.storage.get_local_path(file_path)

        # --- Text extraction (needs local file) ---
        _stage_times: Dict[str, float] = {}
        _t_extract_start = _time.monotonic()
        chunks_with_metadata = None
        parent_chunks_data = None
        text = ""
        text_length = 0

        try:
            use_docling = self._should_use_docling(local_path, config)
            if use_docling:
                try:
                    processor = DoclingProcessor.get_instance(config)
                    chunks_with_metadata, parent_chunks_data, text = processor.process(
                        file_path=local_path,
                        document_id=document.document_id,
                        session_id=document.session_id,
                        filename=document.filename,
                        config=config,
                    )
                    text_length = len(text)
                except Exception as e:
                    logger.warning(f"Docling processing failed, falling back to legacy: {e}")
                    chunks_with_metadata = None

            # --- Legacy path: text extraction + semantic chunking ---
            if chunks_with_metadata is None:
                if _document_processor_override is not None:
                    # A registered processor owns extraction outright: it
                    # decides how bytes become text, which is the stage under
                    # test. Running OCR settings on top of someone else's
                    # extractor would mean the measured system is not the one
                    # that was configured.
                    from ..core.base_components import BaseDocumentProcessor

                    if not isinstance(_document_processor_override, BaseDocumentProcessor):
                        raise TypeError(
                            f"document_processor must implement BaseDocumentProcessor, "
                            f"got {type(_document_processor_override).__name__}. Run "
                            f"`lasagna component validate <module:Class>` to check."
                        )
                    processed = _document_processor_override.extract(local_path)
                    text = getattr(processed, "text", None)
                    if text is None:
                        raise TypeError(
                            "document_processor.extract() must return a "
                            "ProcessedDocument with a `text` attribute."
                        )
                    logger.info(
                        f"Extracted with injected component: {_document_processor_override.name}"
                    )
                else:
                    text = self._extract_text_with_ocr(local_path, document.file_type, config)
                text_length = len(text)

                if not text or not text.strip():
                    raise ValueError("No text could be extracted from document")

                if config.enable_hierarchical:
                    chunk_result = chunk_document(
                        text=text,
                        document_id=document.document_id,
                        session_id=document.session_id,
                        filename=document.filename,
                        embedding_manager=embedder,
                        config=config,
                        return_parents=True,
                        chunker=_chunker_override,
                    )
                    if isinstance(chunk_result, tuple):
                        chunks_with_metadata, parent_chunks_data = chunk_result
                    else:
                        chunks_with_metadata = chunk_result
                else:
                    chunks_with_metadata = chunk_document(
                        text=text,
                        document_id=document.document_id,
                        session_id=document.session_id,
                        filename=document.filename,
                        embedding_manager=embedder,
                        config=config,
                        chunker=_chunker_override,
                    )
        finally:
            # Clean up temp file if we downloaded from cloud storage
            if is_temp:
                try:
                    if os.path.exists(local_path):
                        os.unlink(local_path)
                except (OSError, IOError) as e:
                    logger.warning(f"Failed to clean up temp file {local_path}: {e}")

        if not chunks_with_metadata:
            raise ValueError("No chunks created from document")

        _stage_times["extract_chunk"] = _time.monotonic() - _t_extract_start

        # Preserve original (pre-enrichment) chunk texts for late chunking.
        # Late chunking needs texts that are substrings of full_text for offset mapping.
        _original_chunk_texts = [ct for ct, _ in chunks_with_metadata]

        # Contextual retrieval enrichment (Anthropic-style chunk-level context)
        # Prepends each chunk with an LLM-generated context sentence that
        # situates the chunk within the broader document.
        # Truncate document context to 10K chars to reduce LLM token cost
        # (~5x cheaper than the default 50K truncation in enricher.enrich()).
        if getattr(config, "enable_contextual_retrieval", False) and text:
            try:
                from ..processing.contextual import ContextualEnricher
                from ..providers.llm_provider import SimpleLLMProvider

                llm = SimpleLLMProvider().get_llm()
                enricher = ContextualEnricher(llm_provider=llm)
                doc_context = text[:10000]
                chunk_texts = [ct for ct, _ in chunks_with_metadata]
                enriched_texts = enricher.enrich_batch(doc_context, chunk_texts)
                chunks_with_metadata = [
                    (et, cm) for et, (_, cm) in zip(enriched_texts, chunks_with_metadata)
                ]
                logger.info(
                    f"Contextual enrichment: enriched {len(enriched_texts)} chunks "
                    f"for document {document.document_id} "
                    f"(doc context: {len(doc_context)} chars)"
                )
            except Exception as e:
                logger.warning(f"Contextual enrichment failed, using original chunks: {e}")

        # GraphRAG entity extraction (synchronous, ThreadPoolExecutor for parallelism)
        if getattr(config, "enable_graph_retrieval", False):
            try:
                from ..retrieval.graph import EntityExtractor, get_or_create_session_graph
                from ..providers.llm_provider import SimpleLLMProvider

                extractor = EntityExtractor(
                    entity_types=getattr(config, "graph_entity_types", None),
                    model_name=getattr(config, "graph_extraction_model", None),
                )
                from ..utils.model_utils import get_default_model_name

                extraction_model = (
                    getattr(config, "graph_extraction_model", "") or get_default_model_name()
                )
                llm = SimpleLLMProvider().get_llm(model_name=extraction_model, temperature=0)
                chunk_texts_for_graph = [ct for ct, _ in chunks_with_metadata]
                extractions = extractor.extract_batch_sync(
                    chunk_texts_for_graph, llm, max_workers=4
                )
                graph_store = get_or_create_session_graph(document.session_id)
                for idx, extraction in enumerate(extractions):
                    chunk_id = f"{chunk_prefix}_chunk_{idx}"
                    for entity in extraction.get("entities", []):
                        entity_emb = embedder.encode_document(entity["name"])
                        graph_store.add_entity(
                            entity_id=entity["name"],
                            entity_type=entity.get("type", "UNKNOWN"),
                            embedding=entity_emb,
                            chunk_id=chunk_id,
                        )
                    for rel in extraction.get("relationships", []):
                        graph_store.add_relationship(
                            source_id=rel["source"],
                            target_id=rel["target"],
                            rel_type=rel.get("type", "RELATED"),
                        )
                logger.info(
                    f"GraphRAG: extracted {graph_store.num_entities} entities, "
                    f"{graph_store.num_relationships} relationships "
                    f"for document {document.document_id}"
                )
            except Exception as e:
                logger.warning(f"GraphRAG entity extraction failed: {e}")

        # Generate embeddings in batch (or via late chunking)
        _t_embed_start = _time.monotonic()
        chunk_texts = [ct for ct, _ in chunks_with_metadata]
        if getattr(config, "enable_late_chunking", False) and text:
            # Late chunking uses ORIGINAL (pre-enrichment) chunk texts for
            # offset mapping against full_text.  The resulting embeddings
            # capture full-document context regardless of enrichment.
            logger.info(
                f"Using late chunking for document {document.document_id} "
                f"({len(text)} chars, {len(chunk_texts)} chunks)"
            )
            embeddings = embedder.encode_documents_late_chunking(
                full_text=text,
                chunk_texts=_original_chunk_texts,
            )
        else:
            embeddings = embedder.encode_documents_batch(chunk_texts, show_progress=True)

        # RAPTOR: build recursive summary tree and add summary nodes as extra chunks
        if getattr(config, "enable_raptor", False):
            try:
                from ..retrieval.raptor import RaptorTreeBuilder
                from ..providers.llm_provider import SimpleLLMProvider

                llm = SimpleLLMProvider().get_llm(temperature=0)
                builder = RaptorTreeBuilder(
                    embedding_manager=embedder,
                    llm_provider=llm,
                    max_cluster_size=getattr(config, "raptor_max_cluster_size", 10),
                    max_levels=getattr(config, "raptor_max_levels", 5),
                )
                tree = builder.build(
                    texts=chunk_texts,
                    embeddings=embeddings,
                )
                # Add non-leaf summary nodes as extra chunks
                raptor_extra = 0
                for node in tree.all_nodes():
                    if node.level == 0:
                        continue  # leaf nodes are already in chunks_with_metadata
                    summary_meta = {
                        "document_id": document.document_id,
                        "session_id": document.session_id,
                        "filename": document.filename,
                        "raptor_level": node.level,
                        "raptor_node_id": node.node_id,
                        "raptor_children": len(node.children),
                    }
                    chunks_with_metadata.append((node.text, summary_meta))
                    embeddings.append(node.embedding)
                    raptor_extra += 1
                logger.info(
                    f"RAPTOR: added {raptor_extra} summary nodes "
                    f"({tree.max_level} levels) for document {document.document_id}"
                )
            except Exception as e:
                logger.warning(f"RAPTOR tree building failed: {e}")

        _stage_times["embed"] = _time.monotonic() - _t_embed_start

        # Resolve backend for this session
        _t_insert_start = _time.monotonic()
        backend = self._collection_manager.resolve_backend(document.session_id)

        # Compute sparse tokens for OpenSearch 3-way hybrid (when enabled)
        sparse_tokens_list: Optional[List[Optional[Dict]]] = None
        if backend.backend_name == "opensearch":
            try:
                from ..vector_backends.opensearch_backend import _neural_sparse_enabled

                if _neural_sparse_enabled() and hasattr(embedder, "encode_sparse"):
                    sparse_result = embedder.encode_sparse(chunk_texts)
                    if sparse_result is not None:
                        sparse_tokens_list = sparse_result
                        logger.debug(f"Generated sparse tokens for {len(chunk_texts)} chunks")
            except Exception as e:
                logger.debug(f"Sparse token generation skipped: {e}")

        # Resolve tenant partition key for partition-based isolation
        tenant_mgr = get_tenant_manager()
        partition_key = tenant_mgr.resolve_partition_key(document.session_id)
        if partition_key:
            try:
                tenant_mgr.ensure_partition(backend, config.collection_name, partition_key)
            except Exception as e:
                raise ValueError(
                    f"Failed to create partition for session {document.session_id}: {e}"
                ) from e

        # Store chunks in vector database
        chunk_entries = []
        for i, ((chunk_text, metadata), embedding) in enumerate(
            zip(chunks_with_metadata, embeddings)
        ):
            content = chunk_text[: config.max_content_length]
            sparse_tokens = (
                sparse_tokens_list[i]
                if sparse_tokens_list and i < len(sparse_tokens_list)
                else None
            )
            chunk_entries.append((content, metadata, embedding, sparse_tokens))

        # Quota: check chunk limit before insert (a new copy adds the difference)
        quota_mgr = get_quota_manager()
        replaced_chunks = int((replacing or {}).get("chunks_count") or 0)
        quota_mgr.check_chunks(document.session_id, len(chunk_entries) - replaced_chunks)
        old_copy = (
            self._replaced_chunks(replacing, config, backend, partition_key) if replacing else None
        )
        new_ids = [f"{chunk_prefix}_chunk_{idx}" for idx in range(len(chunk_entries))]
        if clear_previous_attempt and not replacing:
            self._clear_previous_attempt(document, backend, config, partition_key)

        def _insert_all_chunks() -> tuple[int, list[dict]]:
            batch_data = []
            chunks_for_bm25 = []
            for idx, (content, metadata, embedding, sparse_tokens) in enumerate(chunk_entries):
                chunk_id = new_ids[idx]
                insert_doc = {
                    "chunk_id": chunk_id,
                    "document_id": document.document_id,
                    "session_id": metadata.get("session_id", ""),
                    "content": content,
                    "embedding": embedding.tolist(),
                    "metadata": metadata,
                }
                if sparse_tokens:
                    insert_doc["sparse_tokens"] = sparse_tokens
                batch_data.append(insert_doc)
                chunks_for_bm25.append(
                    {
                        # The stored id, which dense hits carry: indexed by its
                        # position, the chunk never fused with its dense hit and,
                        # once the index read the store, was in it twice.
                        "chunk_id": chunk_id,
                        "document_id": document.document_id,
                        "content": content,
                        "metadata": metadata,
                    }
                )

            if batch_data:
                backend.insert(
                    collection_name=config.collection_name,
                    data=batch_data,
                    partition_key=partition_key,
                )

            return len(batch_data), chunks_for_bm25

        inserted = False
        try:
            chunk_count, chunks_for_bm25 = self._insert_or_rebuild(
                _insert_all_chunks, document, config, backend
            )
            inserted = True
            if replacing:
                self._retire_replaced_chunks(
                    document, old_copy, config, backend, new_ids, partition_key
                )
                # From here the old copy's chunks are gone: the row is pointed at
                # the new ones in one write, and what follows it cannot undo it.
                processing_time = _time.monotonic() - start_time
                self._commit_replacement(
                    document,
                    file_path,
                    config,
                    backend,
                    chunk_count=chunk_count,
                    processing_time=processing_time,
                    text_length=text_length,
                    preview_text=self._generate_preview_text(text),
                )
            else:
                self._finish_new_document(
                    document,
                    config,
                    chunks_for_bm25,
                    parent_chunks_data,
                    chunk_count=chunk_count,
                    start_time=start_time,
                    text_length=text_length,
                    preview_text=self._generate_preview_text(text),
                )
        except Exception as exc:
            self._discard_new_chunks(
                document,
                backend,
                config.collection_name,
                new_ids,
                partition_key,
                # A replacement indexes keywords only once committed; the old
                # copy's entries stay with the old copy.
                indexed_here=config.enable_hybrid_search and not replacing,
                quiet=not inserted,
            )
            if inserted and not isinstance(exc, DocumentRemoved):
                logger.info(
                    f"Removed the chunks inserted for {document.document_id} after "
                    f"{type(exc).__name__}"
                )
            raise

        if replacing:
            # The row names the new copy: what is left only maintains what is
            # derived from it, so a failure is logged and not raised.
            self._refresh_derived_state(document, config, chunks_for_bm25, parent_chunks_data)

        _stage_times["insert"] = _time.monotonic() - _t_insert_start
        processing_time = _time.monotonic() - start_time
        logger.debug(
            f"[DOC PROFILE] {document.document_id}: "
            f"extract={_stage_times.get('extract_chunk', 0):.2f}s "
            f"embed={_stage_times.get('embed', 0):.2f}s "
            f"insert={_stage_times.get('insert', 0):.2f}s "
            f"total={processing_time:.2f}s "
            f"chunks={chunk_count}"
        )
        METRICS.counter("lasagna_ingests", 1, tags={"outcome": "ready"})

        logger.info(
            f"Document processed: {document.document_id} "
            f"({chunk_count} chunks, {processing_time:.2f}s)"
        )

    def _insert_or_rebuild(
        self, insert_all, document: Document, config: RAGConfig, backend: Any
    ) -> tuple:
        """``insert_all()``, rebuilding an empty collection of the wrong size or schema once."""
        _is_dim_mismatch_error = is_dim_mismatch_error

        def _is_schema_type_mismatch_error(exc: Exception) -> bool:
            message = str(exc)
            return (
                "field should be a int64" in message.lower()
                or "DataNotMatchException" in type(exc).__name__
            )

        def _is_recoverable_schema_error(exc: Exception) -> bool:
            return _is_dim_mismatch_error(exc) or _is_schema_type_mismatch_error(exc)

        try:
            return insert_all()
        except Exception as exc:
            if not _is_recoverable_schema_error(exc):
                raise

            # Serialize mismatch detection to prevent concurrent uploads from
            # both rebuilding. Use _collection_manager's own lock via its
            # thread-safe methods — do NOT hold _rebuild_lock while calling
            # into collection manager (would risk deadlock with
            # _collection_init_lock).
            with self._rebuild_lock:
                if self._rebuilding:
                    raise RuntimeError(
                        f"Collection '{config.collection_name}' is already being rebuilt "
                        f"by another thread. Please retry your upload."
                    )

                # Any rows, or rows that cannot be counted, belong to some
                # session; an upload must not drop them. A schema-type mismatch
                # rebuilt even a populated collection.
                has_other_docs = not self._collection_manager.collection_is_empty(
                    config.collection_name, backend=backend
                )

                if has_other_docs:
                    mismatch = (
                        "EMBEDDING DIMENSION"
                        if _is_dim_mismatch_error(exc)
                        else "COLLECTION SCHEMA"
                    )
                    session = document.session_id
                    error_msg = (
                        f"{mismatch} MISMATCH during insert: collection "
                        f"{config.collection_name!r} holds documents embedded another way, "
                        f"so this document was not added and nothing was deleted. To keep "
                        f"those documents, switch the session back to the profile they were "
                        f"embedded with (lasagna configure --profile <profile> -s {session}). "
                        f"To use this embedding model, upload into a new session "
                        f"(lasagna upload <file> -s <new-session>)."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg) from exc

                self._rebuilding = True

            # Rebuild outside _rebuild_lock to avoid deadlock with _collection_init_lock
            try:
                mismatch_type = (
                    "schema type" if _is_schema_type_mismatch_error(exc) else "embedding dimension"
                )
                logger.warning(
                    f"Insert failed due to {mismatch_type} mismatch. "
                    f"Rebuilding collection '{config.collection_name}' and retrying."
                )
                self._collection_manager.recreate_collection_for_config(config, backend=backend)
                if config.enable_hybrid_search:
                    self.hybrid_searcher.clear_session(document.session_id)
                return insert_all()
            finally:
                with self._rebuild_lock:
                    self._rebuilding = False

    def _clear_previous_attempt(
        self, document: Document, backend: Any, config: RAGConfig, partition_key: Optional[str]
    ) -> None:
        """Remove the chunks an earlier attempt at this new document left in its collection.

        Raised, not logged: indexing on top of them would serve each passage twice.
        """
        try:
            backend.delete(
                collection_name=config.collection_name,
                filter_expr=FilterExpression.eq("document_id", document.document_id),
                partition_key=partition_key,
            )
        except Exception as exc:
            raise RuntimeError(
                f"The chunks an earlier attempt at {document.document_id} left in "
                f"{config.collection_name!r} could not be removed ({type(exc).__name__}: "
                f"{exc}), so it was not indexed again."
            ) from exc

    def _finish_new_document(
        self,
        document: Document,
        config: RAGConfig,
        chunks_for_bm25: List[Dict[str, Any]],
        parent_chunks_data: Any,
        *,
        chunk_count: int,
        start_time: float,
        text_length: int,
        preview_text: str,
    ) -> None:
        """Index a new document's chunks for keyword search, store its parents, mark it ready."""
        from lasagna.core.database import get_rag_database

        if config.enable_hybrid_search:
            self.hybrid_searcher.index_chunks(chunks_for_bm25, document.session_id, build_now=False)
        self._record_content_change(document.session_id, indexed_here=config.enable_hybrid_search)

        # Store parent chunks in SQLite for auto-merge retrieval
        if parent_chunks_data:
            parent_count = get_rag_database().save_parent_chunks(parent_chunks_data)
            logger.info(f"Stored {parent_count} parent chunks for document {document.document_id}")

        recorded = self._update_document_status(
            document.document_id,
            DocumentStatus.READY,
            chunks_count=chunk_count,
            processing_time=_time.monotonic() - start_time,
            text_length=text_length,
            preview_text=preview_text,
        )
        if recorded is False:
            # Deleted while it was processed (by a path that does not wait for
            # the document's lock): its chunks would stay with no row.
            raise DocumentRemoved(
                f"Document {document.document_id} was deleted while it was being processed, "
                "so the chunks just made for it were removed."
            )

    def _commit_replacement(
        self,
        document: Document,
        file_path: str,
        config: RAGConfig,
        backend: Any,
        *,
        chunk_count: int,
        processing_time: float,
        text_length: int,
        preview_text: str,
    ) -> None:
        """Point the row at the new copy (file, collection and counts) in one write.

        Runs after the old copy's chunks are retired, so a failure here would
        leave a row naming chunks that are gone: the new chunks are removed by
        the caller and the document is marked failed, with the way out, when
        that can still be written.
        """
        from lasagna.core.database import get_rag_database

        # An archive made while this ran (by a process that did not wait for
        # the document's lock) is kept: READY was written over it.
        try:
            now = get_rag_database().get_document(document.document_id) or {}
        except Exception:  # noqa: BLE001 - the write below says what the store can do
            now = {}
        status = (
            DocumentStatus.SOFT_DELETED.value
            if str(now.get("status") or "").lower() == "soft_deleted"
            else DocumentStatus.READY.value
        )
        try:
            recorded = get_rag_database().update_document_fields(
                document.document_id,
                {
                    "filename": document.filename,
                    "file_path": file_path,
                    "file_size": document.file_size,
                    "file_type": document.file_type,
                    "upload_date": document.upload_date.isoformat(),
                    "status": status,
                    "error_message": None,
                    "chunks_count": chunk_count,
                    "processing_time": processing_time,
                    "text_length": text_length,
                    "preview_text": preview_text,
                    "vector_backend_name": backend.backend_name,
                    "storage_backend_name": self.storage.backend_name,
                    "collection_name": config.collection_name,
                },
            )
        except Exception as exc:
            reason = (
                f"Its previous chunks were removed, and the new copy could not be recorded "
                f"({type(exc).__name__}: {exc}). Reindex it again."
            )
            try:
                self._update_document_status(
                    document.document_id, DocumentStatus.ERROR, error_message=reason
                )
            except Exception as status_exc:
                logger.warning(f"Could not mark {document.document_id} failed: {status_exc}")
            raise ReplacementNotRecorded(
                f"Reindex of {document.document_id} failed. {reason}"
            ) from exc
        if recorded is False:
            raise DocumentRemoved(
                f"Document {document.document_id} was deleted while it was being processed, "
                "so the chunks just made for it were removed."
            )

    def _refresh_derived_state(
        self,
        document: Document,
        config: RAGConfig,
        chunks_for_bm25: List[Dict[str, Any]],
        parent_chunks_data: Any,
    ) -> None:
        """The keyword index and parent chunks of a replaced document; logged, never raised."""
        from lasagna.core.database import get_rag_database

        indexed_here = config.enable_hybrid_search
        if indexed_here:
            try:
                self.hybrid_searcher.remove_document(document.document_id, document.session_id)
                self.hybrid_searcher.index_chunks(
                    chunks_for_bm25, document.session_id, build_now=False
                )
            except Exception as exc:
                # Recorded as a change made elsewhere, so this process rebuilds too.
                indexed_here = False
                logger.warning(
                    f"Keyword index not updated for {document.document_id}: {exc}; it is "
                    "rebuilt from the vector store on the next search."
                )
        self._record_content_change(document.session_id, indexed_here=indexed_here)
        try:
            get_rag_database().delete_parent_chunks_by_document(document.document_id)
            if parent_chunks_data:
                parent_count = get_rag_database().save_parent_chunks(parent_chunks_data)
                logger.info(
                    f"Stored {parent_count} parent chunks for document {document.document_id}"
                )
        except Exception as exc:
            logger.warning(
                f"Parent chunks of {document.document_id} not stored ({exc}); auto-merge "
                "returns its chunks unmerged until it is reindexed."
            )

    # File extensions natively supported by Docling's DocumentConverter.
    # Plain text (.txt), JSON, and XML are NOT supported — they fall back
    # to the legacy pipeline without the noisy Docling error.
    _DOCLING_SUPPORTED_EXTENSIONS = frozenset(
        {
            ".pdf",
            ".docx",
            ".pptx",
            ".html",
            ".htm",
            ".md",
            ".csv",
            ".xlsx",
            ".asciidoc",
            ".adoc",
            ".latex",
            ".tex",
            ".vtt",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".gif",
        }
    )

    @staticmethod
    def _should_use_docling(file_path: str, config: RAGConfig) -> bool:
        """Determine if Docling should process this document."""
        if not config.docling_enabled:
            return False
        if not is_docling_available():
            # The settings file turns Docling on and falls back without it, by
            # design: a warning per document on every core install was noise,
            # also for the speed profile, which turns Docling off.
            global _docling_fallback_said
            if not _docling_fallback_said:
                _docling_fallback_said = True
                logger.info(
                    "Docling is not installed, so documents are read with the built-in "
                    "extraction. To use it: pip install 'lasagna-rag[docling]'"
                )
            return False
        # Skip Docling for file types it doesn't support (e.g. .txt, .json)
        ext = Path(file_path).suffix.lower()
        if ext and ext not in DocumentPipeline._DOCLING_SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping Docling for unsupported format {ext}, using legacy pipeline")
            return False
        return True

    def _extract_text_with_ocr(
        self,
        file_path: str,
        file_type: str,
        config: RAGConfig,
    ) -> str:
        """Extract text with OCR fallback for scanned documents."""
        try:
            text = self.document_processor.extract_text(file_path, file_type)
            if text and len(text.strip()) > 100:
                return text
        except Exception as e:
            logger.warning(f"Normal text extraction failed: {e}")

        if file_type == ".pdf" and config.enable_ocr:
            from ..processing.ocr_processor import is_scanned_pdf, get_ocr_processor

            if is_scanned_pdf(file_path):
                logger.info(f"Detected scanned PDF, using OCR: {file_path}")
                ocr = get_ocr_processor()
                ocr_text = ocr.process_pdf(file_path)
                if ocr_text:
                    return ocr_text

        return self.document_processor.extract_text(file_path, file_type)

    def _generate_preview_text(self, text: str, token_count: int = 150) -> str:
        """Generate a preview snippet from document text."""
        if not text:
            return ""

        words = text.split()
        snippet_words = words[:token_count]
        snippet = " ".join(snippet_words)
        snippet = " ".join(snippet.split())

        if len(words) > token_count:
            snippet += "..."

        return snippet

    def _update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunks_count: Optional[int] = None,
        processing_time: Optional[float] = None,
        text_length: Optional[int] = None,
        error_message: Optional[str] = None,
        preview_text: Optional[str] = None,
    ) -> Optional[bool]:
        """Update the document's status; False when it has no row any more."""
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()
        return rag_db.update_document_status(
            document_id=document_id,
            status=status.value,
            error_message=error_message,
            chunks_count=chunks_count,
            processing_time=processing_time,
            text_length=text_length,
            preview_text=preview_text,
        )

    # =========================================================================
    # Document Management
    # =========================================================================

    def get_documents(self, session_id: str) -> List[DocumentMetadata]:
        """Get all documents for a session."""
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()
        rows = rag_db.get_session_documents(session_id)

        documents = []
        for row in rows:
            doc = DocumentMetadata(
                document_id=row["document_id"],
                session_id=row["session_id"],
                filename=row["filename"],
                file_path=row["file_path"],
                file_size=row["file_size"],
                file_type=row["file_type"],
                status=DocumentStatus(row["status"]),
                chunks_count=row["chunks_count"],
                upload_date=datetime.fromisoformat(row["upload_date"])
                if isinstance(row["upload_date"], str)
                else row["upload_date"],
                error_message=row.get("error_message"),
                processing_time=row.get("processing_time"),
                text_length=row.get("text_length"),
                vector_backend_name=row.get("vector_backend_name", "milvus"),
                storage_backend_name=row.get("storage_backend_name", "local"),
            )
            documents.append(doc)

        return documents

    def get_document_by_id(self, document_id: str, session_id: str) -> Optional[DocumentMetadata]:
        """Get a single document by ID with session verification (O(1) lookup).

        Args:
            document_id: Document identifier.
            session_id: Session ID for authorization — the document must
                belong to this session.

        Returns:
            DocumentMetadata or None if the document does not exist or
            belongs to a different session.
        """
        from lasagna.core.database import get_rag_database

        if not is_safe_document_id(document_id) or not is_valid_session_id(session_id):
            return None

        rag_db = get_rag_database()
        row = rag_db.get_document(document_id)
        if not row or row.get("session_id") != session_id:
            return None

        upload_date = row["upload_date"]
        if isinstance(upload_date, str):
            upload_date = datetime.fromisoformat(upload_date)

        return DocumentMetadata(
            document_id=row["document_id"],
            session_id=row["session_id"],
            filename=row["filename"],
            file_path=row.get("file_path", ""),
            file_size=row.get("file_size", 0),
            file_type=row.get("file_type", ""),
            status=DocumentStatus(row.get("status", "processing")),
            chunks_count=row.get("chunks_count", 0),
            upload_date=upload_date,
            error_message=row.get("error_message"),
            processing_time=row.get("processing_time"),
            text_length=row.get("text_length"),
            vector_backend_name=row.get("vector_backend_name", "milvus"),
            storage_backend_name=row.get("storage_backend_name", "local"),
        )

    def _record_content_change(self, session_id: str, *, indexed_here: bool) -> None:
        """Bump the session's content generation after its chunks changed.

        Every process keeps its own BM25 index; the others rebuild theirs when
        they see the generation move. ``indexed_here``: this process's index
        already has the change, so it is not rebuilt for it.
        """
        from lasagna.core.database import get_rag_database

        try:
            bump = getattr(get_rag_database(), "bump_content_generation", None)
            generation = bump(session_id) if bump is not None else None
        except Exception as e:
            logger.warning(
                f"Could not record the change to session {session_id[:8]}'s documents: {e}. "
                "Other API or worker processes keep their keyword index until the next change."
            )
            return
        if indexed_here:
            self.hybrid_searcher.note_local_change(session_id, generation)

    def delete_document(
        self, document_id: str, session_id: str, *, keep_file: bool = False
    ) -> bool:
        """Delete a document and its chunks.

        Args:
            document_id: Document ID to delete.
            session_id: Session ID for authorization.
            keep_file: If True, skip deleting the stored file from the storage
                backend.  Used by reindex flows that need the source file to
                survive deletion of the old index data.
        """
        from lasagna.api._profile_guard import refuse_during_switch

        refuse_during_switch(session_id, "The delete")
        with _document_lock(document_id):
            refuse_during_switch(session_id, "The delete")
            deleted = self._delete_document(
                document_id, session_id, keep_file=keep_file, clean_session=False
            )
        if deleted:
            # After the document's lock: a session's lock is never taken
            # under a document's (a switch holds the session's and waits for
            # each document's).
            self._remove_from_session_focus(document_id, session_id)
        return deleted

    def _delete_document(
        self, document_id: str, session_id: str, *, keep_file: bool, clean_session: bool = True
    ) -> bool:
        """Delete the document; ``clean_session`` also drops it from the session's focus
        (only with the session's lock held, or no document's)."""
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()

        doc = rag_db.get_document(document_id)
        if not doc or doc.get("session_id") != session_id:
            logger.warning(f"Document not found: {document_id}")
            return False

        file_path = doc.get("file_path")
        file_size = doc.get("file_size", 0)
        chunks_count = doc.get("chunks_count", 0)

        # Resolve tenant partition key for scoped deletion
        tenant_mgr = get_tenant_manager()
        partition_key = tenant_mgr.resolve_partition_key(session_id)

        # Delete from database (cascades to citations)
        rag_db.delete_document(document_id)

        # Delete vectors from the backend that actually holds them
        doc_backend_name = doc.get(
            "vector_backend_name", self._collection_manager._default_backend_name
        )
        try:
            target_backend = get_backend(doc_backend_name)
            if not target_backend.is_connected():
                target_backend.connect()
            target_backend.delete(
                collection_name=doc.get("collection_name")
                or self._get_session_config(session_id).collection_name,
                filter_expr=FilterExpression.eq("document_id", document_id),
                partition_key=partition_key,
            )
        except Exception as e:
            logger.warning(f"Could not delete vectors from {doc_backend_name}: {e}")

        # Remove from BM25 fallback index
        if rag_config.enable_hybrid_search:
            self.hybrid_searcher.remove_document(document_id, session_id)
        self._record_content_change(session_id, indexed_here=rag_config.enable_hybrid_search)

        # Clean up parent chunks from SQLite
        parent_deleted = rag_db.delete_parent_chunks_by_document(document_id)
        if parent_deleted > 0:
            logger.info(f"Deleted {parent_deleted} parent chunks for document {document_id}")

        # Delete file from storage backend (skip when reindexing)
        if file_path and not keep_file:
            try:
                self.storage.delete(file_path)
                logger.info(f"Deleted document files: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete document files: {e}")

        # Clean up session-level context awareness
        if clean_session:
            self._remove_from_session_focus(document_id, session_id)

        # Invalidate search cache
        if rag_config.enable_search_cache:
            try:
                search_cache = get_search_cache(rag_config)
                invalidated = search_cache.invalidate_session(session_id)
                if invalidated > 0:
                    logger.debug(
                        f"Search cache invalidated after delete: {invalidated} entries for session {session_id[:8]}..."
                    )
            except Exception as e:
                logger.warning(f"Failed to invalidate search cache after delete: {e}")

        # Record deletion for quota tracking (frees quota). A failed document
        # is not counted: its reservation was released when it failed, and
        # GC phase 6 counts documents that did not fail.
        if doc.get("status") != DocumentStatus.ERROR.value:
            quota_mgr = get_quota_manager()
            quota_mgr.record_delete(session_id, file_size_bytes=file_size, chunks=chunks_count)

        logger.info(f"Document deleted: {document_id}")
        return True

    def get_document_chunks(
        self,
        document_id: str,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        if not is_safe_document_id(document_id):
            logger.warning(f"Invalid document_id: {document_id!r}")
            return []

        if not is_valid_session_id(session_id):
            logger.warning("Invalid session_id format in get_document_chunks")
            return []

        try:
            backend = self._collection_manager.resolve_backend(session_id)
            collection = self._get_session_config(session_id).collection_name

            filter_expr = FilterExpression.and_(
                FilterExpression.eq("document_id", document_id),
                backend.session_filter(collection, session_id),
            )
            results = backend.query(
                collection_name=collection,
                filter_expr=filter_expr,
                output_fields=["content", "metadata"],
                limit=limit + offset,
            )

            if not results:
                return []

            sorted_results = sorted(
                results, key=lambda x: x.get("metadata", {}).get("chunk_index", 0)
            )
            paginated = sorted_results[offset : offset + limit]

            return [
                {
                    "content": chunk.get("content", ""),
                    "chunk_index": chunk.get("metadata", {}).get("chunk_index", i),
                    "metadata": chunk.get("metadata", {}),
                }
                for i, chunk in enumerate(paginated)
            ]

        except Exception as e:
            # Raised, not answered with []: an empty list reads as "this document
            # has no chunks", which is a different and alarming claim about the
            # data, made with a 200 while the store was unreachable.
            raise ChunkStoreUnavailable(
                f"The chunks of {document_id} could not be read from the vector store "
                f"({type(e).__name__}: {e})."
            ) from e

    def get_document_file_path(self, document_id: str, session_id: str) -> Optional[str]:
        """Get the file path for a document."""
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()
        doc = rag_db.get_document(document_id)

        if doc and doc.get("session_id") == session_id:
            return doc.get("file_path")
        return None

    # =========================================================================
    # Metadata Helpers
    # =========================================================================

    def _get_document_metadata(self, document_id: str) -> Optional[Document]:
        """Get document metadata from SQLite."""
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()
        row = rag_db.get_document(document_id)

        if not row:
            return None

        upload_date = row["upload_date"]
        if isinstance(upload_date, str):
            upload_date = datetime.fromisoformat(upload_date)

        return Document(
            document_id=row["document_id"],
            session_id=row["session_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            status=DocumentStatus(row["status"]),
            chunks_count=row["chunks_count"],
            upload_date=upload_date,
            error_message=row.get("error_message"),
        )

    def registered_document_ids(self, document_ids: List[str]) -> List[str]:
        """Those of ``document_ids`` that have a row, in any status."""
        from lasagna.core.database import get_rag_database

        valid = [d for d in document_ids if is_safe_document_id(d)]
        rows = get_rag_database().get_documents_batch(valid, include_soft_deleted=True)
        return [r["document_id"] for r in rows]

    def get_documents_metadata_batch(self, document_ids: List[str]) -> Dict[str, Document]:
        """The documents search may return chunks of, by id, in one query: ready ones.

        Search attaches each chunk to its document's row through this, and a
        chunk without one is dropped. Archived documents were left out, while
        a failed document, or one still processing, was served: its chunks
        stay in the store when the failure came after the insert, or until
        its row is marked ready.
        """
        from lasagna.core.database import get_rag_database

        if not document_ids:
            return {}

        unique_ids = list(dict.fromkeys(document_ids))
        valid_ids = [doc_id for doc_id in unique_ids if is_safe_document_id(doc_id)]
        if not valid_ids:
            return {}

        rag_db = get_rag_database()
        result = {}

        for row in rag_db.get_documents_batch(valid_ids):
            if row.get("status", DocumentStatus.READY.value) != DocumentStatus.READY.value:
                continue
            upload_date = row.get("upload_date")
            if isinstance(upload_date, str):
                upload_date = datetime.fromisoformat(upload_date)

            doc = Document(
                document_id=row["document_id"],
                session_id=row["session_id"],
                filename=row["filename"],
                file_type=row.get("file_type", ""),
                file_size=row.get("file_size", 0),
                status=DocumentStatus(row.get("status", "processing")),
                chunks_count=row.get("chunks_count", 0),
                upload_date=upload_date,
                error_message=row.get("error_message"),
            )
            result[doc.document_id] = doc

        return result

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _get_session_config(session_id: Optional[str] = None) -> RAGConfig:
        """Get a session-aware RAGConfig instance."""
        if session_id:
            return rag_config.with_session(session_id)
        return rag_config

    # =========================================================================
    # Lifecycle Helpers (called by DocumentLifecycleManager)
    # =========================================================================

    def _invalidate_search_cache(self, session_id: str) -> None:
        """Invalidate the search cache for a session.

        Called by the lifecycle manager after soft-delete, restore, and
        bulk-delete operations to ensure stale cache entries are evicted.
        """
        if rag_config.enable_search_cache:
            try:
                search_cache = get_search_cache(rag_config)
                invalidated = search_cache.invalidate_session(session_id)
                if invalidated > 0:
                    logger.debug(
                        f"Search cache invalidated: {invalidated} entries "
                        f"for session {session_id[:8]}..."
                    )
            except Exception as e:
                logger.warning(f"Failed to invalidate search cache for session {session_id}: {e}")

    def _remove_from_session_focus(self, document_id: str, session_id: str) -> None:
        """Remove a document from the session's document-focus / awareness state.

        Called by the lifecycle manager after soft-delete so that the document
        no longer appears in the session's focused document list.
        """
        try:
            from ..core.session_config import get_knowledge_config_manager

            config_manager = get_knowledge_config_manager()
            config_manager.remove_document_from_session_state(document_id, session_id)
            logger.debug(f"Removed document {document_id} from session focus: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to remove document {document_id} from session focus: {e}")
