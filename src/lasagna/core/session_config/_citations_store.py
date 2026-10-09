"""Citations store mixin for session configuration.

Handles storage and retrieval of last search results (citations and chunks)
for cross-process sharing between MCP subprocess and main server.
"""

import os
import json
import tempfile
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from ._data_models import (
    MAX_CITATIONS_PER_SESSION,
    MAX_CHUNKS_PER_SESSION,
    MAX_CHUNK_CONTENT_LENGTH,
)


class CitationStoreMixin:
    """Mixin for citations and search results storage.

    Expects the host class to provide:
    - self._lock: RLock
    - self._data_dir: str
    - self.get_session_state(): method (from SessionStateMixin)
    - self._persist_session_states(): method (from PersistenceMixin)
    """

    def _get_citations_file_path(self, session_id: str) -> str:
        """Get the file path for persisting citations for a session.

        Applies a regex allowlist, collision-safe hashing, length capping,
        and path containment check to prevent path traversal and filename
        collision attacks.
        """
        import re
        import hashlib

        # Allowlist: keep only alphanumeric, underscore, hyphen
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)

        # Prevent collision: append short hash when sanitization changed the ID
        # (e.g. "a/b" and "a_b" would otherwise map to the same filename)
        if safe != session_id:
            suffix = hashlib.sha256(session_id.encode()).hexdigest()[:8]
            safe = f"{safe}_{suffix}"

        # Enforce max filename length (255 - len("citations_") - len(".json") = 241)
        if len(safe) > 200:
            safe = safe[:192] + "_" + hashlib.sha256(session_id.encode()).hexdigest()[:8]

        path = os.path.join(self._data_dir, f"citations_{safe}.json")

        # Defense-in-depth: verify resolved path stays within data_dir
        real_path = os.path.realpath(path)
        real_data = os.path.realpath(self._data_dir)
        if not real_path.startswith(real_data + os.sep) and real_path != real_data:
            raise ValueError(f"Session ID resolves outside data directory: {session_id!r}")

        return path

    def _persist_citations_to_file(
        self,
        session_id: str,
        state: Any,
    ) -> None:
        """Persist citations and chunks to file for cross-process access.

        Uses atomic write (tempfile + os.replace) to prevent corruption.
        """
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            citations_file = self._get_citations_file_path(session_id)
            data = {
                "session_id": session_id,
                "timestamp": time.time(),
                "citations": state.last_citations,
                "chunks": state.last_chunks,
            }
            logger.info(f"[CITATIONS-WRITE] Writing to file: {citations_file}")
            fd, tmp_path = tempfile.mkstemp(dir=self._data_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, citations_file)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            logger.info(
                f"[CITATIONS-WRITE] Persisted {len(state.last_citations)} citations and "
                f"{len(state.last_chunks)} chunks to {citations_file}"
            )
        except Exception as e:
            logger.error(f"[CITATIONS-WRITE] Failed to persist citations to file: {e}")

    def remove_document_from_session_state(
        self,
        document_id: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Remove a deleted document from all session-level tracking."""
        session_id = session_id or "default"
        state = self.get_session_state(session_id)

        with self._lock:
            # Remove from focused document IDs
            if document_id in state.focused_document_ids:
                state.focused_document_ids = [
                    doc_id for doc_id in state.focused_document_ids if doc_id != document_id
                ]
                logger.debug(
                    f"Removed document {document_id} from focused_document_ids "
                    f"for session {session_id}"
                )
                self._persist_session_states()

            # Filter out stale citations and chunks
            original_citations_count = len(state.last_citations)
            original_chunks_count = len(state.last_chunks)

            state.last_citations = [
                citation
                for citation in state.last_citations
                if citation.get("document_id") != document_id
            ]
            state.last_chunks = [
                chunk for chunk in state.last_chunks if chunk.get("document_id") != document_id
            ]

            citations_removed = original_citations_count - len(state.last_citations)
            chunks_removed = original_chunks_count - len(state.last_chunks)

            if citations_removed > 0 or chunks_removed > 0:
                logger.debug(
                    f"Removed {citations_removed} citations and {chunks_removed} chunks "
                    f"for deleted document {document_id} from session {session_id}"
                )
                self._persist_citations_to_file(session_id, state)

    def set_last_search_results(
        self,
        citations: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> None:
        """Store last RAG search results for a session."""
        session_id = session_id or "default"
        state = self.get_session_state(session_id)

        with self._lock:
            # Enforce memory limits
            limited_citations = (citations or [])[:MAX_CITATIONS_PER_SESSION]
            limited_chunks = (chunks or [])[:MAX_CHUNKS_PER_SESSION]

            # Truncate chunk content to prevent memory exhaustion
            for chunk in limited_chunks:
                if "content" in chunk and isinstance(chunk["content"], str):
                    if len(chunk["content"]) > MAX_CHUNK_CONTENT_LENGTH:
                        chunk["content"] = (
                            chunk["content"][:MAX_CHUNK_CONTENT_LENGTH] + "...[truncated]"
                        )

            if len(citations or []) > MAX_CITATIONS_PER_SESSION:
                logger.warning(
                    f"Citations truncated from {len(citations)} to {MAX_CITATIONS_PER_SESSION} "
                    f"for session {session_id[:8]}..."
                )
            if len(chunks or []) > MAX_CHUNKS_PER_SESSION:
                logger.warning(
                    f"Chunks truncated from {len(chunks)} to {MAX_CHUNKS_PER_SESSION} "
                    f"for session {session_id[:8]}..."
                )

            # Ensure citations are plain dicts for JSON serialization
            serializable_citations = []
            for c in limited_citations:
                if isinstance(c, dict):
                    serializable_citations.append(c)
                elif hasattr(c, "to_dict"):
                    serializable_citations.append(c.to_dict())
                elif hasattr(c, "__dataclass_fields__"):
                    from dataclasses import asdict

                    serializable_citations.append(asdict(c))
                else:
                    serializable_citations.append(c)
            limited_citations = serializable_citations

            state.last_citations = limited_citations
            state.last_chunks = limited_chunks

            # Persist to file for cross-process access
            self._persist_citations_to_file(session_id, state)

            # Update mtime tracker so same-process reads don't unnecessarily re-read
            try:
                citations_file = self._get_citations_file_path(session_id)
                state._citations_file_mtime = os.path.getmtime(citations_file)
            except OSError:
                state._citations_file_mtime = time.time()

    def get_last_search_results(
        self,
        session_id: Optional[str] = None,
    ) -> tuple:
        """Get last RAG search results for a session."""
        session_id = session_id or "default"
        state = self.get_session_state(session_id)

        # Always check the file for freshness (cross-process case).
        # The MCP subprocess writes citations to file; the main server
        # may have stale in-memory data from a previous query.
        try:
            citations_file = self._get_citations_file_path(session_id)
            if os.path.exists(citations_file):
                file_mtime = os.path.getmtime(citations_file)
                if file_mtime > state._citations_file_mtime:
                    # File has been updated since we last read it
                    with open(citations_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    citations = data.get("citations", [])
                    chunks = data.get("chunks", [])
                    if citations:
                        logger.info(
                            f"[CITATIONS-READ] Refreshed {len(citations)} citations "
                            f"from file for session {session_id}"
                        )
                        state.last_citations = citations
                        state.last_chunks = chunks
                        state._citations_file_mtime = file_mtime
                        return citations, chunks
        except Exception as e:
            logger.warning(f"[CITATIONS-READ] Error reading citations from file: {e}")

        # Return in-memory state (populated by same-process writes or previous file reads)
        if state.last_citations:
            return state.last_citations, state.last_chunks or []

        return [], []

    def clear_last_search_results(
        self,
        session_id: Optional[str] = None,
    ) -> None:
        """Clear last search results for a session."""
        session_id = session_id or "default"
        self.set_last_search_results([], [], session_id)

        # Also remove the persisted file
        try:
            citations_file = self._get_citations_file_path(session_id)
            if os.path.exists(citations_file):
                os.remove(citations_file)
        except Exception as e:
            logger.debug(f"Could not remove citations file: {e}")
