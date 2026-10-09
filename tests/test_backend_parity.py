"""Backend parity tests for SQLite (RAGDatabase) vs PostgreSQL (PostgresRAGDatabase).

Tests verify that both database backends satisfy the DatabaseBackend protocol
contract with identical behavior. PostgreSQL tests are skipped when no
DATABASE_DSN environment variable is set.

Parity contract (from core/docs/AUDIT_FINDINGS.md):
1. store_citations() with N > limit must retain only the most recent max_citations per session
2. store_citations() on conflict must update ALL mutable fields
3. get_stats() must return shared key structure
4. All 16 DatabaseBackend Protocol methods produce identical outputs for identical inputs
5. Schema DDL for shared tables must be semantically equivalent
"""

import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from lasagna.core.database import RAGDatabase

# PostgreSQL backend is optional
_PG_AVAILABLE = False
try:
    from lasagna.core.database_pg import PostgresRAGDatabase

    _PG_AVAILABLE = True
except ImportError:
    pass

_DATABASE_DSN = os.environ.get("DATABASE_DSN", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_document(session_id: str = "test_session") -> Dict[str, Any]:
    """Create a test document dict."""
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    return {
        "document_id": doc_id,
        "session_id": session_id,
        "filename": "test.pdf",
        "file_size": 1024,
        "file_type": ".pdf",
        "status": "ready",
        "chunks_count": 10,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "vector_backend_name": "milvus",
        "storage_backend_name": "local",
    }


def _make_citation(
    session_id: str,
    document_id: str,
    index: int,
    query_id: str = "q1",
) -> Dict[str, Any]:
    """Create a test citation dict."""
    return {
        "citation_id": f"cite_{uuid.uuid4().hex[:12]}",
        "session_id": session_id,
        "query_id": query_id,
        "chunk_id": f"chunk_{index}",
        "document_id": document_id,
        "document_name": "test.pdf",
        "content_preview": f"Content for citation {index}",
        "content_hash": f"hash_{index}_{uuid.uuid4().hex[:8]}",
        "chunk_index": index,
        "page_number": index + 1,
        "section_header": f"Section {index}",
        "score": 0.95 - (index * 0.01),
        "normalized_score": 0.9 - (index * 0.01),
        "hop_index": 0,
    }


# ---------------------------------------------------------------------------
# Parametrized backend fixture (SQLite always, PG when available)
# ---------------------------------------------------------------------------


def _backend_params():
    """Return pytest params: always sqlite, add pg when DSN is set."""
    params = [pytest.param("sqlite", id="sqlite")]
    if _PG_AVAILABLE and _DATABASE_DSN:
        params.append(pytest.param("pg", id="pg"))
    else:
        params.append(
            pytest.param(
                "pg",
                id="pg",
                marks=pytest.mark.skip(
                    reason="PostgreSQL not available (set DATABASE_DSN env var)"
                ),
            )
        )
    return params


@pytest.fixture(params=_backend_params())
def backend(request, tmp_path):
    """Yield a DatabaseBackend implementation — runs each test on both backends."""
    if request.param == "sqlite":
        db_path = str(tmp_path / "test_rag.db")
        yield RAGDatabase(db_path)
    elif request.param == "pg":
        if not _PG_AVAILABLE or not _DATABASE_DSN:
            pytest.skip("PostgreSQL not available (set DATABASE_DSN env var)")
        db = PostgresRAGDatabase(dsn=_DATABASE_DSN, pool_min=1, pool_max=3)
        yield db
        # Cleanup: drop test data (but not tables)
        try:
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM citations")
                    cur.execute("DELETE FROM parent_chunks")
                    cur.execute("DELETE FROM rag_documents")
                conn.commit()
        except Exception:
            pass
    else:
        pytest.skip("Unsupported backend param")


# =========================================================================
# PARITY TEST: Document CRUD
# =========================================================================


class TestDocumentCRUD:
    """Verify document operations produce identical results on both backends."""

    def test_save_and_retrieve_document(self, backend):
        doc = _make_document()
        assert backend.save_document(doc, "/tmp/test.pdf") is True

        retrieved = backend.get_document(doc["document_id"])
        assert retrieved is not None
        assert retrieved["document_id"] == doc["document_id"]
        assert retrieved["session_id"] == doc["session_id"]
        assert retrieved["filename"] == "test.pdf"
        assert retrieved["status"] == "ready"

    def test_get_nonexistent_document(self, backend):
        assert backend.get_document("nonexistent_id") is None

    def test_get_documents_by_session(self, backend):
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        for _ in range(3):
            doc = _make_document(session_id)
            backend.save_document(doc, "/tmp/test.pdf")

        docs = backend.get_documents(session_id)
        assert len(docs) == 3

    def test_get_documents_batch(self, backend):
        doc_ids = []
        for _ in range(5):
            doc = _make_document()
            backend.save_document(doc, "/tmp/test.pdf")
            doc_ids.append(doc["document_id"])

        batch = backend.get_documents_batch(doc_ids[:3])
        assert len(batch) == 3

    def test_update_document_status(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        updated = backend.update_document_status(
            doc["document_id"], "error", error_message="Test error"
        )
        assert updated is True

        retrieved = backend.get_document(doc["document_id"])
        assert retrieved["status"] == "error"
        assert retrieved["error_message"] == "Test error"

    def test_delete_document_cascades(self, backend):
        """Verify document deletion cascades to citations and parent chunks."""
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        # Store citations and parent chunks
        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(5)]
        backend.store_citations(citations, "test_session")

        chunks = [
            {
                "stable_id": f"pc_{i}",
                "document_id": doc["document_id"],
                "session_id": "test_session",
                "content": f"Parent chunk {i}",
                "chunk_index": i,
            }
            for i in range(3)
        ]
        backend.save_parent_chunks(chunks)

        # Delete document
        assert backend.delete_document(doc["document_id"]) is True

        # Verify cascaded deletion
        assert backend.get_document(doc["document_id"]) is None
        session_citations = backend.get_session_citations("test_session")
        for c in session_citations:
            assert c["document_id"] != doc["document_id"]


# =========================================================================
# PARITY TEST: Citation Limit Enforcement
# =========================================================================


class TestCitationLimitEnforcement:
    """Verify store_citations enforces per-session limit atomically."""

    def test_store_citations_basic(self, backend):
        """Basic citation storage and retrieval."""
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(10)]
        stored = backend.store_citations(citations, "test_session")
        assert stored == 10

        retrieved = backend.get_session_citations("test_session")
        assert len(retrieved) == 10

    def test_store_citations_with_query_filter(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations_q1 = [
            _make_citation("test_session", doc["document_id"], i, "q1") for i in range(5)
        ]
        citations_q2 = [
            _make_citation("test_session", doc["document_id"], i, "q2") for i in range(3)
        ]
        backend.store_citations(citations_q1, "test_session", query_id="q1")
        backend.store_citations(citations_q2, "test_session", query_id="q2")

        all_cites = backend.get_session_citations("test_session")
        assert len(all_cites) == 8

        q1_cites = backend.get_session_citations("test_session", query_id="q1")
        assert len(q1_cites) == 5

    def test_enforce_citation_limit_standalone(self, backend):
        """Test the standalone enforce_citation_limit method."""
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(50)]
        backend.store_citations(citations, "test_session")

        deleted = backend.enforce_citation_limit("test_session", max_citations=20)
        assert deleted == 30

        remaining = backend.get_session_citations("test_session", limit=100)
        assert len(remaining) == 20

    def test_citation_expiry(self, backend):
        """Test TTL-based citation expiry via cleanup_citations_by_age."""
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(5)]
        backend.store_citations(citations, "test_session")

        # Backdate created_at to 2 hours ago so cleanup_citations_by_age can find them.
        # Use backend-specific raw SQL since this is a test-only operation.
        if isinstance(backend, RAGDatabase):
            with backend.transaction() as conn:
                conn.execute(
                    "UPDATE citations SET created_at = datetime('now', '-2 hours') "
                    "WHERE session_id = 'test_session'"
                )
        else:
            # PostgreSQL backend
            with backend._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE citations SET created_at = NOW() - INTERVAL '2 hours' "
                        "WHERE session_id = 'test_session'"
                    )
                conn.commit()

        # cleanup_citations_by_age(max_age_hours=1) deletes citations older than 1 hour
        cleaned = backend.cleanup_citations_by_age(max_age_hours=1)
        assert cleaned >= 5

    def test_clear_session_citations(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(10)]
        backend.store_citations(citations, "test_session")

        cleared = backend.clear_session_citations("test_session")
        assert cleared == 10

        remaining = backend.get_session_citations("test_session")
        assert len(remaining) == 0


# =========================================================================
# PARITY TEST: Citation Conflict (Upsert)
# =========================================================================


class TestCitationConflict:
    """Verify store_citations on conflict updates ALL mutable fields."""

    def test_store_citations_conflict_updates_all_fields(self, backend):
        """When re-inserting a citation with same citation_id, all fields must update."""
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citation_id = f"cite_{uuid.uuid4().hex[:12]}"
        original = {
            "citation_id": citation_id,
            "session_id": "test_session",
            "query_id": "q_original",
            "chunk_id": "chunk_0",
            "document_id": doc["document_id"],
            "document_name": "original.pdf",
            "content_preview": "Original content",
            "content_hash": "hash_original",
            "chunk_index": 0,
            "page_number": 1,
            "section_header": "Original Section",
            "score": 0.5,
            "normalized_score": 0.4,
            "hop_index": 0,
        }
        backend.store_citations([original], "test_session")

        # Re-insert with updated fields
        updated = {
            "citation_id": citation_id,  # Same ID
            "session_id": "test_session",
            "query_id": "q_updated",
            "chunk_id": "chunk_99",
            "document_id": doc["document_id"],
            "document_name": "updated.pdf",
            "content_preview": "Updated content",
            "content_hash": "hash_updated",
            "chunk_index": 99,
            "page_number": 42,
            "section_header": "Updated Section",
            "score": 0.99,
            "normalized_score": 0.95,
            "hop_index": 2,
        }
        backend.store_citations([updated], "test_session")

        # Retrieve and verify ALL fields updated
        retrieved = backend.get_citation(citation_id)
        assert retrieved is not None
        assert retrieved["query_id"] == "q_updated", "query_id not updated"
        assert retrieved["score"] == 0.99, "score not updated"
        # These fields were the parity gap in PG (Fix 2.3) — verify both update them
        assert retrieved["document_name"] == "updated.pdf", "document_name not updated on conflict"
        assert retrieved["content_preview"] == "Updated content", "content_preview not updated"
        assert retrieved["content_hash"] == "hash_updated", "content_hash not updated"
        assert retrieved["chunk_index"] == 99, "chunk_index not updated"
        assert retrieved["page_number"] == 42, "page_number not updated"
        assert retrieved["section_header"] == "Updated Section", "section_header not updated"
        assert retrieved["normalized_score"] == 0.95, "normalized_score not updated"
        assert retrieved["hop_index"] == 2, "hop_index not updated"


# =========================================================================
# PARITY TEST: Parent Chunks
# =========================================================================


class TestParentChunks:
    """Verify parent chunk CRUD."""

    def test_parent_chunk_roundtrip(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        chunks = [
            {
                "stable_id": f"pc_{i}",
                "document_id": doc["document_id"],
                "session_id": "test_session",
                "content": f"Parent chunk content {i}",
                "chunk_index": i,
                "start_char": i * 100,
                "end_char": (i + 1) * 100,
                "metadata": {"section": f"Section {i}"},
            }
            for i in range(5)
        ]

        count = backend.save_parent_chunks(chunks)
        assert count == 5

        # Retrieve by stable_ids
        content_map = backend.get_parent_chunks(["pc_0", "pc_2", "pc_4"])
        assert len(content_map) == 3
        assert content_map["pc_0"] == "Parent chunk content 0"
        assert content_map["pc_4"] == "Parent chunk content 4"

    def test_parent_chunk_with_metadata(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        chunks = [
            {
                "stable_id": "pc_meta",
                "document_id": doc["document_id"],
                "session_id": "test_session",
                "content": "Content with metadata",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 100,
                "metadata": {"section": "Intro", "page": 1},
            }
        ]
        backend.save_parent_chunks(chunks)

        result = backend.get_parent_chunks_with_metadata(["pc_meta"])
        assert "pc_meta" in result
        assert result["pc_meta"]["content"] == "Content with metadata"
        assert result["pc_meta"]["chunk_index"] == 0
        assert result["pc_meta"]["start_char"] == 0
        assert result["pc_meta"]["end_char"] == 100
        assert result["pc_meta"]["metadata"]["section"] == "Intro"

    def test_delete_parent_chunks_by_document(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        chunks = [
            {
                "stable_id": f"pc_del_{i}",
                "document_id": doc["document_id"],
                "session_id": "test_session",
                "content": f"Content {i}",
                "chunk_index": i,
            }
            for i in range(3)
        ]
        backend.save_parent_chunks(chunks)

        deleted = backend.delete_parent_chunks_by_document(doc["document_id"])
        assert deleted == 3

        remaining = backend.get_parent_chunks(["pc_del_0", "pc_del_1", "pc_del_2"])
        assert len(remaining) == 0


# =========================================================================
# PARITY TEST: Statistics
# =========================================================================


class TestGetStats:
    """Verify get_stats returns the contract-required shared keys."""

    def test_get_stats_shared_keys(self, backend):
        """Both backends must return these keys."""
        stats = backend.get_stats()

        # Shared required keys
        assert "documents" in stats
        assert "total" in stats["documents"]
        assert "sessions" in stats["documents"]
        assert "by_status" in stats["documents"]

        assert "citations" in stats
        assert "total" in stats["citations"]
        assert "expired_pending_cleanup" in stats["citations"]
        assert "sessions" in stats["citations"]

    def test_get_stats_reflects_data(self, backend):
        """Stats should reflect actual data."""
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(5)]
        backend.store_citations(citations, "test_session")

        stats = backend.get_stats()
        assert stats["documents"]["total"] >= 1
        assert stats["citations"]["total"] >= 5


# =========================================================================
# PARITY TEST: Schema DDL Consistency
# =========================================================================


class TestSchemaDDLConsistency:
    """Verify schema.py CREATE_DOCUMENTS_TABLE matches database.py."""

    def test_schema_ddl_matches_database(self):
        """The CREATE_DOCUMENTS_TABLE constant in schema.py must have the same
        columns as the one in database.py (canonical source of truth)."""
        from lasagna.core.schema import (
            CREATE_DOCUMENTS_TABLE as schema_ddl,
        )
        from lasagna.core.database import (
            CREATE_DOCUMENTS_TABLE as db_ddl,
        )

        def _extract_columns(ddl: str) -> set:
            """Extract column names from CREATE TABLE DDL."""
            # Find content between first ( and last )
            match = re.search(r"\((.*)\)", ddl, re.DOTALL)
            if not match:
                return set()
            body = match.group(1)
            columns = set()
            for line in body.split("\n"):
                line = line.strip().rstrip(",")
                if not line or line.startswith("--") or line.upper().startswith("FOREIGN"):
                    continue
                # First word is column name
                parts = line.split()
                if parts and parts[0].upper() not in ("CREATE", "TABLE", "IF", "PRIMARY"):
                    columns.add(parts[0].lower())
            return columns

        schema_cols = _extract_columns(schema_ddl)
        db_cols = _extract_columns(db_ddl)

        missing_in_schema = db_cols - schema_cols
        extra_in_schema = schema_cols - db_cols

        # This test is expected to FAIL before Fix 2.4 (schema drift fix)
        assert not missing_in_schema, (
            f"schema.py is missing columns present in database.py: {missing_in_schema}"
        )
        assert not extra_in_schema, (
            f"schema.py has extra columns not in database.py: {extra_in_schema}"
        )


# =========================================================================
# PARITY TEST: Citation Content Hash Lookup
# =========================================================================


class TestCitationContentHash:
    """Verify content hash lookup works identically."""

    def test_get_citation_by_content_hash(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citation = _make_citation("test_session", doc["document_id"], 0)
        citation["content_hash"] = "unique_hash_123"
        backend.store_citations([citation], "test_session")

        result = backend.get_citation_by_content_hash("unique_hash_123", "test_session")
        assert result is not None
        assert result["content_hash"] == "unique_hash_123"

    def test_get_citation_by_content_hash_not_found(self, backend):
        result = backend.get_citation_by_content_hash("nonexistent_hash", "test_session")
        assert result is None


# =========================================================================
# PARITY TEST: Aggregate Stats
# =========================================================================


class TestCitationAggregateStats:
    """Verify citation aggregate statistics."""

    def test_get_citation_aggregate_stats(self, backend):
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        citations = [_make_citation("test_session", doc["document_id"], i) for i in range(5)]
        backend.store_citations(citations, "test_session")

        stats = backend.get_citation_aggregate_stats()
        assert "avg_score" in stats
        assert "missing_page" in stats
        assert isinstance(stats["avg_score"], float)


# =========================================================================
# PARITY TEST: Citation Cap Enforcement (stress)
# =========================================================================


class TestCitationCapEnforcement:
    """Verify store_citations enforces the 5000 per-session cap."""

    @pytest.mark.slow
    def test_store_citations_enforces_5000_cap(self, backend):
        """Insert >5000 citations and verify exactly 5000 remain.

        This directly validates the off-by-one fix (F12).  Inserts in
        batches to keep memory reasonable.
        """
        doc = _make_document()
        backend.save_document(doc, "/tmp/test.pdf")

        batch_size = 500
        total_to_insert = 5050

        for batch_start in range(0, total_to_insert, batch_size):
            batch_end = min(batch_start + batch_size, total_to_insert)
            citations = [
                _make_citation("test_session", doc["document_id"], i)
                for i in range(batch_start, batch_end)
            ]
            backend.store_citations(citations, "test_session")

        # After inserting 5050, the cap should keep exactly 5000
        remaining = backend.get_session_citations("test_session", limit=6000)
        assert len(remaining) == 5000, (
            f"Expected exactly 5000 citations after cap enforcement, got {len(remaining)}"
        )
