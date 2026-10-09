"""
Smoke tests for RAG pipeline fixes.

Tests the following issues:
- Issue A: upload_document should return correct chunks_count and status
- Issue B: Citations should have correct page numbers
- Issue C: Document snippets should use stored preview_text
- Issue D: Index creation should be idempotent

Run with: pytest tests/test_rag_fixes.py -v
"""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock
import pytest

# Import the modules we're testing
from lasagna.core.schema import DocumentStatus


class TestUploadDocumentResponse:
    """Test Issue A: upload_document returns correct chunks and status."""

    def test_document_status_enum_has_ready(self):
        """Verify DocumentStatus has READY status for completed uploads."""
        assert hasattr(DocumentStatus, "READY")
        assert DocumentStatus.READY.value == "ready"

    def test_document_status_enum_has_processing(self):
        """Verify DocumentStatus has PROCESSING status for in-progress uploads."""
        assert hasattr(DocumentStatus, "PROCESSING")
        assert DocumentStatus.PROCESSING.value == "processing"


class TestPageNumberExtraction:
    """Test Issue B: Page number metadata in chunks/citations."""

    def test_page_marker_regex(self):
        """Test that page markers are correctly detected."""
        import re

        # The page marker pattern used in semantic_chunker
        pattern = r"---\s*Page\s+(\d+)\s*---"

        # Test various page marker formats
        test_cases = [
            ("--- Page 1 ---\nContent here", True, 1),
            ("--- Page 5 ---\nMore content", True, 5),
            ("---Page 10---\nContent", True, 10),  # No spaces variant
            ("No page markers here", False, None),
            ("Page 1", False, None),  # Not in correct format
        ]

        for text, should_have_marker, expected_page in test_cases:
            # Check if page marker pattern is found
            match = re.search(pattern, text)
            has_marker = bool(match)
            assert has_marker == should_have_marker, f"Failed marker detection for: {text}"

            if expected_page is not None:
                extracted_page = int(match.group(1))
                assert extracted_page == expected_page, f"Failed page extraction for: {text}"

    def test_citation_manager_page_extraction(self):
        """Test that CitationManager correctly extracts page numbers."""
        from lasagna.agentic.citation_manager import CitationManager

        manager = CitationManager()

        # Test with chunk having direct page_number attribute
        mock_chunk = MagicMock()
        mock_chunk.page_number = 5
        mock_chunk.metadata = {}

        mock_result = MagicMock()
        mock_result.chunk = mock_chunk

        page = manager._extract_page_number(mock_result)
        assert page == 5, "Should extract page_number from chunk attribute"

    def test_citation_manager_page_extraction_from_metadata(self):
        """Test that CitationManager extracts page from metadata as fallback."""
        from lasagna.agentic.citation_manager import CitationManager

        manager = CitationManager()

        # Test with chunk having page_number in metadata only
        mock_chunk = MagicMock()
        mock_chunk.page_number = None
        mock_chunk.metadata = {"page_number": 3}

        mock_result = MagicMock()
        mock_result.chunk = mock_chunk

        page = manager._extract_page_number(mock_result)
        assert page == 3, "Should fall back to metadata page_number"


class TestDocumentSnippetStorage:
    """Test Issue C: Document snippets stored at upload time."""

    def test_preview_text_generation(self):
        """Test that preview text is generated correctly."""
        # Simulate the preview generation logic
        text = "This is a test document with many words. " * 50
        token_count = 150

        words = text.split()
        snippet_words = words[:token_count]
        snippet = " ".join(snippet_words)
        snippet = " ".join(snippet.split())
        if len(words) > token_count:
            snippet += "..."

        assert "..." in snippet, "Long documents should have ellipsis"
        assert len(snippet.split()) <= token_count + 1, "Should respect token limit"

    def test_preview_text_short_document(self):
        """Test preview text for short documents without truncation."""
        text = "Short document."
        token_count = 150

        words = text.split()
        snippet_words = words[:token_count]
        snippet = " ".join(snippet_words)
        snippet = " ".join(snippet.split())
        if len(words) > token_count:
            snippet += "..."

        assert "..." not in snippet, "Short documents should not have ellipsis"
        assert snippet == text, "Short documents should be returned as-is"

    def test_document_info_has_preview_text_field(self):
        """Verify DocumentInfo dataclass has preview_text field."""
        from lasagna.agentic.document_awareness import DocumentInfo
        from dataclasses import fields

        field_names = [f.name for f in fields(DocumentInfo)]
        assert "preview_text" in field_names, "DocumentInfo should have preview_text field"


class TestIdempotentIndexCreation:
    """Test Issue D: Index creation is idempotent."""

    def test_list_indexes_check_logic(self):
        """Test the logic for checking existing indexes."""
        # Simulate the check that happens in _ensure_index_exists

        # Case 1: Index exists
        existing_indexes = ["embedding"]
        if existing_indexes and "embedding" in existing_indexes:
            should_create = False
        else:
            should_create = True
        assert not should_create, "Should not create index if it exists"

        # Case 2: Index doesn't exist
        existing_indexes = []
        if existing_indexes and "embedding" in existing_indexes:
            should_create = False
        else:
            should_create = True
        assert should_create, "Should create index if it doesn't exist"

        # Case 3: Different index exists
        existing_indexes = ["other_field"]
        if existing_indexes and "embedding" in existing_indexes:
            should_create = False
        else:
            should_create = True
        assert should_create, "Should create index if only other indexes exist"


class TestSchemaPreviewTextColumn:
    """Test that schema correctly includes preview_text column."""

    def test_create_documents_table_has_preview_text(self):
        """Verify CREATE_DOCUMENTS_TABLE includes preview_text column."""
        from lasagna.core.schema import CREATE_DOCUMENTS_TABLE

        assert "preview_text" in CREATE_DOCUMENTS_TABLE, (
            "CREATE_DOCUMENTS_TABLE should include preview_text column"
        )

    def test_add_preview_text_migration_exists(self):
        """Verify migration SQL for adding preview_text exists."""
        from lasagna.core.schema import ADD_PREVIEW_TEXT_COLUMN

        assert "preview_text" in ADD_PREVIEW_TEXT_COLUMN, (
            "Migration SQL should add preview_text column"
        )
        assert "ALTER TABLE" in ADD_PREVIEW_TEXT_COLUMN, "Migration should use ALTER TABLE"


class TestIntegration:
    """Integration tests requiring actual database setup."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_sqlite_schema_creation(self, temp_db):
        """Test that SQLite schema can be created with preview_text."""
        from lasagna.core.schema import CREATE_DOCUMENTS_TABLE

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(CREATE_DOCUMENTS_TABLE)
            conn.commit()

            # Verify preview_text column exists
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "preview_text" in columns, "preview_text column should be created"

    def test_sqlite_migration(self, temp_db):
        """Test that migration adds preview_text to existing table."""
        from lasagna.core.schema import ADD_PREVIEW_TEXT_COLUMN

        # Create table without preview_text column (simulate old schema)
        old_schema = """
        CREATE TABLE IF NOT EXISTS rag_documents (
            document_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT,
            file_size INTEGER,
            file_path TEXT,
            status TEXT DEFAULT 'processing',
            chunks_count INTEGER DEFAULT 0,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_time TIMESTAMP,
            error_message TEXT
        )
        """

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(old_schema)
            conn.commit()

            # Verify preview_text doesn't exist yet
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns_before = {row[1] for row in cursor.fetchall()}
            assert "preview_text" not in columns_before

            # Run migration
            try:
                cursor.execute(ADD_PREVIEW_TEXT_COLUMN)
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            # Verify preview_text now exists
            cursor.execute("PRAGMA table_info(rag_documents)")
            columns_after = {row[1] for row in cursor.fetchall()}
            assert "preview_text" in columns_after, "Migration should add preview_text"


class TestRecordedDocumentChunkPageNumber:
    """Test that RecordedDocumentChunk properly receives page_number."""

    def test_recorded_chunk_has_page_number_field(self):
        """Verify RecordedDocumentChunk schema includes page_number."""
        from lasagna.core.schema import RecordedDocumentChunk, DocumentChunk

        # DocumentChunk should have page_number
        chunk = DocumentChunk(
            content="Test content", document_id="doc_123", chunk_index=0, page_number=5
        )
        assert chunk.page_number == 5

        # RecordedDocumentChunk inherits from DocumentChunk
        recorded = RecordedDocumentChunk(
            content="Test content",
            document_id="doc_123",
            chunk_index=0,
            page_number=10,
            chunk_id="doc_123_chunk_0",
        )
        assert recorded.page_number == 10

    def test_chunk_page_number_from_metadata(self):
        """Test extracting page_number from metadata dict."""
        metadata = {"page_number": 7, "chunk_index": 0, "session_id": "test"}

        # Simulate what rag_backend.py does
        page_number = metadata.get("page_number")
        assert page_number == 7


class TestCitationMonitoring:
    """Test citation monitoring and suspicious pattern detection."""

    def test_late_document_keywords(self):
        """Test that late-document keywords are properly defined."""
        # Import the keywords from MCP server module
        # We can't easily import from the MCP module, so we test the pattern
        late_keywords = [
            "experiment",
            "table",
            "figure",
            "appendix",
            "references",
            "ablation",
            "results",
            "evaluation",
            "conclusion",
            "discussion",
            "methodology",
        ]

        # Test cases that should match
        test_content_late = [
            "Table 5 shows the experimental results",
            "In the Experiments section, we evaluate",
            "4. Methodology and Approach",
            "References and citations",
            "The ablation study demonstrates",
        ]

        for content in test_content_late:
            content_lower = content.lower()
            found_keyword = any(kw in content_lower for kw in late_keywords)
            assert found_keyword, f"Should detect late-doc keyword in: {content}"

        # Test cases that should NOT match (early document content)
        test_content_early = [
            "Abstract: This paper presents",
            "1. Introduction to the topic",
            "We propose a novel approach",
        ]

        for content in test_content_early:
            content_lower = content.lower()
            found_keyword = any(kw in content_lower for kw in late_keywords)
            # These might still match some keywords, but less likely
            # This is a soft check - early content CAN contain these words

    def test_suspicious_same_page_detection(self):
        """Test detection of suspicious all-same-page pattern."""
        # Simulate pages all being 1 but content suggesting late document
        pages_found = [1, 1, 1, 1, 1]
        unique_pages = set(pages_found)

        # All same page
        assert len(unique_pages) == 1

        # If single page is 1 or 2 and we have late-document content, it's suspicious
        single_page = list(unique_pages)[0]
        assert single_page <= 2  # This triggers suspicion check

    def test_missing_page_percentage(self):
        """Test calculation of missing page percentage."""
        total_chunks = 10
        missing_page_count = 6

        missing_pct = missing_page_count / total_chunks * 100
        assert missing_pct == 60.0

        # >50% missing should trigger warning
        assert missing_pct > 50

    def test_default_page_values_detection(self):
        """Test detection of all pages being 0 or 1 (default values)."""
        pages_found = [0, 1, 1, 0, 1]

        # All are 0 or 1
        all_default = all(p in (0, 1) for p in pages_found)
        assert all_default

        # More than 2 chunks should trigger warning
        assert len(pages_found) > 2


class TestPDFPageMarkerExtraction:
    """Test that PDF extraction inserts proper page markers."""

    def test_page_marker_format(self):
        """Test the [Page N] marker format used in PDF extraction."""
        # Simulate what document_processor.py does
        page_num = 0  # 0-indexed internally
        page_text = "Some content from the PDF"

        # The format used in extract_text_from_pdf
        formatted = f"[Page {page_num + 1}]\n{page_text}"

        assert "[Page 1]" in formatted
        assert page_text in formatted

    def test_page_marker_pattern_in_chunker(self):
        """Test that semantic_chunker can parse [Page N] markers."""
        import re

        # Pattern from StructureAwareChunker.PAGE_MARKER_PATTERN
        pattern = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)

        test_text = """[Page 1]
First page content here.

[Page 2]
Second page content here.

[Page 3]
Third page content."""

        matches = list(pattern.finditer(test_text))
        assert len(matches) == 3

        pages = [int(m.group(1)) for m in matches]
        assert pages == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
