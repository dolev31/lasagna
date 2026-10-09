"""Tests for Docling document processor integration.

Unit tests (TestDocling*) run without docling installed — they mock everything.
Integration tests (TestDoclingIntegration) require docling and are skipped otherwise.

Run with:
    pytest tests/test_docling_processor.py -v
"""

import hashlib
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from lasagna.processing.docling_processor import (
    DoclingProcessor,
    _reset_availability_cache,
    is_docling_available,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Ensure each test starts with a fresh processor singleton."""
    DoclingProcessor.reset()
    _reset_availability_cache()
    yield
    DoclingProcessor.reset()
    _reset_availability_cache()


def _make_mock_config(**overrides) -> MagicMock:
    """Create a mock RAGConfig with Docling defaults."""
    defaults = {
        "docling_enabled": True,
        "docling_mode": "local",
        "docling_pdf_backend": "pypdfium2",
        "docling_ocr_enabled": False,
        "docling_table_structure": True,
        "docling_max_tokens": 512,
        "docling_timeout": 300,
        "citation_id_length": 8,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_mock_chunk(
    text: str,
    headings: List[str] = None,
    label: str = "TEXT",
    page_no: int = None,
    charspan: tuple = None,
):
    """Build a minimal chunk-like object matching Docling's output.

    Optionally attach provenance data (page_no, charspan) to mirror
    Docling's native ``doc_item.prov`` structure.
    """
    prov_list = []
    if page_no is not None or charspan is not None:
        prov_item = SimpleNamespace(
            page_no=page_no,
            charspan=charspan,
        )
        prov_list.append(prov_item)

    item = SimpleNamespace(label=SimpleNamespace(name=label), prov=prov_list)
    meta = SimpleNamespace(
        headings=headings or [],
        doc_items=[item],
    )
    chunk = SimpleNamespace(meta=meta, _text=text)
    return chunk


# ===========================================================================
# A. Unit tests — no docling installation required
# ===========================================================================


class TestDoclingAvailability:
    """Verify the cached availability check."""

    def test_returns_bool(self):
        result = is_docling_available()
        assert isinstance(result, bool)

    def test_cache_is_stable(self):
        a = is_docling_available()
        b = is_docling_available()
        assert a == b

    @patch.dict("sys.modules", {"docling": None})
    def test_false_when_import_fails(self):
        _reset_availability_cache()
        # Force ImportError by patching the module to None
        with patch("builtins.__import__", side_effect=ImportError("no docling")):
            _reset_availability_cache()
            assert is_docling_available() is False


class TestDoclingProcessorSingleton:
    """Verify singleton lifecycle."""

    def test_get_instance_returns_same_object(self):
        cfg = _make_mock_config()
        a = DoclingProcessor.get_instance(cfg)
        b = DoclingProcessor.get_instance(cfg)
        assert a is b

    def test_reset_clears_instance(self):
        cfg = _make_mock_config()
        a = DoclingProcessor.get_instance(cfg)
        DoclingProcessor.reset()
        b = DoclingProcessor.get_instance(cfg)
        assert a is not b


class TestDoclingModeResolution:
    """Verify deployment mode auto-detection."""

    def test_local_mode_explicit(self):
        cfg = _make_mock_config(docling_mode="local")
        proc = DoclingProcessor(cfg)
        assert proc._resolve_mode() == "local"

    def test_production_mode_explicit(self):
        cfg = _make_mock_config(docling_mode="production")
        proc = DoclingProcessor(cfg)
        assert proc._resolve_mode() == "production"

    @patch("lasagna.processing.docling_processor.DoclingProcessor._resolve_mode")
    def test_auto_mode_no_gpu(self, mock_resolve):
        """When no GPU available, auto should resolve to local."""
        mock_resolve.return_value = "local"
        cfg = _make_mock_config(docling_mode="auto")
        proc = DoclingProcessor(cfg)
        assert proc._resolve_mode() == "local"

    def test_auto_mode_without_torch(self):
        """If torch is not importable, auto falls back to local."""
        cfg = _make_mock_config(docling_mode="auto")
        proc = DoclingProcessor(cfg)
        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert proc._resolve_mode() == "local"


class TestDoclingChunkTypeDetection:
    """Verify chunk type mapping from Docling labels."""

    def test_table_detection(self):
        chunk = _make_mock_chunk("data", label="TABLE")
        assert DoclingProcessor._detect_chunk_type(chunk) == "table"

    def test_code_detection(self):
        chunk = _make_mock_chunk("code", label="CODE")
        assert DoclingProcessor._detect_chunk_type(chunk) == "code"

    def test_header_detection(self):
        chunk = _make_mock_chunk("title", label="SECTION_HEADER")
        assert DoclingProcessor._detect_chunk_type(chunk) == "header"

    def test_title_detection(self):
        chunk = _make_mock_chunk("title", label="TITLE")
        assert DoclingProcessor._detect_chunk_type(chunk) == "header"

    def test_formula_detection(self):
        chunk = _make_mock_chunk("eq", label="FORMULA")
        assert DoclingProcessor._detect_chunk_type(chunk) == "formula"

    def test_text_fallback(self):
        chunk = _make_mock_chunk("text", label="TEXT")
        assert DoclingProcessor._detect_chunk_type(chunk) == "text"

    def test_no_doc_items(self):
        chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[]))
        assert DoclingProcessor._detect_chunk_type(chunk) == "text"

    def test_no_meta(self):
        chunk = SimpleNamespace(meta=None)
        assert DoclingProcessor._detect_chunk_type(chunk) == "text"

    def test_picture_detection(self):
        chunk = _make_mock_chunk("img", label="PICTURE")
        assert DoclingProcessor._detect_chunk_type(chunk) == "image"

    def test_figure_detection(self):
        chunk = _make_mock_chunk("fig", label="FIGURE")
        assert DoclingProcessor._detect_chunk_type(chunk) == "image"


class TestDoclingProvenanceExtraction:
    """Verify native Docling provenance extraction."""

    def test_page_number_from_prov(self):
        chunk = _make_mock_chunk("text", page_no=3, charspan=(100, 250))
        page, start, end = DoclingProcessor._extract_provenance(chunk)
        assert page == 3

    def test_charspan_from_prov(self):
        chunk = _make_mock_chunk("text", page_no=1, charspan=(100, 250))
        page, start, end = DoclingProcessor._extract_provenance(chunk)
        assert start == 100
        assert end == 250

    def test_multiple_doc_items_min_page(self):
        """When chunk spans multiple pages, return the lowest page number."""
        prov1 = SimpleNamespace(page_no=3, charspan=(200, 400))
        prov2 = SimpleNamespace(page_no=2, charspan=(50, 199))
        item1 = SimpleNamespace(
            label=SimpleNamespace(name="TEXT"),
            prov=[prov1],
        )
        item2 = SimpleNamespace(
            label=SimpleNamespace(name="TEXT"),
            prov=[prov2],
        )
        chunk = SimpleNamespace(
            meta=SimpleNamespace(headings=[], doc_items=[item1, item2]),
        )
        page, start, end = DoclingProcessor._extract_provenance(chunk)
        assert page == 2
        assert start == 50
        assert end == 400

    def test_no_prov_returns_defaults(self):
        """No provenance data → (None, 0, 0)."""
        item = SimpleNamespace(label=SimpleNamespace(name="TEXT"), prov=[])
        chunk = SimpleNamespace(
            meta=SimpleNamespace(headings=[], doc_items=[item]),
        )
        page, start, end = DoclingProcessor._extract_provenance(chunk)
        assert page is None
        assert start == 0
        assert end == 0

    def test_no_meta_returns_defaults(self):
        chunk = SimpleNamespace(meta=None)
        page, start, end = DoclingProcessor._extract_provenance(chunk)
        assert page is None
        assert start == 0
        assert end == 0

    def test_page_number_in_process_output(self):
        """process() should include page_number in metadata when prov has it."""
        cfg = _make_mock_config()

        mock_dl_doc = MagicMock()
        mock_dl_doc.export_to_markdown.return_value = "# Title\n\nContent on page 3."
        mock_result = MagicMock()
        mock_result.document = mock_dl_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        mock_chunk = _make_mock_chunk("Content", page_no=3, charspan=(10, 50))
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_chunker.contextualize.return_value = "Content"

        proc = DoclingProcessor(cfg)
        proc._converter = mock_converter
        proc._chunker_cache[512] = mock_chunker

        chunks, _, _ = proc.process(
            "/tmp/test.pdf",
            "doc_1",
            "sess_1",
            "test.pdf",
            cfg,
        )
        _, meta = chunks[0]
        assert meta["page_number"] == 3
        assert meta["start_char"] == 10
        assert meta["end_char"] == 50

    def test_no_page_number_when_prov_empty(self):
        """process() should NOT include page_number when prov is empty."""
        cfg = _make_mock_config()

        mock_dl_doc = MagicMock()
        mock_dl_doc.export_to_markdown.return_value = "# Title\n\nContent."
        mock_result = MagicMock()
        mock_result.document = mock_dl_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        # Chunk without provenance
        mock_chunk = _make_mock_chunk("Content")
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_chunker.contextualize.return_value = "Content"

        proc = DoclingProcessor(cfg)
        proc._converter = mock_converter
        proc._chunker_cache[512] = mock_chunker

        chunks, _, _ = proc.process(
            "/tmp/test.pdf",
            "doc_1",
            "sess_1",
            "test.pdf",
            cfg,
        )
        _, meta = chunks[0]
        assert "page_number" not in meta


class TestDoclingOutputContract:
    """Verify the output format matches the chunk_document() contract."""

    def _run_mocked_process(self, chunks_data: List[Dict[str, Any]]):
        """Run process() with mocked Docling internals."""
        cfg = _make_mock_config()

        # Mock converter
        mock_dl_doc = MagicMock()
        mock_dl_doc.export_to_markdown.return_value = "# Title\n\nSome content here."
        mock_result = MagicMock()
        mock_result.document = mock_dl_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        # Mock chunker
        mock_chunks = []
        for cd in chunks_data:
            mock_chunks.append(
                _make_mock_chunk(cd["text"], cd.get("headings", []), cd.get("label", "TEXT"))
            )

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.contextualize.side_effect = [cd["text"] for cd in chunks_data]

        proc = DoclingProcessor(cfg)
        proc._converter = mock_converter
        proc._chunker_cache[512] = mock_chunker

        return proc.process(
            file_path="/tmp/test.pdf",
            document_id="doc_123",
            session_id="sess_456",
            filename="test.pdf",
            config=cfg,
        )

    def test_output_is_correct_tuple(self):
        result = self._run_mocked_process([{"text": "Hello world"}])
        assert isinstance(result, tuple)
        assert len(result) == 3
        chunks, parents, full_text = result
        assert isinstance(chunks, list)
        assert parents is None
        assert isinstance(full_text, str)

    def test_chunks_are_tuples_of_str_dict(self):
        chunks, _, _ = self._run_mocked_process(
            [
                {"text": "Chunk one"},
                {"text": "Chunk two"},
            ]
        )
        assert len(chunks) == 2
        for text, meta in chunks:
            assert isinstance(text, str)
            assert isinstance(meta, dict)

    def test_metadata_has_required_keys(self):
        chunks, _, _ = self._run_mocked_process([{"text": "test"}])
        _, meta = chunks[0]
        required = {
            "document_id",
            "session_id",
            "filename",
            "chunk_index",
            "total_chunks",
            "stable_id",
            "chunk_type",
            "start_char",
            "end_char",
            "line_start",
            "line_end",
            "processor",
        }
        assert required.issubset(set(meta.keys()))

    def test_metadata_values(self):
        chunks, _, _ = self._run_mocked_process([{"text": "test"}])
        _, meta = chunks[0]
        assert meta["document_id"] == "doc_123"
        assert meta["session_id"] == "sess_456"
        assert meta["filename"] == "test.pdf"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 1
        assert meta["processor"] == "docling"

    def test_stable_id_is_content_hash(self):
        text = "Hello world"
        chunks, _, _ = self._run_mocked_process([{"text": text}])
        _, meta = chunks[0]
        expected = hashlib.sha256(text.encode()).hexdigest()[:8]
        assert meta["stable_id"] == expected

    def test_headings_in_metadata(self):
        chunks, _, _ = self._run_mocked_process(
            [
                {"text": "content", "headings": ["Chapter 1", "Section 1.2"]},
            ]
        )
        _, meta = chunks[0]
        assert meta["section_header"] == "Section 1.2"
        assert meta["headings"] == ["Chapter 1", "Section 1.2"]

    def test_metadata_is_json_serializable(self):
        chunks, _, _ = self._run_mocked_process(
            [
                {"text": "test", "headings": ["H1"]},
            ]
        )
        for _, meta in chunks:
            json.dumps(meta)  # must not raise

    def test_empty_text_raises(self):
        cfg = _make_mock_config()
        mock_dl_doc = MagicMock()
        mock_dl_doc.export_to_markdown.return_value = ""
        mock_result = MagicMock()
        mock_result.document = mock_dl_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        proc = DoclingProcessor(cfg)
        proc._converter = mock_converter

        with pytest.raises(ValueError, match="no content"):
            proc.process("/tmp/test.pdf", "doc_1", "sess_1", "test.pdf", cfg)

    def test_no_chunks_raises(self):
        cfg = _make_mock_config()
        mock_dl_doc = MagicMock()
        mock_dl_doc.export_to_markdown.return_value = "Some text"
        mock_result = MagicMock()
        mock_result.document = mock_dl_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        proc = DoclingProcessor(cfg)
        proc._converter = mock_converter
        proc._chunker_cache[512] = mock_chunker

        with pytest.raises(ValueError, match="no chunks"):
            proc.process("/tmp/test.pdf", "doc_1", "sess_1", "test.pdf", cfg)


class TestShouldUseDocling:
    """Verify the pipeline routing logic."""

    def test_disabled_returns_false(self):
        from lasagna.api._document_pipeline import DocumentPipeline

        cfg = _make_mock_config(docling_enabled=False)
        assert DocumentPipeline._should_use_docling("/tmp/test.pdf", cfg) is False

    @patch(
        "lasagna.api._document_pipeline.is_docling_available",
        return_value=False,
    )
    def test_not_installed_returns_false(self, _mock):
        from lasagna.api._document_pipeline import DocumentPipeline

        cfg = _make_mock_config(docling_enabled=True)
        assert DocumentPipeline._should_use_docling("/tmp/test.pdf", cfg) is False

    @patch(
        "lasagna.api._document_pipeline.is_docling_available",
        return_value=True,
    )
    def test_enabled_and_installed_returns_true(self, _mock):
        from lasagna.api._document_pipeline import DocumentPipeline

        cfg = _make_mock_config(docling_enabled=True)
        assert DocumentPipeline._should_use_docling("/tmp/test.pdf", cfg) is True


# ===========================================================================
# B. Integration tests — require docling installed
# ===========================================================================

try:
    import docling  # noqa: F401

    _HAS_DOCLING = True
except ImportError:
    _HAS_DOCLING = False

# Docling fetches its layout models from the Hugging Face Hub on first use, so
# with the hub offline these fail on a model download that says nothing about
# the code under test. CI runs them online.
_HUB_OFFLINE = any(
    os.environ.get(var, "").strip().lower() in {"1", "true", "yes", "on"}
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
)


@pytest.mark.skipif(not _HAS_DOCLING, reason="docling not installed")
@pytest.mark.skipif(
    _HUB_OFFLINE, reason="Docling downloads its layout models on first use; the hub is offline"
)
class TestDoclingIntegration:
    """End-to-end tests with real Docling processing."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a minimal PDF for testing."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            pdf_path = tmp_path / "test_document.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(72, 720, "Introduction")
            c.drawString(72, 700, "This is a test document for Docling integration.")
            c.drawString(72, 680, "It contains simple text content.")
            c.drawString(72, 640, "Section Two")
            c.drawString(72, 620, "More content in the second section of this document.")
            c.save()
            return pdf_path
        except ImportError:
            pytest.skip("reportlab not installed — cannot create test PDF")

    def test_process_returns_chunks(self, sample_pdf):
        cfg = _make_mock_config(docling_ocr_enabled=False)
        proc = DoclingProcessor(cfg)
        chunks, parents, text = proc.process(
            file_path=str(sample_pdf),
            document_id="doc_integ",
            session_id="sess_integ",
            filename="test_document.pdf",
            config=cfg,
        )
        assert len(chunks) > 0
        assert parents is None
        assert len(text) > 0

    def test_chunks_have_correct_format(self, sample_pdf):
        cfg = _make_mock_config(docling_ocr_enabled=False)
        proc = DoclingProcessor(cfg)
        chunks, _, _ = proc.process(
            file_path=str(sample_pdf),
            document_id="doc_integ",
            session_id="sess_integ",
            filename="test_document.pdf",
            config=cfg,
        )
        for chunk_text, meta in chunks:
            assert isinstance(chunk_text, str)
            assert len(chunk_text) > 0
            assert meta["document_id"] == "doc_integ"
            assert meta["session_id"] == "sess_integ"
            assert meta["processor"] == "docling"
            assert "stable_id" in meta
            # Verify JSON-serializable
            json.dumps(meta)

    def test_chunk_texts_are_embeddable(self, sample_pdf):
        """All chunk texts should be non-empty strings suitable for embedding."""
        cfg = _make_mock_config(docling_ocr_enabled=False)
        proc = DoclingProcessor(cfg)
        chunks, _, _ = proc.process(
            file_path=str(sample_pdf),
            document_id="doc_integ",
            session_id="sess_integ",
            filename="test_document.pdf",
            config=cfg,
        )
        texts = [t for t, _ in chunks]
        assert all(isinstance(t, str) and len(t.strip()) > 0 for t in texts)

    def test_chunks_have_page_numbers(self, sample_pdf):
        """Docling extracts page numbers from PDF via native provenance."""
        cfg = _make_mock_config(docling_ocr_enabled=False)
        proc = DoclingProcessor(cfg)
        chunks, _, _ = proc.process(
            file_path=str(sample_pdf),
            document_id="doc_integ",
            session_id="sess_integ",
            filename="test_document.pdf",
            config=cfg,
        )
        # At least one chunk should have a page_number from Docling's prov
        pages = [m.get("page_number") for _, m in chunks if m.get("page_number") is not None]
        assert len(pages) > 0, (
            "Expected at least one chunk with page_number from Docling provenance"
        )
        assert all(isinstance(p, int) and p > 0 for p in pages)

    def test_charspan_is_populated(self, sample_pdf):
        """start_char and end_char come from Docling's charspan, not hardcoded 0."""
        cfg = _make_mock_config(docling_ocr_enabled=False)
        proc = DoclingProcessor(cfg)
        chunks, _, _ = proc.process(
            file_path=str(sample_pdf),
            document_id="doc_integ",
            session_id="sess_integ",
            filename="test_document.pdf",
            config=cfg,
        )
        # At least one chunk should have non-zero charspan from Docling
        has_nonzero = any(m["start_char"] > 0 or m["end_char"] > 0 for _, m in chunks)
        assert has_nonzero, (
            "Expected at least one chunk with non-zero charspan from Docling provenance"
        )
