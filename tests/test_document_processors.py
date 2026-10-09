"""The docling, markitdown and pymupdf4llm document processors.

Each is checked against a stand-in for its library (so the wiring is tested
everywhere) and, when the library is installed, against the real one on tiny
files written here: text, Markdown, HTML, and a one-page PDF built by hand.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.processing import docling_processor
from lasagna.processing.docling_processor import DoclingDocumentProcessor
from lasagna.processing.markdown_converters import MarkItDownProcessor, PyMuPDF4LLMProcessor

NAMES = ["docling", "markitdown", "pymupdf4llm"]


def minimal_pdf(lines):
    """A valid one-page PDF showing ``lines`` in Helvetica, with no PDF library."""
    content = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({x}) Tj T*" for x in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


PDF_LINES = ["Late chunking keeps context.", "Second line here."]
HTML = (
    "<html><head><title>Doc</title></head><body><h1>Intro</h1><p>Hello <b>world</b>.</p>"
    "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>"
)


@pytest.fixture
def files(tmp_path):
    paths = {
        "pdf": tmp_path / "one.pdf",
        "html": tmp_path / "page.html",
        "txt": tmp_path / "plain.txt",
        "md": tmp_path / "notes.md",
    }
    paths["pdf"].write_bytes(minimal_pdf(PDF_LINES))
    paths["html"].write_text(HTML)
    paths["txt"].write_text("Plain line one.\nLine two.\n")
    paths["md"].write_text("# Title\n\nBody text.\n")
    return {k: str(v) for k, v in paths.items()}


@pytest.mark.parametrize("name", NAMES)
def test_the_processor_passes_the_stage_contract(name):
    processor = registry.get("document_processor", name)
    report = ComponentTestHarness().validate(processor)
    assert report.passed, report.summary()
    assert processor.name == name


@pytest.mark.parametrize("name", NAMES)
def test_every_processor_is_registered_with_a_reference(name):
    entry = next(e for e in registry.list_document_processors() if e["name"] == name)
    assert entry["implements_contract"] and entry.get("paper")


def test_the_handmade_pdf_is_readable():
    from io import BytesIO

    import pypdf

    text = pypdf.PdfReader(BytesIO(minimal_pdf(PDF_LINES))).pages[0].extract_text()
    assert text.split("\n")[:2] == PDF_LINES


class TestMarkItDown:
    def test_a_missing_package_names_the_install(self, monkeypatch, files):
        monkeypatch.setitem(sys.modules, "markitdown", None)
        with pytest.raises(ImportError, match=r"lasagna-rag\[markitdown\]"):
            MarkItDownProcessor().extract(files["txt"])

    def test_it_returns_the_markdown_with_plugins_off(self, monkeypatch, files):
        made = {}

        class Converter:
            def __init__(self, **kwargs):
                made.update(kwargs)

            def convert(self, path):
                return SimpleNamespace(markdown=f"# from {path}", text_content="", title="T")

        monkeypatch.setitem(sys.modules, "markitdown", SimpleNamespace(MarkItDown=Converter))
        doc = MarkItDownProcessor().extract(files["txt"])
        assert doc.text == f"# from {files['txt']}"
        assert doc.metadata == {"source": files["txt"], "processor": "markitdown", "title": "T"}
        assert made == {"enable_plugins": False}

    def test_the_real_converter_keeps_structure(self, files):
        pytest.importorskip("markitdown")
        processor = MarkItDownProcessor()
        html = processor.extract(files["html"])
        assert html.text == "# Intro\n\nHello **world**.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"
        assert html.metadata["title"] == "Doc"
        assert processor.extract(files["txt"]).text == "Plain line one.\nLine two.\n"
        assert processor.extract(files["md"]).text == "# Title\n\nBody text.\n"

    def test_the_real_converter_reads_a_pdf(self, files):
        pytest.importorskip("markitdown")
        pytest.importorskip("pdfminer")
        text = MarkItDownProcessor().extract(files["pdf"]).text
        assert [line for line in text.splitlines() if line] == PDF_LINES


class TestPyMuPDF4LLM:
    def test_a_missing_package_names_the_install_and_licence(self, monkeypatch, files):
        monkeypatch.setitem(sys.modules, "pymupdf4llm", None)
        with pytest.raises(ImportError, match=r"AGPL-3.0.*lasagna-rag\[pymupdf4llm\]"):
            PyMuPDF4LLMProcessor().extract(files["pdf"])

    def test_pages_are_kept_and_joined_in_order(self, monkeypatch, files):
        calls = []

        def to_markdown(path, **kwargs):
            calls.append(kwargs)
            return [{"text": "# Page one\n\n"}, {"text": "Page two.\n"}]

        monkeypatch.setitem(sys.modules, "pymupdf4llm", SimpleNamespace(to_markdown=to_markdown))
        doc = PyMuPDF4LLMProcessor().extract(files["pdf"])
        assert doc.pages == ["# Page one\n\n", "Page two.\n"]
        assert doc.text == "# Page one\n\nPage two.\n"
        assert doc.metadata["page_count"] == 2
        assert calls == [{"page_chunks": True}]

    def test_the_real_converter_reads_a_pdf(self, files):
        pytest.importorskip("pymupdf4llm")
        doc = PyMuPDF4LLMProcessor().extract(files["pdf"])
        assert doc.metadata["page_count"] == 1
        assert all(line in doc.text for line in PDF_LINES)


class TestDocling:
    def test_without_docling_it_names_the_install(self, monkeypatch, files):
        monkeypatch.setattr(docling_processor, "is_docling_available", lambda: False)
        with pytest.raises(ImportError, match=r"lasagna-rag\[docling\]"):
            DoclingDocumentProcessor().extract(files["pdf"])

    def test_it_exports_markdown_and_tables_without_chunking(self, monkeypatch, files):
        class Table:
            def export_to_markdown(self, doc=None):
                return "| a |\n| - |"

        document = SimpleNamespace(export_to_markdown=lambda: "# Doc\n\nText.", tables=[Table()])

        class Converter:
            def convert(self, path):
                return SimpleNamespace(document=document)

        stub = SimpleNamespace(_get_converter=lambda: Converter())
        monkeypatch.setattr(docling_processor, "is_docling_available", lambda: True)
        monkeypatch.setattr(
            docling_processor.DoclingProcessor,
            "get_instance",
            classmethod(lambda cls, c=None: stub),
        )
        doc = DoclingDocumentProcessor().extract(files["pdf"])
        assert doc.text == "# Doc\n\nText."
        assert doc.tables == ["| a |\n| - |"]
        assert doc.metadata == {"source": files["pdf"], "processor": "docling"}

    def test_the_real_converter_reads_a_pdf(self, files):
        if not docling_processor.is_docling_available():
            pytest.skip("docling is not installed")
        text = DoclingDocumentProcessor().extract(files["pdf"]).text
        assert all(line in text for line in PDF_LINES)
