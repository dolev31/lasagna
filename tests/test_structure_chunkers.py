"""The sentence-window and Markdown-header chunkers: boundaries, offsets, metadata."""

from __future__ import annotations

import pytest

from lasagna.core.config import rag_config
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.processing.chunker.markdown import MarkdownHeaderChunker
from lasagna.processing.chunker.semantic import SentenceWindowChunkerAdapter, chunk_document


@pytest.fixture(autouse=True)
def _no_nltk_download(monkeypatch):
    """Sentence splitting uses installed NLTK data or its regex fallback, never a download."""
    monkeypatch.setattr("lasagna.processing.chunker.semantic._NLTK_CHECKED", True)


@pytest.mark.parametrize("name", ["sentence_window", "markdown_header"])
def test_the_chunker_passes_the_stage_contract(name):
    report = ComponentTestHarness().validate(registry.get("chunker", name))
    assert report.passed, report.summary()


class TestSentenceWindows:
    TEXT = "One. Two. Three. Four. Five."

    def test_windows_share_the_overlap_and_span_their_sentences(self):
        chunks = SentenceWindowChunkerAdapter(window_size=3, sentence_overlap=1).chunk(self.TEXT)
        assert [(c.content, c.start_char, c.end_char) for c in chunks] == [
            ("One. Two. Three.", 0, 16),
            ("Three. Four. Five.", 10, 28),
        ]

    def test_no_window_follows_the_one_that_reaches_the_end(self):
        # A third window, "Five.", lay wholly inside the second.
        chunks = SentenceWindowChunkerAdapter(window_size=3, sentence_overlap=1).chunk(self.TEXT)
        assert chunks[-1].content.endswith("Five.") and len(chunks) == 2

    def test_it_reads_the_arms_window_size(self):
        arm = rag_config.with_query_overrides({"window_size": 2, "sentence_overlap": 0})
        out = chunk_document(
            self.TEXT, "d", "s", "f", config=arm, chunker=SentenceWindowChunkerAdapter()
        )
        assert [c for c, _ in out] == ["One. Two.", "Three. Four.", "Five."]


DOC = (
    "Preface line.\n\n"
    "# Install\n\nRun pip.\n\n"
    "## Windows\n\nUse WSL.\n```\n# not a header\n```\n#### deep note\n\n"
    "# Usage\nCall it.\n"
)


def _md(**kwargs):
    kwargs.setdefault("split_long_sections", False)
    return MarkdownHeaderChunker(**kwargs)


class TestMarkdownHeaders:
    def test_each_section_is_one_chunk_with_its_header_path(self):
        chunks = _md(header_levels=3).chunk(DOC)
        assert [(c.content, c.start_char, c.end_char) for c in chunks] == [
            ("Preface line.", 0, 13),
            ("Run pip.", 26, 34),
            ("Use WSL.\n```\n# not a header\n```\n#### deep note", 48, 94),
            ("Call it.", 104, 112),
        ]
        assert [c.metadata["header_path"] for c in chunks] == [
            [],
            ["Install"],
            ["Install", "Windows"],
            ["Usage"],
        ]
        assert chunks[2].metadata["headers"] == {"h1": "Install", "h2": "Windows"}
        assert chunks[2].section_header == "Windows"

    def test_a_header_inside_a_code_fence_is_text(self):
        chunks = _md().chunk(DOC)
        assert all("not a header" not in c.metadata["header_path"] for c in chunks)

    def test_headers_deeper_than_the_setting_stay_in_the_text(self):
        doc = "## W\ntext\n#### deep\nmore"
        assert [c.content for c in _md(header_levels=3).chunk(doc)] == ["text\n#### deep\nmore"]
        four = _md(header_levels=4).chunk(doc)
        assert [(c.content, c.metadata["headers"]) for c in four] == [
            ("text", {"h2": "W"}),
            ("more", {"h2": "W", "h4": "deep"}),
        ]

    def test_kept_headers_open_their_chunk_and_a_bare_header_joins_its_child(self):
        chunks = _md(strip_headers=False).chunk("# A\n## B\n### C\ntext c\n## D\ntext d")
        assert [(c.content, c.metadata["header_path"]) for c in chunks] == [
            ("# A\n## B\n### C\ntext c", ["A", "B", "C"]),
            ("## D\ntext d", ["A", "D"]),
        ]

    def test_a_long_section_is_split_and_every_piece_keeps_the_headers(self):
        doc = "# Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 15 for i in range(6))
        arm = rag_config.with_query_overrides(
            {"chunk_size": 200, "chunk_overlap": 0, "chunk_unit": "chars"}
        )
        chunks = MarkdownHeaderChunker(split_long_sections=True).chunk(doc, config=arm)
        assert len(chunks) > 1
        assert all(c.metadata["header_path"] == ["Big"] for c in chunks)
        assert all(doc[c.start_char : c.end_char] == c.content for c in chunks)
        assert max(len(c.content) for c in chunks) <= 200

    def test_sections_and_headers_match_langchain(self):
        splitters = pytest.importorskip("langchain_text_splitters")
        theirs = splitters.MarkdownHeaderTextSplitter(
            [("#", "h1"), ("##", "h2"), ("###", "h3")]
        ).split_text(DOC)
        ours = _md(header_levels=3).chunk(DOC)
        assert [c.metadata["headers"] for c in ours] == [d.metadata for d in theirs]
        # LangChain re-joins a section's stripped lines; the text is otherwise the same.
        assert [c.content for c in ours] == [d.page_content for d in theirs]
