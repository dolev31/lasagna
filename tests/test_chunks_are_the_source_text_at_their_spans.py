"""Every built-in chunk is the source text, at the span it reports.

The semantic and structure-aware chunkers joined sentences with single spaces,
so "**Is the widget free?** Yes" came back as "**Is the widget free? ** Yes"
and paragraph breaks as spaces; a one-sentence text got a chunk with no span;
a section's span counted the whitespace stripped from it; and a fixed chunk
of a 233-character text reported 0-500 (``start_char=end_char=250`` on a
65-character chunk was another such span). Citations highlight these spans.
"""

from __future__ import annotations

import numpy as np
import pytest

from lasagna.core.config import rag_config
from lasagna.processing.chunker.semantic import (
    HierarchicalChunker,
    SemanticChunker,
    StructureAwareChunker,
)

FAQ = """# Widget FAQ

**Is the widget free?** Yes, the widget is free for personal use.

**Does it need an account?** No. You can use it without signing up.

## Pricing

| Plan | Seat |
|------|------|
| Team | $10  |

Teams pay per seat. Each seat costs ten dollars a month, billed yearly.
"""


class _SameEverywhere:
    """Every sentence embeds alike, so nothing splits on meaning."""

    def encode_documents_batch(self, texts, show_progress=False):
        return np.ones((len(texts), 3))

    def similarity(self, a, b):
        return 1.0


def _chunker(**overrides):
    config = rag_config.with_query_overrides(
        {"preserve_tables": True, "min_chunk_size": 10, "max_chunk_size": 1500, **overrides}
    )
    return SemanticChunker(embedding_manager=_SameEverywhere(), config=config)


def _assert_spans(chunks, text):
    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.content, (
            chunk.start_char,
            chunk.end_char,
            chunk.content[:60],
        )


def test_semantic_chunks_keep_the_source_text():
    chunks = _chunker().chunk_semantic(FAQ)
    _assert_spans(chunks, FAQ)
    assert "**Is the widget free?** Yes" in chunks[0].content
    assert "? **" not in chunks[0].content


def test_one_sentence_gets_its_span():
    text = "  Just one sentence here.  "
    (chunk,) = _chunker().chunk_semantic(text)
    _assert_spans([chunk], text)


def test_structure_aware_spans_point_into_the_document():
    chunks = StructureAwareChunker(_chunker()).chunk_with_structure(FAQ)
    _assert_spans(chunks, FAQ)
    contents = [c.content for c in chunks]
    assert any(c.startswith("**Is the widget free?** Yes") for c in contents)
    assert any(c.startswith("Teams pay per seat.") for c in contents), contents


@pytest.mark.parametrize("size", [40, 500])
def test_fixed_and_hierarchical_spans_end_inside_the_text(size):
    semantic = _chunker()
    _assert_spans(semantic.chunk_fixed(FAQ, chunk_size=size, overlap=10), FAQ)
    parents, children = HierarchicalChunker(semantic).chunk_hierarchical(
        FAQ, parent_size=size * 2, child_size=size
    )
    _assert_spans(parents, FAQ)
    _assert_spans(children, FAQ)
    assert max(c.end_char for c in parents + children) <= len(FAQ)


class _TwoTopics:
    """Every third sentence on another topic, so meaning splits happen."""

    def encode_documents_batch(self, texts, show_progress=False):
        return np.array([[1.0, 0.0] if i % 3 else [0.0, 1.0] for i, _ in enumerate(texts)])

    def similarity(self, a, b):
        return float(np.dot(a, b))


#: A text with double spaces, line breaks and "?**", as several sentences.
_SPLIT_TEXT = (
    "Alpha begins here.  Beta follows\nright after. Gamma closes the first part.\n\n"
    "**Is it free?** Yes, it is. Delta opens part two. Epsilon ends it.\n"
) * 4

#: What the chunkers produced before the change (e498b34^): the fixed chunks
#: exactly, the semantic and structure-aware chunks with whitespace removed (the
#: change restores the source's whitespace and nothing else). Sentences split
#: differently with nltk's punkt_tab and with the regex fallback, so each
#: splitter has its own digest; both were recorded from the code before the change.
_BEFORE = {
    "fixed": "56a28313f3bfe509",  # pragma: allowlist secret
    "semantic": {
        "nltk": "282dd8a52821d42f",  # pragma: allowlist secret
        "regex": "c67ba52a13c5b596",  # pragma: allowlist secret
    },
}


def _digest(contents, keep_whitespace=False):
    import hashlib
    import json

    shown = contents if keep_whitespace else ["".join(c.split()) for c in contents]
    return hashlib.sha256(json.dumps(shown).encode()).hexdigest()[:16]


@pytest.mark.parametrize("splitter", ["nltk", "regex"])
def test_the_chunk_boundaries_are_the_ones_from_before(splitter, monkeypatch):
    """Same chunks as before, split at the same places: only whitespace returned.

    The regex splitter is forced, so it runs on every machine (CI has no
    punkt_tab); nltk's runs wherever its data is installed.
    """
    from lasagna.processing.chunker import semantic as semantic_module

    if splitter == "regex":
        monkeypatch.setattr(semantic_module, "punkt_available", lambda: False)
    elif not semantic_module.punkt_available():
        pytest.skip(f"nltk's punkt_tab is not installed ({semantic_module.PUNKT_DOWNLOAD_COMMAND})")
    config = rag_config.with_query_overrides(
        {"min_chunk_size": 20, "max_chunk_size": 120, "preserve_tables": True}
    )
    semantic = SemanticChunker(embedding_manager=_TwoTopics(), config=config)
    fixed = semantic.chunk_fixed(_SPLIT_TEXT, chunk_size=90, overlap=15)
    assert _digest([c.content for c in fixed], keep_whitespace=True) == _BEFORE["fixed"]
    chunks = semantic.chunk_semantic(_SPLIT_TEXT)
    assert _digest([c.content for c in chunks]) == _BEFORE["semantic"][splitter]
    structured = StructureAwareChunker(semantic).chunk_with_structure("# Head\n\n" + _SPLIT_TEXT)
    assert _digest([c.content for c in structured]) == _BEFORE["semantic"][splitter]
    _assert_spans(chunks, _SPLIT_TEXT)
