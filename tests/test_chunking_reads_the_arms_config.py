"""The built-in chunking strategies chunk with the arm's settings.

``chunk_document`` took the arm's config and then built a ``SemanticChunker``
that read ``chunk_size``, ``chunk_overlap``, ``window_size`` and the semantic
thresholds from the process-wide config. Two arms sweeping ``chunk_size`` got
byte-identical chunks, so the sweep reported that chunk size does not matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from lasagna.core.config import rag_config
from lasagna.processing.chunker.semantic import (
    HierarchicalChunkerAdapter,
    SemanticChunkerAdapter,
    StructureAwareChunkerAdapter,
    chunk_document,
)


@pytest.fixture(autouse=True)
def _no_nltk_download(monkeypatch):
    """Sentence splitting uses installed NLTK data or its regex fallback, never a download."""
    monkeypatch.setattr("lasagna.processing.chunker.semantic._NLTK_CHECKED", True)


_FLAT = {"enable_hierarchical": False, "preserve_headers": False, "preserve_tables": False}

TEXT = "Alpha beta gamma delta. " * 200


def _arm(**overrides):
    return rag_config.with_query_overrides({**_FLAT, **overrides})


def test_two_arms_sweeping_chunk_size_get_different_chunks():
    small = chunk_document(
        TEXT, "d", "s", "f", config=_arm(chunking_strategy="fixed", chunk_size=200)
    )
    large = chunk_document(
        TEXT, "d", "s", "f", config=_arm(chunking_strategy="fixed", chunk_size=1000)
    )
    assert max(len(c) for c, _ in small) <= 200
    assert len(small) > len(large)


def test_a_sentence_overlap_of_zero_is_honoured():
    text = "One. Two. Three. Four."
    out = chunk_document(
        text,
        "d",
        "s",
        "f",
        config=_arm(chunking_strategy="sentence_window", window_size=2, sentence_overlap=0),
    )
    # `overlap or default` read 0 as "unset" and overlapped by one sentence.
    assert [c for c, _ in out] == ["One. Two.", "Three. Four."]


class _OrthogonalEmbedder:
    """Every sentence points somewhere new, so every gap is a breakpoint."""

    def encode_documents_batch(self, texts, show_progress=False):
        return [np.eye(len(texts))[i] for i in range(len(texts))]

    def similarity(self, a, b):
        return float(np.dot(a, b))


def test_the_semantic_adapter_accepts_its_size_keywords():
    chunker = SemanticChunkerAdapter(embedding_manager=_OrthogonalEmbedder())
    text = "Short one. Another short one. A third short sentence here."
    # min_chunk_size used to be forwarded under that name and raise TypeError.
    chunks = chunker.chunk(text, threshold=0.5, min_chunk_size=1, max_chunk_size=10_000)
    assert [c.content for c in chunks] == [
        "Short one.",
        "Another short one.",
        "A third short sentence here.",
    ]


def test_the_hierarchical_component_sizes_children_by_the_arm():
    arm = rag_config.with_query_overrides({"parent_chunk_size": 400, "child_chunk_size": 100})
    out = chunk_document(TEXT, "d", "s", "f", config=arm, chunker=HierarchicalChunkerAdapter())
    assert max(len(c) for c, _ in out) <= 100


def test_the_structure_aware_component_chunks_text_semantically():
    text = "# Pets\n\nA cat sat. The cat purred.\n\n# Cars\n\nA car sped. The car honked."
    chunks = StructureAwareChunkerAdapter(embedding_manager=_OrthogonalEmbedder()).chunk(
        text, config=rag_config.with_query_overrides({"min_chunk_size": 1, "preserve_tables": True})
    )
    # Every sentence pair is dissimilar to the stub, so each sentence is a
    # chunk; size-based chunking would have kept each section whole.
    assert [(c.content, c.section_header) for c in chunks] == [
        ("A cat sat.", "Pets"),
        ("The cat purred.", "Pets"),
        ("A car sped.", "Cars"),
        ("The car honked.", "Cars"),
    ]
