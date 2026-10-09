"""Semantic chunking with Kamradt's breakpoint methods.

The embedder is stubbed so the distances, and therefore the breakpoints, can
be worked out by hand: sentences about cats embed to one axis and sentences
about cars to the other, so the only large distance is at the topic change.
"""

from __future__ import annotations

import numpy as np
import pytest

from lasagna.core.config import rag_config
from lasagna.core.testing import ComponentTestHarness
from lasagna.processing.chunker.semantic import (
    SemanticChunkerAdapter,
    breakpoint_threshold,
    chunk_document,
)

TEXT = (
    "A cat sat. The cat purred. Cats nap. "  # three about cats
    "A car sped. The car honked. Cars rust."  # three about cars
)


class TopicEmbedder:
    """[mentions of cat, mentions of car], and a record of what it was asked."""

    def __init__(self):
        self.seen = []

    def encode_documents_batch(self, texts, show_progress=False):
        self.seen.append(list(texts))
        return [
            np.array([t.lower().count("cat"), t.lower().count("car")], dtype=float) for t in texts
        ]

    def similarity(self, a, b):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm else 0.0


def _chunk(method, amount=None, buffer_size=0, embedder=None):
    chunker = SemanticChunkerAdapter(embedding_manager=embedder or TopicEmbedder())
    return chunker.chunk(
        TEXT, breakpoint_method=method, breakpoint_amount=amount, buffer_size=buffer_size
    )


def test_the_chunker_passes_the_stage_contract():
    report = ComponentTestHarness().validate(
        SemanticChunkerAdapter(embedding_manager=TopicEmbedder())
    )
    assert report.passed, report.summary()


def test_the_percentile_method_breaks_at_the_topic_change():
    # distances [0, 0, 1, 0, 0]; the 95th percentile is 0.8; only gap 2 exceeds it.
    chunks = _chunk("percentile")
    assert [(c.content, c.start_char, c.end_char) for c in chunks] == [
        ("A cat sat. The cat purred. Cats nap.", 0, 36),
        ("A car sped. The car honked. Cars rust.", 37, 75),
    ]


@pytest.mark.parametrize(
    "method,amount,n_chunks",
    [
        ("standard_deviation", 3.0, 1),  # mean 0.2 + 3 * std 0.4 = 1.4: nothing exceeds it
        ("standard_deviation", 1.0, 2),  # 0.2 + 0.4 = 0.6: the topic change does
        ("interquartile", 1.5, 2),  # IQR is 0, so the threshold is the mean, 0.2
    ],
)
def test_the_threshold_follows_the_method_and_amount(method, amount, n_chunks):
    assert len(_chunk(method, amount)) == n_chunks


def test_the_gradient_method_breaks_where_distance_rises_fastest():
    # gradient of [0, 0, 1, 0, 0] is [0, .5, 0, -.5, 0]; its 95th percentile is
    # 0.4, so the break comes one gap early, as in LangChain.
    chunks = _chunk("gradient")
    assert [c.content for c in chunks] == [
        "A cat sat. The cat purred.",
        "Cats nap. A car sped. The car honked. Cars rust.",
    ]


def test_each_sentence_is_embedded_with_its_buffer():
    embedder = TopicEmbedder()
    _chunk("percentile", buffer_size=1, embedder=embedder)
    windows = embedder.seen[0]
    assert windows[0] == "A cat sat. The cat purred."
    assert windows[1] == "A cat sat. The cat purred. Cats nap."
    assert windows[-1] == "The car honked. Cars rust."


def test_the_threshold_matches_langchains_formulas():
    d = [0.1, 0.2, 0.9, 0.15, 0.3]
    assert breakpoint_threshold(d, "percentile", 95)[0] == pytest.approx(np.percentile(d, 95))
    assert breakpoint_threshold(d, "standard_deviation", 3)[0] == pytest.approx(
        np.mean(d) + 3 * np.std(d)
    )
    q1, q3 = np.percentile(d, [25, 75])
    assert breakpoint_threshold(d, "interquartile", 1.5)[0] == pytest.approx(
        np.mean(d) + 1.5 * (q3 - q1)
    )
    with pytest.raises(ValueError, match="semantic_breakpoint_method"):
        breakpoint_threshold(d, "median", 1)


def test_the_arms_method_reaches_the_built_in_semantic_strategy():
    arm = rag_config.with_query_overrides(
        {
            "chunking_strategy": "semantic",
            "semantic_breakpoint_method": "percentile",
            "semantic_buffer_size": 0,
            "enable_hierarchical": False,
            "preserve_headers": False,
            "preserve_tables": False,
        }
    )
    out = chunk_document(TEXT, "d", "s", "f", embedding_manager=TopicEmbedder(), config=arm)
    assert [c for c, _ in out] == [
        "A cat sat. The cat purred. Cats nap.",
        "A car sped. The car honked. Cars rust.",
    ]


def test_without_a_model_it_raises_instead_of_running_fixed_chunking(monkeypatch):
    from lasagna.infrastructure.embedding import manager

    def no_model(config=None):
        raise RuntimeError("Could not load any embedding model")

    monkeypatch.setattr(manager, "shared_embedding_manager", no_model)
    with pytest.raises(RuntimeError, match="embedding model"):
        SemanticChunkerAdapter().chunk(TEXT)


def _cached(repo):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.skipif(not _cached(MINILM), reason=f"{MINILM} is not cached")
def test_a_real_model_separates_two_unrelated_topics():
    from lasagna.infrastructure.embedding.manager import shared_embedding_manager

    arm = rag_config.with_query_overrides(
        {"embedding_model": MINILM, "use_gpu": False, "enable_embedding_cache": False}
    )
    text = (
        "The violin has four strings. Violinists draw a bow across the strings. "
        "A violin's body is carved from spruce and maple. "
        "Photosynthesis turns light into sugar in plant leaves. "
        "Chlorophyll in leaves absorbs red and blue light. "
        "Plants release oxygen during photosynthesis."
    )
    # Sentences embedded alone: with a buffer, the windows either side of the
    # change each hold both topics and the largest gap can move by one.
    chunks = SemanticChunkerAdapter(embedding_manager=shared_embedding_manager(arm)).chunk(
        text, breakpoint_method="percentile", buffer_size=0
    )
    assert len(chunks) == 2
    assert chunks[0].content.endswith("spruce and maple.")
