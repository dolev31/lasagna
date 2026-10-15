"""Each preset profile says which chunker it runs.

The presets declared ``chunking_strategy: semantic`` (or inherited it) while
their structural settings ran another chunker: `speed` and `standard` keep
headers and tables, so structure-aware chunking ran, and `balanced` and
`max_quality` are hierarchical. Every upload on `speed` warned that `semantic`
was being overridden. The presets now declare what runs.

Declaring it must not change a single chunk. An arm built on a preset that
turns the structural settings off ran `semantic` before, and still does, and
benchmark ingest still hands the chunker the embedding model.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.core.session_config._profiles import ProfileMixin
from lasagna.processing.chunker import semantic
from lasagna.processing.chunker.semantic import (
    ChunkingStrategy,
    chunk_document,
    resolve_effective_strategy,
)

PRESETS = sorted(ProfileMixin._PROFILE_SPECS)


@pytest.fixture(scope="module")
def presets():
    from lasagna.core.session_config import get_knowledge_config_manager

    manager = get_knowledge_config_manager()
    return {p.name: p.config for p in manager._default_profiles if p.name in PRESETS}


@pytest.fixture
def warnings():
    from loguru import logger

    semantic._WARNED_OVERRIDES.clear()
    captured: list = []
    sink = logger.add(captured.append, level="WARNING")
    yield captured
    logger.remove(sink)


@pytest.mark.parametrize("name", PRESETS)
def test_the_declared_strategy_is_the_one_that_runs(presets, name, warnings):
    config = SimpleNamespace(**presets[name])
    assert resolve_effective_strategy(config) == config.chunking_strategy
    assert not warnings, warnings


def test_structure_aware_is_a_named_strategy():
    assert ChunkingStrategy("structure_aware") is ChunkingStrategy.STRUCTURE_AWARE


class _Embeddings:
    """Deterministic sentence vectors, so semantic chunking runs without a model."""

    def encode_documents_batch(self, texts, show_progress=False):
        return np.array([self._vector(t) for t in texts])

    @staticmethod
    def _vector(text):
        digest = hashlib.sha256(text.encode()).digest()
        return np.frombuffer(digest, dtype=np.uint8).astype(float)

    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


DOCUMENT = (
    "# Install\n\nFirst sentence here. Second one follows. A third about pears.\n\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
    "## Use\n\nSearch the corpus. Read the answer. Compare two runs. " * 6
)

#: Every combination an arm can turn the structural settings into.
FLAGS = [
    {"enable_hierarchical": h, "preserve_headers": p, "preserve_tables": t}
    for h in (True, False)
    for p in (True, False)
    for t in (True, False)
]


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("flags", FLAGS, ids=lambda f: "-".join(k for k, v in f.items() if v))
def test_declaring_it_changes_no_chunk(presets, name, flags):
    """The same chunks as when the preset said `semantic`, whatever an arm turns off."""
    from lasagna.core.config import rag_config

    declared = rag_config.with_query_overrides({**presets[name], **flags})
    before = rag_config.with_query_overrides(
        {**presets[name], **flags, "chunking_strategy": "semantic"}
    )
    for manager in (None, _Embeddings()):
        now = chunk_document(DOCUMENT, "d", "s", "f", embedding_manager=manager, config=declared)
        was = chunk_document(DOCUMENT, "d", "s", "f", embedding_manager=manager, config=before)
        assert [(c, m["start_char"], m.get("section_header")) for c, m in now] == [
            (c, m["start_char"], m.get("section_header")) for c, m in was
        ]


@pytest.mark.parametrize("name", PRESETS)
def test_benchmark_ingest_still_hands_over_the_embedding_model(presets, name):
    """Fast ingest passed the model only for `semantic`; the sections of a
    structure-aware preset were chunked semantically with it."""
    from lasagna.processing.chunker.semantic import chunker_needs_embeddings

    assert chunker_needs_embeddings(SimpleNamespace(**presets[name]))
    assert chunker_needs_embeddings(SimpleNamespace(chunking_strategy="semantic"))
    assert not chunker_needs_embeddings(SimpleNamespace(chunking_strategy="fixed"))


@pytest.mark.parametrize("name", PRESETS)
def test_the_declaration_passes_config_validation(presets, name):
    """Sessions and sweep arms validate their config; an arm on `speed` was
    refused with "Invalid value 'structure_aware'" before the schema knew it."""
    from lasagna.configurator.validation import validate_config

    result = validate_config({"chunking_strategy": presets[name]["chunking_strategy"]})
    assert result.valid, result.errors
