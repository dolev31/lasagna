"""A run must record the stand-ins that ran because an extra was missing.

Without nltk the chunker splits sentences naively and BM25 runs unstemmed; the
same named configuration then produces different numbers on a core install
and on a full one, and the two bundles were indistinguishable.
"""

from __future__ import annotations

import importlib.util

from lasagna.core import provenance
from lasagna.knowledge_evaluation_sdk.bundle import _integrity


def test_missing_nltk_and_rank_bm25_are_named(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name in ("nltk", "rank_bm25") else real(name, *a),
    )
    fallbacks = provenance._active_fallbacks()
    assert {"sentence_split_without_nltk", "bm25_without_stemming", "hybrid_without_bm25"} <= set(
        fallbacks
    )


def test_a_bundle_with_fallbacks_says_so():
    meta = {"environment": {"active_fallbacks": ["bm25_without_stemming"]}}
    assert "ACTIVE_FALLBACKS" in _integrity([], meta)["warnings"]
    assert "ACTIVE_FALLBACKS" not in _integrity([], {"environment": {}})["warnings"]


def test_capture_records_the_list():
    assert isinstance(provenance.capture(include_git=False)["active_fallbacks"], list)


def test_missing_sentence_tokenizer_data_is_named(monkeypatch):
    from lasagna.processing.chunker import semantic

    monkeypatch.setattr(semantic, "punkt_available", lambda: False)
    assert "sentence_split_without_punkt" in provenance._active_fallbacks()
    monkeypatch.setattr(semantic, "punkt_available", lambda: True)
    assert "sentence_split_without_punkt" not in provenance._active_fallbacks()
