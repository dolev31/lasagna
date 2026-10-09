"""`results validate --strict` can pass on the documented install.

It never could: docling, requested by the default settings and irrelevant to
a benchmark corpus, raised ACTIVE_FALLBACKS, and the smoke benchmark that
ships inside the package was UNPINNED_DATASET.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.bundle import _integrity, exercised_fallbacks


def _row(hybrid=None, chunker=None, **extra):
    stages = {}
    if hybrid is not None:
        stages["retriever"] = {"settings": {"enable_hybrid_search": hybrid}}
    if chunker is not None:
        stages["chunker"] = {"settings": chunker}
    return {
        "error": None,
        "resolved_mode": "live",
        "hf_revision": "x",
        "system": {"stages": stages},
        **extra,
    }


def test_docling_never_counts_for_a_benchmark():
    integ = _integrity(
        [_row(hybrid=True)],
        {"environment": {"active_fallbacks": ["docling_requested_but_missing"]}},
    )
    assert "ACTIVE_FALLBACKS" not in integ["warnings"] and integ["fallbacks"] == []


def test_bm25_stand_ins_count_only_when_a_task_used_hybrid_search():
    fb = ["bm25_without_stemming"]
    assert exercised_fallbacks([_row(hybrid=False)], fb) == []
    assert exercised_fallbacks([_row(hybrid=True)], fb) == fb
    assert exercised_fallbacks([_row()], fb) == fb, "no manifest: assume it might have"


def test_the_sentence_splitter_counts_only_when_chunking_ran_it():
    fb = ["sentence_split_without_nltk"]
    fixed = {"chunking_strategy": "fixed", "preserve_headers": False, "preserve_tables": False}
    assert exercised_fallbacks([_row(chunker=fixed)], fb) == []
    assert exercised_fallbacks([_row(chunker={"enable_hierarchical": True})], fb) == []
    assert exercised_fallbacks([_row(chunker={"preserve_headers": True})], fb) == fb
    assert exercised_fallbacks([_row(chunker={**fixed, "chunking_strategy": "semantic"})], fb) == fb


def test_an_exercised_fallback_says_how_to_remove_it():
    integ = _integrity(
        [_row(hybrid=True)], {"environment": {"active_fallbacks": ["bm25_without_stemming"]}}
    )
    assert "ACTIVE_FALLBACKS" in integ["warnings"]
    assert "pip install" in integ["fallbacks"][0]


def test_the_packaged_smoke_benchmark_is_pinned_by_its_content():
    pin = benchmark_registry.get("smoke").hf_revision
    assert pin.startswith("sha256:") and len(pin) > 10


def test_missing_punkt_data_is_named_with_its_fix_and_only_when_used():
    fb = ["sentence_split_without_punkt"]
    fixed = {"chunking_strategy": "fixed", "preserve_headers": False, "preserve_tables": False}
    assert exercised_fallbacks([_row(chunker=fixed)], fb) == []
    integ = _integrity(
        [_row(chunker={"preserve_headers": True})], {"environment": {"active_fallbacks": fb}}
    )
    assert "punkt_tab" in integ["fallbacks"][0]
