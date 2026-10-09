"""Pure helper functions for experiment evaluation.

This module has **zero** third-party dependencies — only stdlib types are
used.  This allows test code to import these helpers without triggering
the heavy transitive imports that ``experiment.py`` pulls in (loguru,
pydantic, sentence-transformers, etc.).

Production code in ``experiment.py`` re-exports from here so callers are
unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def build_bm25_docs(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build BM25 chunk dicts from raw Milvus query results.

    Handles missing or ``None`` chunk_id values by falling back to a
    positional name (``chunk_0``, ``chunk_1``, …).  The ``or`` operator
    covers both the *key-absent* and *key-present-but-None* cases —
    ``dict.get("chunk_id", default)`` alone would return ``None`` for
    the latter.
    """
    return [
        {
            "chunk_id": c.get("chunk_id") or f"chunk_{i}",
            "content": c.get("content", ""),
            "document_id": c.get("document_id", ""),
            "metadata": c.get("metadata", {}),
        }
        for i, c in enumerate(chunks)
    ]


def filter_nonempty_agentic_indices(
    gen_agentic_answers: List[str],
) -> List[int]:
    """Return indices of non-empty agentic answers.

    Empty or falsy strings are excluded so that RAGAS is only evaluated
    on samples that actually have an agentic-generated answer.
    """
    return [i for i, a in enumerate(gen_agentic_answers) if a]


def apply_canonical_metric_mapping(
    per_sample_scores: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    base_keys: Sequence[str] = ("faithfulness", "context_recall", "context_precision"),
) -> None:
    """Emit canonical unsuffixed metric keys for agentic configs.

    For each sample, the canonical per-sample score is the ``_agentic``
    variant when available, falling back to ``_regen``.  The canonical
    **aggregate** is the mean of these per-sample canonical values over
    *all* N samples — **not** the subset average from
    ``faithfulness_agentic`` (which may cover fewer samples when some
    agentic answers are empty).

    *base_keys* are the judged metrics the run requested; a metric missing
    from it got ``_regen``/``_agentic`` scores and no canonical one, so a
    significance test on it found nothing to compare.

    Mutates *per_sample_scores* and *metrics* in place.
    """
    for ps in per_sample_scores:
        for base_key in base_keys:
            if f"{base_key}_agentic" in ps:
                ps[base_key] = ps[f"{base_key}_agentic"]
            elif f"{base_key}_regen" in ps:
                ps[base_key] = ps[f"{base_key}_regen"]

    for base_key in base_keys:
        vals = [ps[base_key] for ps in per_sample_scores if base_key in ps]
        if vals:
            metrics[base_key] = sum(vals) / len(vals)
