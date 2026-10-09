"""Two arms with corpora of their own never share document ids.

Document ids were derived from the benchmark's id alone, so two per-arm corpus
sessions in one collection (a chunk_size sweep, a chunker or embedder arm)
shared document and chunk ids. The documents table keeps one session per id
and the vector store one row per chunk id, so whichever arm ingested last took
the other's documents: that arm's searches came back empty and it scored 0.0,
or a mix of both corpora, depending on task order.
"""

from __future__ import annotations

import hashlib

from lasagna.knowledge_evaluation_sdk.experiment import _corpus_doc_id


def test_per_arm_sessions_get_their_own_ids():
    a = _corpus_doc_id("tardigrade_1", "smoke|chunk_size=400")
    b = _corpus_doc_id("tardigrade_1", "smoke|chunk_size=800")
    assert a != b and a.startswith("doc_") and len(a) == len("doc_") + 16


def test_the_shared_corpus_keeps_its_ids():
    # Corpora ingested before per-session ids existed must still be found.
    legacy = "doc_" + hashlib.md5(b"tardigrade_1").hexdigest()[:16]
    assert _corpus_doc_id("tardigrade_1") == legacy


def test_ids_are_stable_across_runs():
    assert _corpus_doc_id("d", "k") == _corpus_doc_id("d", "k")
