"""The cross-encoder loads when it first reranks, not when the backend starts.

The reranker is built at backend start; loading it there downloaded the
default 870 MB model even for the speed profile, which turns reranking off,
so the documented 80 MB first run was nearly a gigabyte.
"""

from __future__ import annotations

from types import SimpleNamespace

from lasagna.retrieval.search import ranker


def test_building_the_reranker_loads_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(ranker.CrossEncoderReranker, "_load_model", lambda self: calls.append(1))
    ranker.CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert calls == []


def test_reranking_with_reranking_off_never_loads(monkeypatch):
    calls = []
    monkeypatch.setattr(ranker.CrossEncoderReranker, "_load_model", lambda self: calls.append(1))
    rr = ranker.CrossEncoderReranker(model_name="x")
    off = SimpleNamespace(enable_reranking=False, rerank_top_k=5)
    rr.rerank("q", [{"id": "a", "content": "a", "score": 1.0, "metadata": {}}], top_k=1, config=off)
    assert calls == []


def test_first_rerank_with_reranking_on_loads_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ranker.CrossEncoderReranker, "_load_model", lambda self: calls.append(1))
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    rr = ranker.CrossEncoderReranker(model_name="x")
    on = SimpleNamespace(enable_reranking=True, rerank_top_k=5)
    hit = [{"id": "a", "content": "a", "score": 1.0, "metadata": {}}]
    rr.rerank("q", hit, top_k=1, config=on)
    assert calls == [1]  # attempted on first use (the stub leaves model None)
