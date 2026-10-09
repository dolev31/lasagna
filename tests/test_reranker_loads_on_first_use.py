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


def _session(model="x", on=True):
    return SimpleNamespace(
        enable_reranking=on,
        reranker_model=model,
        reranker_batch_size=8,
        rerank_top_k=5,
        enable_mmr=False,
    )


def test_reranking_with_reranking_off_never_loads(monkeypatch):
    calls = []
    monkeypatch.setattr(ranker.CrossEncoderReranker, "_load_model", lambda self: calls.append(1))
    rr = ranker.CrossEncoderReranker(model_name="x")
    off = _session(on=False)
    rr.rerank("q", [{"id": "a", "content": "a", "score": 1.0, "metadata": {}}], top_k=1, config=off)
    assert calls == []


def test_first_rerank_with_reranking_on_loads_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ranker.CrossEncoderReranker, "_load_model", lambda self: calls.append(1))
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    rr = ranker.CrossEncoderReranker(model_name="x")
    on = _session()
    hit = [{"id": "a", "content": "a", "score": 1.0, "metadata": {}}]
    rr.rerank("q", hit, top_k=1, config=on)
    assert calls == [1]  # attempted on first use (the stub leaves model None)


class _FakeCrossEncoder:
    def __init__(self, name, device=None):
        self.name = name

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        return [float(len(doc)) for _, doc in pairs]


def test_the_session_reranker_model_is_the_one_that_runs(monkeypatch):
    # A sweep arm naming reranker_model got whatever model the backend had
    # loaded at start; the arm's model must be the one scoring.
    monkeypatch.setattr(ranker, "CrossEncoder", _FakeCrossEncoder)
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    rr = ranker.CrossEncoderReranker()
    hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
    rr.rerank("q", hit, top_k=1, config=_session("small-model"))
    assert rr.model.name == "small-model"
    rr.rerank("q", hit, top_k=1, config=_session("other-model"))
    assert rr.model.name == "other-model"


def test_an_ensemble_keeps_its_members_and_their_models(monkeypatch):
    # Lazy loading must not empty the ensemble (it kept members whose model
    # had loaded), and a member keeps the model it was built for.
    monkeypatch.setattr(ranker, "CrossEncoder", _FakeCrossEncoder)
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    ens = ranker.EnsembleReranker(
        model_configs=[
            ranker.RerankerModelConfig(model_name="m1", weight=0.5),
            ranker.RerankerModelConfig(model_name="m2", weight=0.5),
        ]
    )
    assert [r.model.name for r, _ in ens.rerankers] == ["m1", "m2"]
    hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
    ens.rerank("q", hit, top_k=1, config=_session("session-default"))
    assert [r.model.name for r, _ in ens.rerankers] == ["m1", "m2"]


def test_the_adapted_cross_encoder_runs_the_arms_model(monkeypatch):
    # {reranker: cross_encoder_adapted, reranker_model: MiniLM} memory-mapped
    # the 870 MB global default: the adapter never saw the session config.
    monkeypatch.setattr(ranker, "CrossEncoder", _FakeCrossEncoder)
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    adapted = ranker.CrossEncoderRerankerAdapter()
    pipeline_side = ranker.adapt_reranker_for_pipeline(adapted)
    hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
    pipeline_side.rerank(query="q", results=hit, top_k=1, config=_session("small-model"))
    assert adapted._inner.model.name == "small-model"


def test_a_user_reranker_is_not_handed_the_session_config():
    seen = {}

    class Mine(ranker.BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            seen.update(kwargs)
            return documents[:top_k]

    hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
    ranker.adapt_reranker_for_pipeline(Mine()).rerank(
        query="q", results=hit, top_k=1, config=_session()
    )
    assert seen == {}
