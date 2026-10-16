"""The ``mmr``, ``combined`` and ``ensemble`` rerankers rank, and a pass-through is flagged.

All three returned their input order or the cross-encoder's in a sweep on the
speed profile. MMR never received the pipeline's embedder (the orchestrator
binds it through a wrapper it did not look inside) and was skipped when
``enable_mmr`` was off; ``combined`` skipped its MMR step for the same reason
and scored exactly like ``cross_encoder``; ``ensemble`` loaded the settings
file's reranker model instead of the arm's, failed offline, and passed its
input through. In the README's two-arm sweep nothing said so, since each arm
was compared only with the other reranker.
"""

from __future__ import annotations

import numpy as np
import pytest

from lasagna.core.base_components import ScoredDocument
from lasagna.core.config import rag_config
from lasagna.core.strict import observe_reranking, strict
from lasagna.knowledge_evaluation_sdk.stages import isolation_report
from lasagna.retrieval.search import ranker
from lasagna.retrieval.search.ranker import (
    CombinedRerankerAdapter,
    CrossEncoderRerankerAdapter,
    EnsembleRerankerAdapter,
    MMRRerankerAdapter,
    adapt_reranker_for_pipeline,
    search_embedder,
)

# Two near-duplicates first, then a different passage: what MMR exists to fix.
_VECTORS = {
    "alpha one": [1.0, 0.0],
    "alpha one again": [0.99, 0.14],
    "beta two": [0.0, 1.0],
}


class _Embedder:
    def encode_documents_batch(self, texts, show_progress=False):
        return [np.array(_VECTORS[t]) for t in texts]

    def similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class _Model:
    """A cross-encoder that scores from a table."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        return [self.scores[text] for _q, text in pairs]


def _docs(*pairs):
    return [
        ScoredDocument(content=text, score=score, document_id=f"d{i}", chunk_id=text)
        for i, (text, score) in enumerate(pairs)
    ]


def _speed_like(**extra):
    """An arm's config with reranking on and MMR off, as under the speed profile."""
    return rag_config.with_query_overrides(
        {"enable_reranking": True, "enable_mmr": False, "mmr_lambda": 0.5, **extra}
    )


@pytest.fixture
def models(monkeypatch):
    """Cross-encoders by name; any other name fails to load, as offline."""
    table = {}
    loaded = []

    def _model_for(self, name, trust_remote_code=False):
        loaded.append(name)
        if name not in table:
            raise RuntimeError(f"Cannot load reranker model {name!r}: not cached")
        return table[name]

    monkeypatch.setattr(ranker.CrossEncoderReranker, "_model_for", _model_for)
    table["loaded"] = loaded
    return table


def _order(docs):
    return [d.chunk_id for d in docs]


def test_mmr_moves_a_near_duplicate_down_with_mmr_off_in_the_profile():
    docs = _docs(("alpha one", 0.9), ("alpha one again", 0.85), ("beta two", 0.5))
    mmr = adapt_reranker_for_pipeline(MMRRerankerAdapter())
    # The orchestrator hands the search's embedder to the wrapped component.
    with search_embedder(_Embedder()):
        out = mmr.rerank(
            "q", [ranker.scored_document_to_dict(d) for d in docs], 3, config=_speed_like()
        )

    assert [r.chunk_id for r in out] == ["alpha one", "beta two", "alpha one again"]


def test_mmr_without_an_embedder_says_so_instead_of_returning_its_input():
    docs = _docs(("alpha one", 0.9), ("beta two", 0.5))
    with pytest.raises(ValueError, match="no embedding model"):
        MMRRerankerAdapter().rerank("q", docs, top_k=2, config=_speed_like())


def test_combined_differs_from_cross_encoder_with_mmr_off_in_the_profile(models):
    models["cross-encoder/ms-marco-MiniLM-L-6-v2"] = _Model(
        {"alpha one": 3.0, "alpha one again": 2.9, "beta two": 1.0}
    )
    config = _speed_like(reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    docs = _docs(("beta two", 0.9), ("alpha one again", 0.8), ("alpha one", 0.7))

    cross_encoder = CrossEncoderRerankerAdapter().rerank("q", docs, top_k=3, config=config)
    combined = adapt_reranker_for_pipeline(CombinedRerankerAdapter())
    with search_embedder(_Embedder()):
        out = combined.rerank(
            "q", [ranker.scored_document_to_dict(d) for d in docs], 3, config=config
        )

    assert _order(cross_encoder) == ["alpha one", "alpha one again", "beta two"]
    assert [r.chunk_id for r in out] == ["alpha one", "beta two", "alpha one again"]


def test_ensemble_ranks_with_the_arms_members(models):
    models["m/a"] = _Model({"p1": 3.0, "p2": 2.0, "p3": 0.0})
    models["m/b"] = _Model({"p1": 0.0, "p2": 2.0, "p3": 3.0})
    docs = _docs(("p3", 0.9), ("p1", 0.8), ("p2", 0.7))
    config = _speed_like(ensemble_reranker_models=["m/a", "m/b"])

    with strict():
        out = EnsembleRerankerAdapter().rerank("q", docs, top_k=3, config=config)

    assert _order(out) == ["p2", "p1", "p3"]  # neither member's order, nor the input's


def test_ensemble_reads_the_arms_model_not_the_settings_file(models):
    models["m/arm"] = _Model({"p1": 0.0, "p2": 1.0})
    config = _speed_like(reranker_model="m/arm")
    assert rag_config.reranker_model != "m/arm"

    out = EnsembleRerankerAdapter().rerank("q", _docs(("p1", 0.9), ("p2", 0.1)), 2, config=config)

    assert set(models["loaded"]) == {"m/arm"}
    assert _order(out) == ["p2", "p1"]


def test_under_evaluation_an_ensemble_of_one_or_with_a_missing_member_fails(models):
    models["m/a"] = _Model({"p1": 1.0})
    docs = _docs(("p1", 0.9))
    with strict(), pytest.raises(ValueError, match="Name two or more"):
        EnsembleRerankerAdapter().rerank("q", docs, 1, config=_speed_like(reranker_model="m/a"))
    with strict(), pytest.raises(RuntimeError, match="'m/missing'"):
        EnsembleRerankerAdapter().rerank(
            "q", docs, 1, config=_speed_like(ensemble_reranker_models=["m/a", "m/missing"])
        )


def test_a_reranker_call_records_whether_it_moved_anything():
    class _Keep(ranker.BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            return list(documents)[:top_k]

    class _Flip(ranker.BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            return list(reversed(documents))[:top_k]

    dicts = [ranker.scored_document_to_dict(d) for d in _docs(("a", 0.9), ("b", 0.5))]
    with observe_reranking() as moved:
        adapt_reranker_for_pipeline(_Keep()).rerank("q", dicts, 2)
        adapt_reranker_for_pipeline(_Flip()).rerank("q", dicts, 2)
    assert moved == [False, True]


def _row(name, reranker, moved):
    return {
        "benchmark_name": "smoke",
        "config_name": name,
        "profile": "speed",
        "config_overrides": {"reranker": reranker},
        "system": {"stages": {"reranker": {"digest": name, "component": reranker}}},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "_retrieved_ids": [name, str(i)], "_reranker_moved": m}
            for i, m in enumerate(moved)
        ],
    }


def test_a_pass_through_arm_is_flagged_without_a_control_arm():
    rows = [
        _row("reranker=cross_encoder", "cross_encoder", [True, False, True]),
        _row("reranker=mine", "mine", [False, False, False]),
    ]
    inert = isolation_report(rows)["inert"]
    assert len(inert) == 1
    assert "'reranker=mine'" in inert[0] and "on all 3 queries" in inert[0]


def test_a_sweep_arm_names_the_ensembles_members_and_the_run_records_them():
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    # The YAML refused these keys as unknown, and the bundle recorded "ensemble".
    arm = {"reranker": "ensemble", "ensemble_reranker_models": ["m/a", "m/b"]}
    exp._check_override_keys([{"name": "ens", "overrides": arm}])
    assert exp._reranker_that_ran("ensemble", arm) == "ensemble:m/a+m/b"
