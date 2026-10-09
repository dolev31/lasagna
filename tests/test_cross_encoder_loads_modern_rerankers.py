"""The cross-encoder reranker loads current rerankers by name.

BAAI/bge-reranker-v2-m3 is a sequence classifier and loads as any
cross-encoder does. mixedbread-ai/mxbai-rerank-base-v2 is a causal LM scored
from its true/false token logits, which sentence-transformers supports from
5.4; an older release would load it with an untrained classification head and
score noise, so that is refused. Checkpoints with their own modelling code
load when ``reranker_trust_remote_code`` allows it. Nothing is downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.core.base_components import ScoredDocument
from lasagna.core.registry import registry
from lasagna.knowledge_evaluation_sdk.stages import stage_of
from lasagna.retrieval.search import ranker


def _docs(*texts):
    return [
        ScoredDocument(
            content=t, score=1.0 - i / 10, document_id=f"d{i}", chunk_id=f"c{i}", metadata={"i": i}
        )
        for i, t in enumerate(texts)
    ]


def _order(out):
    return [d.chunk_id for d in out]


def test_trusting_remote_code_is_a_reranker_setting():
    assert stage_of("reranker_trust_remote_code") == "reranker"


class _RecordingCrossEncoder:
    built = []

    def __init__(self, name, device=None, **kwargs):
        _RecordingCrossEncoder.built.append((name, kwargs))

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        return [float(len(doc)) for _, doc in pairs]


def _session(**extra):
    return SimpleNamespace(
        enable_reranking=True,
        reranker_model="org/reranker",
        reranker_batch_size=8,
        rerank_top_k=5,
        **extra,
    )


class TestCrossEncoderLoadsModernRerankers:
    @pytest.fixture(autouse=True)
    def _fake(self, monkeypatch):
        monkeypatch.setattr(ranker, "CrossEncoder", _RecordingCrossEncoder)
        monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
        _RecordingCrossEncoder.built.clear()

    def test_remote_code_runs_only_when_the_setting_allows_it(self):
        rr = ranker.CrossEncoderReranker()
        hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
        rr.rerank("q", hit, top_k=1, config=_session())
        rr.rerank("q", hit, top_k=1, config=_session(reranker_trust_remote_code=True))
        assert _RecordingCrossEncoder.built == [
            ("org/reranker", {}),
            ("org/reranker", {"trust_remote_code": True}),
        ]

    def test_a_causal_lm_reranker_is_refused_where_it_would_score_noise(self, monkeypatch):
        import importlib.metadata

        import transformers

        monkeypatch.setattr(importlib.metadata, "version", lambda name: "5.1.2")
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **k: SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
        )
        rr = ranker.CrossEncoderReranker()
        hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
        with pytest.raises(RuntimeError, match=r"sentence-transformers>=5\.4"):
            rr.rerank("q", hit, top_k=1, config=_session())
        assert _RecordingCrossEncoder.built == []

    def test_a_current_sentence_transformers_loads_it_without_asking(self, monkeypatch):
        import importlib.metadata

        import transformers

        monkeypatch.setattr(importlib.metadata, "version", lambda name: "6.1.0")

        def unexpected(*a, **k):
            raise AssertionError("the architecture check ran on a version that scores it")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unexpected)
        rr = ranker.CrossEncoderReranker()
        hit = [{"id": "a", "content": "aa", "score": 1.0, "metadata": {}}]
        assert rr.rerank("q", hit, top_k=1, config=_session())


# ---------------------------------------------------------------------------
# Real models, when cached
# ---------------------------------------------------------------------------


def _cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(name, "config.json"), str)


QUESTION = "what is the capital of france"
PASSAGES = ("Bananas are rich in potassium.", "Paris is the capital and largest city of France.")


@pytest.mark.skipif(
    not _cached("cross-encoder/ms-marco-MiniLM-L-6-v2"), reason="MiniLM cross-encoder not cached"
)
def test_the_cross_encoder_component_loads_a_reranker_by_name():
    out = registry.get("reranker", "cross_encoder").rerank(
        QUESTION,
        _docs(*PASSAGES),
        top_k=2,
        config=SimpleNamespace(
            enable_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_batch_size=8,
            rerank_top_k=2,
            reranker_trust_remote_code=False,
        ),
    )
    assert _order(out) == ["c1", "c0"]
