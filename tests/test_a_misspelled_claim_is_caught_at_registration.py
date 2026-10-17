"""A misspelled claim is caught when its component registers.

`claims={"beir/scifcat": {"ndgc@10": 0.5}}` registered, and its Claimed column
never appeared: no run records `ndgc@10`, and no benchmark is named
`beir/scifcat`. A metric name no run records is now an error naming the
closest one. A benchmark no registry has is a warning naming the closest one,
since a plugin may register it.
"""

from __future__ import annotations

import warnings

import pytest

from lasagna import components
from lasagna.core.registry import registry

NAME = "_misspelled_claim_probe"


@pytest.fixture(autouse=True)
def _unregistered():
    yield
    registry._unregister("reranker", NAME)


def _registered() -> bool:
    return NAME in {c["name"] for c in registry.list_components("reranker")}


def _register(claims):
    @components.reranker(NAME, paper="Doe 2026", claims=claims)
    def keep(query, documents):
        return documents


def test_a_metric_no_run_records_is_refused_with_the_closest_one():
    with pytest.raises(ValueError) as refused:
        _register({"beir/scifact": {"ndgc@10": 0.5}})
    message = str(refused.value)
    assert message.startswith(
        "claims for 'beir/scifact': no metric named 'ndgc@10' (did you mean 'ndcg@10'?)"
    )
    assert "ndcg@k" in message and "faithfulness" in message
    assert not _registered()


def test_a_benchmark_no_registry_has_is_a_warning_with_the_closest_one():
    with pytest.warns(UserWarning) as warned:
        _register({"beir/scifcat": {"ndcg@10": 0.5}})
    (message,) = [str(w.message) for w in warned if "scifcat" in str(w.message)]
    assert "'beir/scifcat', which no registry has (did you mean 'beir/scifact'?)" in message
    assert "lasagna benchmark list" in message
    assert _registered()


def test_a_claim_on_a_real_benchmark_and_metric_registers_quietly():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _register({"beir/scifact": {"NDCG@10": 0.5, "recall@100": 0.9, "split": "test"}})
    assert _registered()


def test_every_metric_a_run_records_can_be_claimed():
    from lasagna.core.claims import normalize_claims
    from lasagna.knowledge_evaluation_sdk import experiment
    from lasagna.knowledge_evaluation_sdk.ir_metrics import corpus_metrics
    from lasagna.knowledge_evaluation_sdk.ragas_evaluator import METRIC_MAP

    recorded = set(corpus_metrics([{"d": 1}], [["d"]], k=10))
    recorded |= set(corpus_metrics([{"d": 1}], [["d"]], k=100))
    recorded |= set(METRIC_MAP) | set(experiment._ANSWER_METRICS) | {"exact_match", "token_f1"}
    for metric in sorted(recorded):
        normalize_claims({"beir/scifact": {metric: 0.5}})


def test_no_check_waits_on_a_registry_that_is_loading_its_plugins(monkeypatch):
    """A plugin that registers a component while the benchmark registry loads it."""
    from lasagna.core.claims import unregistered_benchmarks
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry as reg

    monkeypatch.setattr(reg, "_builtins_registered", False)
    with reg._lock:
        assert unregistered_benchmarks({"beir/scifcat": {"ndcg@10": 0.5}}) == []
