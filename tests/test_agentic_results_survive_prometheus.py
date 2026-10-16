"""An agentic search with Prometheus on returns the agent's results.

``rag.search.total_ms`` was recorded without labels by the standard path and
with ``tier``/``pipeline`` labels by the agentic one. With a real
``prometheus_client`` (on in compose and k8s) the second registration raised
``Duplicated timeseries``, the routing ``except`` caught it, and the standard
pipeline ran again: the agent's results and its LLM spend were thrown away,
with nothing in ``degraded_stages``. Metrics now keep one label set per name,
and a metrics failure is logged, never raised into a search.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from lasagna.api import _search_orchestrator as orch
from lasagna.core.config import RAGConfig
from lasagna.utils import degraded
from lasagna.utils import monitoring

SRC = Path(__file__).resolve().parents[1] / "src" / "lasagna"


@pytest.fixture
def prometheus(monkeypatch):
    """The real Prometheus backend, its metrics removed from the registry after."""
    prometheus_client = pytest.importorskip("prometheus_client")
    backend = monitoring._PrometheusBackend()
    monkeypatch.setattr(monitoring.METRICS, "_backend", backend)
    yield prometheus_client.REGISTRY
    for metric in [*backend._counters.values(), *backend._histograms.values()]:
        prometheus_client.REGISTRY.unregister(metric)


def _metric_calls():
    """Every ``METRICS.counter/histogram(...)`` call in the package, by AST."""
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("counter", "histogram")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "METRICS"
            ):
                continue
            where = f"{path.relative_to(SRC)}:{node.lineno}"
            name = node.args[0]
            assert isinstance(name, ast.Constant), f"{where}: metric name must be a literal"
            tags = node.args[2] if len(node.args) > 2 else None
            for kw in node.keywords:
                if kw.arg == "tags":
                    tags = kw.value
            if tags is None or (isinstance(tags, ast.Constant) and tags.value is None):
                keys = ()
            else:
                assert isinstance(tags, ast.Dict) and all(
                    isinstance(k, ast.Constant) for k in tags.keys
                ), f"{where}: tags must be a dict literal so its label names can be checked"
                keys = tuple(sorted(k.value for k in tags.keys))
            yield node.func.attr, name.value, keys, where


def test_every_metric_name_has_one_label_set():
    seen = {}
    calls = list(_metric_calls())
    assert len(calls) > 20, "the scan found too few metric calls to be looking at the package"
    for kind, name, keys, where in calls:
        sanitized = name.replace(".", "_").replace("-", "_")
        first = seen.setdefault(sanitized, (kind, keys, where))
        assert (kind, keys) == first[:2], (
            f"{name} is a {kind} with labels {keys} at {where}, "
            f"but a {first[0]} with labels {first[1]} at {first[2]}"
        )


def test_every_metric_registers_with_the_real_prometheus_client(prometheus):
    backend = monitoring.METRICS._backend
    for kind, name, keys, _where in _metric_calls():
        tags = {k: "x" for k in keys} or None
        getattr(backend, kind)(name, 1, tags)  # the backend raises; the facade would not


def _orchestrator(monkeypatch):
    o = orch.SearchOrchestrator.__new__(orch.SearchOrchestrator)
    o.reranker = None
    o.query_enhancer = None
    o._get_session_config = lambda sid: RAGConfig()

    def standard_pipeline_ran(enhancer):
        raise AssertionError("the standard pipeline ran: the agent's results were dropped")

    # The first step of the standard pipeline after routing.
    monkeypatch.setattr(orch, "_is_public_enhancer", standard_pipeline_ran)
    monkeypatch.setattr(
        orch,
        "get_quota_manager",
        lambda: SimpleNamespace(check_search=lambda s: None, record_search=lambda s: None),
    )
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda: None)
    )

    class _Agent:
        def __init__(self, **kw):
            pass

        async def retrieve(self, **kw):
            chunk = {"chunk_id": "c1", "content": "agentic", "document_id": "d1", "score": 0.9}
            return SimpleNamespace(chunks=[chunk], agent_summary="the agent's answer")

    import lasagna.agentic.agentic_rag as agentic_rag

    monkeypatch.setattr(agentic_rag, "AgenticRAG", _Agent)
    return o


def test_an_agentic_search_with_prometheus_on_returns_the_agents_results(monkeypatch, prometheus):
    o = _orchestrator(monkeypatch)
    # What a standard search records first (its labels are pinned above).
    monitoring.METRICS.histogram("rag.search.total_ms", 3.0, tags={"tier": "standard"})

    with degraded.collecting() as stages:
        results = o.search("why?", "s1", routing_tier="comprehensive")

    meta = results[0].chunk.metadata
    assert meta["_routing_tier_used"] == "comprehensive"
    assert meta["_agentic_answer"] == "the agent's answer"
    assert stages == []
    assert prometheus.get_sample_value("rag_search_total_ms_count", {"tier": "comprehensive"}) == 1


def test_a_failing_metrics_backend_never_changes_a_search(monkeypatch):
    class _Broken(monitoring._MetricsBackend):
        def counter(self, name, value=1, tags=None):
            raise RuntimeError("registry full")

        histogram = counter

    monkeypatch.setattr(monitoring.METRICS, "_backend", _Broken())
    warnings = []
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        o = _orchestrator(monkeypatch)
        results = o.search("why?", "s1", routing_tier="comprehensive")
        o.search("why?", "s2", routing_tier="comprehensive")
    finally:
        logger.remove(sink)

    assert results[0].chunk.metadata["_routing_tier_used"] == "comprehensive"
    about_total = [w for w in warnings if "rag.search.total_ms" in w]
    assert len(about_total) == 1, warnings  # logged once, not once per search
