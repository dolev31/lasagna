"""A stage that falls back during a benchmark is named in the run's bundle (#50).

Hybrid search, the graph expansion and other stages catch their own failure,
carry on with less, and call ``degraded.record``. REST, MCP and the SDK report
those stages to their caller; the benchmark harness never collected them. In the
Compose stack every hybrid search tried the native call, failed and fell back,
253 times in one sweep under ``standard``, and the bundle said
``active_fallbacks: []`` and validated, ``--strict`` included. Under ``speed``
the same path ran a hybrid arm dense-only (#52), with nothing in the record.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import bundle as bundle_mod
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
from lasagna.utils import degraded

_REASON = "Backend hybrid search failed: fieldName(sparse_embedding) not found"
_DENSE = "BM25 index empty; dense results only"


class _Pipeline:
    """Stands in for KnowledgePipeline: returns one document, records fallbacks.

    A question in ``dense_only`` takes the path #52 took: the native call
    fails, then the BM25 index is empty (one stage, two reasons, and the query
    counts once). One in ``fusion`` only fails the native call, as under
    ``standard`` in #50. ``when`` decides from the query's config whether to
    fall back at all.
    """

    def __init__(self, dense_only=(), fusion=(), when=None):
        self._config = {"enable_hybrid_search": False}
        self._component_names = {}
        self._instances = {}
        self._dense_only = set(dense_only)
        self._fusion = set(fusion)
        self._when = when

    def query(self, question, top_k, session_id, config):
        if self._when is not None:
            time.sleep(0.01)  # let two tasks' worker threads interleave
            if not self._when(config):
                return self._results()
        if question in self._dense_only | self._fusion:
            degraded.record("hybrid", _REASON)
        if question in self._dense_only:
            degraded.record("hybrid", _DENSE)
        return self._results()

    @staticmethod
    def _results():
        return [SimpleNamespace(chunk=SimpleNamespace(document_id="v1", content="x"), score=1.0)]


def _samples(n):
    return [
        BenchmarkSample(
            query_id=f"q{i}",
            question=f"question {i}?",
            expected_docs={"d1": 1},
            ground_truth="tardigrade",
        )
        for i in range(n)
    ]


@pytest.fixture
def live(monkeypatch):
    """`_eval_live` over a fake corpus, profile and backend; returns a runner."""
    from lasagna.api import rag_backend
    from lasagna.core import pipeline as pipeline_mod

    async def _ingested(*a, **k):
        return {
            "session_id": "bench_corpus_test",
            "corpus_to_valid": {"d1": "v1"},
            "valid_to_corpus": {"v1": "d1"},
            "corpus_fingerprint": "fp",
        }

    async def _released(*a, **k):
        return None

    backend = SimpleNamespace(
        embedding_manager=SimpleNamespace(model_name="m", dimension=4),
    )
    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _ingested)
    monkeypatch.setattr(exp, "_release_corpus", _released)
    # The arm's overrides reach the config each query is made with, as they
    # do through the real pipeline.
    monkeypatch.setattr(
        exp,
        "_apply_overrides_to_pipeline",
        lambda pipeline, overrides, session_id: pipeline._config.update(overrides or {}),
    )
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: backend)

    def task(config_name, track="retrieval", overrides=None):
        return BenchmarkTask(
            benchmark_name="smoke",
            config_name=config_name,
            run_index=0,
            mode="live",
            config_overrides=overrides or {},
            tracks=(track,),
        )

    def run(config_name, n=4, track="retrieval", **fake):
        pipeline = _Pipeline(**fake)
        monkeypatch.setattr(
            pipeline_mod.KnowledgePipeline, "from_profile", classmethod(lambda cls, p: pipeline)
        )
        return asyncio.run(exp._eval_live(task(config_name, track), _samples(n)))

    def run_together(tasks, n=6, **fake):
        monkeypatch.setattr(
            pipeline_mod.KnowledgePipeline,
            "from_profile",
            classmethod(lambda cls, p: _Pipeline(**fake)),
        )

        async def _all():
            return await asyncio.gather(*(exp._eval_live(t, _samples(n)) for t in tasks))

        return asyncio.run(_all())

    run.task = task
    run.together = run_together
    return run


class _Answers:
    """An answer model that always answers; the judge it also serves scores nothing."""

    async def ainvoke(self, prompt):
        return SimpleNamespace(content="tardigrade")

    def invoke(self, prompt):
        return SimpleNamespace(content="tardigrade")


def test_a_stage_that_fell_back_is_recorded_on_the_tasks_row(live):
    """With the queries each reason covered: which ran dense-only is the point."""
    result = live("hybrid", dense_only={"question 0?", "question 2?"}, fusion={"question 3?"})
    assert result.error is None
    assert result.system["degraded_stages"] == {
        "retrieval": {
            "hybrid": {
                "queries": 3,
                "of": 4,
                "reasons": [
                    {"reason": _REASON, "queries": 3},
                    {"reason": _DENSE, "queries": 2},
                ],
            }
        }
    }


def test_two_tasks_running_at_once_keep_their_own_fallbacks(live):
    """Tasks of a sweep run concurrently; each counts only its own queries."""
    hybrid, control = live.together(
        [
            live.task("hybrid", overrides={"hybrid_alpha": 0.25}),
            live.task("control"),
        ],
        dense_only={f"question {i}?" for i in range(6)},
        when=lambda config: config.get("hybrid_alpha") == 0.25,
    )
    assert hybrid.system["degraded_stages"]["retrieval"]["hybrid"]["queries"] == 6
    assert "degraded_stages" not in control.system


def test_a_stage_recorded_in_a_nested_block_still_reaches_the_task():
    """A search path that opens its own block must not hide fallbacks from the harness."""
    tally = exp._DegradedTally()
    with tally.query():
        with degraded.collecting():
            degraded.record("graphrag", "graph store unavailable")
    assert tally.to_dict()["graphrag"]["queries"] == 1


def test_reasons_are_capped_cut_and_carry_no_home_directory(monkeypatch):
    import os

    tally = exp._DegradedTally()
    home = os.path.expanduser("~")
    for i in range(5):
        with tally.query():
            degraded.record("hybrid", f"failure {i} in {home}/project/x.py " + "e" * 400)
    reasons = tally.to_dict()["hybrid"]["reasons"]
    assert len(reasons) == exp._DegradedTally._REASONS_KEPT
    assert all(len(r["reason"]) <= exp._DegradedTally._REASON_CHARS for r in reasons)
    assert not any(home in r["reason"] for r in reasons)
    assert tally.to_dict()["hybrid"]["queries"] == 5
    assert tally.to_dict()["hybrid"]["other_reasons"] == 2
    (line,) = bundle_mod.degraded_findings([_row("hybrid", {"retrieval": tally.to_dict()})])
    assert line.endswith("; 2 queries: other reasons, not kept)")


def test_a_record_the_reader_cannot_parse_is_skipped_not_fatal():
    """`results bundle` and `results view` read rows from any results.json."""
    odd = {"retrieval": {"hybrid": {"queries": 1, "of": 2, "reasons": ["plain text"]}, "x": 3}}
    (line,) = bundle_mod.degraded_findings([_row("hybrid", odd)])
    assert line.endswith("1 of 2 queries")


def test_the_generation_tracks_searches_are_counted_on_their_own(live, monkeypatch):
    import lasagna.providers as providers

    model = _Answers()
    monkeypatch.setattr(
        providers,
        "get_llm_provider",
        lambda: SimpleNamespace(
            get_llm=lambda *a, **k: model, get_chat_model=lambda *a, **k: model
        ),
    )
    result = live("hybrid", n=3, track="generation", dense_only={"question 1?"})
    assert result.system["degraded_stages"] == {
        "generation": {
            "hybrid": {
                "queries": 1,
                "of": 3,
                "reasons": [
                    {"reason": _REASON, "queries": 1},
                    {"reason": _DENSE, "queries": 1},
                ],
            }
        }
    }


def test_a_task_where_nothing_fell_back_records_nothing(live):
    """No new key on a clean row: its bundle, run id included, is what it was."""
    result = live("control")
    assert result.error is None
    assert "degraded_stages" not in result.system


def test_the_row_survives_results_json():
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    row = TaskResult(
        task_key="k",
        benchmark_name="smoke",
        config_name="hybrid",
        run_index=0,
        system={"degraded_stages": {"retrieval": {"hybrid": {"queries": 2, "of": 4}}}},
    )
    again = TaskResult.from_dict(json.loads(json.dumps(row.to_dict())))
    assert again.system["degraded_stages"] == row.system["degraded_stages"]


def _row(config, degraded_stages=None, run_index=0, benchmark="smoke"):
    system = {"degraded_stages": degraded_stages} if degraded_stages else {}
    return {
        "benchmark_name": benchmark,
        "config_name": config,
        "run_index": run_index,
        "metrics": {},
        "system": system,
    }


_HYBRID = {
    "retrieval": {
        "hybrid": {
            "queries": 12,
            "of": 12,
            "reasons": [{"reason": _REASON, "queries": 12}, {"reason": _DENSE, "queries": 12}],
        }
    }
}


def test_the_bundle_warns_and_names_the_arm_track_and_share_of_queries():
    integrity = bundle_mod._integrity([_row("control"), _row("hybrid", _HYBRID)], {})
    assert "ACTIVE_FALLBACKS" in integrity["warnings"]
    (line,) = integrity["fallbacks"]
    assert line == (
        "hybrid fell back during the run: smoke, arm 'hybrid', retrieval track, "
        f"12 of 12 queries (12 queries: {_REASON}; 12 queries: {_DENSE})"
    )


def test_runs_of_one_arm_add_up_into_one_line():
    def entry(q, dense):
        return {
            "retrieval": {
                "hybrid": {
                    "queries": q,
                    "of": 12,
                    "reasons": [{"reason": _REASON, "queries": q}]
                    + ([{"reason": _DENSE, "queries": dense}] if dense else []),
                }
            }
        }

    rows = [_row("hybrid", entry(12, 1), 0), _row("hybrid", entry(3, 0), 1)]
    rows.append(_row("hybrid", None, run_index=2))
    assert bundle_mod.degraded_findings(rows) == [
        "hybrid fell back during the run: smoke, arm 'hybrid', retrieval track, "
        f"15 of 24 queries in 2 of its 3 runs (15 queries: {_REASON}; 1 query: {_DENSE})"
    ]


def test_a_bundle_where_nothing_fell_back_has_no_warning():
    integrity = bundle_mod._integrity([_row("control"), _row("hybrid")], {})
    assert "ACTIVE_FALLBACKS" not in integrity["warnings"]
    assert integrity["fallbacks"] == []


def _bundle_file(tmp_path, rows):
    from lasagna.knowledge_evaluation_sdk import results as results_mod

    raw = {
        "experiment_config": {
            "metrics_version": bundle_mod.METRICS_VERSION,
            "environment": {"lasagna_version": "0", "python": "3", "platform": "x"},
            "n_runs": 1,
        },
        "results": [
            {"task_key": f"t{i}", "duration_seconds": 0.0, **row} for i, row in enumerate(rows)
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(raw))
    loaded = results_mod.BenchmarkResults.load(str(path))
    out = tmp_path / "run.json"
    out.write_text(json.dumps(bundle_mod.build_bundle(loaded)))
    return str(out)


def test_validate_prints_the_line_and_strict_refuses_it(tmp_path, monkeypatch):
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(emoji=False, width=300))
    path = _bundle_file(tmp_path, [_row("control"), _row("hybrid", _HYBRID)])

    plain = CliRunner().invoke(cli.app, ["results", "validate", path])
    assert plain.exit_code == 0, plain.output
    assert "ACTIVE_FALLBACKS" in plain.output
    assert "smoke, arm 'hybrid', retrieval track, 12 of 12 queries" in plain.output

    strict = CliRunner().invoke(cli.app, ["results", "validate", path, "--strict"])
    assert strict.exit_code == 1, strict.output
    assert "integrity (strict): ACTIVE_FALLBACKS" in strict.output
    assert "smoke, arm 'hybrid', retrieval track, 12 of 12 queries" in strict.output


def test_the_run_summary_names_it(monkeypatch):
    from rich.console import Console

    printed = Console(emoji=False, width=300, record=True)
    monkeypatch.setattr(cli, "console", printed)
    cli._print_fallbacks(SimpleNamespace(_results=[_row("control"), _row("hybrid", _HYBRID)]))
    out = printed.export_text()
    assert "Some stages fell back while the queries ran" in out
    assert "  - hybrid fell back during the run: smoke, arm 'hybrid'" in out
    printed = Console(emoji=False, width=300, record=True)
    monkeypatch.setattr(cli, "console", printed)
    cli._print_fallbacks(SimpleNamespace(_results=[_row("control")]))
    assert printed.export_text() == ""


def test_the_viewer_names_it():
    from lasagna.knowledge_evaluation_sdk import viewer

    html = viewer._fallbacks([_row("control"), _row("hybrid", _HYBRID)])
    assert "smoke, arm &#x27;hybrid&#x27;, retrieval track, 12 of 12 queries" in html
    assert viewer._fallbacks([_row("control")]) == ""


def test_every_reason_a_stage_fell_back_for_is_kept_and_the_first_still_leads():
    """The first reason is what one search's caller is shown; the harness keeps all."""
    with degraded.collecting():
        degraded.record("hybrid", _REASON)
        degraded.record("hybrid", _DENSE)
        degraded.record("hybrid", _REASON)
        assert degraded.reasons() == {"hybrid": _REASON}
        assert degraded.all_reasons() == {"hybrid": [_REASON, _DENSE]}
    assert degraded.all_reasons() == {}
