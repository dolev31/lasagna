"""What a researcher reads about a run names what actually ran and what was checked.

* The short-list warning on a dense ``speed`` run blamed "BM25 and SPLADE",
  neither of which ran: the corpus held fewer chunks than were requested.
* The inert check called ``cross_encoder`` "not a reranking" on a two-question
  run of one's own data, and the bundle was marked INERT_ARM; on so few
  queries a reranker can keep an order that was already right.
* ``component validate cross_encoder`` built it with the settings file's
  881 MB reranker and said nothing; the speed profile's is 92 MB.
* The markdown leaderboard of a run's results.json said "machine-validated"
  for a file nobody validated, and linked it by a path on this machine.
* docs/own-data.md said a file registers as ``local/<file name>@hash``; it is
  ``local/<stem>@<hash>``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import experiment
from lasagna.knowledge_evaluation_sdk import stages
from lasagna.knowledge_evaluation_sdk.stages import isolation_report
from tests._paths import PACKAGE_DIR


def test_a_dense_run_that_ran_short_names_dense_search():
    why = experiment._short_retrieval({"enable_hybrid_search": False})
    msg = experiment._short_list_warning(
        short=2, exhausted=2, n_queries=2, k=10, overfetch=4, ran_short=why
    )
    assert "BM25" not in msg and "SPLADE" not in msg
    assert "dense search returns at most the chunks the corpus holds" in msg


def test_hybrid_and_lexical_retrievers_are_named_for_what_they_are():
    assert "BM25" in experiment._short_retrieval({"enable_hybrid_search": True})
    assert "SPLADE" in experiment._short_retrieval(
        {"enable_hybrid_search": True, "sparse_method": "splade"}
    )
    lexical = experiment._short_retrieval({}, retriever="bm25")
    assert "bm25 retriever returns only passages that share a term" in lexical


def _row(name, reranker, moved):
    return {
        "benchmark_name": "local/mine@abc",
        "config_name": name,
        "profile": "speed",
        "config_overrides": {"reranker": reranker} if reranker else {},
        "system": {"stages": {"reranker": {"digest": name, "component": reranker}}},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "_retrieved_ids": [name, str(i)], "_reranker_moved": m}
            for i, m in enumerate(moved)
        ],
    }


def test_two_queries_are_too_few_to_call_a_reranker_inert():
    rows = [_row("cross_encoder", "cross_encoder", [False, False])]
    report = isolation_report(rows)
    assert report["inert"] == []
    (line,) = report["unverified"]
    assert "too few queries to tell" in line and "on all 2 queries" in line
    assert f"at least {stages.MIN_QUERIES_TO_CALL_INERT}" in line

    from lasagna.knowledge_evaluation_sdk.bundle import _integrity

    integrity = _integrity(rows, {})
    assert "INERT_ARM" not in integrity["warnings"]


def test_enough_queries_still_call_it_inert():
    rows = [_row("mine", "mine", [False] * stages.MIN_QUERIES_TO_CALL_INERT)]
    assert len(isolation_report(rows)["inert"]) == 1


def test_component_validate_builds_with_the_speed_profiles_model(monkeypatch):
    from lasagna.core.config import rag_config
    from lasagna.core.testing import ComponentTestHarness

    seen = []

    def validate(self, instance):
        seen.append(rag_config.reranker_model)
        return SimpleNamespace(
            component_type="reranker", checks=[], failures=[], latency_ms=None, passed=True
        )

    monkeypatch.setattr(ComponentTestHarness, "validate", validate)
    before = rag_config.reranker_model
    result = CliRunner().invoke(cli.app, ["component", "validate", "cross_encoder"])
    words = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert seen == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    assert (
        "the speed profile's reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2 (92 MB" in words
    )
    assert rag_config.reranker_model == before  # restored afterwards

    result = CliRunner().invoke(
        cli.app, ["component", "validate", "cross_encoder", "--profile", "standard"]
    )
    assert seen[-1] == "mixedbread-ai/mxbai-rerank-large-v1"
    assert "(881 MB" in " ".join(result.output.split())


def _results_file(tmp_path):
    from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

    path = tmp_path / "run" / "results.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "experiment_name": "mine",
                "metrics_version": METRICS_VERSION,
                "experiment_config": {
                    "metrics_version": METRICS_VERSION,
                    "environment": {
                        "lasagna_version": "0.1.0",
                        "python": "3.12.0",
                        "platform": "test",
                    },
                },
                "results": [
                    {
                        "task_key": "smoke|speed|run_0",
                        "benchmark_name": "smoke",
                        "config_name": "speed",
                        "run_index": 0,
                        "metrics": {"ndcg@10": 0.9},
                        "n_samples": 12,
                        "config_overrides": {},
                        "system": {"embedder_model": "all-MiniLM-L6-v2"},
                    }
                ],
            }
        )
    )
    return path


def test_a_leaderboard_of_a_results_file_claims_no_validation_and_no_path(tmp_path):
    from lasagna.knowledge_evaluation_sdk.leaderboard import build

    source = _results_file(tmp_path)
    build(source, output=tmp_path / "pages")
    page = (tmp_path / "pages" / "smoke.md").read_text()
    assert "machine-validated" not in page
    assert "not validated" in page and "lasagna results validate" in page
    assert "results.json)" not in page and "](" not in page


def test_a_leaderboard_of_bundles_links_each_and_says_what_validate_checks(tmp_path):
    from lasagna.knowledge_evaluation_sdk.leaderboard import build
    from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

    runs = tmp_path / "runs"
    runs.mkdir()
    bundle = BenchmarkResults.load(str(_results_file(tmp_path))).to_bundle()
    (runs / "mine.json").write_text(json.dumps(bundle))
    build(runs, output=tmp_path / "pages")
    page = (tmp_path / "pages" / "smoke.md").read_text()
    assert "machine-validated" not in page
    assert "](../runs/mine.json)" in page
    assert "`lasagna results validate` recomputes" in page


def test_own_data_docs_name_the_benchmark_as_it_registers(tmp_path):
    from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.local_files import (
        read_local_data,
    )

    data = tmp_path / "my_data.jsonl"
    data.write_text(json.dumps({"question": "q", "answer": "a", "contexts": ["c"]}) + "\n")
    assert read_local_data(data).benchmark_name().startswith("local/my_data@")
    docs = (PACKAGE_DIR.parents[1] / "docs" / "own-data.md").read_text()
    assert "local/<stem>@<12 hex digits>" in docs
    assert "local/<file name>" not in docs
