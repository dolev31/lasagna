"""Messages and tables the first hour with Lasagna got wrong, each now right.

A malformed sweep printed a 164-line Rich traceback. A misspelled built-in
(`cross_encodr`) was told to install an entry point, and a misspelled arm key
was answered with all ~400 known keys, `celery_broker_url` among them. The CLI
printed loguru's timestamp, module and line on every warning, wrapped each
task hash over two lines, and put the isolation verdict after "Next".
`significance` without `--all` printed an empty table header and "1
non-significant pairs hidden". The LaTeX leaderboard repeated "Config" as
"Varied", printed "anonymous" and an empty "Verified" for a local run, and the
results tables printed "0.9806 ± 0.0000" for one run. The dry run said
"downloaded" for a model it had not downloaded. `component validate nope:x`
printed "nope❌", a reranker returning None failed with a bare float()
TypeError, and success said "Next: register it" while the README says a sweep
needs no registration.
"""

from __future__ import annotations

import json
import re

from loguru import logger
from typer.testing import CliRunner

from lasagna import cli


def _say(args, tmp_path=None):
    result = CliRunner().invoke(cli.app, args, terminal_width=200)
    return result, " ".join(result.output.split())


def test_a_malformed_sweep_is_one_line_with_its_position(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nbenchmarks: [{name: smoke}\npipeline:\n  rerankers: [a]\n")
    result, out = _say(["benchmark", "experiment", str(bad)])
    assert result.exit_code == 1
    assert "not valid YAML at line 3, column 1" in out
    assert "Traceback" not in result.output


def test_a_misspelled_component_gets_the_likely_name(tmp_path):
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    try:
        exp._check_components_exist([{"name": "a", "overrides": {"reranker": "cross_encodr"}}])
    except ValueError as err:
        text = str(err)
    assert "did you mean 'cross_encoder'?" in text
    assert "entry point" not in text


def test_a_misspelled_arm_key_gets_the_likely_key_and_no_wall_of_keys():
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    try:
        exp._check_override_keys([{"name": "a", "overrides": {"enable_rerankng": True}}])
    except ValueError as err:
        text = str(err)
    assert "did you mean 'enable_reranking'?" in text
    assert "celery_broker_url" not in text and len(text) < 600


def test_cli_warnings_are_one_plain_line(capsys):
    import sys

    cli._configure_cli_logging()
    try:
        logger.warning("12 of 12 queries had fewer than 10 documents")
        err = capsys.readouterr().err
    finally:
        logger.remove()
        logger.add(sys.__stderr__)
    assert err == "WARNING: 12 of 12 queries had fewer than 10 documents\n"


def _results(tmp_path, per_query):
    rows = [
        {
            "task_key": f"bench|{config}|cfg|corpus|live|run_0",
            "run_index": 0,
            "benchmark_name": "bench",
            "config_name": config,
            "metrics": {"ndcg@10": sum(scores) / len(scores)},
            "per_sample_scores": [
                {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)
            ],
            "error": None,
        }
        for config, scores in per_query.items()
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": rows}))
    return str(path)


def test_significance_with_nothing_significant_prints_no_empty_table(tmp_path):
    same = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6]
    path = _results(tmp_path, {"a": same, "b": list(reversed(same))})
    result, out = _say(["results", "significance", path, "-m", "ndcg@10"])
    assert result.exit_code == 0, result.output
    assert not re.search(r"^\s*pair\s+diff", result.output, re.M)
    assert "1 non-significant pair is not shown" in out


def _row(config, n_runs=1, submitter="anonymous"):
    return {
        "config": config,
        "varied_levels": {},
        "varied_dimensions": [config],
        "mean": 0.9806,
        "std": 0.0,
        "n_runs": n_runs,
        "submitter": submitter,
        "verified": False,
        "held_fixed": {},
        "comparability_key": "k",
        "stages_recorded": True,
    }


def test_the_latex_leaderboard_for_a_local_run_has_no_empty_columns():
    from lasagna.knowledge_evaluation_sdk.leaderboard import render_latex

    tex = render_latex({"smoke": {"k": [_row("reranker=mmr")]}}, "ndcg@10")
    header = next(line for line in tex.splitlines() if line.startswith("Config"))
    assert header == "Config & Runs & ndcg@10 \\\\"
    assert "\\pm" not in tex and "anonymous" not in tex


def test_one_run_has_no_spread_in_the_results_tables(tmp_path):
    from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

    results = BenchmarkResults.load(_results(tmp_path, {"a": [0.9806, 0.9806]}))
    assert "±" not in results.to_markdown("ndcg@10")
    assert "\\pm" not in results.to_latex("ndcg@10")


def test_the_dry_run_says_cached():
    import inspect

    from lasagna.knowledge_evaluation_sdk import cost

    source = inspect.getsource(cost)
    assert '{True: "cached"' in source and '{True: "downloaded"' not in source


def test_a_name_with_colons_prints_literally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, out = _say(["component", "validate", "nope:x"])
    assert "nope:x" in out and "❌" not in result.output


def test_a_reranker_returning_none_is_named(tmp_path):
    import lasagna
    from lasagna.core.base_components import ScoredDocument
    from lasagna.core.registry import registry

    @lasagna.reranker("t_returns_none")
    def returns_none(query, text):
        return None

    docs = [ScoredDocument(content="a", score=1.0, document_id="d", chunk_id="c")]
    try:
        registry.get("reranker", "t_returns_none").rerank("q", docs)
    except TypeError as err:
        text = str(err)
    assert "'t_returns_none' returned None" in text and "number" in text


def test_validate_points_at_a_sweep_not_at_registration(tmp_path, monkeypatch):
    (tmp_path / "my_ideas.py").write_text(
        "import lasagna\n\n\n@lasagna.reranker\ndef fine(query, text):\n    return len(text)\n"
    )
    monkeypatch.chdir(tmp_path)
    result, out = _say(["component", "validate", "my_ideas.py:fine"])
    assert result.exit_code == 0, result.output
    assert "register it" not in out
    assert 'rerankers: [cross_encoder, "my_ideas:fine"]' in out
