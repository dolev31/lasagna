"""Every results command reads `--benchmark` the same way, and an unknown one says so.

A sweep entry `{path: tickets.jsonl, name: tickets}` registers as
`tickets@<content hash>`. `results significance <run> --benchmark tickets` then
said "No metric 'ndcg@10' ... They record no metrics", and so did `power` and
`interaction`; `per-query` said "No per-query scores in this run". Only `view`
named the run's benchmarks. One resolver now serves them all: it takes the id
or the name the sweep gave, and anything else is an error that lists the
run's benchmarks with the closest one.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

runner = CliRunner()

TICKETS = "tickets@7d85fdce34c4"
#: The four cells of a 2x2, so `results interaction` finds them by name.
ARMS = {"neither": 0.0, "a": 0.1, "b": 0.05, "a+b": 0.12}


@pytest.fixture
def run(tmp_path) -> str:
    rows = []
    for bench in ("smoke", TICKETS):
        for config, lift in ARMS.items():
            scores = [0.3 + lift + 0.02 * (i % 5) for i in range(12)]
            rows.append(
                {
                    "task_key": f"{bench}|{config}|run_0",
                    "benchmark_name": bench,
                    "config_name": config,
                    "run_index": 0,
                    "metrics": {"ndcg@10": sum(scores) / len(scores)},
                    "per_sample_scores": [
                        {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)
                    ],
                }
            )
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"metrics_version": METRICS_VERSION, "results": rows}))
    return str(path)


COMMANDS = [
    ["significance", "{run}", "--all"],
    ["power", "{run}", "neither", "a"],
    ["interaction", "{run}"],
    ["per-query", "{run}"],
    ["view", "{run}", "-o", "{tmp}/view.html"],
]


def _invoke(argv, run, tmp_path):
    args = ["results", *(a.format(run=run, tmp=tmp_path) for a in argv)]
    return runner.invoke(cli.app, args, terminal_width=200)


@pytest.mark.parametrize("argv", COMMANDS, ids=lambda a: a[0])
def test_the_sweeps_name_for_a_benchmark_is_accepted(argv, run, tmp_path):
    result = _invoke(argv + ["--benchmark", "tickets"], run, tmp_path)
    assert result.exit_code == 0, result.output
    if argv[0] in ("significance", "power", "interaction"):
        assert TICKETS in result.output


@pytest.mark.parametrize("argv", COMMANDS, ids=lambda a: a[0])
def test_an_unknown_benchmark_lists_the_runs_benchmarks(argv, run, tmp_path):
    result = _invoke(argv + ["--benchmark", "tickts"], run, tmp_path)
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert "No benchmark 'tickts' in these results (did you mean 'tickets'?)" in words
    assert f"They have: smoke, {TICKETS}." in words
    assert "No metric" not in words and "No per-query scores" not in words


def test_the_resolver_takes_an_id_a_display_name_or_a_file_stem():
    from lasagna.knowledge_evaluation_sdk.results import resolve_benchmark

    names = ["beir/scifact", "local/my_data@0123456789ab", TICKETS]
    assert resolve_benchmark("beir/scifact", names) == "beir/scifact"
    assert resolve_benchmark("tickets", names) == TICKETS
    assert resolve_benchmark(TICKETS, names) == TICKETS
    assert resolve_benchmark("my_data", names) == "local/my_data@0123456789ab"
    assert resolve_benchmark("local/my_data", names) == "local/my_data@0123456789ab"


def test_a_name_two_versions_share_must_be_given_in_full():
    from lasagna.knowledge_evaluation_sdk.results import resolve_benchmark

    names = ["tickets@aaaaaaaaaaaa", "tickets@bbbbbbbbbbbb"]
    with pytest.raises(ValueError, match="names 2 benchmarks.*Pass one of them in full"):
        resolve_benchmark("tickets", names)


def test_an_unknown_config_in_per_query_names_the_runs_configs(run, tmp_path):
    result = _invoke(["per-query", "{run}", "--config", "nether"], run, tmp_path)
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert "No config 'nether' in this run (did you mean 'neither'?)" in words
