"""The CLI's options, names and help read the same way in every command.

Found in a first-hour walkthrough: `-s` meant `--session` in seven commands and
`--split` in `benchmark run`; `benchmark compare` and `results compare` did
different things under one name; help texts quoted their own history ("has
done this since the beginning and was reachable only from Python") and printed
reStructuredText backticks; `results interaction --bootstrap` set the draws of
a sign-flip test; a typo in a benchmark, profile or reranker name ran for
seconds and failed inside a task, or listed every name without suggesting the
one meant; there was no way to list components at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from lasagna import cli
from tests._paths import PACKAGE_DIR

ROOT = typer.main.get_command(cli.app)

#: Phrases that narrate the code's past instead of telling a user what to do.
HISTORY = ("since the beginning", "reachable only from", "nothing joined", "previously", "used to")


def _commands(group=ROOT, path=()):
    """Every command and subcommand, as (path, click command), hidden ones too."""
    yield path, group
    for name, command in getattr(group, "commands", {}).items():
        yield from _commands(command, (*path, name))


COMMANDS = [(path, cmd) for path, cmd in _commands() if path]
VISIBLE = [
    (p, c)
    for p, c in COMMANDS
    if not c.hidden and not any(ROOT.commands.get(p[0]).hidden for _ in [0])
]


def _ids(items):
    return [" ".join(p) for p, _ in items]


def _help(*path: str) -> str:
    result = CliRunner().invoke(cli.app, [*path, "--help"], terminal_width=100)
    assert result.exit_code == 0, result.output
    return result.output


# -- options -----------------------------------------------------------------


@pytest.mark.parametrize("path,command", COMMANDS, ids=_ids(COMMANDS))
def test_s_always_means_session(path, command):
    for param in command.params:
        opts = list(getattr(param, "opts", [])) + list(getattr(param, "secondary_opts", []))
        if "--session" in opts:
            assert "-s" in opts, f"{' '.join(path)}: --session has no -s"
        elif "-s" in opts:
            pytest.fail(f"{' '.join(path)}: -s means {opts[0]} here, --session everywhere else")


def test_benchmark_run_keeps_split_and_mode_as_long_options():
    run = ROOT.commands["benchmark"].commands["run"]
    options = {p.name: p.opts for p in run.params}
    assert options["split"] == ["--split"]
    assert options["mode"] == ["--mode"]


@pytest.mark.parametrize("path,command", COMMANDS, ids=_ids(COMMANDS))
def test_every_option_and_argument_says_what_it_is(path, command):
    missing = [
        p.name
        for p in command.params
        if not getattr(p, "hidden", False) and not (getattr(p, "help", None) or "").strip()
    ]
    assert not missing, f"{' '.join(path)}: no help for {missing}"


@pytest.mark.parametrize("path,command", VISIBLE, ids=_ids(VISIBLE))
def test_help_is_written_for_users(path, command):
    text = _help(*path)
    assert "``" not in text, "reStructuredText backticks reached the terminal"
    assert "—" not in text, "em dash in help text"
    for phrase in HISTORY:
        assert phrase not in text.lower(), f"history in help: {phrase!r}"


def test_the_interaction_resamples_option_says_what_it_draws():
    text = " ".join(_help("results", "interaction").split())
    assert "--resamples" in text and "--bootstrap" not in text
    assert "sign-flip" in text and "t interval" in text


def test_bootstrap_still_works_for_one_release(monkeypatch, tmp_path):
    seen = {}

    def fake_test(self, benchmark, metric, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop here")

    from lasagna.knowledge_evaluation_sdk import results as results_module

    monkeypatch.setattr(results_module.BenchmarkResults, "interaction_test", fake_test)
    monkeypatch.setattr(
        results_module.BenchmarkResults,
        "load",
        classmethod(
            lambda cls, path: SimpleNamespace(
                _results=[
                    SimpleNamespace(benchmark_name="b", config_name=n, error=None)
                    for n in ("neither", "a", "b", "a+b")
                ],
                interaction_test=lambda *a, **k: fake_test(None, *a, **k),
                require_metric=lambda *a, **k: None,
            )
        ),
    )
    run = tmp_path / "r.json"
    run.write_text("{}")  # read by the stub above; the command checks that it exists
    CliRunner().invoke(cli.app, ["results", "interaction", str(run), "--bootstrap", "77"])
    assert seen.get("n_bootstrap") == 77


@pytest.mark.parametrize(
    "args,tracks",
    [
        ([], ["retrieval"]),
        (["--generator", "concise"], ["retrieval", "generation"]),
        (["--tracks", "generation"], ["generation"]),
    ],
)
def test_benchmark_run_measures_retrieval_unless_asked(monkeypatch, args, tracks):
    """Generation needs a key and the scoring models, so it is opt-in; naming a
    generator opts in, or `--generator` would run nothing."""
    from lasagna.knowledge_evaluation_sdk import experiment

    seen = {}

    class _Stop(Exception):
        pass

    def fake_experiment(config):
        seen["tracks"] = config.tracks
        raise _Stop

    import lasagna.providers as providers

    # Generation is checked for an LLM key before anything runs.
    monkeypatch.setattr(providers, "get_llm_provider", lambda: object())
    monkeypatch.setattr(experiment, "Experiment", fake_experiment)
    CliRunner().invoke(cli.app, ["benchmark", "run", "smoke", *args])
    assert seen["tracks"] == tracks


# -- benchmark compare / leaderboard -------------------------------------------


@pytest.mark.parametrize("name", ["compare", "leaderboard"])
def test_the_old_benchmark_spellings_are_hidden(name):
    assert ROOT.commands["benchmark"].commands[name].hidden
    assert f" {name} " not in _help("benchmark")


def test_benchmark_compare_points_at_results_compare(monkeypatch):
    called = {}
    monkeypatch.setattr(
        cli, "results_compare", lambda a, b, metric: called.update(a=a, b=b, m=metric)
    )
    result = CliRunner().invoke(cli.app, ["benchmark", "compare", "x.json", "y.json", "-m", "mrr"])
    assert result.exit_code == 0, result.output
    assert called == {"a": "x.json", "b": "y.json", "m": "mrr"}
    assert result.output.count("deprecated") == 1
    assert "lasagna results compare" in result.output


def test_benchmark_leaderboard_points_at_results_leaderboard(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        cli,
        "results_leaderboard",
        lambda directory, metric, output, fmt: called.update(d=directory, fmt=fmt),
    )
    result = CliRunner().invoke(cli.app, ["benchmark", "leaderboard", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert called == {"d": str(tmp_path), "fmt": "markdown"}
    assert result.output.count("deprecated") == 1


# -- component list --------------------------------------------------------------


def _run_cli(tmp_path, *args):
    code = (
        "import sys\n"
        "sys.argv = ['lasagna', *sys.argv[1:]]\n"
        "from lasagna.cli import app\n"
        "try:\n"
        "    app()\n"
        "finally:\n"
        "    print('TORCH', 'torch' in sys.modules)\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update({"HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent), "COLUMNS": "200"})
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=300,
    )


def test_component_list_shows_every_stage_without_loading_models(tmp_path):
    out = _run_cli(tmp_path, "component", "list")
    assert out.returncode == 0, out.stdout + out.stderr
    sections = {}
    for line in out.stdout.splitlines():
        title = line.strip().split(" (")[0]
        if title in {"reranker", "retriever", "chunker", "generator", "agentic_pipeline"}:
            stage = title
        elif line.startswith("│"):
            sections.setdefault(stage, []).append(line.split("│")[1].strip())
    for stage, name in [
        ("reranker", "cross_encoder"),
        ("retriever", "bm25"),
        ("chunker", "structure_aware"),
        ("generator", "concise"),
        ("agentic_pipeline", "self_ask"),
    ]:
        assert name in sections.get(stage, []), (stage, name, sections)
    assert "arxiv:2004.12832" in out.stdout, "papers are listed"
    assert "TORCH False" in out.stdout


def test_component_list_filters_by_stage(tmp_path):
    out = _run_cli(tmp_path, "component", "list", "retriever")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "bm25" in out.stdout and "cross_encoder" not in out.stdout


# -- names are checked before anything loads ----------------------------------------


@pytest.mark.parametrize(
    "args,suggestion",
    [
        (["benchmark", "run", "smok"], "'smoke'"),
        (["benchmark", "run", "smoke", "--profile", "sped"], "'speed'"),
        (["benchmark", "run", "smoke", "--reranker", "cross_encodr"], "'cross_encoder'"),
        (["benchmark", "run", "smoke", "--generator", "concis"], "'concise'"),
        (["benchmark", "info", "beir/scifac"], "'beir/scifact'"),
        (["configure", "--profile", "sped"], "'speed'"),
        (["component", "list", "rerankr"], "'reranker'"),
    ],
    ids=lambda v: " ".join(v) if isinstance(v, list) else v,
)
def test_a_typo_is_one_error_with_a_suggestion(tmp_path, args, suggestion):
    out = _run_cli(tmp_path, *args)
    text = " ".join((out.stdout + out.stderr).split())
    assert out.returncode != 0, text
    assert text.count(f"did you mean {suggestion}?") == 1, text
    assert "RuntimeWarning" not in text and "Traceback" not in text, text
    assert "entry point" not in text, "a typo is not a missing plugin"
    assert "TORCH False" in out.stdout, "the name was checked after loading models"


def test_a_name_that_looks_like_a_plugin_gets_the_plugin_advice(tmp_path):
    out = _run_cli(tmp_path, "benchmark", "run", "smoke", "--reranker", "my_pkg.MyReranker")
    text = " ".join((out.stdout + out.stderr).split())
    assert out.returncode != 0
    assert "entry point" in text and "module:attribute" in text, text


# -- benchmark experiment ------------------------------------------------------


def test_an_experiment_ends_with_the_leaderboard(monkeypatch, tmp_path):
    from lasagna.knowledge_evaluation_sdk import experiment

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text("name: x\n")

    class _Results:
        total_tasks, failed_tasks, _results = 2, [], []

        def get_metric_names(self):
            return ["exact_match", "ndcg@10"]

        def leaderboard(self, metric):
            return f"LEADERBOARD BY {metric}"

    async def run(progress_callback=None):
        return _Results()

    config = SimpleNamespace(
        name="x",
        benchmarks=[],
        n_runs=1,
        mode="auto",
        max_concurrency=1,
        output_dir=str(tmp_path / "out"),
        checkpoint_dir="",
        effective_output_dir=lambda: str(tmp_path / "out"),
    )
    fake = SimpleNamespace(config=config, generate_tasks=lambda: [], run=run)
    monkeypatch.setattr(experiment.Experiment, "from_file", classmethod(lambda cls, path: fake))
    monkeypatch.setattr(cli, "_release_backend", lambda: None)
    result = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep)])
    assert result.exit_code == 0, result.output
    assert "LEADERBOARD BY ndcg@10" in result.output


# -- component validate --------------------------------------------------------------


def test_a_file_target_is_named_as_a_sweep_names_it(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "my_ideas.py").write_text(
        "def keyword_overlap(query, text):\n"
        "    return len(set(query.split()) & set(text.split()))\n"
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli.app,
        ["component", "validate", "pkg/my_ideas.py:keyword_overlap", "--stage", "reranker"],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    assert "my_ideas:keyword_overlap (reranker)" in result.output
    assert "https://github.com/dolev31/lasagna/blob/main/docs/guide.md#" in result.output
    assert "my_ideas_py" not in result.output


def test_every_command_group_lists_its_commands_without_a_click_error():
    """A guard for the walk above: it must have found the groups it checks."""
    groups = {" ".join(p) for p, c in COMMANDS if hasattr(c, "commands")}
    assert {"benchmark", "component", "results", "plugins", "skills", "pipelines"} <= groups
