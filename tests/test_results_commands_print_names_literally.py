"""`lasagna results ...` must print config names as text, never as Rich markup.

`results validate` printed its problem lines through Rich markup: an arm named
`baseline [old]` printed as `baseline `, and one named `a[/b]` raised
MarkupError, so the command crashed instead of saying why a bundle was invalid.
Arm names are free text chosen by the researcher.
"""

from __future__ import annotations

import json

import pytest

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION  # noqa: E402

runner = typer_testing.CliRunner()

NAMES = ("baseline [old]", "a[/b]")


def _results_file(tmp_path, headline_off_by=0.0):
    rows = []
    for name in NAMES:
        scores = [0.4 + 0.01 * (i % 7) + (0.1 if name == "a[/b]" else 0.0) for i in range(12)]
        rows.append(
            {
                "task_key": name,
                "run_index": 0,
                "benchmark_name": "bench",
                "config_name": name,
                "metrics": {"ndcg@10": sum(scores) / len(scores) + headline_off_by},
                "per_sample_scores": [
                    {"_sample_id": f"q{i}", "ndcg@10": s} for i, s in enumerate(scores)
                ],
                "error": None,
            }
        )
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"metrics_version": METRICS_VERSION, "results": rows}))
    return str(path)


def _bundle(tmp_path, **kwargs):
    out = tmp_path / "run.json"
    result = runner.invoke(
        app, ["results", "bundle", _results_file(tmp_path, **kwargs), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    return str(out), result.output


def test_bundle_prints_each_config_name(tmp_path):
    _, output = _bundle(tmp_path)
    for name in NAMES:
        assert name in output


def test_validate_names_the_arm_whose_aggregate_was_edited(tmp_path):
    path, _ = _bundle(tmp_path, headline_off_by=0.05)
    result = runner.invoke(app, ["results", "validate", path])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert "'baseline [old]'" in result.output
    assert "'a[/b]'" in result.output


@pytest.mark.parametrize(
    "command",
    [
        ["significance", "--all"],
        ["per-query"],
        ["power", NAMES[0], NAMES[1]],
        ["compare"],
    ],
)
def test_other_results_commands_print_names_literally(tmp_path, command):
    path = _results_file(tmp_path)
    name, *rest = command
    args = ["results", name, path, *rest]
    if name == "compare":
        args.append(path)
    result = runner.invoke(app, args)
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 0, result.output
    for config in NAMES:
        assert config in result.output, f"{config!r} missing from `results {name}`"


def _bundle_with_fallbacks(tmp_path, fallbacks):
    from lasagna.knowledge_evaluation_sdk.bundle import _content_hash

    # A current bundle: `results validate` refuses one with no metrics_version.
    bundle = {
        "metrics_version": METRICS_VERSION,
        "schema_version": "1.0",
        "experiment": {"environment": {"lasagna_version": "x", "python": "3", "platform": "p"}},
        "configs": [],
        "results": [{"config_name": "a", "metrics": {}, "per_sample_scores": []}],
        "integrity": {"warnings": ["ACTIVE_FALLBACKS"], "blocking": [], "fallbacks": fallbacks},
    }
    bundle["run_id"] = _content_hash(bundle)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(bundle))
    return str(path)


FALLBACK = (
    "sentence_split_without_nltk: sentences are split by punctuation; "
    "pip install 'lasagna-rag[eval]'"
)


@pytest.mark.parametrize("strict", [False, True])
def test_validate_says_which_fallbacks_were_active(tmp_path, strict):
    """ACTIVE_FALLBACKS alone does not say what to install."""
    args = ["results", "validate", _bundle_with_fallbacks(tmp_path, [FALLBACK])]
    result = runner.invoke(app, args + (["--strict"] if strict else []))
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert "pip install 'lasagna-rag[eval]'" in " ".join(result.output.split())
    if strict:
        assert "1 problem(s)" in result.output


def test_an_older_bundle_without_the_list_still_validates(tmp_path):
    path = _bundle_with_fallbacks(tmp_path, [])
    data = json.loads(open(path).read())
    del data["integrity"]["fallbacks"]
    from lasagna.knowledge_evaluation_sdk.bundle import _content_hash

    data["run_id"] = _content_hash(data)
    open(path, "w").write(json.dumps(data))
    result = runner.invoke(app, ["results", "validate", path])
    assert result.exit_code == 0, result.output
