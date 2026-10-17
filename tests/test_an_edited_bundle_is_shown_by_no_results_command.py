"""Every results command that reads a bundle checks it as `results validate` does.

With one number edited, `results validate` said INVALID and `results compare`
refused the bundle, but `results leaderboard --format latex` printed the edited
number, the `-o` page ranked it, `results view` showed it next to the original
run id, and `results bundle` gave it a fresh run id. Each command now refuses
it with the problems listed; `--allow-invalid` renders it only marked INVALID,
in every format. A results.json is not a bundle and is shown unvalidated.
"""

from __future__ import annotations

import json
import re

import pytest

from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION, InvalidBundleError
from lasagna.knowledge_evaluation_sdk.stages import OTHER, STAGES

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402

runner = typer_testing.CliRunner()

N = 20
MARK = "INVALID: edited after it was made"


def _task(config: str, base: float) -> dict:
    manifest = {s: {"digest": "same", "component": None, "settings": {}} for s in (*STAGES, OTHER)}
    manifest["reranker"] = {"digest": f"d-{config}", "component": config, "settings": {}}
    scores = [base + 0.001 * i for i in range(N)]
    return {
        "task_key": f"bench|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": config,
        "metrics": {"ndcg@10": sum(scores) / N},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "ndcg@10": s, "_retrieved_ids": [f"doc{i}", f"{config}-{i}"]}
            for i, s in enumerate(scores)
        ],
        "error": None,
        "config_overrides": {"reranker": config},
        "profile": "speed",
        "system": {"stages": manifest},
    }


@pytest.fixture
def files(tmp_path):
    """A results.json, its valid bundle, and a copy with one number edited."""
    config = {
        "experiment_name": "two arms",
        "n_runs": 1,
        "tracks": ["retrieval"],
        # Retrieval only: one run is complete, so a matching number verifies.
        "nothing_samples": True,
        "environment": {"lasagna_version": "0.1.0", "python": "3.12.0", "platform": "test"},
        "metrics_version": METRICS_VERSION,
    }
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "experiment_name": "two arms",
                "experiment_config": config,
                "results": [_task("control", 0.50), _task("mine", 0.60)],
            }
        )
    )
    bundle = tmp_path / "run.json"
    made = runner.invoke(app, ["results", "bundle", str(results), "-o", str(bundle)])
    assert made.exit_code == 0, made.output
    data = json.loads(bundle.read_text())
    data["results"][0]["metrics"]["ndcg@10"] = 0.5
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(data))
    assert runner.invoke(app, ["results", "validate", str(bundle)]).exit_code == 0
    assert runner.invoke(app, ["results", "validate", str(edited)]).exit_code == 1
    return {"results": results, "bundle": bundle, "edited": edited, "dir": tmp_path}


def _words(result) -> str:
    return " ".join(result.output.split())


def _refused(result, *, allow: bool) -> None:
    words = _words(result)
    assert result.exit_code == 1, words
    assert "0.5000" not in words
    assert "is not a valid bundle, so its numbers cannot be" in words
    # Every problem, not only the first.
    assert "run_id does not match the bundle contents" in words
    assert "aggregate disagrees with per-query scores" in words
    assert "lasagna results validate" in words
    assert ("--allow-invalid" in words) is allow


@pytest.mark.parametrize(
    "args",
    [
        ["leaderboard", "{edited}"],
        ["leaderboard", "{edited}", "-o", "{dir}/pages"],
        ["leaderboard", "{edited}", "--format", "latex"],
        ["leaderboard", "{edited}", "--format", "latex", "-o", "{dir}/tex"],
        ["view", "{edited}", "-o", "{dir}/view.html"],
    ],
    ids=["leaderboard", "leaderboard-o", "latex", "latex-o", "view"],
)
def test_a_command_that_renders_a_bundle_refuses_an_edited_one(files, args):
    argv = ["results", *(a.format(**files) for a in args)]
    _refused(runner.invoke(app, argv), allow=True)
    assert not (files["dir"] / "pages").exists() and not (files["dir"] / "tex").exists()
    assert not (files["dir"] / "view.html").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["bundle", "{edited}", "-o", "{dir}/again.json"],
        ["compare", "{edited}", "{bundle}"],
        ["power", "{edited}", "control", "mine"],
        ["significance", "{edited}"],
        ["per-query", "{edited}"],
        ["interaction", "{edited}"],
    ],
    ids=["bundle", "compare", "power", "significance", "per-query", "interaction"],
)
def test_a_command_that_reads_a_bundle_refuses_an_edited_one(files, args):
    argv = ["results", *(a.format(**files) for a in args)]
    _refused(runner.invoke(app, argv), allow=False)
    # Re-bundling gave an edited bundle a fresh, matching run id.
    assert not (files["dir"] / "again.json").exists()


def test_the_deprecated_leaderboard_spelling_refuses_it_too(files):
    _refused(runner.invoke(app, ["benchmark", "leaderboard", str(files["edited"])]), allow=True)


def test_a_folder_leaves_the_edited_bundle_out_and_says_so(files, tmp_path):
    folder = tmp_path / "runs"
    folder.mkdir()
    (folder / "run.json").write_text(files["bundle"].read_text())
    (folder / "edited.json").write_text(files["edited"].read_text())

    out = tmp_path / "pages"
    page = runner.invoke(app, ["results", "leaderboard", str(folder), "-o", str(out)])
    words = _words(page)
    assert page.exit_code == 0, words
    assert "2 rows from 1 bundle" in words
    assert (
        f"Left out, rejected by `lasagna results validate` (--allow-invalid shows them marked INVALID): edited.json: {MARK}"
        in words
    )
    md = (out / "bench.md").read_text()
    assert "0.5000" not in md and "edited.json" not in md
    index = json.loads((out / "index.json").read_text())
    assert index["invalid"][0]["path"] == "edited.json"
    assert index["invalid"][0]["shown"] is False

    latex = runner.invoke(app, ["results", "leaderboard", str(folder), "--format", "latex"])
    assert latex.exit_code == 0, latex.output
    assert "$0.5000$" not in latex.output
    # A comment, so `> tables.tex` stays valid LaTeX.
    assert latex.output.rstrip().splitlines()[-1].startswith(f"% Left out edited.json: {MARK}.")


def test_allow_invalid_marks_every_format(files, tmp_path):
    edited = str(files["edited"])

    latex = runner.invoke(
        app, ["results", "leaderboard", edited, "--format", "latex", "--allow-invalid"]
    )
    assert latex.exit_code == 0, latex.output
    caption = next(line for line in latex.output.splitlines() if line.startswith("\\caption"))
    assert caption.startswith(f"\\caption{{\\textbf{{{MARK}.}}")
    row = next(line for line in latex.output.splitlines() if line.startswith("control &"))
    assert f"$0.5000$ \\textbf{{{MARK}}}" in row

    out = tmp_path / "pages"
    written = runner.invoke(
        app, ["results", "leaderboard", edited, "-o", str(out), "--allow-invalid"]
    )
    assert written.exit_code == 0, written.output
    assert f"Shown marked INVALID, rejected by `lasagna results validate`: edited.json: {MARK}" in (
        _words(written)
    )
    md = (out / "bench.md").read_text()
    assert md.splitlines()[2].startswith(f"**{MARK}.**")
    assert f"| 0.5000 **{MARK}** |" in md
    assert "| yes |" not in md
    assert json.loads((out / "index.json").read_text())["invalid"][0]["shown"] is True

    tex = tmp_path / "tex"
    as_tex = runner.invoke(
        app,
        ["results", "leaderboard", edited, "--format", "latex", "-o", str(tex), "--allow-invalid"],
    )
    assert as_tex.exit_code == 0, as_tex.output
    assert f"$0.5000$ \\textbf{{{MARK}}}" in (tex / "bench.tex").read_text()

    html_file = tmp_path / "view.html"
    view = runner.invoke(app, ["results", "view", edited, "-o", str(html_file), "--allow-invalid"])
    assert view.exit_code == 0, view.output
    page = html_file.read_text()
    assert f"<title>two arms ({MARK})</title>" in page
    banner = page.index('<div class="note bad" role="alert">')
    assert banner < page.index("<header>") and MARK in page[banner : page.index("<header>")]
    assert "run_id does not match the bundle contents" in page
    assert f'<h2>bench <span class="chip bad">{MARK}</span></h2>' in page


@pytest.mark.parametrize("theirs_valid", [True, False])
def test_an_invalid_row_is_never_verified_and_verifies_nothing(files, tmp_path, theirs_valid):
    from lasagna.knowledge_evaluation_sdk.bundle import _content_hash
    from lasagna.knowledge_evaluation_sdk.leaderboard import grouped_rows

    folder = tmp_path / "runs"
    folder.mkdir()
    # Another lab publishing the same numbers: a reproduction, when its
    # bundle is valid.
    theirs = json.loads(files["bundle"].read_text())
    theirs["submitter"] = {"name": "another lab"}
    if theirs_valid:
        theirs["run_id"] = _content_hash(theirs)
    (folder / "mine.json").write_text(files["bundle"].read_text())
    (folder / "theirs.json").write_text(json.dumps(theirs))
    rows = [
        r for groups in grouped_rows(folder, allow_invalid=True)["bench"].values() for r in groups
    ]
    assert len(rows) == 4
    assert all(r["verified"] for r in rows) is theirs_valid
    assert any(r["verified"] for r in rows) is theirs_valid


def test_the_library_refuses_too(files):
    from lasagna.knowledge_evaluation_sdk.leaderboard import build, grouped_rows
    from lasagna.knowledge_evaluation_sdk.viewer import render_html

    for call in (
        lambda: render_html(files["edited"]),
        lambda: grouped_rows(files["edited"]),
        lambda: build(files["edited"]),
    ):
        with pytest.raises(InvalidBundleError) as raised:
            call()
        assert any(p.startswith("run_id does not match") for p in raised.value.problems)
    assert MARK in render_html(files["edited"], allow_invalid=True)


def test_a_results_json_is_still_shown_unvalidated(files, tmp_path):
    out = tmp_path / "pages"
    page = runner.invoke(app, ["results", "leaderboard", str(files["results"]), "-o", str(out)])
    assert page.exit_code == 0, page.output
    md = (out / "bench.md").read_text()
    assert "not validated" in md and "INVALID" not in md
    view = runner.invoke(
        app, ["results", "view", str(files["results"]), "-o", str(tmp_path / "v.html")]
    )
    assert view.exit_code == 0, view.output
    assert "INVALID" not in (tmp_path / "v.html").read_text()
    assert not re.search("INVALID", _words(page))
