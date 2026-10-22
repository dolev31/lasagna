"""What a researcher meets first is plain, consistent and true.

Found by the sixth persona audit:

* `component validate cross_encoder` suggested `rerankers: [cross_encoder,
  "cross_encoder"]`, an arm compared with itself.
* An empty sweep file gave "'NoneType' object is not iterable", and
  `n_runs: lots` gave "'str' object cannot be interpreted as an integer",
  naming neither the file nor the key.
* `benchmark synthesize --dry-run --n 200` planned 200 questions from three
  passages without a word.
* `pipelines list` marked `legacy_react` active, the settings file's value,
  while a session on `standard` runs `default`.
* The example reranker had three names: `keyword_overlap` in the demo GIF, the
  CLI help and the component guide, `longer_first` in the README, `coverage`
  in the notebook. It is `keyword_overlap` everywhere.
* The notebook ships without outputs (its test keeps them cleared), and the
  README did not say so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lasagna import cli

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.mark.parametrize(
    "stage, name, expected",
    [
        ("reranker", "cross_encoder", "rerankers: [flashrank, cross_encoder]"),
        (
            "reranker",
            "my_ideas:keyword_overlap",
            'rerankers: [cross_encoder, "my_ideas:keyword_overlap"]',
        ),
        ("chunker", "fixed", "chunkers: [recursive, fixed]"),
        (
            "retriever",
            "bm25",
            "config_overrides: [{name: dense, retriever: dense}, {name: bm25, retriever: bm25}]",
        ),
    ],
)
def test_validate_suggests_a_control_other_than_the_component(stage, name, expected):
    example = cli._sweep_example(stage, name)
    assert example == expected
    arms = re.findall(r"[\w:\"]+", example.split(":", 1)[1])
    assert len(set(arms)) > 1


@pytest.mark.parametrize(
    "text, words",
    [
        ("", "sweep.yaml: the file is empty."),
        ("- a\n- b\n", "sweep.yaml: its top level is a list, not a mapping of keys."),
        (
            "name: x\nbenchmarks: [{name: smoke}]\nn_runs: lots\n",
            "sweep.yaml: `n_runs:` at the top level takes a whole number, got 'lots' (at least 1).",
        ),
        (
            "name: x\nbenchmarks: [{name: smoke, n_samples: 0}]\n",
            "sweep.yaml: `n_samples:` in benchmark 'smoke' must be at least 1, got 0.",
        ),
        (
            "name: x\nbenchmarks: smoke\n",
            "sweep.yaml: `benchmarks:` at the top level takes a list, got 'smoke'.",
        ),
    ],
)
def test_a_bad_sweep_file_names_the_file_and_the_key(tmp_path, monkeypatch, text, words):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sweep.yaml").write_text(text)
    result = runner.invoke(cli.app, ["benchmark", "experiment", "sweep.yaml"], terminal_width=300)
    out = " ".join(result.output.split())
    assert result.exit_code == 1, out
    assert words in out
    assert "object is not iterable" not in out and "cannot be interpreted" not in out


def _corpus(tmp_path, n_docs: int) -> Path:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    sentence = "Gallium melts at 29.76 degrees Celsius, tin at 232 degrees and lead at 327. "
    for i in range(n_docs):
        (corpus / f"p{i}.txt").write_text(f"Document {i}. " + sentence * 3)
    return corpus


def test_synthesize_warns_when_the_corpus_cannot_supply_that_many_questions(tmp_path):
    corpus = _corpus(tmp_path, 3)
    result = runner.invoke(
        cli.app,
        ["benchmark", "synthesize", str(corpus), "--n", "200", "--dry-run"],
        terminal_width=400,
    )
    out = " ".join(result.output.split())
    assert result.exit_code == 0, out
    assert "Warning: 200 questions from 3 passages" in out
    assert "--n 6 or fewer" in out

    fits = runner.invoke(cli.app, ["benchmark", "synthesize", str(corpus), "--n", "6", "--dry-run"])
    assert fits.exit_code == 0 and "Warning" not in fits.output, fits.output


def test_pipelines_list_marks_what_the_session_runs():
    from lasagna.core.session_config import get_knowledge_config_manager

    for session, profile, active in (
        ("pipelines-standard", "standard", "default"),
        ("pipelines-speed", "speed", None),
    ):
        get_knowledge_config_manager(session_id=session).set_active_profile(
            profile, session_id=session, force=True
        )
        result = runner.invoke(cli.app, ["pipelines", "list", "-s", session], terminal_width=200)
        out = " ".join(result.output.split())
        assert result.exit_code == 0, out
        if active:
            assert f"Active in session {session} ({profile} profile): {active}" in out
        else:
            assert f"Agentic retrieval is off in session {session} ({profile} profile)" in out


def test_the_example_reranker_has_one_name_everywhere():
    pages = {
        "README.md": (REPO / "README.md").read_text(),
        "docs/contributing-components.md": (REPO / "docs/contributing-components.md").read_text(),
        "examples/notebooks/compare_your_reranker.py": (
            REPO / "examples/notebooks/compare_your_reranker.py"
        ).read_text(),
        "examples/notebooks/compare_your_reranker.ipynb": json.dumps(
            json.loads((REPO / "examples/notebooks/compare_your_reranker.ipynb").read_text())
        ),
        "component validate --help": runner.invoke(
            cli.app, ["component", "validate", "--help"], terminal_width=200
        ).output,
    }
    for page, text in pages.items():
        assert "keyword_overlap" in text, page
        for other in ("longer_first", "my_ideas:coverage", "def coverage("):
            assert other not in text, (page, other)


def test_the_readme_says_the_notebook_stores_no_outputs():
    text = re.sub(r"\n> ?", " ", (REPO / "README.md").read_text())  # blockquote lines
    words = " ".join(text.split())
    assert "It is stored without outputs: run it on Colab, or locally with Jupyter" in words
