"""The Colab notebook stays runnable: valid, compilable, and its sweeps parse.

``examples/notebooks/compare_your_reranker.ipynb`` is the first thing many
people will run, on a machine nobody here sees. Running it for real needs a
dataset download and two models, so that is done before a release through
``compare_your_reranker.py`` next to it, which runs the same steps. What needs
neither is checked here, on every commit:

- it is a valid notebook, committed with its outputs cleared;
- every code cell compiles as Python once its ``!`` and ``%`` lines are set
  aside, and so does the module it writes;
- each YAML file it writes goes through the sweep parser, which refuses an
  unknown key, and plans the arms the text describes;
- the technique it writes passes the reranker contract;
- the script writes exactly the notebook's files, so running one tests the other;
- every ``lasagna`` command and option, component name, extra and repository
  link it shows exists.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import nbformat
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "examples" / "notebooks" / "compare_your_reranker.ipynb"
SCRIPT = NOTEBOOK.with_suffix(".py")

_WRITEFILE = re.compile(r"\A%%writefile\s+(\S+)\n")


@pytest.fixture(scope="module")
def notebook():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    return nb


@pytest.fixture(scope="module")
def written(notebook) -> dict:
    """The files the notebook's ``%%writefile`` cells write, by name."""
    files = {}
    for cell in notebook.cells:
        match = _WRITEFILE.match(cell.source) if cell.cell_type == "code" else None
        if match:
            files[match.group(1)] = cell.source[match.end() :]
    return files


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("_colab_notebook_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def _as_python(source: str) -> str:
    """The cell with each shell (``!``) and magic (``%``) line replaced by ``pass``.

    Replaced rather than dropped, so a magic inside an ``if`` block still leaves
    the block a body.
    """
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _shell_lines(nb):
    for cell in _code_cells(nb):
        if _WRITEFILE.match(cell.source):
            continue
        for line in cell.source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!"):
                yield stripped[1:]


def test_it_is_committed_with_outputs_cleared(notebook):
    dirty = [
        i
        for i, cell in enumerate(_code_cells(notebook))
        if cell.outputs or cell.execution_count is not None
    ]
    assert not dirty, f"code cells {dirty} carry outputs; clear them before committing"


def test_every_code_cell_compiles(notebook, written):
    for i, cell in enumerate(_code_cells(notebook)):
        if cell.source.startswith("%%"):
            continue  # a cell magic: its body is checked as the file it writes
        compile(_as_python(cell.source), f"<code cell {i}>", "exec")
    for name, body in written.items():
        if name.endswith(".py"):
            compile(body, name, "exec")


def test_it_writes_the_files_the_script_tests(written, script):
    expected = {
        "my_ideas.py": script.MY_IDEAS,
        "sweep.yaml": script.SWEEP,
        "compose.yaml": script.COMPOSE,
    }
    assert sorted(written) == sorted(expected)
    for name, body in expected.items():
        assert written[name].strip("\n") == body.strip("\n"), f"{name} differs from the script's"


def _write(tmp_path, written, name) -> Path:
    path = tmp_path / name
    path.write_text(written[name])
    return path


def test_the_sweep_parses_and_each_arm_is_one_change_from_the_last(tmp_path, written):
    from lasagna.knowledge_evaluation_sdk import Experiment
    from lasagna.knowledge_evaluation_sdk.experiment import _generate_pipeline_configs

    config = Experiment.from_file(str(_write(tmp_path, written, "sweep.yaml"))).config
    assert [b.name for b in config.benchmarks] == ["beir/scifact"]
    assert config.tracks == ["retrieval"]  # keyless: no LLM, no judge
    assert config.pipeline.base_profile == "speed"

    arms = {c["name"]: c["overrides"] for c in _generate_pipeline_configs(config.pipeline)}
    assert list(arms) == ["cross_encoder", "keyword_overlap", "keyword_overlap_bm25"]

    def changed(arm, base):
        return {k for k in set(arm) | set(base) if arm.get(k) != base.get(k)}

    # What the notebook says: each arm is one change away from the one before.
    assert arms["cross_encoder"]["reranker"] == "cross_encoder"
    assert changed(arms["keyword_overlap"], arms["cross_encoder"]) == {"reranker"}
    assert arms["keyword_overlap"]["reranker"] == "my_ideas:keyword_overlap"
    assert changed(arms["keyword_overlap_bm25"], arms["keyword_overlap"]) == {"retriever"}
    assert arms["keyword_overlap_bm25"]["retriever"] == "bm25"


def test_the_factorial_parses_into_four_cells(tmp_path, written):
    from lasagna.knowledge_evaluation_sdk import Experiment
    from lasagna.knowledge_evaluation_sdk.experiment import _generate_pipeline_configs

    config = Experiment.from_file(str(_write(tmp_path, written, "compose.yaml"))).config
    cells = _generate_pipeline_configs(config.pipeline)
    assert len(cells) == 4
    # Held fixed means the same in every cell, including the chunking that
    # lets the factorial reuse the sweep's corpus.
    sweep_chunking = {"chunker": "fixed", "chunk_size": 2000}
    for cell in cells:
        assert cell["overrides"]["reranker"] == "my_ideas:keyword_overlap"
        assert {k: cell["overrides"][k] for k in sweep_chunking} == sweep_chunking


def test_the_technique_passes_the_reranker_contract(tmp_path, written):
    from lasagna import components
    from lasagna.core.registry import registry
    from lasagna.core.testing import ComponentTestHarness

    path = _write(tmp_path, written, "my_ideas.py")
    try:
        # By file path, as a private module: a bare `my_ideas` would stay in
        # sys.modules and collide with another test's module of that name.
        fn = components.import_reference(f"{path}:keyword_overlap")
        stage, _ = fn.__lasagna_component__
        cls = components.as_component_class(stage, fn, "my_ideas:keyword_overlap")
        report = ComponentTestHarness().validate(cls())
    finally:
        registry.unregister_reranker("keyword_overlap")  # registered by its decorator
    assert stage == "reranker"
    assert report.passed, report.failures


def _cli_command(path):
    import typer

    from lasagna.cli import app

    command = typer.main.get_command(app)
    for name in path:
        command = command.commands[name]
    return command


def test_every_lasagna_command_and_option_it_runs_exists(notebook):
    seen = []
    for line in _shell_lines(notebook):
        words = shlex.split(line.split("|")[0].split("2>&1")[0])
        if words[0] != "lasagna":
            continue
        group = _cli_command(words[1:2])
        path = words[1:3] if hasattr(group, "commands") else words[1:2]
        command = _cli_command(path)
        # A hidden command is a deprecated spelling: it prints a notice in
        # the middle of the notebook's output and goes away in 0.2.0.
        assert not command.hidden, f"`{line}`: `lasagna {' '.join(path)}` is deprecated"
        known = {opt for p in command.params for opt in getattr(p, "opts", [])}
        options = [w for w in words if w.startswith("-")]
        unknown = [o for o in options if o not in known]
        assert not unknown, f"`{line}`: {unknown} not options of `lasagna {' '.join(path)}`"
        seen.append(" ".join(path))
    assert {"component validate", "benchmark experiment", "results significance"} <= set(seen)


def test_the_stages_table_names_real_keys_components_and_decorators(notebook):
    import lasagna
    from lasagna.core.registry import registry
    from lasagna.knowledge_evaluation_sdk.experiment import _COMPONENT_KEYS

    text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    rows = [r for r in text.splitlines() if r.startswith("| ") and "`" in r]
    assert len(rows) == 7, "one row per sweepable stage"
    for row in rows:
        stage, key, builtins, own = [c.strip() for c in row.strip("|").split("|")]
        stage = stage.replace(" ", "_")
        assert key == f"`{stage}`" and stage in _COMPONENT_KEYS, row
        for name in re.findall(r"`(\w+)`", builtins):
            assert registry.has(stage, name), f"{stage} {name!r} is not registered"
        for decorator in re.findall(r"@lasagna\.(\w+)", own):
            assert decorator == stage and callable(getattr(lasagna, decorator)), row


def test_every_extra_and_repository_link_exists(notebook):
    with (REPO / "pyproject.toml").open("rb") as f:
        extras = set(tomllib.load(f)["project"]["optional-dependencies"])
    text = "\n".join(c.source for c in notebook.cells)

    named = {e for group in re.findall(r"lasagna-rag\[([\w,-]+)\]", text) for e in group.split(",")}
    assert named and named <= extras, named - extras

    links = re.findall(r"https://github\.com/dolev31/lasagna/(?:blob|tree)/main/([^)#\s]+)", text)
    assert links
    missing = [link for link in links if not (REPO / link).exists()]
    assert not missing, missing
