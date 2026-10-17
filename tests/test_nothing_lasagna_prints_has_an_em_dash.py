"""Nothing Lasagna prints, writes or describes to a user carries an em dash.

The house style is plain punctuation (a colon, a comma, parentheses), and the
em dash was everywhere the user looks: the leaderboard's title, empty table
cells, `results compare`, warnings and errors, benchmark descriptions in
`benchmark list` and DATASETS.md, the configurator, the REST API docs and the
MCP tool descriptions an agent reads.

Two halves. The rendered output: every command's `--help`, the significance
box, the leaderboards, the HTML view, the OpenAPI document, the MCP tools and
the generated pages. And the source: every string literal that is not a
docstring, plus the docstrings that become help, tool or route descriptions.
Comments and other docstrings may keep theirs. So may the agentic prompts,
which are model input: rewording one changes what the model answers and the
prompt fingerprint in the comparability key.
"""

from __future__ import annotations

import ast
import re
import asyncio
import json

import pytest
import typer
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from tests._paths import PACKAGE_DIR

DASH = "—"
REPO = PACKAGE_DIR.parents[1]
runner = CliRunner()

#: Model input, left as it is: see the module docstring.
PROMPTS = {
    ("agentic/nodes/generate.py", "_GENERATION_PROMPT"),
    ("agentic/nodes/quality_gates.py", "_HALLUCINATION_PROMPT"),
    ("agentic/nodes/quality_gates.py", "_ANSWER_QUALITY_PROMPT"),
    ("agentic/nodes/rewrite_query.py", "_build_missing_info"),
}

#: Decorators whose function's docstring a user reads: command help, MCP tool
#: descriptions, REST route descriptions.
_SHOWN = (".command", ".callback", "tool", "router.", "app.get", "app.post")


def _commands(group, path=()):
    yield path, group
    for name, command in getattr(group, "commands", {}).items():
        yield from _commands(command, (*path, name))


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def _owner(parents, node):
    """The top-level name a node sits under: an assignment target or a function."""
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
    return ""


def test_no_string_in_the_source_has_one():
    found = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if DASH not in source:
            continue
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        tree = ast.parse(source)
        docstrings = _docstring_nodes(tree)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                shown = any(
                    marker in ast.unparse(d) for d in node.decorator_list for marker in _SHOWN
                )
                doc = ast.get_docstring(node, clean=False) or ""
                if shown and DASH in doc:
                    found.append(f"{rel}:{node.lineno} docstring of {node.name}")
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if DASH not in node.value or id(node) in docstrings:
                continue
            if (rel, _owner(parents, node)) in PROMPTS:
                continue
            found.append(f"{rel}:{node.lineno} {node.value.strip()[:60]!r}")
    assert not found, "\n".join(found)


def test_no_command_help_has_one():
    root = typer.main.get_command(cli.app)
    with_dash = []
    for path, _ in _commands(root):
        result = runner.invoke(cli.app, [*path, "--help"], terminal_width=200)
        assert result.exit_code == 0, (path, result.output)
        if DASH in result.output:
            with_dash.append(" ".join(path) or "lasagna")
    assert not with_dash, with_dash


def test_the_mcp_server_help_and_tools_have_none(monkeypatch, capsys):
    pytest.importorskip("mcp")
    from lasagna.mcp import server

    monkeypatch.setattr("sys.argv", ["lasagna-mcp", "--help"])
    with pytest.raises(SystemExit):
        server.main()
    assert DASH not in capsys.readouterr().out

    srv = server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
    for tool in asyncio.run(srv.list_tools()):
        assert DASH not in (tool.description or ""), tool.name
        assert DASH not in json.dumps(tool.inputSchema, ensure_ascii=False), tool.name


def test_the_rest_api_document_has_none(monkeypatch):
    for name in ("RAG_EMBED_MCP", "RAG_CONFIGURATOR", "RAG_REQUIRE_AUTH"):
        monkeypatch.delenv(name, raising=False)
    from lasagna.api.rest.app import create_app

    assert DASH not in json.dumps(create_app().openapi(), ensure_ascii=False)


@pytest.mark.parametrize(
    "snapshot", ["tests/snapshots/mcp_tools.json", "tests/snapshots/openapi.json"]
)
def test_the_snapshots_have_none(snapshot):
    text = (REPO / snapshot).read_text(encoding="utf-8")
    assert DASH not in json.dumps(json.loads(text), ensure_ascii=False)


def _task(config, scores, extra=None):
    return {
        "task_key": f"bench|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": config,
        "metrics": {"ndcg@10": sum(scores) / len(scores), "latency_ms": 10.0, **(extra or {})},
        "per_sample_scores": [
            {
                "_sample_id": f"q{i}",
                "ndcg@10": s,
                "_retrieved_ids": [f"{config}-{i}"],
                "_question": f"question {i}",
                "_retrieved_contexts": [f"passage {i}"],
            }
            for i, s in enumerate(scores)
        ],
        "error": None,
        "config_overrides": {"reranker": config},
        "profile": "speed",
    }


@pytest.fixture
def run(tmp_path):
    """Two arms, one of which records a metric the other does not, so the
    tables have empty cells."""
    config = {
        "experiment_name": "dashes",
        "created_at": "2026-10-15T18:51:38+00:00",
        "n_runs": 1,
        "tracks": ["retrieval"],
        "metrics_version": METRICS_VERSION,
    }
    tasks = [
        _task("control", [0.4 + 0.01 * i for i in range(30)]),
        _task("mine", [0.5 + 0.01 * i for i in range(30)], {"mrr": 0.7}),
    ]
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"experiment_name": "dashes", "experiment_config": config, "results": tasks})
    )
    return path


def test_the_reports_of_a_run_have_none(run, tmp_path):
    from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
    from lasagna.knowledge_evaluation_sdk.viewer import render_html

    other = tmp_path / "other.json"
    data = json.loads(run.read_text())
    data["results"] = data["results"][:1]
    other.write_text(json.dumps(data))
    commands = [
        ["results", "significance", str(run), "--all"],
        ["results", "leaderboard", str(run)],
        ["results", "leaderboard", str(run), "--format", "latex"],
        ["results", "leaderboard", str(run), "-m", "mrr"],
        ["results", "compare", str(run), str(other)],
        ["results", "per-query", str(run)],
        ["benchmark", "compare", str(run), str(other), "--format", "latex"],
    ]
    for args in commands:
        result = runner.invoke(cli.app, args, terminal_width=200)
        assert result.exit_code == 0, (args, result.output)
        assert DASH not in result.output, (args, result.output)

    results = BenchmarkResults.load(str(run))
    for text in (
        results.leaderboard("mrr"),
        results.to_markdown("mrr"),
        results.to_latex("mrr"),
        results.compare(BenchmarkResults.load(str(other))),
        render_html(run),
        render_html(run, metric="mrr"),
    ):
        assert DASH not in text


def test_the_generated_pages_have_none(tmp_path):
    out = tmp_path / "DATASETS.md"
    result = runner.invoke(cli.app, ["benchmark", "licenses", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert DASH not in out.read_text(encoding="utf-8")
    listed = runner.invoke(cli.app, ["benchmark", "list"], terminal_width=200)
    assert listed.exit_code == 0 and DASH not in listed.output
    for page in (
        "src/lasagna/knowledge_evaluation_sdk/research/DATASETS.md",
        "docs/reference/settings.md",
    ):
        assert DASH not in (REPO / page).read_text(encoding="utf-8"), page


#: Shipped text a user reads as written: every page of the docs, the legal
#: notice, and the issue forms GitHub shows a reporter. The changelog quotes
#: old text, so it is left out.
_SHIPPED_TEXT = sorted(
    {
        *REPO.glob("*.md"),
        *REPO.glob("docs/**/*.md"),
        *REPO.glob("examples/**/*.md"),
        *REPO.glob("hub/*.md"),
        *PACKAGE_DIR.rglob("*.md"),
        REPO / "NOTICE",
        *REPO.glob(".github/ISSUE_TEMPLATE/*.yml"),
    }
    - {REPO / "CHANGELOG.md"}
)

#: " -- " used as a dash between words; an SQL comment ("REAL,  -- DOUBLE")
#: has spaces and capitals around it.
_DOUBLE_HYPHEN = re.compile(r"[\w\"')] -- [a-z]")


@pytest.mark.parametrize("path", _SHIPPED_TEXT, ids=lambda p: str(p.relative_to(REPO)))
def test_no_shipped_page_or_form_has_one(path):
    """The guide's code comments read "handed -- the", and the issue forms and
    NOTICE had em dashes, while the changelog said none were left."""
    text = path.read_text(encoding="utf-8")
    assert DASH not in text
    lines = [line for line in text.splitlines() if _DOUBLE_HYPHEN.search(line)]
    assert not lines, lines
