"""`lasagna component list` puts a paper under "Paper" and a docs page under "Docs".

The "Paper" column used to show the OpenAI and Cohere docs pages, LangChain
GitHub links and a blog post, while the README said 30 of 55 components cite
the paper they implement. One function now decides what a paper is, and the
list and the README count both use it.
"""

from __future__ import annotations

from typer.testing import CliRunner

from lasagna.cli import app
from lasagna.core.registry import is_paper_citation, registry


def test_what_counts_as_a_paper():
    assert is_paper_citation("arxiv:2004.04906")
    assert is_paper_citation("doi:10.1561/1500000019")
    assert is_paper_citation("Wei et al. 2022, Chain-of-Thought Prompting")
    assert not is_paper_citation("https://platform.openai.com/docs/guides/embeddings")
    assert not is_paper_citation("https://github.com/microsoft/markitdown")
    assert not is_paper_citation("")
    assert not is_paper_citation(None)


def test_every_listed_source_is_in_the_column_that_names_it(monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    result = CliRunner().invoke(app, ["component", "list", "embedder"])
    assert result.exit_code == 0, result.output
    header = next(line for line in result.output.splitlines() if "Paper" in line)
    assert "Docs" in header
    paper_at, docs_at = header.index("Paper"), header.index("Docs")
    for entry in registry.list_components("embedder"):
        source = entry.get("paper") or ""
        row = next(line for line in result.output.splitlines() if f" {entry['name']} " in line)
        if is_paper_citation(source):
            assert source in row[paper_at:docs_at], row
        else:
            assert source in row[docs_at:], row
    assert "openai.com/docs" in result.output
    assert "arxiv:2402.03216" in result.output
