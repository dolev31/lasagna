"""`lasagna component validate` checks the component a sweep would run.

Step 2 of the documented path rejected the component written in step 1: it
could not import from the working directory, and a function registered with
`@lasagna.reranker` came back as a bare function ("does not inherit from any
known base class"). And the contract claimed to check input mutation, which
it did not, so a reranker that rewrote its input passed both `validate` and
the sweep pre-flight.
"""

from __future__ import annotations

import textwrap

import pytest
from typer.testing import CliRunner

from lasagna import cli

_IDEAS = textwrap.dedent(
    """
    import lasagna


    @lasagna.reranker("t_validate_overlap")
    def keyword_overlap(query, text):
        return len(set(query.lower().split()) & set(text.lower().split()))


    def plain(query):
        return [query + " extra"]


    def first_context(query, contexts):
        return contexts[0] if contexts else ""
    """
)

_MUTATING = textwrap.dedent(
    """
    from lasagna.core.base_components import BaseReranker


    class Mutator(BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            for d in documents:
                d.score = -1.0
            return list(documents)[:top_k]
    """
)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    import sys

    (tmp_path / "t_validate_ideas.py").write_text(_IDEAS)
    (tmp_path / "t_validate_mutating.py").write_text(_MUTATING)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    # Each test writes the modules into its own folder; a process keeps one
    # file per module name, so forget them between tests.
    for name in ("t_validate_ideas", "t_validate_mutating"):
        sys.modules.pop(name, None)


class _Result:
    def __init__(self, result):
        self.exit_code = result.exit_code
        self.output = " ".join(result.output.split())  # Rich wraps long lines


def _validate(*args):
    return _Result(CliRunner().invoke(cli.app, ["component", "validate", *args]))


def test_a_decorated_function_in_the_working_directory_passes(workdir):
    result = _validate("t_validate_ideas:keyword_overlap")
    assert result.exit_code == 0, result.output
    assert "Contract satisfied" in result.output
    assert "does not modify its input" in result.output


def test_a_file_path_reference_works_too(workdir):
    result = _validate("./t_validate_ideas.py:keyword_overlap")
    assert result.exit_code == 0, result.output


def test_a_plain_function_needs_its_stage(workdir):
    result = _validate("t_validate_ideas:plain")
    assert result.exit_code != 0 and "--stage" in result.output
    result = _validate("t_validate_ideas:plain", "--stage", "query_enhancer")
    assert result.exit_code == 0, result.output


def test_a_reranker_that_rewrites_its_input_fails(workdir):
    result = _validate("t_validate_mutating:Mutator")
    assert result.exit_code == 1
    assert "modified the documents it was handed" in result.output
    assert "pass  rerank() does not modify" not in result.output


def test_the_sweep_pre_flight_refuses_it_too(workdir):
    from lasagna.knowledge_evaluation_sdk.experiment import (
        _check_components_conform,
        _load_component_references,
    )

    configs = [{"name": "m", "overrides": {"reranker": "t_validate_mutating:Mutator"}}]
    _load_component_references(configs, [str(workdir)])
    with pytest.raises(ValueError, match="modified the documents"):
        _check_components_conform(configs)


def test_a_plain_generator_function_validates_with_its_stage(workdir):
    result = _validate("t_validate_ideas:first_context", "--stage", "generator")
    assert result.exit_code == 0, result.output
    assert "generate() does not modify its input" in result.output
