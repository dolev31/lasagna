"""Agentic components must be checked, not rejected as an unknown stage."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.agentic.agentic_rag import AgenticRAGResult
from lasagna.core.base_components import BaseAgenticPipeline
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness


class GoodPipeline(BaseAgenticPipeline):
    async def run(self, input):
        return AgenticRAGResult(
            formatted_context="", chunks=[], citations=[], agent_summary="", search_count=0
        )


class WrongResult(GoodPipeline):
    async def run(self, input):
        return "not an AgenticRAGResult"


class Mutator(GoodPipeline):
    async def run(self, input):
        input.document_filter.append("changed")
        input.session_config._query_overrides["changed"] = True
        return await super().run(input)


def test_an_agentic_pipeline_validates():
    report = ComponentTestHarness().validate(GoodPipeline())
    assert report.component_type == "agentic_pipeline"
    assert report.passed, report.summary()
    assert any("does not modify its input" in c for c in report.checks)


def test_wrong_result_names_the_expected_type():
    report = ComponentTestHarness().validate(WrongResult())
    assert not report.passed
    assert any("Expected AgenticRAGResult, got str" in f for f in report.failures)


def test_nested_input_mutation_is_rejected():
    report = ComponentTestHarness().validate(Mutator())
    assert not report.passed
    assert any("modified" in f for f in report.failures)


async def test_validation_also_works_inside_a_running_event_loop():
    report = ComponentTestHarness().validate(GoodPipeline())
    assert report.passed, report.summary()


def test_infrastructure_errors_are_not_reported_as_contract_failures():
    class Unavailable(GoodPipeline):
        async def run(self, input):
            raise ConnectionError("backend unavailable")

    with pytest.raises(ConnectionError, match="backend unavailable"):
        ComponentTestHarness().validate(Unavailable())


def test_self_ask_validates_from_its_import_path():
    result = CliRunner().invoke(
        cli.app,
        ["component", "validate", "lasagna.agentic.loops.self_ask:SelfAskPipeline"],
    )
    assert result.exit_code == 0, result.output
    assert "Contract satisfied" in result.output


def test_registered_agentic_stage_is_accepted():
    result = CliRunner().invoke(
        cli.app, ["component", "validate", "self_ask", "--stage", "agentic_pipeline"]
    )
    assert result.exit_code == 0, result.output


def test_plain_agentic_functions_are_refused_with_the_class_to_implement(tmp_path):
    module = tmp_path / "agentic_function.py"
    module.write_text("async def run(input):\n    return None\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.app, ["component", "validate", f"{module}:run", "--stage", "agentic_pipeline"]
    )
    assert result.exit_code == 1
    assert "BaseAgenticPipeline" in result.output
    assert "class" in result.output


@pytest.mark.parametrize(
    "name", ["default", "legacy_react", "self_ask", "ircot", "flare", "adaptive", "self_rag"]
)
def test_built_in_pipeline_against_stub_dependencies(name, monkeypatch):
    from tests._loop_stubs import install_backend

    install_backend(monkeypatch, {})
    report = ComponentTestHarness().validate(registry.get("agentic_pipeline", name))
    assert report.passed, report.summary()
