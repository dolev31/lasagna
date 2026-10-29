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


def test_pipeline_can_read_an_ordinary_config_setting():
    from lasagna.core.config import RAGConfig

    class ReadsConfig(GoodPipeline):
        async def run(self, input):
            assert isinstance(input.session_config, RAGConfig)
            assert isinstance(input.session_config.enable_hyde, bool)
            return await super().run(input)

    report = ComponentTestHarness().validate(ReadsConfig())
    assert report.passed, report.summary()


def test_replacing_the_config_reference_is_rejected():
    from lasagna.core.config import RAGConfig

    class ReplacesConfig(GoodPipeline):
        async def run(self, input):
            input.session_config = RAGConfig()
            return await super().run(input)

    report = ComponentTestHarness().validate(ReplacesConfig())
    assert not report.passed
    assert any("modified" in f for f in report.failures)


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


def test_registered_agentic_name_without_a_stage_is_accepted():
    result = CliRunner().invoke(cli.app, ["component", "validate", "self_ask"])
    assert result.exit_code == 0, result.output
    assert "Contract satisfied" in result.output


def test_ambiguous_registered_name_still_requires_a_stage():
    result = CliRunner().invoke(cli.app, ["component", "validate", "default"])
    assert result.exit_code == 1
    assert "agentic_pipeline" in result.output
    assert "pick one with --stage" in result.output


@pytest.mark.parametrize(
    "setting, value, name",
    [
        ("LASAGNA_AGENTIC_MODELS", '{"self_ask": "some-real-model"}', "self_ask"),
        ("LASAGNA_ADAPTIVE_RAG_CLASSIFIER_MODEL", "some-real-model", "adaptive"),
    ],
)
def test_model_settings_cannot_replace_the_validation_stub(monkeypatch, setting, value, name):
    monkeypatch.setenv(setting, value)
    result = CliRunner().invoke(cli.app, ["component", "validate", name])
    assert result.exit_code == 0, result.output
    assert "Contract satisfied" in result.output


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
    "name",
    ["default", "legacy_react", "self_ask", "ircot", "flare", "adaptive", "self_rag", "search_o1"],
)
def test_built_in_pipeline_against_stub_dependencies(name, monkeypatch):
    from tests._loop_stubs import install_backend

    install_backend(monkeypatch, {})
    report = ComponentTestHarness().validate(registry.get("agentic_pipeline", name))
    assert report.passed, report.summary()
