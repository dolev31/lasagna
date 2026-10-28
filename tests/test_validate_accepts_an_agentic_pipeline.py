"""`lasagna component validate` has a case for an `agentic_pipeline`.

The stage was registrable and documented, but the validator knew seven base
classes and not `BaseAgenticPipeline`, so a correct pipeline failed with "does
not inherit from any known base class" followed by a list that left out the one
it did inherit from. The CLI's registered-name lookup also left the stage out,
so `component validate self_ask` said no such component was registered.

A pipeline cannot be run in isolation (it needs a model and a search index), so
the check is its shape: `run` is `async def`, because the orchestrator awaits it,
and takes the input.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from lasagna.core.base_components import BaseAgenticPipeline
from lasagna.core.testing import ComponentTestHarness


class _Good(BaseAgenticPipeline):
    def __init__(self, session_config=None, **kwargs):
        pass

    async def run(self, input):
        return object()


class _SyncRun(BaseAgenticPipeline):
    def run(self, input):  # the orchestrator awaits this, so it cannot be a plain method
        return object()


class _NoInput(BaseAgenticPipeline):
    async def run(self):
        return object()


def test_a_correct_agentic_pipeline_passes():
    report = ComponentTestHarness().validate(_Good())

    assert report.passed, report.failures
    assert report.component_type == "agentic_pipeline"
    assert "PASS: run() is a coroutine function" in report.checks
    assert "PASS: run() accepts the AgenticPipelineInput" in report.checks


def test_a_run_that_is_not_async_fails_and_says_why():
    report = ComponentTestHarness().validate(_SyncRun())

    assert not report.passed
    assert report.component_type == "agentic_pipeline"
    assert any("async def" in failure for failure in report.failures), report.failures


def test_a_run_that_takes_no_input_fails():
    report = ComponentTestHarness().validate(_NoInput())

    assert not report.passed
    assert any("accepts the AgenticPipelineInput" in failure for failure in report.failures)


def test_the_failure_for_a_non_component_still_names_the_agentic_base():
    class NotAComponent:
        pass

    report = ComponentTestHarness().validate(NotAComponent())

    assert not report.passed
    assert "BaseAgenticPipeline" in report.failures[0]


# --- the command, as a contributor runs it -----------------------------------

_PIPELINES = textwrap.dedent(
    """
    from lasagna.core.base_components import BaseAgenticPipeline


    class GoodPipeline(BaseAgenticPipeline):
        def __init__(self, session_config=None, **kwargs):
            pass

        async def run(self, input):
            return None


    class SyncPipeline(BaseAgenticPipeline):
        def run(self, input):
            return None
    """
)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    (tmp_path / "t_agentic_pipelines.py").write_text(_PIPELINES)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    sys.modules.pop("t_agentic_pipelines", None)


def _validate(*args):
    from typer.testing import CliRunner

    from lasagna import cli

    result = CliRunner().invoke(cli.app, ["component", "validate", *args])
    return result.exit_code, " ".join(result.output.split())  # Rich wraps long lines


def test_the_command_accepts_a_pipeline_class(workdir):
    code, output = _validate("t_agentic_pipelines:GoodPipeline")
    assert code == 0, output
    assert "Contract satisfied" in output
    assert "(agentic_pipeline)" in output


def test_the_command_refuses_a_pipeline_whose_run_is_not_async(workdir):
    code, output = _validate("t_agentic_pipelines:SyncPipeline")
    assert code == 1, output
    assert "async def" in output


@pytest.mark.parametrize("name", ["legacy_react", "self_ask"])
def test_the_command_finds_a_registered_pipeline_by_name(name):
    code, output = _validate(name)
    assert code == 0, output
    assert "Contract satisfied" in output


def test_stage_picks_the_pipeline_when_a_name_is_registered_for_several_stages():
    # `default` is an embedder, a document processor and an agentic pipeline.
    code, output = _validate("default")
    assert code != 0 and "--stage" in output

    code, output = _validate("default", "--stage", "agentic_pipeline")
    assert code == 0, output
    assert "(agentic_pipeline)" in output
