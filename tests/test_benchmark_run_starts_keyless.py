"""``lasagna benchmark run`` starts from the keyless speed profile, and says each failure once.

It defaulted to ``standard`` (BGE-M3, about 3 GB, and an agentic loop that
needs an LLM) while every document said speed, so ``lasagna benchmark run
smoke`` on a fresh install downloaded gigabytes or failed offline. A run whose
every task failed also ended with a ``RuntimeWarning`` and the source line that
raised it, repeating the failure the CLI had already printed.
"""

from __future__ import annotations

import warnings

import pytest
from typer.testing import CliRunner

from lasagna import cli


class _Stop(Exception):
    pass


def _profile_used(monkeypatch, *args):
    from lasagna.knowledge_evaluation_sdk import experiment

    seen = {}

    def fake_experiment(config):
        seen["profile"] = config.pipeline.base_profile
        raise _Stop

    monkeypatch.setattr(experiment, "Experiment", fake_experiment)
    CliRunner().invoke(cli.app, ["benchmark", "run", "smoke", *args])
    return seen["profile"]


def test_the_default_profile_is_speed(monkeypatch):
    assert _profile_used(monkeypatch) == "speed"


def test_standard_is_still_there_for_who_asks(monkeypatch):
    assert _profile_used(monkeypatch, "--profile", "standard") == "standard"


def test_the_cli_does_not_repeat_an_all_failed_run_as_a_warning(monkeypatch):
    monkeypatch.delenv("RAG_SESSION_BACKEND", raising=False)  # the callback sets it
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli._cli(verbose=False, version=False)
        warnings.warn("All 1 task(s) of 'benchmark_smoke' failed. First: ...", RuntimeWarning)
        warnings.warn("something else", RuntimeWarning)
    assert [str(w.message) for w in caught] == ["something else"]


@pytest.fixture(autouse=True)
def _restore_warning_filters():
    saved = warnings.filters[:]
    yield
    warnings.filters[:] = saved
