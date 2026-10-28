"""The CLI's isolation line claims only what was compared.

It printed "arms differ only in the stages they name" after runs where every
task failed, or only one arm finished: a green line about nothing.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

from rich.console import Console

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.stages import stage_manifest


class _Config:
    def __init__(self, **values):
        self._config = values


def _row(name, overrides, error=None, manifest=True):
    return {
        "benchmark_name": "b",
        "config_name": name,
        "config_overrides": overrides,
        "profile": "speed",
        "error": error,
        "system": (
            {
                "stages": stage_manifest(
                    _Config(chunk_size=800), {"reranker": overrides.get("reranker")}
                )
            }
            if manifest
            else {}
        ),
    }


def _summary(monkeypatch, rows):
    out = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, width=200))
    cli._print_isolation(SimpleNamespace(_results=rows))
    return out.getvalue()


def test_two_clean_arms_get_the_green_line(monkeypatch):
    text = _summary(monkeypatch, [_row("a", {}), _row("b", {"reranker": "x"})])
    assert "differs only in the settings the arms name (1 pair compared)" in text


def test_one_finished_arm_is_not_called_isolated(monkeypatch):
    text = _summary(monkeypatch, [_row("a", {}), _row("b", {"reranker": "x"}, error="boom")])
    assert "differ only" not in text and "not checked" in text


def test_runs_without_manifests_say_so(monkeypatch):
    text = _summary(monkeypatch, [_row("a", {}, manifest=False), _row("b", {}, manifest=False)])
    assert "differ only" not in text and "unverified" in text
