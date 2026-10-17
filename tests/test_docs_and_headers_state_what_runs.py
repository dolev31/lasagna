"""Docs, help and headers say what runs, and only the numbers that were measured.

* configuration.md's `default` row left out the hierarchical chunking the
  settings file turns on.
* `configure --help` listed four presets; `default` works too.
* `benchmark run` starts from speed and a sweep without `base_profile:` from
  standard, and neither `--dry-run` nor the run header said which an arm used.
* The fixes in error messages send programs to `POST
  /api/config/knowledge/reset`, unversioned while everything else is under
  `/api/v1`, and the API docs did not say why.
* `init` ended "Setup complete!".
* The settings file and the SDK presets stated latencies and Adaptive-RAG
  percentages that nothing in the repository measured or cited.
"""

from __future__ import annotations

import asyncio
import re
import sys

import pytest
import yaml
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import cost
from lasagna.knowledge_evaluation_sdk import experiment as exp
from tests._paths import PACKAGE_DIR

REPO = PACKAGE_DIR.parents[1]
runner = CliRunner()


def test_the_default_row_names_hierarchical_chunking_as_the_settings_file_runs_it():
    from lasagna.core.config import rag_config

    assert rag_config.enable_hierarchical is True
    docs = (REPO / "docs" / "configuration.md").read_text(encoding="utf-8")
    row = next(line for line in docs.splitlines() if line.startswith("| `default` |"))
    assert "hierarchical chunking" in row


def test_configure_help_lists_the_default_profile():
    out = runner.invoke(cli.app, ["configure", "--help"], env={"COLUMNS": "300"})
    words = " ".join(out.output.replace("│", " ").split())
    assert "speed, standard, balanced, max_quality, or default (the settings file)" in words


def _sweep(tmp_path, **pipeline):
    path = tmp_path / "sweep.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "starts",
                "benchmarks": [{"name": "smoke", "n_samples": 2}],
                "tracks": ["retrieval"],
                "pipeline": pipeline,
            }
        )
    )
    return path


def test_the_dry_run_names_the_profile_every_arm_starts_from(tmp_path, monkeypatch):
    monkeypatch.setattr(cost, "_hf_cached", lambda name: True)
    sweep = exp.Experiment.from_file(str(_sweep(tmp_path, rerankers=["cross_encoder", "mmr"])))
    lines = cost.dry_run_lines(sweep, probe=False)
    assert (
        "Base profile: standard, for every arm (pipeline.base_profile, standard when unset; "
        "an arm's own profile: wins)"
    ) in lines

    mixed = exp.Experiment.from_file(
        str(
            _sweep(
                tmp_path,
                base_profile="speed",
                config_overrides=[{"name": "a"}, {"name": "b", "profile": "standard"}],
            )
        )
    )
    assert "Base profiles: a from speed, b from standard" in cost.dry_run_lines(mixed, probe=False)


def test_the_run_header_names_it_too(tmp_path, monkeypatch):
    async def stop(self, progress_callback=None):
        raise RuntimeError("stopped after the header")

    monkeypatch.setattr(exp.Experiment, "run", stop)
    monkeypatch.setattr(cli, "_require_vector_backend", lambda: None)
    out = runner.invoke(
        cli.app,
        ["benchmark", "experiment", str(_sweep(tmp_path, base_profile="speed"))],
        env={"COLUMNS": "300", "HOME": str(tmp_path)},
    )
    assert "stopped after the header" in out.output, out.output
    assert "Base profile: speed, for every arm" in out.output


def test_the_unversioned_reset_route_is_documented_as_the_configurators():
    endpoints = (PACKAGE_DIR / "api" / "docs" / "endpoints.md").read_text(encoding="utf-8")
    words = " ".join(endpoints.split())
    assert (
        "One has no `/api/v1` counterpart in 0.1.0: clearing a session's overrides is "
        "`POST /api/config/knowledge/reset`"
    ) in words
    source = (PACKAGE_DIR / "api" / "rest" / "routes" / "config.py").read_text()
    assert "POST /api/config/knowledge/reset (the configurator's" in source


INIT = """
import sys
import questionary
from typer.testing import CliRunner

class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value

answers = iter(["milvus", "speed"])
questionary.select = lambda *a, **k: _Answer(next(answers))
from lasagna import cli

print(CliRunner().invoke(cli.app, ["init"], env={"COLUMNS": "200"}).output)
"""


def test_init_ends_plainly(tmp_path):
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent))
    done = subprocess.run(
        [sys.executable, "-c", INIT], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert "Setup done." in done.stdout, done.stdout + done.stderr[-2000:]
    assert "Setup complete!" not in done.stdout


#: Claims of speed or gain that nothing in the repository measured.
_UNSOURCED = re.compile(
    r"Target latency|~\d+-\d+ ?ms|\+\d+-\d+%|\+\d+-\d+ms|30-40% latency|28% cost|97\.9%|<1ms"
)


@pytest.mark.parametrize("path", ["settings.lasagna.toml", "sdk/config.py"])
def test_no_unmeasured_latency_or_gain_is_stated(path):
    text = (PACKAGE_DIR / path).read_text(encoding="utf-8")
    found = [line.strip() for line in text.splitlines() if _UNSOURCED.search(line)]
    assert not found, found


def test_the_routing_comment_cites_adaptive_rag_and_claims_nothing_for_the_rules():
    text = (PACKAGE_DIR / "settings.lasagna.toml").read_text(encoding="utf-8")
    assert "arXiv:2403.14403" in text
    assert "measured with a sweep, not assumed" in text
    assert asyncio  # the module is imported for the run-header stub
