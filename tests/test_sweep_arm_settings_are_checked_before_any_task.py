"""An arm's settings are checked before any corpus is ingested, against one set of ranges.

`{chunk_size: -5}` and `{hybrid_alpha: 7}` passed `--dry-run` with exit 0. The
real run then failed two tasks of three, each when it applied its settings,
and still printed "Next: lasagna results significance". The ranges also
disagreed: an arm ran `chunk_size: 10`, while `RAG_CHUNK_SIZE=10` was refused
with "takes 50 to 100000". Every arm is now validated up front by the
validator its task applies, and the environment's ranges are the schema's.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.experiment import (
    Experiment,
    _generate_pipeline_configs,
    _parse_config,
)

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()

BAD = {
    "name": "bad arms",
    "benchmarks": [{"name": "smoke"}],
    "tracks": ["retrieval"],
    "pipeline": {
        "base_profile": "speed",
        "config_overrides": [
            {"name": "ok"},
            {"name": "neg_chunk", "chunk_size": -5},
            {"name": "alpha7", "hybrid_alpha": 7},
        ],
    },
}


def _check_override_values(configs):
    from lasagna.knowledge_evaluation_sdk.experiment import _check_override_values

    _check_override_values(configs)


def _sweep_file(tmp_path, data) -> str:
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump({**data, "output_dir": str(tmp_path / "runs")}))
    return str(path)


def test_generate_tasks_names_each_bad_arm_and_setting():
    with pytest.raises(ValueError) as raised:
        Experiment(config=_parse_config(copy.deepcopy(BAD))).generate_tasks()
    message = str(raised.value)
    assert "arm 'neg_chunk': chunk_size: Value -5 is below minimum 1" in message
    assert "arm 'alpha7': hybrid_alpha: Value 7 exceeds maximum 1.0" in message
    assert "'ok'" not in message


@pytest.mark.parametrize("extra", [["--dry-run", "--no-probe"], []], ids=["dry run", "run"])
def test_the_dry_run_and_the_run_refuse_before_any_folder_is_made(tmp_path, extra):
    result = runner.invoke(cli.app, ["benchmark", "experiment", _sweep_file(tmp_path, BAD), *extra])
    words = " ".join(result.output.split())
    assert result.exit_code == 1, words
    assert "arm 'neg_chunk': chunk_size" in words and "arm 'alpha7': hybrid_alpha" in words
    assert "Starting experiment" not in words and "Next:" not in words
    assert not (tmp_path / "runs").exists()


def test_a_small_chunk_size_is_valid_for_an_arm_and_for_the_environment():
    from lasagna.core.config import _parse_env_value

    arm = copy.deepcopy(BAD)
    arm["pipeline"]["config_overrides"] = [{"name": "tiny", "chunk_size": 10}]
    _check_override_values(_generate_pipeline_configs(_parse_config(arm).pipeline))
    assert _parse_env_value("chunk_size", "10", 800) == 10


def test_the_environment_and_the_schema_have_the_same_ranges():
    from lasagna.configurator.schema import PARAMETERS
    from lasagna.core.config import _VALID_RANGES

    schema = {p.key: (p.min_value, p.max_value) for p in PARAMETERS}
    for key, bounds in _VALID_RANGES.items():
        if key in schema:
            assert tuple(bounds) == schema[key], key


def _shipped_sweeps():
    files = sorted((REPO / "examples").rglob("*.yaml"))
    files += sorted(
        (REPO / "src/lasagna/knowledge_evaluation_sdk/research/experiments").glob("*.yaml")
    )
    for path in files:
        yield str(path.relative_to(REPO)), yaml.safe_load(path.read_text())
    docs = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]
    docs += sorted((REPO / "src/lasagna/skills").rglob("SKILL.md"))
    for doc in docs:
        for n, block in enumerate(re.findall(r"```yaml\n(.*?)```", doc.read_text(), re.S)):
            data = yaml.safe_load(block)
            if isinstance(data, dict) and "benchmarks" in data:
                yield f"{doc.relative_to(REPO)} (yaml block {n + 1})", data


def test_no_shipped_sweep_is_refused():
    sweeps = list(_shipped_sweeps())
    assert any("reproduction/beir_tier1.yaml" in name for name, _ in sweeps)
    for name, data in sweeps:
        configs = _generate_pipeline_configs(_parse_config(data).pipeline)
        try:
            _check_override_values(configs)
        except ValueError as exc:
            pytest.fail(f"{name}: {exc}")


@pytest.mark.parametrize("profile", ["default", "speed", "standard", "balanced", "max_quality"])
def test_every_profile_passes_the_check_it_is_merged_into(profile):
    from lasagna.configurator.validation import validate_config
    from lasagna.core.pipeline import KnowledgePipeline

    assert validate_config(dict(KnowledgePipeline.from_profile(profile)._config)).errors == []
