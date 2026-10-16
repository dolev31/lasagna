"""A run that generated answers says, beside its leaderboard, what the LLM cache replayed.

The hit rate was recorded in results.json and the bundle, but nothing printed
it, so a warm re-run that replayed every answer in seconds read exactly like
one that called the model for each. A retrieval-only run prints nothing.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("langchain_core")

from lasagna import cli  # noqa: E402
from tests.test_llm_cache_is_invisible_on_a_cold_cache import _cache_at, calls  # noqa: E402, F401


def _experiment(tmp_path, *args):
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "name": "cli",
                "benchmarks": [{"name": "smoke", "n_samples": 2}],
                "mode": "generation_only",
                "output_dir": str(tmp_path / "out"),
                "llm_cache": _cache_at(tmp_path),
            }
        )
    )
    result = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep), *args])
    assert result.exit_code == 0, result.output
    return result.output


def test_a_cold_then_a_warm_run_say_what_they_replayed(tmp_path, calls):  # noqa: F811
    cold = _experiment(tmp_path)
    assert "LLM cache: 0 hits, 2 misses (0%)" in cold
    assert cold.index("Results (") < cold.index("LLM cache:")
    warm = _experiment(tmp_path)
    assert "LLM cache: 2 hits, 0 misses (100%)" in warm
    assert len(calls) == 2


def test_with_the_cache_off_nothing_is_said(tmp_path, calls):  # noqa: F811
    assert "LLM cache" not in _experiment(tmp_path, "--no-cache")


@pytest.mark.parametrize(
    "meta, line",
    [
        (
            {"tracks": ["generation"], "llm_cache": {"enabled": True, "hits": 412, "misses": 37}},
            "LLM cache: 412 hits, 37 misses (92%)",
        ),
        (
            {
                "tracks": ["retrieval", "generation"],
                "llm_cache": {"enabled": True, "hits": 1, "misses": 1},
            },
            "LLM cache: 1 hit, 1 miss (50%)",
        ),
        # Retrieval only: an LLM query enhancer may have looked things up; still nothing.
        ({"tracks": ["retrieval"], "llm_cache": {"enabled": True, "hits": 3, "misses": 1}}, ""),
        ({"tracks": ["generation"], "llm_cache": {"enabled": True, "hits": 0, "misses": 0}}, ""),
        ({"tracks": ["generation"], "llm_cache": {"enabled": False, "hits": 0, "misses": 0}}, ""),
        ({"tracks": ["generation"]}, ""),
    ],
)
def test_the_line(meta, line):
    assert cli._llm_cache_line(meta) == line
