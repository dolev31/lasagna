"""A component's claimed number appears beside the measured one.

The README and CONTRIBUTING promise that ``paper`` and ``claims`` "travel with
the component", "so a results table can put the claimed number beside the one
you reproduced". No table did: the run metadata did not record them, and a
sweep naming ``my_ideas:longer_first`` re-registered the function under that
name without the claims its decorator declared.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from lasagna import components
from lasagna.cli import app
from lasagna.core.registry import registry
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from lasagna.knowledge_evaluation_sdk.run_provenance import claimed, component_provenance
from tests._paths import PACKAGE_DIR

MY_IDEAS = """
import lasagna


@lasagna.reranker(paper="Doe et al. 2026", claims={"smoke": {"ndcg@10": 0.456}})
def longer_first(query, text):
    return len(text.split())
"""

CLAIM = {"stage": "reranker", "paper": "Doe et al. 2026", "claims": {"bench": {"ndcg@10": 0.456}}}
# What the README sweep records: the claim names the benchmark it runs.
SMOKE_CLAIM = {**CLAIM, "claims": {"smoke": {"ndcg@10": 0.456}}}


@pytest.fixture
def my_ideas(tmp_path):
    (tmp_path / "my_ideas.py").write_text(MY_IDEAS)
    yield tmp_path
    for name in ("longer_first", "my_ideas:longer_first"):
        registry._unregister("reranker", name)
    sys.modules.pop("my_ideas", None)
    components._LOADED_FROM.pop(("reranker", "my_ideas:longer_first"), None)


def test_a_reference_keeps_the_claims_its_decorator_declared(my_ideas):
    components.load("reranker", "my_ideas:longer_first", [my_ideas])
    declared = registry.provenance("reranker", "my_ideas:longer_first")
    assert declared == {"paper": "Doe et al. 2026", "claims": {"smoke": {"ndcg@10": 0.456}}}
    rows = [{"system": {"stages": {"reranker": {"component": "my_ideas:longer_first"}}}}]
    assert component_provenance(rows) == {"my_ideas:longer_first": SMOKE_CLAIM}


def test_a_claim_matches_its_metric_and_benchmark_only():
    record = {"mine": CLAIM}
    system = {"stages": {"reranker": {"component": "mine"}}}
    assert claimed(record, system, "NDCG@10", "bench")["value"] == 0.456
    assert claimed(record, system, "recall@10", "bench") is None
    assert claimed(record, system, "ndcg@10", "smoke") is None
    other = {"stages": {"reranker": {"component": "other"}}}
    assert claimed(record, other, "ndcg@10", "bench") is None


def _task(config: str, component: str, score: float) -> dict:
    return {
        "task_key": f"bench|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": config,
        "metrics": {"ndcg@10": score},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "ndcg@10": score, "_retrieved_ids": [f"{config}-{i}"]}
            for i in range(10)
        ],
        "error": None,
        "config_overrides": {"reranker": component},
        "profile": "speed",
        "system": {"stages": {"reranker": {"component": component}}},
    }


@pytest.fixture
def results_json(tmp_path):
    path = tmp_path / "run" / "results.json"
    path.parent.mkdir()
    config = {
        "experiment_name": "claims",
        "n_runs": 1,
        "tracks": ["retrieval"],
        "metrics_version": METRICS_VERSION,
        "components": {"my_ideas:longer_first": CLAIM},
    }
    tasks = [
        _task("control", "cross_encoder", 0.50),
        _task("mine", "my_ideas:longer_first", 0.41),
    ]
    path.write_text(
        json.dumps({"experiment_name": "claims", "experiment_config": config, "results": tasks})
    )
    return path


def test_every_results_table_shows_the_claim_beside_the_measurement(results_json, tmp_path):
    runner = CliRunner()
    text = runner.invoke(app, ["results", "leaderboard", str(results_json), "-m", "ndcg@10"])
    assert text.exit_code == 0, text.output
    mine = next(line for line in text.output.splitlines() if "mine" in line)
    control = next(line for line in text.output.splitlines() if "control" in line)
    assert "0.4100" in mine and "claimed 0.4560 (Doe et al. 2026)" in mine
    assert "claimed" not in control

    latex = runner.invoke(app, ["results", "leaderboard", str(results_json), "--format", "latex"])
    assert latex.exit_code == 0, latex.output
    assert "Claimed" in latex.output and "0.4560 (Doe et al. 2026)" in latex.output

    pages = tmp_path / "pages"
    md = runner.invoke(app, ["results", "leaderboard", str(results_json), "-o", str(pages)])
    assert md.exit_code == 0, md.output
    page = (pages / "bench.md").read_text()
    assert "| Claimed |" in page and "0.4560 (Doe et al. 2026)" in page

    html = tmp_path / "view.html"
    view = runner.invoke(app, ["results", "view", str(results_json), "-o", str(html)])
    assert view.exit_code == 0, view.output
    assert "<th>Claimed</th>" in html.read_text()
    assert "0.4560 (Doe et al. 2026)" in html.read_text()

    bundle = tmp_path / "run.json"
    made = runner.invoke(app, ["results", "bundle", str(results_json), "-o", str(bundle)])
    assert made.exit_code == 0, made.output
    assert json.loads(bundle.read_text())["experiment"]["components"] == {
        "my_ideas:longer_first": CLAIM
    }


SWEEP = """
name: does longer_first help
benchmarks: [{name: smoke}]
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, "my_ideas:longer_first"]
n_runs: 1
output_dir: out
"""


def _cached(model: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


@pytest.mark.skipif(
    not (
        _cached("sentence-transformers/all-MiniLM-L6-v2")
        and _cached("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ),
    reason="the speed profile's models are not cached",
)
def test_the_readme_sweep_records_and_prints_the_claim(my_ideas):
    (my_ideas / "sweep.yaml").write_text(SWEEP)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(my_ideas / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="200",
    )
    run = subprocess.run(
        [sys.executable, "-m", "lasagna", "benchmark", "experiment", "sweep.yaml"],
        cwd=my_ideas,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "claimed 0.4560 (Doe et al. 2026)" in run.stdout
    saved = json.loads(next((my_ideas / "out").rglob("results.json")).read_text())
    assert saved["experiment_config"]["components"] == {"my_ideas:longer_first": SMOKE_CLAIM}
