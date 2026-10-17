"""Five messages that said something vague or wrong now say what they mean.

* `--dry-run` under HF_HUB_OFFLINE=1 said a missing model "downloads on first
  use", which offline it cannot: it says so, with the command that fixes it.
* A `base_profile` typo was reported "in arm 'reranker=cross_encoder'", an arm
  that names no profile: it names the key.
* `results power` with a misspelled arm said "No tasks for (smoke,
  cross_encoder)": it lists the arms and the closest one.
* `results per-query` showed complete_recall@10 and f1@10 first, the first
  names alphabetically: it leads with the run's primary metric.
* The interaction table labelled its first cell "neither  neither": each cell
  is labelled by its factor levels.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna.cli import app
from lasagna.knowledge_evaluation_sdk import cost
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

runner = CliRunner()
N = 20


def _sweep(**pipeline):
    return {
        "name": "messages",
        "benchmarks": [{"name": "smoke", "n_samples": 2}],
        "tracks": ["retrieval"],
        "pipeline": {"base_profile": "standard", **pipeline},
    }


@pytest.mark.parametrize("offline", [True, False])
def test_a_dry_run_says_an_uncached_model_cannot_download_offline(monkeypatch, offline):
    monkeypatch.setattr(cost, "_hf_cached", lambda name: False)
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    lines = cost.dry_run_lines(exp.Experiment.from_dict(_sweep()), probe=False)
    (embedder,) = [line for line in lines if line.strip().startswith("embedder")]
    if offline:
        assert "not cached, and HF_HUB_OFFLINE is set, so it cannot download" in embedder
        assert "run hf download BAAI/bge-m3 --include modules.json" in embedder
        assert "or unset HF_HUB_OFFLINE" in embedder
        assert "downloads on first use" not in embedder
    else:
        assert embedder.endswith("not cached: downloads on first use")


def test_a_base_profile_typo_names_the_key():
    sweep = _sweep(base_profile="spede", rerankers=["cross_encoder", "none"])
    with pytest.raises(ValueError) as refused:
        exp.Experiment.from_dict(sweep).generate_tasks()
    assert str(refused.value).startswith(
        "Unknown profile 'spede' (did you mean 'speed'?) in the sweep's pipeline.base_profile."
    )
    assert "in arm" not in str(refused.value)


def test_an_arms_profile_typo_names_the_arm_and_its_key():
    sweep = _sweep(config_overrides=[{"name": "a", "profile": "bogus"}])
    with pytest.raises(ValueError, match="Unknown profile 'bogus' in the profile: key of arm 'a'"):
        exp.Experiment.from_dict(sweep).generate_tasks()


def _task(config, scores, overrides=None, metrics=("ndcg@10",)):
    samples = [
        {"_sample_id": f"q{i}", **{m: s for m in metrics}, "_retrieved_ids": [f"{config}{i}"]}
        for i, s in enumerate(scores)
    ]
    return {
        "task_key": f"bench|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": config,
        "metrics": {m: sum(scores) / len(scores) for m in metrics},
        "per_sample_scores": samples,
        "error": None,
        "config_overrides": overrides or {},
        "profile": "speed",
    }


def _results_file(tmp_path, tasks):
    path = tmp_path / "results.json"
    config = {"experiment_name": "msgs", "n_runs": 1, "metrics_version": METRICS_VERSION}
    path.write_text(
        json.dumps({"experiment_name": "msgs", "experiment_config": config, "results": tasks})
    )
    return path


def test_a_misspelled_arm_lists_the_arms_and_the_closest(tmp_path):
    path = _results_file(tmp_path, [_task("control", [0.5] * N), _task("cross_encoder", [0.6] * N)])
    results = BenchmarkResults.load(str(path))
    with pytest.raises(ValueError) as refused:
        results.minimum_detectable_effect("bench", "control", "cross_encodr", "ndcg@10")
    assert str(refused.value) == (
        "No arm named 'cross_encodr' (did you mean 'cross_encoder'?) on bench. "
        "Its arms: control, cross_encoder."
    )
    out = runner.invoke(app, ["results", "power", str(path), "control", "cross_encodr"])
    assert out.exit_code == 1
    assert "Its arms: control, cross_encoder." in " ".join(out.output.split())


def test_per_query_leads_with_the_runs_primary_metric(tmp_path):
    metrics = ("complete_recall@10", "f1@10", "hit_rate@10", "map@10", "mrr@10", "ndcg@10")
    path = _results_file(tmp_path, [_task("a", [0.5] * 3, metrics=metrics)])
    out = runner.invoke(app, ["results", "per-query", str(path)], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    header = next(line for line in out.output.splitlines() if "sample" in line)
    columns = [c.strip() for c in header.replace("┃", "|").split("|") if c.strip()]
    assert columns[:3] == ["config", "sample", "ndcg@10"]


def test_the_interaction_table_labels_cells_by_their_levels(tmp_path):
    tasks = [
        _task("neither", [0.40 + 0.001 * i for i in range(N)]),
        _task("reranking", [0.50 + 0.002 * i for i in range(N)]),
        _task("hyde", [0.45 + 0.001 * i for i in range(N)]),
        _task("reranking+hyde", [0.52 + 0.003 * i for i in range(N)]),
    ]
    path = _results_file(tmp_path, tasks)
    out = runner.invoke(app, ["results", "interaction", str(path)], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    text = out.output
    assert "neither  neither" not in text
    for label in (
        "reranking off, hyde off  neither",
        "reranking on, hyde off  reranking",
        "reranking off, hyde on  hyde",
        "reranking on, hyde on  reranking+hyde",
    ):
        assert label in text, label
