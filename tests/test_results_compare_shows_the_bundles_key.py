"""`results compare` shows the comparability key every other command shows.

For one run, `results bundle`, `results leaderboard` and the LaTeX captions
printed one key, and `results compare` another: it hashed a held-fixed block of
its own, without the run's metadata when it read a bundle. It said "Comparable:
yes" for a bundle `results validate` calls INVALID (a number edited after the
run), and its header read "Comparison:  vs " for a bundle, which keeps its
experiment name elsewhere. Compare now takes the keys from the function the
bundle uses, refuses an invalid bundle, and names both inputs.
"""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

runner = CliRunner()

SYSTEM = {
    "embedder_model": "all-MiniLM-L6-v2",
    "embedder_dim": 384,
    "chunker": "structure_aware",
    "retriever_top_k": 40,
    "vector_backend": "milvus",
}


def _row(benchmark: str, config: str, ndcg: float) -> dict:
    return {
        "task_key": f"{benchmark}|{config}|run_0",
        "benchmark_name": benchmark,
        "config_name": config,
        "run_index": 0,
        "metrics": {"ndcg@10": ndcg},
        "per_sample_scores": [{"_sample_id": "q1", "ndcg@10": ndcg}],
        "n_samples": 1,
        "split": "test",
        "sampling_seed": 42,
        "sample_id_digest": f"digest-{benchmark}",
        "corpus_fingerprint": f"corpus-{benchmark}",
        "corpus_provenance": "native",
        "resolved_mode": "live",
        "config_overrides": {"retriever": config},
        "system": dict(SYSTEM),
    }


def _run(tmp_path) -> str:
    """Two benchmarks, two arms, and the run metadata a bundle keeps under `experiment`."""
    path = tmp_path / "results.json"
    meta = {
        "experiment_name": "two arms",
        "metrics_version": METRICS_VERSION,
        "package_version": "0.1.0",
        # In the key: a bundle read without its metadata hashed these as None.
        "judge_model": "gpt-4.1",
        "similarity_model": "all-MiniLM-L6-v2",
        "environment": {"lasagna_version": "0.1.0", "python": "3.12.9", "platform": "darwin"},
    }
    rows = [
        _row(bench, config, score)
        for bench in ("smoke", "tickets@abc")
        for config, score in (("dense", 0.9), ("bm25", 0.8))
    ]
    path.write_text(
        json.dumps(
            {
                "experiment_name": "two arms",
                "metrics_version": METRICS_VERSION,
                "package_version": "0.1.0",
                "experiment_config": meta,
                "results": rows,
            }
        )
    )
    return str(path)


def _bundle(tmp_path, results: str) -> tuple:
    out = tmp_path / "run.json"
    made = runner.invoke(cli.app, ["results", "bundle", results, "-o", str(out)])
    assert made.exit_code == 0, made.output
    keys = dict(re.findall(r"  (\S+): comparability key (\w+)", made.output))
    return str(out), keys


def test_compare_prints_the_key_the_bundle_records_for_each_benchmark(tmp_path):
    results = _run(tmp_path)
    bundle, keys = _bundle(tmp_path, results)
    assert keys["smoke/dense"] != keys["tickets@abc/dense"]

    for a, b in ((results, results), (bundle, bundle), (results, bundle)):
        out = runner.invoke(cli.app, ["results", "compare", a, b], terminal_width=200)
        assert out.exit_code == 0, out.output
        assert f"smoke: Comparable: yes (key {keys['smoke/dense']})" in out.output
        assert f"tickets@abc: Comparable: yes (key {keys['tickets@abc/dense']})" in out.output


def test_the_header_names_both_inputs(tmp_path):
    results = _run(tmp_path)
    bundle, _ = _bundle(tmp_path, results)
    out = runner.invoke(cli.app, ["results", "compare", results, bundle], terminal_width=400)
    assert out.exit_code == 0, out.output
    assert f'A: {results}, experiment "two arms"' in out.output
    assert f'B: {bundle}, experiment "two arms"' in out.output
    assert " vs \n" not in out.output


def test_an_edited_bundle_is_refused_not_called_comparable(tmp_path):
    results = _run(tmp_path)
    bundle, _ = _bundle(tmp_path, results)
    data = json.loads(open(bundle).read())
    data["results"][0]["metrics"]["ndcg@10"] = 0.1
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(data))

    assert runner.invoke(cli.app, ["results", "validate", str(edited)]).exit_code == 1
    out = runner.invoke(cli.app, ["results", "compare", str(edited), bundle])
    words = " ".join(out.output.split())
    assert out.exit_code == 1, words
    assert "Comparable: yes" not in words
    assert "is not a valid bundle, so its numbers cannot be compared" in words
    assert "lasagna results validate" in words


def test_different_keys_say_not_comparable_and_name_the_field(tmp_path):
    results = _run(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    data = json.loads(open(results).read())
    for row in data["results"]:
        row["system"]["embedder_model"] = "BAAI/bge-small-en-v1.5"
    (other / "results.json").write_text(json.dumps(data))

    out = runner.invoke(cli.app, ["results", "compare", results, str(other)], terminal_width=200)
    assert out.exit_code == 0, out.output
    assert "smoke: Comparable: NO" in out.output
    assert "embedder_model: 'all-MiniLM-L6-v2' vs 'BAAI/bge-small-en-v1.5'" in out.output
