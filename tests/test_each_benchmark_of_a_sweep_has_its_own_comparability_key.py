"""Each benchmark of a multi-benchmark run carries its own comparability key.

The bundle kept one key per config name (``if name in configs: continue``),
computed from the config's first result. A sweep over ``smoke`` and a file of
two questions, rendered with ``lasagna results leaderboard <run> --format
latex``, captioned the smoke table "n_samples=2" with the file's key; the
README's ``beir_tier1.yaml`` gave NFCorpus SciFact's key, and the hub would
have split comparable runs. The key is now per (benchmark, config) everywhere:
the bundle, the leaderboard's tables and captions, ``results bundle``,
``results validate`` and ``results compare``. A single-benchmark run's key is
unchanged.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION, comparability_key
from lasagna.knowledge_evaluation_sdk.leaderboard import grouped_rows, render_latex
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

SYSTEM = {
    "embedder_model": "all-MiniLM-L6-v2",
    "embedder_dim": 384,
    "chunker": "structure_aware",
    "retriever_top_k": 40,
    "vector_backend": "milvus",
}


def _row(benchmark: str, n: int, config: str = "speed", ndcg: float = 0.9) -> dict:
    return {
        "task_key": f"{benchmark}|{config}|run_0",
        "benchmark_name": benchmark,
        "config_name": config,
        "run_index": 0,
        "metrics": {"ndcg@10": ndcg},
        "per_sample_scores": [],
        "n_samples": n,
        "split": "test",
        "sampling_seed": 42,
        "sample_id_digest": f"digest-{benchmark}",
        "corpus_fingerprint": f"corpus-{benchmark}",
        "corpus_provenance": "native",
        "resolved_mode": "live",
        "config_overrides": {},
        "system": dict(SYSTEM),
    }


def _results(tmp_path, *rows) -> str:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "experiment_name": "two benchmarks",
                "metrics_version": METRICS_VERSION,
                "package_version": "0.1.0",
                "experiment_config": {
                    "metrics_version": METRICS_VERSION,
                    "environment": {
                        "lasagna_version": "0.1.0",
                        "python": "3.12.9",
                        "platform": "darwin",
                    },
                },
                "results": list(rows),
            }
        )
    )
    return str(path)


def test_the_bundle_has_one_entry_per_benchmark_and_config(tmp_path):
    path = _results(tmp_path, _row("smoke", 12), _row("local/my_data@abc", 2))
    bundle = BenchmarkResults.load(path).to_bundle()
    entries = {(c["benchmark"], c["name"]): c for c in bundle["configs"]}
    assert set(entries) == {("smoke", "speed"), ("local/my_data@abc", "speed")}
    smoke, mine = entries[("smoke", "speed")], entries[("local/my_data@abc", "speed")]
    assert smoke["held_fixed"]["n_samples"] == 12 and mine["held_fixed"]["n_samples"] == 2
    assert smoke["comparability_key"] != mine["comparability_key"]
    assert smoke["comparability_key"] == comparability_key(smoke["held_fixed"])


def test_a_single_benchmark_run_keeps_its_key(tmp_path):
    """The key a config had before: computed from its first result."""
    from lasagna.knowledge_evaluation_sdk.bundle import held_fixed_from

    path = _results(tmp_path, _row("smoke", 12))
    results = BenchmarkResults.load(path)
    (entry,) = results.to_bundle()["configs"]
    row = results._results[0].to_dict()
    held = held_fixed_from(row, results.experiment_config, [row.get("config_overrides") or {}])
    assert entry["comparability_key"] == comparability_key(held)


def test_each_leaderboard_caption_names_its_own_benchmark(tmp_path):
    path = _results(tmp_path, _row("smoke", 12), _row("local/my_data@abc", 2))
    bundle = BenchmarkResults.load(path).to_bundle()
    keys = {c["benchmark"]: c["comparability_key"] for c in bundle["configs"]}
    latex = render_latex(grouped_rows(path, "ndcg@10"), "ndcg@10")
    captions = [line for line in latex.splitlines() if "\\caption{" in line]
    smoke = next(c for c in captions if c.startswith("\\caption{smoke"))
    mine = next(c for c in captions if "my\\_data" in c)
    assert "n\\_samples=12" in smoke and keys["smoke"] in smoke
    assert "n\\_samples=2" in mine and keys["local/my_data@abc"] in mine


def test_bundle_and_validate_name_the_benchmark(tmp_path):
    path = _results(tmp_path, _row("smoke", 12), _row("local/my_data@abc", 2))
    out = tmp_path / "run.json"
    made = CliRunner().invoke(cli.app, ["results", "bundle", path, "-o", str(out)])
    assert made.exit_code == 0, made.output
    assert "smoke/speed: comparability key" in made.output
    checked = CliRunner().invoke(cli.app, ["results", "validate", str(out)])
    assert checked.exit_code == 0, checked.output

    bundle = json.loads(out.read_text())
    bundle["configs"][0]["benchmark"] = "elsewhere"
    from lasagna.knowledge_evaluation_sdk.bundle import _content_hash

    bundle["run_id"] = _content_hash(bundle)
    out.write_text(json.dumps(bundle))
    refused = CliRunner().invoke(cli.app, ["results", "validate", str(out)])
    words = " ".join(refused.output.split())
    assert refused.exit_code != 0
    assert "on elsewhere: held_fixed.benchmark" in words


def test_compare_checks_each_benchmark(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    left = BenchmarkResults.load(_results(a, _row("smoke", 12), _row("local/x", 2)))
    right = BenchmarkResults.load(_results(b, _row("smoke", 12), _row("local/x", 3)))
    text = left.compare(right)
    assert "smoke: Comparable: yes" in text
    assert "local/x: Comparable: NO" in text and "n_samples: 2 vs 3" in text
