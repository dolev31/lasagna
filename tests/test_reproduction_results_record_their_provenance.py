"""examples/reproduction/results.csv says which code and which metric definitions produced it.

Its README said the file records "the package commit" and to rerun "at the
commit in results.csv", while the file held only `package_version 0.1.0`,
dated 2026-10-10, two METRICS_VERSION bumps before the release. The numbers
were rerun at a named commit; the file now records that commit, the metrics
version and the run date, `results_csv.py` writes it from a run, and the
README's tables are the file's numbers.
"""

from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path

import pytest

from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

HERE = Path(__file__).resolve().parents[1] / "examples" / "reproduction"
RESULTS = HERE / "results.csv"
README = HERE / "README.md"


def _rows():
    if not RESULTS.is_file():
        pytest.skip("needs a source checkout")
    with RESULTS.open() as f:
        return list(csv.DictReader(f))


def _script():
    spec = importlib.util.spec_from_file_location("results_csv", HERE / "results_csv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_row_records_the_current_metrics_version_and_one_commit():
    rows = _rows()
    assert len(rows) == 14
    assert {r["metrics_version"] for r in rows} == {METRICS_VERSION}
    commits = {r["commit"] for r in rows}
    assert len(commits) == 1 and re.fullmatch(r"[0-9a-f]{7,40}", commits.pop())
    assert len({r["run_date"] for r in rows}) == 1
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["run_date"]) for r in rows)


def test_the_file_has_the_columns_the_script_writes():
    with RESULTS.open() as f:
        header = next(csv.reader(f))
    assert tuple(header) == _script().COLUMNS


def test_the_script_writes_one_row_per_method_and_dataset_from_a_run():
    script = _script()

    def task(dataset, config, metrics, lists):
        return {
            "benchmark_name": dataset,
            "config_name": config,
            "error": None,
            "split": "test",
            "metrics": metrics,
            "per_sample_scores": [{"_retrieved_ids": ids} for ids in lists],
        }

    results = []
    for dataset in script.DATASETS:
        for method in script.METHODS:
            full, short = [f"d{i}" for i in range(10)], ["d1", "d2"]
            results.append(task(dataset, method, {"ndcg@10": 0.12345}, [full, short]))
            results.append(task(dataset, f"{method}_k100", {"recall@100": 0.9}, [full]))
    run = {
        "experiment_config": {
            "created_at": "2026-10-17T12:00:00+00:00",
            "package_version": "0.1.0",
            "metrics_version": METRICS_VERSION,
            "environment": {"git_commit": "0123456789abcdef"},
        },
        "results": results,
    }
    rows = script.rows(run)
    assert len(rows) == 14
    assert rows[0] == {
        "method": "bm25",
        "dataset": "beir/scifact",
        "split": "test",
        "ndcg@10": "0.1235",
        "recall@100": "0.9000",
        "n_queries": 2,
        "short_lists_at_10": 1,
        "run_date": "2026-10-17",
        "package_version": "0.1.0",
        "metrics_version": METRICS_VERSION,
        "commit": "0123456",
    }
    assert script.rows(run, commit="fedcba9876543210")[0]["commit"] == "fedcba9"


def test_the_readme_says_where_the_numbers_came_from_and_how_to_compare():
    words = " ".join(README.read_text().split())
    assert "the package commit" not in words
    assert "the commit they were produced at" in words
    assert "examples/reproduction/results_csv.py" in words


def _readme_tables():
    """``{(label, dataset): (ndcg, recall)}`` from the README's tables, by results.csv method."""
    methods = {
        "BM25 (k1 0.9, b 0.4)": "bm25",
        "all-MiniLM-L6-v2": "dense_minilm",
        "bge-small-en-v1.5": "dense_bge_small",
        "BM25 + MiniLM-L6 cross-encoder": "bm25_ce",
        "BM25 with Lasagna's defaults (k1 1.2, b 0.75)": "bm25_k1.2_b0.75",
        "Hybrid RRF: BM25 + all-MiniLM-L6-v2": "hybrid_rrf_minilm",
        "Hybrid RRF: BM25 + bge-small-en-v1.5": "hybrid_rrf_bge_small",
    }
    datasets = {"SciFact": "beir/scifact", "NFCorpus": "beir/nfcorpus"}
    text = README.read_text()
    ndcg_table, recall_table, extra_table = re.split(r"\n(?=Recall@100|Arms with no)", text)[:3]
    cells = {}
    for table, metric in ((ndcg_table, "ndcg@10"), (recall_table, "recall@100")):
        for line in table.splitlines():
            parts = [c.strip() for c in line.strip("|").split("|")]
            if len(parts) == 6 and parts[0] in methods and parts[1] in datasets:
                cells[(methods[parts[0]], datasets[parts[1]], metric)] = float(parts[4])
    for line in extra_table.splitlines():
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) == 5 and parts[0] in methods:
            for (dataset, metric), value in zip(
                [
                    ("beir/scifact", "ndcg@10"),
                    ("beir/nfcorpus", "ndcg@10"),
                    ("beir/scifact", "recall@100"),
                    ("beir/nfcorpus", "recall@100"),
                ],
                parts[1:],
            ):
                cells[(methods[parts[0]], dataset, metric)] = float(value)
    return cells


def test_the_reproduction_readme_tables_are_the_files_numbers():
    measured = {(r["method"], r["dataset"]): r for r in _rows()}
    cells = _readme_tables()
    assert len(cells) == 26
    for (method, dataset, metric), value in cells.items():
        # Three decimals of the run's number, which the file keeps to four: a
        # 0.3115 in the file is 0.31151 run, which the table rounds to 0.312.
        assert abs(value - float(measured[(method, dataset)][metric])) <= 0.00051, (
            method,
            dataset,
            metric,
        )
