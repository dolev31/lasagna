"""A sweep over your own data records which file it read, never as an absolute path.

The run metadata named a local benchmark only by its content-hashed name
(``local/faq@<hash>``), which says what the data was but not where to find it
again. Recorded as given, a path would carry the machine's layout into
bundles, which are published, and the bundle's redaction only knows the home
and temp folders. So it is recorded relative to the sweep file's folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _run_metadata


@pytest.fixture(autouse=True)
def _registry_restored():
    """Sweeps register their files in the process-wide registry."""
    benchmark_registry.names()  # built-ins first, so they are in the snapshot
    before = dict(benchmark_registry._adapters)
    yield
    benchmark_registry._adapters.clear()
    benchmark_registry._adapters.update(before)


def _faq(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"q{i}", "question": f"question {i}?", "answer": "a", "contexts": [f"passage {i}"]}
        for i in range(4)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _recorded(sweep: Path, written: str) -> dict:
    sweep.parent.mkdir(parents=True, exist_ok=True)
    sweep.write_text(
        f"name: own\nbenchmarks: [{{path: {json.dumps(written)}}}]\n"
        "tracks: [retrieval]\npipeline: {base_profile: speed}\n"
    )
    exp = Experiment.from_file(str(sweep))
    exp.generate_tasks()
    return _run_metadata(exp.config)


@pytest.mark.parametrize("absolute", [False, True])
def test_the_path_is_relative_to_the_sweep_file(tmp_path, monkeypatch, absolute):
    data = _faq(tmp_path / "study" / "data" / "faq.jsonl")
    monkeypatch.chdir(tmp_path)  # not the sweep's folder
    meta = _recorded(tmp_path / "study" / "sweep.yaml", str(data) if absolute else "data/faq.jsonl")
    (bench,) = meta["benchmarks"]
    assert bench["name"].startswith("local/faq@")
    assert bench["path"] == "data/faq.jsonl"
    assert str(tmp_path) not in json.dumps(meta)


def test_data_outside_the_sweeps_folder_is_still_relative(tmp_path):
    _faq(tmp_path / "shared" / "faq.jsonl")
    meta = _recorded(tmp_path / "study" / "sweep.yaml", str(tmp_path / "shared" / "faq.jsonl"))
    assert meta["benchmarks"][0]["path"] == "../shared/faq.jsonl"


def test_benchmark_run_records_it_relative_to_the_working_directory(tmp_path, monkeypatch):
    _faq(tmp_path / "faq.jsonl")
    monkeypatch.chdir(tmp_path)
    exp = Experiment.from_dict(
        {"name": "own", "benchmarks": [{"path": "./faq.jsonl"}], "tracks": ["retrieval"]}
    )
    exp.generate_tasks()
    assert _run_metadata(exp.config)["benchmarks"][0]["path"] == "faq.jsonl"


def test_a_registered_benchmark_records_no_path():
    exp = Experiment.from_dict({"name": "s", "benchmarks": [{"name": "smoke"}]})
    assert "path" not in _run_metadata(exp.config)["benchmarks"][0]
