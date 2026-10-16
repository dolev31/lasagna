"""--resume restores only results made the same way, and results.json is written whole.

The resume key hashed the metrics version, the package version, the judge and
the overridden settings, but not the libraries that compute the numbers
(torch, transformers, sentence-transformers, ragas, pymilvus), the settings
file's contents, ``variance_source`` or ``max_concurrency``: a checkpoint made
before upgrading torch, or under another settings file, was restored as if it
were this run's. And ``results.json`` was written in place, so a crash or a
full disk mid-write left a truncated file where the last good one had been.
"""

from __future__ import annotations

import json

import pytest

from lasagna.knowledge_evaluation_sdk import results as results_module
from lasagna.knowledge_evaluation_sdk import run_provenance
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults


@pytest.fixture
def versions(monkeypatch):
    installed = {
        "torch": "2.8.0",
        "transformers": "4.57.0",
        "sentence-transformers": "5.1.2",
        "ragas": "0.4.1",
        "pymilvus": "2.6.2",
    }
    monkeypatch.setattr(run_provenance, "_library_version", lambda dist: installed.get(dist))
    return installed


def _fingerprint(settings_file, **kw):
    return run_provenance.resume_fingerprint(
        "judge-a", environ={}, metrics_version="1.5", settings_file=str(settings_file), **kw
    )


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[rag]\nchunk_size = 800\n")
    return path


@pytest.mark.parametrize(
    "library", ["torch", "transformers", "sentence-transformers", "ragas", "pymilvus"]
)
def test_another_library_version_is_another_fingerprint(versions, settings, library):
    before = _fingerprint(settings)
    versions[library] = "9.9.9"
    assert _fingerprint(settings) != before


def test_another_settings_file_content_is_another_fingerprint(versions, settings):
    before = _fingerprint(settings)
    settings.write_text("[rag]\nchunk_size = 600\n")
    assert _fingerprint(settings) != before


def test_variance_source_and_concurrency_are_part_of_it(versions, settings):
    base = _fingerprint(settings, variance_source="run", max_concurrency=1)
    assert _fingerprint(settings, variance_source="sample", max_concurrency=1) != base
    assert _fingerprint(settings, variance_source="run", max_concurrency=4) != base
    assert _fingerprint(settings, variance_source="run", max_concurrency=1) == base


def test_a_crash_while_saving_keeps_the_previous_results(tmp_path, monkeypatch):
    path = tmp_path / "results.json"
    BenchmarkResults([], experiment_name="first").save(str(path))
    good = path.read_text()

    def dies_halfway(obj, fp, **kw):
        fp.write('{"experiment_name": "sec')
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(results_module.json, "dump", dies_halfway)
        with pytest.raises(OSError):
            BenchmarkResults([], experiment_name="second").save(str(path))

    assert path.read_text() == good
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results.json"]


def test_the_directory_is_synced_after_the_rename(tmp_path, monkeypatch):
    import lasagna.knowledge_evaluation_sdk.checkpoint as checkpoint

    synced = []
    monkeypatch.setattr(checkpoint, "_fsync_dir", lambda d: synced.append(str(d)))
    BenchmarkResults([], experiment_name="x").save(str(tmp_path / "results.json"))

    assert synced == [str(tmp_path)]
    assert json.loads((tmp_path / "results.json").read_text())["experiment_name"] == "x"
