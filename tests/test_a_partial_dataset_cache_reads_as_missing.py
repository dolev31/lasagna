"""Offline, a dataset cached in part gets the same one-line message as one not cached.

The offline check counted a dataset as cached when its folder existed. With a
build half there (`mteb___scifact/queries/<version>/<hash>/` without its
`dataset_info.json`, as an interrupted download or a pruned cache leaves it),
every task failed with a raw "[Errno 2] No such file or directory:
'.../mteb___scifact/queries/.../dataset_info.json'", once per arm and again in
the failure list, instead of the message that names the benchmark and the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lasagna.knowledge_evaluation_sdk import datasets

pytest.importorskip("datasets")


def _build(root: Path, config: str, info: bool = True) -> Path:
    build = root / "mteb___scifact" / config / "0.0.0" / "fingerprint"
    build.mkdir(parents=True)
    (build / f"scifact-{config}.arrow").write_bytes(b"\0")
    if info:
        (build / "dataset_info.json").write_text("{}")
    # What `datasets` leaves beside every build.
    (build.parent / "fingerprint_builder.lock").write_text("")
    return build


@pytest.fixture
def cache(tmp_path, monkeypatch):
    import datasets.config as datasets_config
    from huggingface_hub import constants

    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return tmp_path / "datasets"


def test_a_complete_build_counts_as_cached(cache):
    for config in ("corpus", "queries", "default"):
        _build(cache, config)
    assert datasets.dataset_cached("mteb/scifact")
    assert datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"]}) is None


def test_a_build_without_its_info_file_is_missing(cache):
    _build(cache, "corpus")
    _build(cache, "queries", info=False)
    assert not datasets.dataset_cached("mteb/scifact")
    problem = datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"]})
    assert problem is not None
    assert "beir/scifact" in str(problem) and "only partly" in str(problem)


def test_an_interrupted_download_is_missing(cache):
    incomplete = cache / "mteb___scifact" / "corpus" / "0.0.0" / "fingerprint.incomplete"
    incomplete.mkdir(parents=True)
    assert not datasets.dataset_cached("mteb/scifact")


def test_an_empty_folder_is_missing(cache):
    (cache / "mteb___scifact").mkdir(parents=True)
    assert not datasets.dataset_cached("mteb/scifact")


def _snapshot(tmp_path: Path, files: dict, blobs: bool = True) -> None:
    """A hub-cache snapshot of mteb/scifact; each file is a symlink into blobs/, as the hub writes it."""
    repo = tmp_path / "hub" / "datasets--mteb--scifact"
    revision = "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"  # pragma: allowlist secret
    snapshot = repo / "snapshots" / revision
    for name in files:
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        blob = repo / "blobs" / name.replace("/", "_")
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blobs:
            blob.write_text("{}")
        link.symlink_to(blob)


def test_a_snapshot_holding_only_a_readme_is_missing(cache, tmp_path):
    """The hub-cache branch counted any file, so a README alone read as cached and
    the offline load then failed in every task."""
    _snapshot(tmp_path, {"README.md": 1})
    assert not datasets.dataset_cached("mteb/scifact")
    problem = datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"]})
    assert problem is not None and "beir/scifact" in str(problem)


def test_a_snapshot_with_a_loading_script_and_no_data_is_missing(cache, tmp_path):
    _snapshot(tmp_path, {"README.md": 1, "scifact.py": 1, "dataset_infos.json": 1})
    assert not datasets.dataset_cached("mteb/scifact")


def test_a_snapshot_whose_data_was_never_downloaded_is_missing(cache, tmp_path):
    _snapshot(tmp_path, {"README.md": 1, "corpus.jsonl": 1}, blobs=False)
    (tmp_path / "hub" / "datasets--mteb--scifact" / "blobs" / "README.md").write_text("x")
    assert not datasets.dataset_cached("mteb/scifact")


def test_a_snapshot_with_its_data_files_is_cached(cache, tmp_path):
    _snapshot(
        tmp_path,
        {
            "README.md": 1,
            "corpus.jsonl": 1,
            "queries.jsonl": 1,
            "qrels/test.jsonl": 1,
        },
    )
    assert datasets.dataset_cached("mteb/scifact")
    assert datasets.missing_offline_datasets({"beir/scifact": ["mteb/scifact"]}) is None


def test_a_parquet_snapshot_is_cached(cache, tmp_path):
    _snapshot(tmp_path, {"README.md": 1, "corpus/test-00000-of-00001.parquet": 1})
    assert datasets.dataset_cached("mteb/scifact")
