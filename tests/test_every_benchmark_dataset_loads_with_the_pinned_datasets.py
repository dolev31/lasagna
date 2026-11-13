"""Every benchmark loads with the `datasets` release Lasagna pins (5.1.0).

`datasets` 5 refuses a Hub repository named without its owner ("Repository id
must be 'namespace/name'"), and since 4.0 it refuses a dataset that is only a
loading script. Five benchmarks hit one or the other: `triviaqa`, `hotpotqa`,
`kilt/nq` and `kilt/triviaqa` named `trivia_qa`, `hotpot_qa` and
`kilt_tasks`, and `fever` was a script. The loader then caught the error as a
missing split and said "Split 'validation' is not available", which sent the
reader to the wrong cause. CI runs offline and never loads them, so these tests
read the catalogue and the error path instead of the Hub.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk import datasets as ds


def _hub_ids():
    for name, info in ds._BENCHMARKS.items():
        for key in ("hf_dataset", "hf_qrels"):
            if info.get(key):
                yield name, key, info[key]


def test_every_hub_dataset_is_named_with_its_owner():
    unnamed = [(name, key, repo) for name, key, repo in _hub_ids() if "/" not in repo]
    assert unnamed == []


def test_fever_loads_its_parquet_conversion(monkeypatch):
    # The Hub converted the FEVER script to parquet; `datasets` loads that
    # branch at the commit it was checked against.
    calls = []

    def fake_load(repo, split=None, **kwargs):
        calls.append((repo, split, kwargs))
        raise RuntimeError("stop after the call is recorded")

    monkeypatch.setattr(ds, "load_dataset", fake_load, raising=False)
    monkeypatch.setattr("datasets.load_dataset", fake_load)
    with pytest.raises(RuntimeError):
        ds.load_benchmark("fever", split="validation", n_samples=1, cache_dir="/nonexistent")
    repo, split, kwargs = calls[0]
    assert (repo, split) == ("fever/fever", "validation")
    assert kwargs.get("revision") == "5f577157472532aa1d9924d2df63aac44f70cf2b"
    # Each split by name: `datasets` would merge labelled and unlabelled dev.
    assert kwargs.get("data_files") == {
        s: f"v1.0/{s}/*.parquet" for s in ("train", "labelled_dev", "paper_dev", "paper_test")
    }
    assert "name" not in kwargs


def test_a_dataset_that_will_not_load_is_not_called_a_missing_split(monkeypatch, tmp_path):
    def broken(repo, split=None, **kwargs):
        raise ValueError(
            "Invalid HF URI 'hf://datasets/x/.huggingface.yaml'. "
            "Repository id must be 'namespace/name', got 'x'."
        )

    monkeypatch.setattr("datasets.load_dataset", broken)
    monkeypatch.setattr(ds, "_available_splits", lambda *a, **k: [])
    with pytest.raises(ValueError) as caught:
        ds.load_benchmark("hotpotqa", split="validation", n_samples=1, cache_dir=str(tmp_path))
    message = str(caught.value)
    assert "is not available" not in message
    assert "Repository id must be 'namespace/name'" in message


def test_a_split_that_is_really_missing_still_says_so(monkeypatch, tmp_path):
    def no_such_split(repo, split=None, **kwargs):
        raise ValueError(f"Unknown split \"{split}\". Should be one of ['train'].")

    monkeypatch.setattr("datasets.load_dataset", no_such_split)
    monkeypatch.setattr(ds, "_available_splits", lambda *a, **k: ["train"])
    with pytest.raises(ValueError, match="Split 'validation' is not available.*available: train"):
        ds.load_benchmark("hotpotqa", split="validation", n_samples=1, cache_dir=str(tmp_path))


def test_a_benchmark_whose_rows_hold_no_question_says_so(monkeypatch, tmp_path):
    # KILT's TriviaQA comes without its questions: every row converts to
    # nothing, which used to be an empty benchmark with no error.
    import datasets

    rows = datasets.Dataset.from_list([{"id": "q1", "input": "", "output": [{"answer": "x"}]}])
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    with pytest.raises(
        ValueError, match="has 1 rows, but none holds a question.*without its questions"
    ):
        ds.load_benchmark("kilt/triviaqa", split="validation", cache_dir=str(tmp_path))
