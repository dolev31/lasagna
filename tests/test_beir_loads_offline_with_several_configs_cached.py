"""A BEIR benchmark must load offline once its repo has several configs cached.

`mteb/scifact` publishes three configs: `default` (the qrels), `corpus` and
`queries`. The qrels were loaded with no config name, which online resolves to
`default` and offline fails as soon as more than one config is in the cache:
"There are multiple 'mteb/scifact' configurations in the cache". Running a
benchmark caches all three, so a cluster that pre-caches and then runs offline
could not reproduce a `beir/scifact` number at all.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk import datasets as ds


class _Rows(list):
    """A list of rows standing in for a `datasets.Dataset`."""


def _offline_hub(calls):
    """`load_dataset` as it behaves offline with every config cached."""

    def _load_dataset(path, split=None, name=None, **kwargs):
        config = name
        calls.append((path, config, split))
        if config is None:
            raise ValueError(
                f"There are multiple {path!r} configurations in the cache: corpus, queries, default"
            )
        if config == "queries":
            return _Rows([{"_id": "q1", "text": "does x cause y"}])
        if config == "default" and split == "test":
            return _Rows([{"query-id": "q1", "corpus-id": "d1", "score": 1}])
        raise ValueError(f"no {config}/{split}")

    return _load_dataset


@pytest.mark.parametrize("benchmark", ["beir/scifact", "beir/fiqa", "beir/nfcorpus"])
def test_the_qrels_are_loaded_by_config_name(monkeypatch, tmp_path, benchmark):
    import datasets as hf

    calls = []
    monkeypatch.setattr(hf, "load_dataset", _offline_hub(calls), raising=False)
    monkeypatch.setattr(ds, "_cache_path", lambda *a, **k: tmp_path / "cache.json")

    samples = ds.load_benchmark(benchmark, split="test", n_samples=None)

    assert [s["query_id"] for s in samples] == ["q1"]
    qrels_calls = [c for c in calls if c[1] != "queries"]
    assert qrels_calls and all(config == "default" for _, config, _ in qrels_calls)
