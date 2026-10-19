"""The README's demo benchmark records a dataset revision, and loads that revision.

`beir/scifact` had no `hf_revision`, so a SciFact run recorded `hf_revision: ""`
(a moving reference), raised UNPINNED_DATASET, and its comparability key did not
name the data. Recording a pin is not enough: the queries, the qrels and the
corpus must each be requested at it, or the bundle claims data it did not load.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk import datasets
from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.runner import _resolved_revision

SCIFACT = "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"


def test_the_demo_benchmark_records_its_revision():
    adapter = benchmark_registry.get("beir/scifact")
    assert _resolved_revision(adapter, "beir/scifact") == SCIFACT


@pytest.mark.parametrize("name", ["beir/scifact", "beir/hotpotqa"])
def test_both_catalogues_agree_on_a_pinned_revision(name):
    adapter = benchmark_registry.get(name)
    assert adapter.hf_revision
    assert datasets.resolve_dataset_revision(name) == adapter.hf_revision


def test_the_pinned_revision_is_what_every_load_asks_for(monkeypatch, tmp_path):
    asked = []

    def fake_load_dataset(path, *args, **kwargs):
        config = kwargs.get("name") or (args[0] if args else None)
        asked.append((path, config, kwargs.get("revision")))
        if config == "queries":
            return [{"_id": "q1", "text": "a claim"}]
        if config == "corpus" or kwargs.get("split") == "corpus":
            return [{"_id": "d1", "title": "t", "text": "body"}]
        return [{"query-id": "q1", "corpus-id": "d1", "score": 1}]

    import datasets as hf_datasets

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    adapter = benchmark_registry.get("beir/scifact")
    adapter.load(split="test", cache_dir=str(tmp_path))
    adapter.load_corpus()
    assert len(asked) == 3, asked
    assert {revision for _, _, revision in asked} == {SCIFACT}, asked
