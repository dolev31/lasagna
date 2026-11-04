"""The MIRACL Swahili adapter reads both pinned repos and keeps only relevant judgments."""

from __future__ import annotations

import gzip
import json
import re

import pytest

from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.miracl import MiraclSwahiliAdapter

TOPICS = "miracl-v1.0-sw/topics/topics.miracl-v1.0-sw-{}.tsv"
QRELS = "miracl-v1.0-sw/qrels/qrels.miracl-v1.0-sw-{}.tsv"
CORPUS = "miracl-corpus-v1.0-sw/docs-0.jsonl.gz"


@pytest.fixture
def dataset_files(tmp_path, monkeypatch):
    import huggingface_hub

    def write(name, text):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_corpus(docs):
        path = tmp_path / CORPUS
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for docid, title, text in docs:
                f.write(json.dumps({"docid": docid, "title": title, "text": text}) + "\n")

    write(TOPICS.format("dev"), "1\tBandari kubwa iko wapi?\n2\tMlima mrefu ni upi?\n3\tSwali?\n")
    # q3 is judged, but only as non-relevant, so it has nothing to score.
    write(QRELS.format("dev"), "1 Q0 10#0 1\n1 Q0 10#1 0\n2 Q0 20#0 1\n2 Q0 10#0 1\n3 Q0 20#1 0\n")
    write(TOPICS.format("train"), "7\tSwali la mafunzo?\n")
    write(QRELS.format("train"), "7 Q0 30#0 1\n")
    write_corpus(
        [
            ("10#0", "Mombasa", "Bandari kubwa."),
            ("10#1", "Mombasa", "Mji wa pwani."),
            ("20#0", "Kilimanjaro", "Mlima mrefu."),
            ("20#1", "Kilimanjaro", "Theluji."),
            ("30#0", "Nairobi", "Mji mkuu."),
        ]
    )
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls, write


def test_registered_metadata_needs_no_download():
    adapter = benchmark_registry.get("miracl/sw")
    info = adapter.info()
    assert isinstance(adapter, MiraclSwahiliAdapter)
    assert info.license == "Apache-2.0"
    assert info.paper == "arxiv:2210.09984"
    assert adapter.available_splits() == ["dev", "train"]
    assert adapter.corpus_provenance == "native"


def test_hf_revision_names_both_pinned_commits():
    adapter = MiraclSwahiliAdapter()
    for revision in (adapter.TOPICS_REVISION, adapter.CORPUS_REVISION):
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert adapter.hf_revision == (
        f"miracl/miracl@{adapter.TOPICS_REVISION}+miracl/miracl-corpus@{adapter.CORPUS_REVISION}"
    )


def test_each_file_comes_from_its_own_repo_at_its_own_commit(dataset_files):
    calls, _ = dataset_files
    adapter = MiraclSwahiliAdapter()
    adapter.load(cache_dir="chosen-cache")
    adapter.load_corpus(cache_dir="chosen-cache")
    assert [(repo, filename, kwargs["revision"]) for repo, filename, kwargs in calls] == [
        ("miracl/miracl", TOPICS.format("dev"), adapter.TOPICS_REVISION),
        ("miracl/miracl", QRELS.format("dev"), adapter.TOPICS_REVISION),
        ("miracl/miracl-corpus", CORPUS, adapter.CORPUS_REVISION),
    ]
    for _, _, kwargs in calls:
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["cache_dir"] == "chosen-cache"


def test_only_relevant_judgments_are_expected_and_all_are_in_the_corpus(dataset_files):
    adapter = MiraclSwahiliAdapter()
    samples = adapter.load()
    corpus = adapter.load_corpus()
    assert [s.query_id for s in samples] == ["1", "2"]
    assert samples[0].question == "Bandari kubwa iko wapi?"
    assert samples[0].expected_docs == {"10#0": 1}
    assert samples[1].expected_docs == {"20#0": 1, "10#0": 1}
    assert corpus["10#0"] == "Mombasa\n\nBandari kubwa."
    assert all(set(s.expected_docs) <= set(corpus) for s in samples)


def test_sampling_is_repeatable_and_train_reads_its_own_files(dataset_files):
    calls, _ = dataset_files
    adapter = MiraclSwahiliAdapter()
    first = adapter.load(n_samples=1, seed=42)
    assert first == adapter.load(n_samples=1, seed=42)
    assert len(first) == 1
    assert adapter.load(n_samples=0) == []
    assert len(adapter.load(n_samples=10)) == 2
    train = adapter.load(split="train")
    assert [s.query_id for s in train] == ["7"]
    assert calls[-1][1] == QRELS.format("train")


def test_corpus_limit_truncates_and_says_so(dataset_files):
    adapter = MiraclSwahiliAdapter()
    assert list(adapter.load_corpus(max_docs=2)) == ["10#0", "10#1"]
    assert adapter.corpus_truncated
    assert len(adapter.load_corpus(max_docs=5)) == 5
    assert not adapter.corpus_truncated
    assert len(adapter.load_corpus(max_docs=0)) == 5
    assert not adapter.corpus_truncated


@pytest.mark.parametrize(
    "kwargs", [{"split": "test"}, {"split": "test-a"}, {"n_samples": -1}, {"filters": {"x": 1}}]
)
def test_invalid_load_options_fail_before_download(dataset_files, kwargs):
    calls, _ = dataset_files
    with pytest.raises(ValueError):
        MiraclSwahiliAdapter().load(**kwargs)
    assert calls == []


def test_the_test_split_error_names_the_split_to_use(dataset_files):
    with pytest.raises(ValueError, match="--split dev"):
        MiraclSwahiliAdapter().load(split="test")


def test_negative_corpus_limit_is_rejected_before_download(dataset_files):
    calls, _ = dataset_files
    with pytest.raises(ValueError, match="max_docs"):
        MiraclSwahiliAdapter().load_corpus(max_docs=-1)
    assert calls == []


def test_qrels_for_a_query_with_no_topic_is_an_error_not_a_dropped_query(dataset_files):
    _, write = dataset_files
    write(QRELS.format("dev"), "1 Q0 10#0 1\n99 Q0 20#0 1\n")
    with pytest.raises(ValueError, match="99"):
        MiraclSwahiliAdapter().load()
