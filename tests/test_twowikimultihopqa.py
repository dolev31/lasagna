"""The 2Wiki adapter keeps distractors and stable title-based relevance IDs."""

from __future__ import annotations

import json

import pytest

from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.twowikimultihopqa import (
    TwoWikiMultihopQAAdapter,
)


@pytest.fixture
def dataset_files(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow", reason="Parquet fixture requires the eval extra")
    pq = pytest.importorskip("pyarrow.parquet", reason="Parquet fixture requires the eval extra")
    import huggingface_hub

    def row(identifier, context, gold):
        return {
            "_id": identifier,
            "question": f"Question {identifier}?",
            "answer": f"Answer {identifier}",
            "type": "compositional",
            "context": json.dumps(context),
            "supporting_facts": json.dumps([[title, 0] for title in gold]),
        }

    rows = [
        row("q1", [["A", ["First fact."]], ["B", ["Second fact."]], ["X", ["Noise."]]], ["A", "B"]),
        row("q2", [["A", ["First fact."]], ["X", ["Additional evidence."]]], ["A", "X"]),
        row("q3", [["B", ["Second fact."]], ["Y", ["More noise."]]], ["B"]),
    ]
    for filename in ("dev.parquet", "train.parquet"):
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / filename)
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return rows, calls, tmp_path, row


def test_registered_metadata_needs_no_download():
    adapter = benchmark_registry.get("2wikimultihopqa")
    info = adapter.info()
    assert isinstance(adapter, TwoWikiMultihopQAAdapter)
    assert info.license == "Apache-2.0"
    assert info.paper == "arxiv:2011.01060"
    assert adapter.available_splits() == ["train", "validation"]
    assert adapter.corpus_provenance == "native"


def test_samples_and_corpus_keep_gold_and_distractor_paragraphs(dataset_files):
    adapter = TwoWikiMultihopQAAdapter()
    samples = adapter.load()
    corpus = adapter.load_corpus()
    assert [s.query_id for s in samples] == ["q1", "q2", "q3"]
    assert samples[0].ground_truth == "Answer q1"
    assert samples[0].hops == 2
    assert samples[0].metadata["type"] == "compositional"
    assert samples[0].expected_docs == {"A": 1, "B": 1}
    assert set(corpus) == {"A", "B", "X", "Y"}
    assert corpus["X"] == "X\n\nNoise.\n\nAdditional evidence."
    assert corpus["A"].count("First fact.") == 1
    assert all(set(s.expected_docs) <= set(corpus) for s in samples)


def test_both_load_paths_use_the_same_pinned_validation_file(dataset_files):
    _, calls, _, _ = dataset_files
    adapter = TwoWikiMultihopQAAdapter()
    adapter.load(cache_dir="chosen-cache")
    adapter.load_corpus(cache_dir="chosen-cache")
    assert len(calls) == 2
    for repo, filename, kwargs in calls:
        assert repo == "xanhho/2WikiMultihopQA"
        assert filename == "dev.parquet"
        assert kwargs["revision"] == adapter.hf_revision
        assert kwargs["cache_dir"] == "chosen-cache"
        assert kwargs["repo_type"] == "dataset"


def test_sampling_is_repeatable_and_train_does_not_change_the_default_corpus(dataset_files):
    _, calls, _, _ = dataset_files
    adapter = TwoWikiMultihopQAAdapter()
    first = adapter.load(n_samples=2, seed=42)
    second = adapter.load(n_samples=2, seed=42)
    assert first == second
    assert len(first) == 2
    assert len({s.query_id for s in first}) == 2
    assert adapter.load(n_samples=0) == []
    assert len(adapter.load(n_samples=10)) == 3
    adapter.load(split="train")
    assert calls[-1][1] == "train.parquet"
    adapter.load_corpus()
    assert calls[-1][1] == "dev.parquet"


def test_corpus_limit_is_truthful_and_does_not_drop_later_versions(dataset_files):
    adapter = TwoWikiMultihopQAAdapter()
    corpus = adapter.load_corpus(max_docs=3)
    assert set(corpus) == {"A", "B", "X"}
    assert "Additional evidence." in corpus["X"]
    assert adapter.corpus_truncated
    assert len(adapter.load_corpus(max_docs=4)) == 4
    assert not adapter.corpus_truncated
    assert len(adapter.load_corpus(max_docs=0)) == 4
    assert not adapter.corpus_truncated


@pytest.mark.parametrize("kwargs", [{"split": "test"}, {"n_samples": -1}, {"filters": {"x": 1}}])
def test_invalid_load_options_fail_before_download(dataset_files, kwargs):
    _, calls, _, _ = dataset_files
    with pytest.raises(ValueError):
        TwoWikiMultihopQAAdapter().load(**kwargs)
    assert calls == []


def test_negative_corpus_limit_is_rejected_before_download(dataset_files):
    _, calls, _, _ = dataset_files
    with pytest.raises(ValueError, match="max_docs"):
        TwoWikiMultihopQAAdapter().load_corpus(max_docs=-1)
    assert calls == []


def test_missing_gold_paragraph_is_an_error_not_a_silent_metric_change(dataset_files):
    rows, _, folder, _ = dataset_files
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows[0]["supporting_facts"] = json.dumps([["Absent", 0]])
    pq.write_table(pa.Table.from_pylist(rows), folder / "dev.parquet")
    with pytest.raises(ValueError, match="q1.*Absent"):
        TwoWikiMultihopQAAdapter().load()


def test_two_versions_of_a_title_in_the_same_question_are_retained(dataset_files):
    rows, _, folder, _ = dataset_files
    import pyarrow as pa
    import pyarrow.parquet as pq

    context = json.loads(rows[0]["context"])
    context.append(["A", ["Another paragraph from the same article."]])
    rows[0]["context"] = json.dumps(context)
    pq.write_table(pa.Table.from_pylist(rows), folder / "dev.parquet")
    corpus = TwoWikiMultihopQAAdapter().load_corpus()
    assert "First fact." in corpus["A"]
    assert "Another paragraph from the same article." in corpus["A"]
