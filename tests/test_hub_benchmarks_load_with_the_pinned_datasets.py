"""The Hub benchmarks load with the `datasets` release Lasagna pins (5.1.0).

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


@pytest.mark.parametrize("offline", [False, True])
def test_fever_loads_its_parquet_conversion_online_and_from_the_cache(monkeypatch, offline):
    # The Hub converted the FEVER script to parquet. Its splits are fetched at
    # the pinned commit and the requested one read as plain parquet, so a
    # cached split loads offline too.
    downloads, reads = [], []

    def fake_snapshot(repo, **kwargs):
        downloads.append((repo, kwargs))
        return "/hub/fever"

    def fake_load(builder, **kwargs):
        reads.append((builder, kwargs))
        raise RuntimeError("stop after the read is recorded")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    monkeypatch.setattr("datasets.load_dataset", fake_load)
    monkeypatch.setattr(ds, "_offline", lambda: offline)
    with pytest.raises(RuntimeError):
        ds.load_benchmark("fever", split="labelled_dev", n_samples=1, cache_dir="/nonexistent")
    assert downloads == [
        (
            "fever/fever",
            {
                "repo_type": "dataset",
                "revision": "5f577157472532aa1d9924d2df63aac44f70cf2b",  # pragma: allowlist secret
                # Every split, as load_dataset fetches them: the offline check
                # that the dataset is cached then holds for any split.
                "allow_patterns": [
                    f"v1.0/{s}/*.parquet"
                    for s in sorted(("labelled_dev", "paper_dev", "paper_test", "train"))
                ],
                "local_files_only": offline,
            },
        )
    ]
    assert reads == [
        (
            "parquet",
            {
                "data_files": {"labelled_dev": "/hub/fever/v1.0/labelled_dev/*.parquet"},
                "split": "labelled_dev",
            },
        )
    ]


def test_a_split_fever_does_not_have_lists_the_ones_it_has(monkeypatch, tmp_path):
    def no_download(*args, **kwargs):
        raise AssertionError("a missing split downloads nothing")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(ValueError, match="available: train, labelled_dev, paper_dev, paper_test"):
        ds.load_benchmark("fever", split="validation", n_samples=1, cache_dir=str(tmp_path))


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
    # A benchmark whose rows produce zero samples after conversion raises an
    # actionable error instead of returning an empty benchmark.
    import datasets

    rows = datasets.Dataset.from_list([{"id": "q1", "input": "", "output": [{"answer": "x"}]}])
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    with pytest.raises(
        ValueError,
        match="has 1 rows, but the .kilt. converter made no sample from any of them",
    ):
        ds.load_benchmark("kilt/nq", split="validation", cache_dir=str(tmp_path))


def _fake_kilt_and_triviaqa_loader(kilt_rows, triviaqa_rows, calls):
    import datasets

    kilt_ds = datasets.Dataset.from_list(kilt_rows)
    tqa_ds = datasets.Dataset.from_list(triviaqa_rows)

    def _load(repo, split=None, **kwargs):
        calls.append((repo, split, dict(kwargs)))
        if repo == "facebook/kilt_tasks":
            return kilt_ds
        if repo == "mandarjoshi/trivia_qa":
            return tqa_ds
        raise AssertionError(f"Unexpected repo {repo}")

    return _load


def test_kilt_triviaqa_loads_both_pinned_datasets_and_joins_by_id(monkeypatch, tmp_path):
    calls = []
    # Intentionally reversed/non-positional order vs KILT rows.
    kilt_rows = [
        {"id": "sfq_2", "input": "", "meta": {"src": "b"}, "output": [{"answer": "Paris"}]},
        {"id": "sfq_1", "input": "", "meta": {"src": "a"}, "output": [{"answer": "Jim Laker"}]},
    ]
    tqa_rows = [
        {"question_id": "sfq_1", "question": "Who took 19 wickets in a Test match?"},
        {"question_id": "other_9", "question": "Unrelated question?"},
        {"question_id": "sfq_2", "question": "What is the capital of France?"},
    ]
    monkeypatch.setattr(
        "datasets.load_dataset",
        _fake_kilt_and_triviaqa_loader(kilt_rows, tqa_rows, calls),
    )

    samples = ds.load_benchmark("kilt/triviaqa", split="validation", cache_dir=str(tmp_path))

    assert calls == [
        (
            "facebook/kilt_tasks",
            "validation",
            {
                "name": "triviaqa_support_only",
                "revision": "0f7881c6bf693742af91a31b4d8f827db2f41233",  # pragma: allowlist secret
            },
        ),
        (
            "mandarjoshi/trivia_qa",
            "validation",
            {
                "name": "rc.wikipedia.nocontext",
                "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",  # pragma: allowlist secret
            },
        ),
    ]
    assert samples == [
        {"question": "What is the capital of France?", "ground_truth": "Paris"},
        {"question": "Who took 19 wickets in a Test match?", "ground_truth": "Jim Laker"},
    ]


def test_kilt_triviaqa_excludes_unmatched_logs_counts_and_respects_n_samples(monkeypatch, tmp_path):
    from loguru import logger

    calls = []
    kilt_rows = [
        {"id": f"id_{i}", "input": "", "meta": {"i": i}, "output": [{"answer": f"ans_{i}"}]}
        for i in range(7)
    ]
    # 5 matched (id_0..id_4), 2 unmatched (id_5, id_6).
    tqa_rows = [{"question_id": f"id_{i}", "question": f"Question {i}?"} for i in range(5)]
    monkeypatch.setattr(
        "datasets.load_dataset",
        _fake_kilt_and_triviaqa_loader(kilt_rows, tqa_rows, calls),
    )

    records = []
    hid = logger.add(lambda msg: records.append(str(msg)), level="INFO")
    try:
        samples = ds.load_benchmark(
            "kilt/triviaqa", split="validation", n_samples=5, seed=42, cache_dir=str(tmp_path)
        )
    finally:
        logger.remove(hid)

    assert len(samples) == 5
    assert all(
        s["question"].startswith("Question ") and s["ground_truth"].startswith("ans_")
        for s in samples
    )
    joined_logs = "\n".join(records)
    assert "5/7 matched (2 unmatched)" in joined_logs
    assert "skipped 2 of 7" in joined_logs


def test_kilt_triviaqa_raises_when_zero_rows_match(monkeypatch, tmp_path):
    calls = []
    kilt_rows = [{"id": "sfq_1", "input": "", "meta": {}, "output": [{"answer": "x"}]}]
    tqa_rows = [{"question_id": "different_id", "question": "Q?"}]
    monkeypatch.setattr(
        "datasets.load_dataset",
        _fake_kilt_and_triviaqa_loader(kilt_rows, tqa_rows, calls),
    )

    with pytest.raises(ValueError, match="matched 0 of 1 rows"):
        ds.load_benchmark("kilt/triviaqa", split="validation", cache_dir=str(tmp_path))


def test_fill_kilt_questions_preserves_metadata_handles_duplicates_and_does_not_mutate():
    import datasets

    kilt_rows = [
        {"id": "dup_1", "input": "", "meta": {"tag": "keep"}, "output": [{"answer": "A1"}]},
        {"id": "", "input": "", "meta": {"tag": "missing_id"}, "output": [{"answer": "A2"}]},
    ]
    tqa_rows = [
        {"question_id": "dup_1", "question": "First question?"},
        {"question_id": "dup_1", "question": "Second duplicate question?"},
        {"question_id": "empty_q", "question": ""},
        {"question_id": None, "question": "No ID?"},
    ]
    kilt_ds = datasets.Dataset.from_list(kilt_rows)
    tqa_ds = datasets.Dataset.from_list(tqa_rows)

    joined, matched_count, unmatched_count = ds._fill_kilt_questions(kilt_ds, tqa_ds)

    assert matched_count == 1
    assert unmatched_count == 1
    assert len(joined) == 1
    assert joined[0]["id"] == "dup_1"
    assert joined[0]["input"] == "First question?"
    assert joined[0]["meta"] == {"tag": "keep"}
    assert joined[0]["output"] == [{"answer": "A1"}]
    # Original input dataset was not mutated.
    assert kilt_ds[0]["input"] == ""


def test_kilt_triviaqa_uses_actual_split_on_split_fallback(monkeypatch, tmp_path):
    import datasets

    calls = []
    kilt_ds = datasets.Dataset.from_list(
        [{"id": "sfq_1", "input": "", "meta": {}, "output": [{"answer": "42"}]}]
    )
    tqa_ds = datasets.Dataset.from_list([{"question_id": "sfq_1", "question": "What is 6x7?"}])

    def _load(repo, split=None, **kwargs):
        calls.append((repo, split))
        if repo == "facebook/kilt_tasks" and split == "test":
            raise ValueError('Unknown split "test". Should be one of ["validation"].')
        if repo == "facebook/kilt_tasks" and split == "validation":
            return kilt_ds
        if repo == "mandarjoshi/trivia_qa":
            return tqa_ds
        raise AssertionError(f"Unexpected call: {repo}, {split}")

    monkeypatch.setattr("datasets.load_dataset", _load)
    monkeypatch.setattr(ds, "_available_splits", lambda *a, **k: ["validation"])

    samples = ds.load_benchmark(
        "kilt/triviaqa",
        split="test",
        cache_dir=str(tmp_path),
        allow_split_fallback=True,
    )
    assert calls == [
        ("facebook/kilt_tasks", "test"),
        ("facebook/kilt_tasks", "validation"),
        ("mandarjoshi/trivia_qa", "validation"),
    ]
    assert samples == [{"question": "What is 6x7?", "ground_truth": "42"}]


def test_kilt_nq_continues_using_single_dataset_path(monkeypatch, tmp_path):
    import datasets

    calls = []
    rows = datasets.Dataset.from_list(
        [{"id": "nq_1", "input": "Who wrote Hamlet?", "output": [{"answer": "Shakespeare"}]}]
    )

    def _load(repo, split=None, **kwargs):
        calls.append((repo, split, dict(kwargs)))
        return rows

    monkeypatch.setattr("datasets.load_dataset", _load)

    samples = ds.load_benchmark("kilt/nq", split="validation", cache_dir=str(tmp_path))
    assert len(calls) == 1
    assert calls[0][0] == "facebook/kilt_tasks"
    assert samples == [{"question": "Who wrote Hamlet?", "ground_truth": "Shakespeare"}]
