"""Your own questions run as a benchmark, from a file, with no adapter to write.

Before this, a sweep could only name a registered benchmark: measuring a
pipeline on your own documents meant subclassing ``BaseBenchmarkAdapter``,
registering it and keeping its identity honest by hand. Nothing tied a result
to the file it came from, so an edited file resumed the old file's checkpoints
and shared its leaderboard group.

The fixtures are the ``smoke`` benchmark written out in each format, so these
run offline and a BEIR folder of it can be checked against ``smoke`` itself.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
from pathlib import Path

import pytest

from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.local_files import (
    CSV_CONTEXT_SEPARATOR,
    DataFileError,
    read_local_data,
)
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.smoke import _CORPUS, _QUERIES
from lasagna.knowledge_evaluation_sdk.experiment import Experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
OWN_DATA_DOC = REPO_ROOT / "docs" / "own-data.md"


@pytest.fixture(autouse=True)
def _registry_restored():
    """Sweeps register their files in the process-wide registry; other tests
    count its entries, so each test leaves it as it found it."""
    benchmark_registry.names()  # built-ins first, so they are in the snapshot
    before = dict(benchmark_registry._adapters)
    yield
    benchmark_registry._adapters.clear()
    benchmark_registry._adapters.update(before)


def _contexts(query):
    return [_CORPUS[d] for d in sorted(query["expected_docs"])]


def write_jsonl(path: Path, queries=_QUERIES) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            row = {
                "id": q["query_id"],
                "question": q["question"],
                "answer": q["ground_truth"],
                "contexts": _contexts(q),
            }
            f.write(json.dumps(row) + "\n")
    return path


def write_csv(path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "contexts"])
        for q in _QUERIES:
            writer.writerow(
                [
                    q["query_id"],
                    q["question"],
                    q["ground_truth"],
                    CSV_CONTEXT_SEPARATOR.join(_contexts(q)),
                ]
            )
    return path


def write_beir(folder: Path) -> Path:
    (folder / "qrels").mkdir(parents=True)
    with open(folder / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id, text in _CORPUS.items():
            f.write(json.dumps({"_id": doc_id, "title": "", "text": text}) + "\n")
    with open(folder / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in _QUERIES:
            f.write(json.dumps({"_id": q["query_id"], "text": q["question"]}) + "\n")
    with open(folder / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in _QUERIES:
            for doc_id, grade in q["expected_docs"].items():
                f.write(f"{q['query_id']}\t{doc_id}\t{grade}\n")
    return folder


WRITERS = {
    "jsonl": lambda tmp: write_jsonl(tmp / "my_data.jsonl"),
    "csv": lambda tmp: write_csv(tmp / "my_data.csv"),
    "beir": lambda tmp: write_beir(tmp / "my_beir"),
}


def _sweep(path, **extra):
    return Experiment.from_dict(
        {
            "name": "own",
            "benchmarks": [{"path": str(path), **extra.pop("bench", {})}],
            "tracks": extra.pop("tracks", ["retrieval"]),
            "pipeline": {"base_profile": "speed"},
            **extra,
        }
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", sorted(WRITERS))
def test_each_format_loads_with_its_relevance(tmp_path, fmt):
    data = read_local_data(WRITERS[fmt](tmp_path))
    assert data.kind == fmt
    assert len(data.splits["test"]) == len(_QUERIES)
    assert "retrieval" in data.supported_tracks()
    by_id = {r.query_id: r for r in data.splits["test"]}
    if fmt == "beir":
        # Graded judgments and the whole corpus, exactly as the file says.
        assert len(data.corpus) == len(_CORPUS)
        assert by_id["smoke_q1"].expected_docs == _QUERIES[0]["expected_docs"]
        assert data.supported_tracks() == ["retrieval"]
    else:
        # Each row's own contexts are what is relevant to it, in a pool of every
        # row's contexts.
        q1 = by_id["smoke_q1"]
        assert sorted(data.corpus[d] for d in q1.expected_docs) == sorted(_contexts(_QUERIES[0]))
        assert set(data.corpus.values()) == {t for q in _QUERIES for t in _contexts(q)}
        assert q1.answers == [_QUERIES[0]["ground_truth"]]
        assert data.supported_tracks() == ["retrieval", "generation"]


def test_a_sweep_entry_with_a_path_becomes_a_hashed_benchmark(tmp_path):
    path = write_jsonl(tmp_path / "my_data.jsonl")
    exp = _sweep(path)
    tasks = exp.generate_tasks()
    name = tasks[0].benchmark_name
    assert re.fullmatch(r"local/my_data@[0-9a-f]{12}", name), name
    assert exp.config.benchmarks[0].name == name, "the run metadata records the hashed name"
    adapter = benchmark_registry.get(name)
    assert adapter.hf_revision == "sha256:" + read_local_data(path).sha256
    samples = adapter.load(n_samples=5, seed=1)
    assert len(samples) == 5 and all(s.expected_docs for s in samples)


def test_a_named_entry_keeps_its_name_with_the_hash(tmp_path):
    path = write_jsonl(tmp_path / "my_data.jsonl")
    exp = _sweep(path, bench={"name": "acme-faq"})
    first = exp.generate_tasks()[0].benchmark_name
    assert re.fullmatch(r"acme-faq@[0-9a-f]{12}", first), first
    # Planning twice, as the CLI does, must not stack a second hash on the name.
    assert exp.generate_tasks()[0].benchmark_name == first


def test_a_path_resolves_against_the_sweep_files_folder(tmp_path, monkeypatch):
    (tmp_path / "study" / "data").mkdir(parents=True)
    write_jsonl(tmp_path / "study" / "data" / "faq.jsonl")
    sweep = tmp_path / "study" / "sweep.yaml"
    sweep.write_text(
        "name: own\nbenchmarks: [{path: data/faq.jsonl, n_samples: 4}]\n"
        "tracks: [retrieval]\npipeline: {base_profile: speed}\n"
    )
    monkeypatch.chdir(tmp_path)  # not the sweep's folder
    tasks = Experiment.from_file(str(sweep)).generate_tasks()
    assert tasks[0].benchmark_name.startswith("local/faq@")
    assert tasks[0].n_samples == 4


def test_a_jsonl_file_reads_answers_lists_single_context_and_unanswerable(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = [
        {"question": "Who?", "answers": ["Constantine", "Constantine I"], "context": "C."},
        {"question": "Why?", "answer": "", "contexts": ["P."], "metadata": {"unanswerable": True}},
        {"question": "What?", "answer": "glass", "contexts": ["G.", "C."], "id": 7},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    data = read_local_data(path)
    first, second, third = data.splits["test"]
    assert first.answers == ["Constantine", "Constantine I"] and first.query_id == "row1"
    assert second.unanswerable and second.expected_docs == {}, (
        "an unanswerable question has no relevant passage"
    )
    assert third.query_id == "7"
    # "C." is shared by two rows: one document, relevant to both.
    assert len(data.corpus) == 3
    assert set(first.expected_docs) <= set(third.expected_docs)
    s = third.sample()
    assert s.context == "G.\n\nC." and s.ground_truth == "glass"


def test_metadata_keys_are_filters(tmp_path):
    path = tmp_path / "typed.jsonl"
    rows = [
        {"question": f"q{i}", "answer": "a", "metadata": {"topic": "glass" if i % 2 else "coins"}}
        for i in range(6)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    name = _sweep(path, tracks=["generation"]).generate_tasks()[0].benchmark_name
    adapter = benchmark_registry.get(name)
    assert adapter.available_filters()["topic"] == ["coins", "glass"]
    assert len(adapter.load(filters={"topic": "glass"})) == 3
    assert len(adapter.load(filters={"unanswerable": False})) == 6
    with pytest.raises(ValueError, match="cannot filter on 'topc'"):
        adapter.load(filters={"topc": "glass"})


def test_named_documents_pair_with_contexts(tmp_path):
    path = tmp_path / "named.jsonl"
    rows = [
        {"question": "a?", "contexts": ["alpha text"], "expected_docs": ["doc-a"]},
        {"question": "b?", "contexts": ["alpha text"], "expected_docs": ["doc-a"]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    data = read_local_data(path)
    assert data.corpus == {"doc-a": "alpha text"}
    assert all(r.expected_docs == {"doc-a": 1} for r in data.rows())


# ---------------------------------------------------------------------------
# Validation: refused before anything loads, with the row and field named
# ---------------------------------------------------------------------------


def test_a_bad_row_names_its_row_and_field(tmp_path):
    path = write_jsonl(tmp_path / "my_data.jsonl")
    lines = path.read_text().splitlines()
    lines += ["", "", ""]  # blank lines still count, as an editor counts them
    bad = json.loads(lines[0])
    bad["answer"] = ["a list"]
    lines.append(json.dumps(bad | {"id": "q-bad"}))
    path.write_text("\n".join(lines) + "\n")
    assert len(path.read_text().splitlines()) == 16

    with pytest.raises(DataFileError) as info:
        read_local_data(path)
    assert str(info.value) == 'my_data.jsonl row 16: field "answer" must be a string, got list'

    # And a sweep naming it fails at planning, before any model loads.
    with pytest.raises(ValueError, match=r'row 16: field "answer" must be a string, got list'):
        _sweep(path).generate_tasks()


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"question": "q", "anwser": "a"}', 'unknown field "anwser" (did you mean "answer"?)'),
        ('{"answer": "a"}', 'field "question" is missing'),
        ('{"question": 3}', 'field "question" must be a string, got number'),
        ('{"question": "q", "answer": "a", "answers": ["a"]}', 'give "answer" or "answers"'),
        ('{"question": "q", "contexts": "one"}', 'field "contexts" must be a list of strings'),
        ('{"question": "q", "contexts": ["", "x"]}', 'field "contexts" item 1 is empty'),
        ('{"question": "q", "answer": ""}', 'field "answer" is empty'),
        ('{"question": "q", "metadata": {"unanswerable": "yes"}}', "must be true or false"),
        (
            '{"question": "q", "answer": "a", "metadata": {"unanswerable": true}}',
            "cannot also have an answer",
        ),
        ('{"question": "q", "expected_docs": ["d"]}', '"expected_docs" needs "contexts"'),
        ('{"question": "q", "contexts": ["x", "y"], "expected_docs": ["d"]}', "pair up in order"),
        ("[1, 2]", "must be a JSON object, got list"),
        ('{"question": "q"', "not valid JSON"),
    ],
)
def test_each_kind_of_bad_row_is_named(tmp_path, line, message):
    path = tmp_path / "f.jsonl"
    path.write_text('{"question": "fine", "answer": "ok"}\n' + line + "\n")
    with pytest.raises(DataFileError) as info:
        read_local_data(path)
    assert str(info.value).startswith("f.jsonl row 2: "), str(info.value)
    assert message in str(info.value)


def test_empty_files_and_duplicate_ids_are_refused(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(DataFileError, match="empty.jsonl: the file has no rows"):
        read_local_data(empty)

    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        '{"id": "a", "question": "q"}\n{"id": "b", "question": "q"}\n{"id": "a", "question": "r"}\n'
    )
    with pytest.raises(DataFileError, match='dup.jsonl row 3: id "a" is also used by row 1'):
        read_local_data(dup)


def test_a_csv_needs_a_header_and_names_its_rows(tmp_path):
    headerless = tmp_path / "h.csv"
    headerless.write_text("What is it?,glass,some passage\n")
    with pytest.raises(DataFileError, match="h.csv row 1: the first row must be a header"):
        read_local_data(headerless)

    bad = tmp_path / "b.csv"
    bad.write_text(
        f"question,answer,contexts\nok?,yes,passage\nempty?,no,a{CSV_CONTEXT_SEPARATOR}\n"
    )
    with pytest.raises(DataFileError, match=r'b.csv row 3: column "contexts" has an empty passage'):
        read_local_data(bad)

    typo = tmp_path / "t.csv"
    typo.write_text("question,anwser\nq,a\n")
    with pytest.raises(DataFileError, match=r'unknown column "anwser" \(did you mean "answer"\?\)'):
        read_local_data(typo)


def test_beir_judgments_of_unknown_documents_are_refused_with_examples(tmp_path):
    folder = write_beir(tmp_path / "my_beir")
    with open(folder / "qrels" / "test.tsv", "a") as f:
        f.write("smoke_q1\tnot_a_doc\t1\nsmoke_q2\talso_missing\t1\nghost_query\tglass_1\t1\n")
    with pytest.raises(DataFileError) as info:
        read_local_data(folder)
    text = str(info.value)
    assert (
        "my_beir/qrels/test.tsv: 2 judged document id(s) are not in corpus.jsonl, for example "
        '"also_missing", "not_a_doc"'
    ) in text
    assert '1 judged query id(s) are not in queries.jsonl, for example "ghost_query"' in text


def test_a_beir_folder_names_what_it_is_missing(tmp_path):
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "corpus.jsonl").write_text('{"_id": "d", "text": "t"}\n')
    with pytest.raises(DataFileError, match="missing queries.jsonl, qrels/test.tsv"):
        read_local_data(folder)


def test_an_unsupported_file_type_names_the_formats(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("[]")
    with pytest.raises(DataFileError, match=r"cannot read \.json .* Use a \.jsonl file"):
        read_local_data(path)


# ---------------------------------------------------------------------------
# Tracks: what each file can measure
# ---------------------------------------------------------------------------


def test_a_file_without_contexts_cannot_be_scored_on_retrieval(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text('{"question": "Who introduced the solidus?", "answer": "Constantine"}\n')
    data = read_local_data(path)
    assert data.supported_tracks() == ["generation"]
    with pytest.raises(ValueError) as info:
        _sweep(path, tracks=["retrieval"]).generate_tasks()
    message = " ".join(str(info.value).split())
    assert "qa.jsonl cannot be scored on retrieval: no row has contexts" in message
    assert "--tracks generation" in message and "closed_book" in message
    # Generation alone plans fine.
    assert _sweep(path, tracks=["generation"]).generate_tasks()


def test_a_beir_folder_cannot_be_scored_on_generation_alone(tmp_path):
    folder = write_beir(tmp_path / "my_beir")
    with pytest.raises(ValueError, match="relevance judgments but no answers"):
        _sweep(folder, tracks=["generation"]).generate_tasks()


def test_mode_resolution_measures_retrieval_when_there_is_something_to_retrieve(tmp_path):
    from lasagna.knowledge_evaluation_sdk.experiment import resolve_mode

    names = {}
    for fmt, write in WRITERS.items():
        names[fmt] = _sweep(write(tmp_path)).generate_tasks()[0].benchmark_name
    plain = tmp_path / "qa.jsonl"
    plain.write_text('{"question": "q", "answer": "a"}\n')
    names["plain"] = _sweep(plain, tracks=["generation"]).generate_tasks()[0].benchmark_name

    assert resolve_mode("auto", names["beir"]) == (
        "live",
        "benchmark provides a corpus, so retrieval can be measured",
    )
    for fmt in ("jsonl", "csv"):
        mode, reason = resolve_mode("auto", names[fmt])
        assert mode == "live" and "pooled" in reason
        assert benchmark_registry.get(names[fmt]).corpus_provenance == "pooled_gold"
    assert resolve_mode("auto", names["plain"])[0] == "generation_only"


# ---------------------------------------------------------------------------
# Identity: the content is in the name
# ---------------------------------------------------------------------------


def _comparability_key(benchmark_name):
    from lasagna.knowledge_evaluation_sdk.bundle import comparability_key, held_fixed_from
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    row = TaskResult(task_key="t", benchmark_name=benchmark_name, config_name="c", run_index=0)
    return comparability_key(held_fixed_from(row.to_dict(), {}))


def test_two_files_with_different_content_get_different_comparability_keys(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write_jsonl(tmp_path / "a" / "faq.jsonl")
    second = write_jsonl(tmp_path / "b" / "faq.jsonl", queries=_QUERIES[:6])
    copy = tmp_path / "copy" / "faq.jsonl"
    copy.parent.mkdir()
    copy.write_bytes(first.read_bytes())

    name_a = _sweep(first).generate_tasks()[0].benchmark_name
    name_b = _sweep(second).generate_tasks()[0].benchmark_name
    name_copy = _sweep(copy).generate_tasks()[0].benchmark_name
    assert name_a != name_b
    assert _comparability_key(name_a) != _comparability_key(name_b)
    # The same bytes under the same file name are the same benchmark.
    assert name_copy == name_a
    assert _comparability_key(name_copy) == _comparability_key(name_a)


def test_editing_the_file_means_an_old_checkpoint_is_not_resumed(tmp_path):
    from lasagna.knowledge_evaluation_sdk.checkpoint import CheckpointManager

    path = write_jsonl(tmp_path / "faq.jsonl")
    checkpoint = CheckpointManager(str(tmp_path / "out"))
    before = _sweep(path).generate_tasks()
    for task in before:
        checkpoint.save(task.task_key, {"metrics": {"ndcg@10": 0.5}})

    unchanged = {t.task_key for t in _sweep(path).generate_tasks()}
    assert checkpoint.completed_keys() & unchanged == unchanged, "an unedited file resumes"

    with open(path, "a") as f:
        f.write(json.dumps({"question": "A new question?", "answer": "yes"}) + "\n")
    after = _sweep(path).generate_tasks()
    assert after[0].benchmark_name != before[0].benchmark_name
    assert not checkpoint.completed_keys() & {t.task_key for t in after}


def test_the_folder_hash_covers_the_files_read_and_nothing_else(tmp_path):
    folder = write_beir(tmp_path / "my_beir")
    original = read_local_data(folder).sha256
    (folder / "README.md").write_text("notes")
    assert read_local_data(folder).sha256 == original, "a note beside the data changes nothing"
    with open(folder / "qrels" / "test.tsv", "a") as f:
        f.write("smoke_q3\ttardigrade_1\t1\n")
    assert read_local_data(folder).sha256 != original


# ---------------------------------------------------------------------------
# `lasagna benchmark validate-data`
# ---------------------------------------------------------------------------


def _cli(*args):
    from typer.testing import CliRunner

    from lasagna import cli

    result = CliRunner().invoke(cli.app, list(args))
    return result.exit_code, " ".join(result.output.split())


def test_validate_data_summarizes_a_valid_file_and_exits_zero(tmp_path):
    path = write_jsonl(tmp_path / "my_data.jsonl")
    code, out = _cli("benchmark", "validate-data", str(path))
    assert code == 0, out
    assert "my_data.jsonl: valid jsonl data" in out
    assert "Questions: 12" in out and "With answers: 12" in out and "With contexts: 12" in out
    assert "Unanswerable: 0" in out
    assert "Retrieval: yes" in out and "Generation: yes" in out
    assert re.search(r"Benchmark name: local/my_data@[0-9a-f]{12}", out)
    assert "Traceback" not in out


def test_validate_data_lists_every_problem_and_exits_one(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "q", "answer": ["x"]}\n{"question": "r", "anwser": "y"}\n')
    code, out = _cli("benchmark", "validate-data", str(path))
    assert code == 1
    assert 'bad.jsonl row 1: field "answer" must be a string, got list' in out
    assert 'bad.jsonl row 2: unknown field "anwser"' in out
    assert "2 problems" in out and "Traceback" not in out


def test_validate_data_says_which_tracks_a_file_cannot_support(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text('{"question": "q", "answer": "a"}\n')
    code, out = _cli("benchmark", "validate-data", str(path))
    assert code == 0
    assert "Retrieval: no, no row has contexts" in out and "Generation: yes" in out
    assert "closed_book" in out

    nothing = tmp_path / "questions_only.jsonl"
    nothing.write_text('{"question": "q"}\n')
    code, out = _cli("benchmark", "validate-data", str(nothing))
    assert code == 1 and "Nothing can be measured" in out

    code, out = _cli("benchmark", "validate-data", str(write_beir(tmp_path / "my_beir")))
    assert code == 0 and "Retrieval: yes" in out and "Generation: no" in out


def test_validate_data_on_a_missing_path_exits_one(tmp_path):
    code, out = _cli("benchmark", "validate-data", str(tmp_path / "nope.jsonl"))
    assert code == 1 and "no such file or folder" in out


# ---------------------------------------------------------------------------
# The documentation
# ---------------------------------------------------------------------------


def _fenced(lang):
    return re.findall(rf"```{lang}\n(.*?)```", OWN_DATA_DOC.read_text(encoding="utf-8"), re.S)


def test_the_documented_examples_are_valid_data(tmp_path):
    jsonl, csv_blocks = _fenced("jsonl"), _fenced("csv")
    assert jsonl and csv_blocks, "docs/own-data.md shows a JSONL and a CSV example"
    for i, block in enumerate(jsonl):
        path = tmp_path / f"example{i}.jsonl"
        path.write_text(block)
        read_local_data(path)
    for i, block in enumerate(csv_blocks):
        path = tmp_path / f"example{i}.csv"
        path.write_text(block)
        read_local_data(path)


def test_the_documented_commands_exist():
    from lasagna.cli import benchmark_app

    commands = {c.name for c in benchmark_app.registered_commands}
    named = set(re.findall(r"lasagna benchmark ([a-z][a-z-]*)", OWN_DATA_DOC.read_text()))
    assert {"validate-data", "run"} <= named
    assert named <= commands, sorted(named - commands)


# ---------------------------------------------------------------------------
# A real retrieval sweep, speed profile, each format
# ---------------------------------------------------------------------------


def _run(exp, out_dir):
    exp.config.output_dir = str(out_dir)
    return asyncio.run(exp.run())


def _ndcg(results, benchmark):
    rows = [r for r in results._results if r.benchmark_name == benchmark and not r.error]
    assert rows, [r.error for r in results._results]
    return rows[0]


@pytest.mark.needs_embedding_weights
@pytest.mark.parametrize("fmt", sorted(WRITERS))
def test_each_format_runs_a_retrieval_sweep_on_the_speed_profile(tmp_path, fmt):
    path = WRITERS[fmt](tmp_path)
    exp = _sweep(path)
    results = _run(exp, tmp_path / "out")
    name = exp.config.benchmarks[0].name
    row = _ndcg(results, name)
    assert row.n_samples == len(_QUERIES)
    assert row.metrics["ndcg@10"] > 0.5, row.metrics
    assert row.corpus_provenance == ("native" if fmt == "beir" else "pooled_gold")
    assert row.hf_revision == "sha256:" + read_local_data(path).sha256


@pytest.mark.needs_embedding_weights
def test_a_beir_folder_of_smoke_scores_what_smoke_scores(tmp_path):
    """Same documents, queries and judgments: the same numbers, to the digit."""
    folder = write_beir(tmp_path / "smoke_beir")
    exp = Experiment.from_dict(
        {
            "name": "twin",
            "benchmarks": [{"name": "smoke"}, {"path": str(folder)}],
            "tracks": ["retrieval"],
            "pipeline": {"base_profile": "speed"},
        }
    )
    results = _run(exp, tmp_path / "out")
    ours = _ndcg(results, exp.config.benchmarks[1].name)
    theirs = _ndcg(results, "smoke")

    def scores(row):
        return {k: v for k, v in row.metrics.items() if "latency" not in k and "_ms" not in k}

    assert "ndcg@10" in scores(theirs) and "recall@10" in scores(theirs)
    assert scores(ours) == scores(theirs)


@pytest.mark.needs_embedding_weights
def test_benchmark_run_takes_a_file_path(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "my_data.jsonl")
    monkeypatch.chdir(tmp_path)
    code, out = _cli(
        "benchmark",
        "run",
        "./my_data.jsonl",
        "--profile",
        "speed",
        "--tracks",
        "retrieval",
        "-o",
        str(tmp_path / "out"),
    )
    assert code == 0, out
    assert re.search(r"local/my_data@[0-9a-f]{12} 1\. speed", out), out
