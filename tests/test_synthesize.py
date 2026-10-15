"""A development question set, written by an LLM from your own documents.

Without it, measuring a pipeline on your own documents meant writing the
questions by hand first. These tests drive `synthesize` with a scripted LLM, so
they check what is asked of the model and what is done with its replies: the
requested mix, multi-hop questions that really span two documents, the check
that drops unsupported questions, a dry run that costs nothing, and output that
`lasagna benchmark run` takes as it is.
"""

from __future__ import annotations

import asyncio
import collections
import json
import re
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import synthesize as synth
from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.local_files import read_local_data
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.smoke import _CORPUS


@pytest.fixture(autouse=True)
def _registry_restored():
    benchmark_registry.names()
    before = dict(benchmark_registry._adapters)
    yield
    benchmark_registry._adapters.clear()
    benchmark_registry._adapters.update(before)


@pytest.fixture
def docs(tmp_path):
    """The smoke corpus as four documents, one per subject."""
    folder = tmp_path / "docs"
    folder.mkdir()
    by_subject = collections.defaultdict(list)
    for doc_id, text in _CORPUS.items():
        by_subject[doc_id.rsplit("_", 1)[0]].append(text)
    for subject, texts in by_subject.items():
        (folder / f"{subject}.md").write_text("\n\n".join(texts))
    (folder / ".hidden.md").write_text("never read " * 30)
    (folder / "image.png").write_bytes(b"\x89PNG")
    return folder


class ScriptedLLM:
    """Writes a question from the passages it is shown; checks with ``verdict``."""

    model_name = "scripted-test-model"

    def __init__(self, verdict=None, writer=None):
        self.prompts = []
        self.verdict = verdict or (lambda question, kind: "yes" if kind == "supported" else "no")
        self.writer = writer

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("You are checking"):
            kind = "supported" if "Reference answer:" in prompt else "unanswered"
            question = re.search(r"^Question: (.*)$", prompt, re.M).group(1)
            return SimpleNamespace(content=self.verdict(question, kind))
        if self.writer is not None:
            return SimpleNamespace(content=self.writer(prompt))
        passages = re.findall(r"Passage(?: [AB])?:\n(.*?)\n\n", prompt, re.S)
        first = passages[0].split()
        if len(passages) == 2:
            other = passages[1].split()
            reply = {"question": f"How do {first[1]} and {other[1]} relate?", "answer": "x"}
        elif '"answer": "..."' in prompt:
            reply = {"question": f"What about {' '.join(first[:4])}?", "answer": first[0]}
        else:
            reply = {"question": f"What is never said about {first[0]}?"}
        return SimpleNamespace(content="Here it is:\n" + json.dumps(reply))


def _plan(docs, n=10, mix="factoid=0.5,multihop=0.3,unanswerable=0.2", seed=7):
    return synth.plan_synthesis(docs, n, mix, seed, chunk_words=40)


def _types(rows):
    return collections.Counter(r["metadata"]["question_type"] for r in rows)


def test_the_requested_mix_is_written_and_each_row_says_how_it_was_made(docs):
    llm = ScriptedLLM()
    rows = synth.synthesize(
        docs, 10, "factoid=0.5,multihop=0.3,unanswerable=0.2", seed=7, llm=llm, chunk_words=40
    )
    assert _types(rows) == {"factoid": 5, "multihop": 3, "unanswerable": 2}
    assert len(llm.prompts) == 20, "one call writes each question and one checks it"
    assert len({r["id"] for r in rows}) == 10

    prompts = {
        "factoid": synth.FACTOID_PROMPT,
        "multihop": synth.MULTIHOP_PROMPT,
        "unanswerable": synth.UNANSWERABLE_PROMPT,
    }
    for row in rows:
        meta = row["metadata"]
        kind = meta["question_type"]
        assert set(meta) == {
            "question_type",
            "source_doc_ids",
            "prompt_id",
            "prompt_sha256",
            "model",
            "seed",
            "synthetic",
            "unanswerable",
        }
        assert meta["prompt_id"] == prompts[kind].id
        assert meta["prompt_sha256"] == prompts[kind].sha256
        assert len(meta["prompt_sha256"]) == 64
        assert meta["model"] == "scripted-test-model"
        assert meta["seed"] == 7 and meta["synthetic"] is True
        assert meta["unanswerable"] is (kind == "unanswerable")
        assert ("answer" in row) is (kind != "unanswerable")
        assert row["contexts"] and all(row["contexts"])
        assert all(not d.startswith(".") and d.endswith(".md") for d in meta["source_doc_ids"])


def test_multihop_rows_cite_two_documents(docs):
    rows = synth.synthesize(docs, 12, "multihop=1", seed=3, llm=ScriptedLLM(), chunk_words=40)
    assert len(rows) == 12
    for row in rows:
        assert len(row["contexts"]) == 2
        assert len(set(row["metadata"]["source_doc_ids"])) == 2, row["metadata"]


def test_the_check_drops_a_question_the_judge_calls_unsupported(docs):
    plan = _plan(docs)
    rejected = {}

    def verdict(question, kind):
        # The judge rejects the first answerable question (its answer is not
        # supported) and the first unanswerable one (the passages answer it).
        first = rejected.setdefault(kind, question) == question
        if kind == "supported":
            return "no" if first else "yes"
        return "yes" if first else "no"

    result = synth.run_synthesis(plan, ScriptedLLM(verdict))
    assert result.dropped == {"unparseable": 0, "unsupported": 1, "answered": 1, "unclear": 0}
    assert len(result.rows) == 8
    kept = [r["question"] for r in result.rows]
    assert rejected["supported"] not in kept and rejected["unanswered"] not in kept


def test_a_reply_that_is_not_the_asked_for_json_or_verdict_is_dropped_and_counted(docs):
    plan = _plan(docs, n=6, mix="factoid=1")
    replies = iter(["not json at all", '{"question": "Only a question?"}'])

    def writer(prompt):
        return next(replies, json.dumps({"question": "Fine?", "answer": "yes"}))

    result = synth.run_synthesis(plan, ScriptedLLM(lambda q, k: "maybe", writer))
    assert result.dropped["unparseable"] == 2
    assert result.dropped["unclear"] == 4
    assert result.rows == []


def test_an_unanswerable_question_is_checked_against_the_closest_passages_too(docs):
    llm = ScriptedLLM()
    synth.run_synthesis(_plan(docs, n=2, mix="unanswerable=1"), llm)
    checks = [p for p in llm.prompts if p.startswith("You are checking")]
    assert len(checks) == 2
    for prompt in checks:
        n_passages = len(re.findall(r"^\[\d+\] ", prompt, re.M))
        assert n_passages == 1 + synth.UNANSWERED_CHECK_NEIGHBORS


def test_the_same_seed_gives_the_same_set(docs):
    def rows(seed):
        return synth.synthesize(docs, 8, seed=seed, llm=ScriptedLLM(), chunk_words=40)

    assert rows(5) == rows(5)
    assert [i.passages for i in _plan(docs, seed=5).items] != [
        i.passages for i in _plan(docs, seed=6).items
    ]


def test_the_plan_reads_files_only(docs):
    plan = synth.plan_synthesis(docs, 20, None, 42)
    assert plan.documents == ["coin.md", "glass.md", "light.md", "tardigrade.md"]
    assert ("image.png", "unsupported type .png") in plan.skipped
    assert plan.counts == {"factoid": 12, "multihop": 6, "unanswerable": 2}
    assert plan.llm_calls == 40
    tokens_in, tokens_out = plan.token_estimate()
    assert tokens_in > 0 and tokens_out > 0


def test_split_counts_add_up():
    assert synth.split_counts(200, synth.DEFAULT_MIX) == {
        "factoid": 120,
        "multihop": 60,
        "unanswerable": 20,
    }
    for n in range(1, 30):
        assert sum(synth.split_counts(n, synth.DEFAULT_MIX).values()) == n


@pytest.mark.parametrize(
    "mix, message",
    [
        ("factoid=0.5,multihop=0.4", "must add up to 1, got 0.9"),
        ("factiod=1", "Unknown question type 'factiod' in --mix (did you mean 'factoid'?)"),
        ("factoid=1.2,multihop=-0.2", "must not be negative"),
        ("factoid", "type=share pairs"),
        ("factoid=lots", "must be a number"),
    ],
)
def test_a_bad_mix_is_refused(mix, message):
    with pytest.raises(ValueError) as info:
        synth.parse_mix(mix)
    assert message in str(info.value)


def test_a_corpus_that_cannot_support_the_plan_is_refused(tmp_path, docs):
    with pytest.raises(ValueError, match="no such file or folder"):
        synth.plan_synthesis(tmp_path / "missing", 5)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="has no document Lasagna can read"):
        synth.plan_synthesis(empty, 5)
    with pytest.raises(ValueError, match="multihop questions need passages from two documents"):
        synth.plan_synthesis(docs / "glass.md", 5, "multihop=1")
    with pytest.raises(ValueError, match="--n must be at least 1"):
        synth.plan_synthesis(docs, 0)
    # One document is enough without multi-hop.
    assert synth.plan_synthesis(docs / "glass.md", 3, "factoid=1", chunk_words=40).items


def test_the_llm_is_the_configured_provider_at_temperature_zero(monkeypatch):
    import lasagna.providers as providers

    calls = []
    provider = SimpleNamespace(get_llm=lambda **kw: calls.append(kw) or "llm")
    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    assert synth.default_llm() == "llm"
    assert synth.default_llm("some-model") == "llm"
    assert calls == [{"temperature": 0}, {"model_name": "some-model", "temperature": 0}]


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _cli(*args):
    from typer.testing import CliRunner

    from lasagna import cli

    result = CliRunner().invoke(cli.app, list(args))
    return result.exit_code, " ".join(result.output.split())


def test_a_dry_run_calls_no_llm_and_writes_nothing(docs, tmp_path, monkeypatch):
    def no_llm(*args, **kwargs):
        raise AssertionError("the dry run built an LLM")

    monkeypatch.setattr(synth, "default_llm", no_llm)
    out = tmp_path / "plan.jsonl"
    code, text = _cli(
        "benchmark", "synthesize", str(docs), "--n", "20", "--out", str(out), "--dry-run"
    )
    assert code == 0, text
    assert not out.exists()
    assert "Documents: 4" in text and "skipped image.png" in text
    assert "Questions: 20 (factoid 12, multihop 6, unanswerable 2)" in text
    assert "LLM calls: 40" in text
    assert re.search(r"Tokens: about [\d,]+ in and [\d,]+ out \(an estimate", text)
    assert "Dry run: no LLM was called and nothing was written." in text


def test_a_missing_llm_names_the_variables_to_set(docs, tmp_path, monkeypatch):
    import lasagna.providers as providers

    def not_configured():
        raise providers.LLMProviderNotConfigured("No LLMProvider configured.")

    monkeypatch.setattr(providers, "get_llm_provider", not_configured)
    out = tmp_path / "x.jsonl"
    code, text = _cli("benchmark", "synthesize", str(docs), "--n", "4", "--out", str(out))
    assert code == 1 and not out.exists()
    assert "needs an LLM, and none is configured" in text
    assert "OPENAI_API_KEY" in text and "ANTHROPIC_API_KEY" in text and "--dry-run" in text
    assert "Traceback" not in text


def _write_with_cli(docs, out, monkeypatch, *extra):
    llm = ScriptedLLM()
    monkeypatch.setattr(synth, "default_llm", lambda model=None: llm)
    code, text = _cli(
        "benchmark",
        "synthesize",
        str(docs),
        "--n",
        "10",
        "--out",
        str(out),
        "--seed",
        "4",
        *extra,
    )
    return code, text, llm


def test_the_output_validates_and_runs_as_own_data(docs, tmp_path, monkeypatch):
    out = tmp_path / "my_bench.jsonl"
    code, text, llm = _write_with_cli(docs, out, monkeypatch)
    assert code == 0, text
    assert "Wrote 10 questions" in text and "model scripted-test-model" in text
    assert "for development" in text

    code, text = _cli("benchmark", "validate-data", str(out))
    assert code == 0, text
    assert "Questions: 10" in text and "Unanswerable: 1" in text
    assert "Retrieval: yes" in text and "Generation: yes" in text

    from lasagna.knowledge_evaluation_sdk.experiment import Experiment, resolve_mode

    exp = Experiment.from_dict(
        {
            "name": "synthetic",
            "benchmarks": [{"path": str(out), "filters": {"question_type": "multihop"}}],
            "tracks": ["retrieval"],
            "pipeline": {"base_profile": "speed"},
        }
    )
    task = exp.generate_tasks()[0]
    assert resolve_mode("auto", task.benchmark_name)[0] == "live"
    samples = benchmark_registry.get(task.benchmark_name).load(filters=task.filters)
    assert len(samples) == 3 and all(len(s.expected_docs) == 2 for s in samples)
    data = read_local_data(out)
    assert sum(r.unanswerable for r in data.rows()) == 1


def test_an_existing_file_is_not_replaced_without_overwrite(docs, tmp_path, monkeypatch):
    out = tmp_path / "keep.jsonl"
    out.write_text("hand edited\n")
    code, text, llm = _write_with_cli(docs, out, monkeypatch)
    assert code == 1 and "--overwrite" in text
    assert out.read_text() == "hand edited\n" and not llm.prompts
    code, text, _ = _write_with_cli(docs, out, monkeypatch, "--overwrite")
    assert code == 0 and len(out.read_text().splitlines()) == 10


@pytest.mark.needs_embedding_weights
def test_a_synthetic_set_runs_a_retrieval_sweep(docs, tmp_path, monkeypatch):
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment

    out = tmp_path / "my_bench.jsonl"
    code, text, _ = _write_with_cli(docs, out, monkeypatch)
    assert code == 0, text
    exp = Experiment.from_dict(
        {
            "name": "synthetic",
            "benchmarks": [{"path": str(out)}],
            "tracks": ["retrieval"],
            "pipeline": {"base_profile": "speed"},
            "output_dir": str(tmp_path / "run"),
        }
    )
    results = asyncio.run(exp.run())
    (row,) = results._results
    assert row.error is None
    assert row.n_samples == 9, "the unanswerable question has nothing relevant to retrieve"
    assert row.corpus_provenance == "pooled_gold"
    assert 0.0 < row.metrics["ndcg@10"] <= 1.0
