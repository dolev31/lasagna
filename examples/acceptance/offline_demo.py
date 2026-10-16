"""Every feature Lasagna 0.1.0 adds, end to end, offline, in under five minutes.

    python examples/acceptance/offline_demo.py

Each step does what a user would do, through the ``lasagna`` command (run in
this process with Typer's ``CliRunner``, so the stub LLM below answers) or the
public Python API, and checks what that user would see:

1. ``benchmark synthesize --dry-run`` calls no LLM and writes nothing; the real
   run writes 12 questions from the smoke documents, each marked synthetic.
2. ``benchmark validate-data`` accepts that file, and names the row and the
   field of a deliberately broken copy.
3. ``lasagna.from_langchain`` turns a LangChain-style document compressor into
   a reranker.
4. ``benchmark experiment --dry-run`` lists the arms, tasks and models of the
   sweep and writes no results.
5. The sweep runs five arms (dense, the wrapped reranker, ``closed_book``,
   ``long_context`` and ``no_rag: true``) on both tracks; the arms that read no
   passage have no faithfulness, and the leaderboard says why.
6. The same sweep again replays every LLM response from the cache: hit rate
   1.0, no LLM call, the same numbers.
7. ``results bundle``, ``results view`` and ``results leaderboard --format latex``.
8. ``lasagna-mcp --help`` answers in under 2 s, and the server's default tool
   list over stdio has no tool that deletes data.

Nothing here needs a network or an API key. The LLM is a deterministic stub put
in through ``set_llm_provider``: a ``SimpleLLMProvider`` whose network call is
replaced, so token counting, temperature pinning and the response cache run as
they do for a real provider. The RAGAS judge is a stub too, scoring by word
overlap. Retrieval is real: the ``speed`` profile's all-MiniLM-L6-v2 from the
Hugging Face cache, with ``HF_HUB_OFFLINE=1``, and a socket guard refuses every
connection that is not to this machine. The demo makes its own HOME inside its
working folder, so it starts clean and leaves your ``~/.lasagna`` alone.

It prints one line per step with its time, and exits 1 at the first check that
fails, with the reason. Exit 2 means it could not start: a missing extra, or
the embedding model not in the cache (``--download-models`` fetches it once,
online).

    --workdir DIR      run in DIR (default: a new temporary folder)
    --keep             keep the working folder (it is kept anyway on failure)
    --report FILE      also write the step results as JSON
    --download-models  download the models the demo needs, then exit
"""

from __future__ import annotations

import argparse
import atexit
import copy
import html
import importlib.util
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: The models the demo loads: the speed profile's embedder, which is also the
#: answer-similarity model.
MODELS = ("sentence-transformers/all-MiniLM-L6-v2",)
STUB_MODEL = "offline-stub"
STUB_JUDGE = "offline-stub-judge"
N_QUESTIONS = 12
MCP_HELP_BUDGET_S = 2.0
#: Tools that delete data or reset configuration. The default tool set must
#: list none of them; the pattern catches one added later under a new name.
DELETION_TOOLS = {
    "clear_session",
    "bulk_delete_session",
    "rebuild_collection",
    "delete_document",
    "delete_rag_profile",
    "reset_rag_config",
}
DELETION_PATTERN = re.compile(r"^(delete|bulk_delete|clear|reset|rebuild|purge|drop|remove)_")
ARMS = ["dense", "reranked", "closed_book", "long_context", "no_rag"]
NO_CONTEXT_ARMS = ["closed_book", "no_rag"]  # sorted

#: Every prompt the stub LLM was asked, in order.
CALLS: List[str] = []
#: Hosts the socket guard refused.
REFUSED: List[str] = []


class DemoFailure(Exception):
    """A check failed. The message is the one line the demo prints."""


class CannotRun(Exception):
    """The demo cannot start here (exit 2): a missing extra or model."""


def check(condition: Any, reason: str) -> None:
    if not condition:
        raise DemoFailure(reason)


# ---------------------------------------------------------------------------
# The environment: offline, credential-free, a HOME of its own
# ---------------------------------------------------------------------------


def _hf_home() -> str:
    """Where the Hugging Face cache is, resolved before HOME is replaced."""
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "huggingface")


def _prepare_environment(home: Path) -> None:
    """Make the run hermetic before anything imports lasagna.

    Settings and credentials from the caller's shell would change what runs:
    ``RAG_*`` and ``LASAGNA_*`` settings, a ``VECTOR_BACKEND``, an API key that
    a component could reach a provider with. They are dropped. HOME moves to
    the working folder, so the data directory, the LLM cache and the results
    folder all start empty; the model cache stays where it was.
    """
    hf_home = _hf_home()
    # The MCP step starts lasagna-mcp from the working folder: a relative
    # PYTHONPATH (PYTHONPATH=src) would point it at another install.
    if os.environ.get("PYTHONPATH"):
        entries = os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(os.path.abspath(e) for e in entries if e)
    for var in list(os.environ):
        if (
            var.startswith(("RAG_", "LASAGNA_"))
            or var.endswith(("_API_KEY", "_APIKEY", "_TOKEN", "_SECRET", "_BASE_URL"))
            or var in ("VECTOR_BACKEND", "MODEL_NAME", "NLTK_DATA")
        ):
            del os.environ[var]
    home.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HOME": str(home),
            "HF_HOME": hf_home,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "RAG_USE_GPU": "false",
            "TOKENIZERS_PARALLELISM": "false",
            "MODEL_NAME": STUB_MODEL,
            "RAG_EVAL_GENERATOR_MODEL": STUB_MODEL,
            "RAG_EVAL_JUDGE_MODEL": STUB_JUDGE,
            # Wide enough that Rich does not wrap the lines the checks read.
            "COLUMNS": "200",
        }
    )


def _is_local(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    return host in (None, "", "localhost", "::1", "0.0.0.0") or str(host).startswith("127.")


def _guard_network() -> None:
    """Refuse every connection that leaves this machine, and record it.

    ``HF_HUB_OFFLINE`` covers the model hub only. A library that fetches
    something on its own would make this demo pass online and fail on a plane;
    with the guard it fails here, or falls back the way it would offline.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def refuse(host: Any) -> OSError:
        REFUSED.append(str(host))
        return OSError(f"the offline demo refused a network connection to {host}")

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address[0]):
            raise refuse(address[0])
        return real_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address[0]):
            raise refuse(address[0])
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            REFUSED.append(str(host))
            raise socket.gaierror(f"the offline demo refused to resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]


def _missing_models() -> List[str]:
    from huggingface_hub import try_to_load_from_cache

    return [m for m in MODELS if not isinstance(try_to_load_from_cache(m, "config.json"), str)]


def _require_extras() -> None:
    # Looked up, not imported: the `lasagna` command quiets some of these
    # libraries before they load, and the demo's commands should look the same.
    missing = [m for m in ("ragas", "langchain_core", "mcp") if not importlib.util.find_spec(m)]
    if missing:
        raise CannotRun(
            f"{', '.join(missing)} is not installed. The demo needs the eval extra: "
            f"pip install 'lasagna-rag[eval]'"
        )
    models = _missing_models()
    if models:
        raise CannotRun(
            f"{', '.join(models)} is not in the Hugging Face cache at {os.environ['HF_HOME']}. "
            f"Download it once, online: python {Path(__file__).name} --download-models"
        )


# ---------------------------------------------------------------------------
# The stub LLM and the stub judge
# ---------------------------------------------------------------------------


def _first_sentence(text: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.S)
    return " ".join((match.group(1) if match else text).split())


def _opening(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def stub_reply(prompt: str) -> str:
    """What the stub LLM answers, decided by the prompt alone.

    A synthesized factoid's answer is its passage's first sentence, and an
    answer generated from retrieved passages is the first sentence of the top
    passage, so exact match measures whether retrieval put the right passage
    first: the arms' numbers differ for a reason.
    """
    if prompt.startswith("You are checking"):
        # Every answerable question is supported, and no unanswerable one is
        # answered: the check keeps all of them.
        return "yes" if "Reference answer:" in prompt else "no"
    if prompt.startswith("You are writing one test question"):
        a = re.search(r"Passage A:\n(.*?)\n\nPassage B:\n(.*?)\n\nReply with", prompt, re.S)
        if a:
            first, second = a.group(1), a.group(2)
            question = f"What links '{_opening(first, 5)}' and '{_opening(second, 5)}'?"
            return json.dumps({"question": question, "answer": _first_sentence(first)})
        single = re.search(r"Passage:\n(.*?)\n\nReply with", prompt, re.S)
        if not single:
            raise ValueError(f"the stub LLM does not know this prompt: {prompt[:80]!r}")
        passage = single.group(1)
        if '"answer"' in prompt:
            question = f"Which fact starts with '{_opening(passage, 6)}'?"
            return json.dumps({"question": question, "answer": _first_sentence(passage)})
        return json.dumps({"question": f"What is never said about {_opening(passage, 3)}?"})
    context = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.S)
    if context and context.group(1).strip():
        return _first_sentence(context.group(1).split("\n\n")[0])
    return "I do not know."


def _install_stub_llm() -> None:
    """Every chat model the package builds answers with :func:`stub_reply`."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from lasagna.providers import set_llm_provider
    from lasagna.providers.llm_provider import SimpleLLMProvider

    class OfflineStubChatModel(BaseChatModel):
        model_name: str = STUB_MODEL
        temperature: float = 0.0

        @property
        def _llm_type(self) -> str:
            return "offline-stub"

        @property
        def _identifying_params(self) -> Dict[str, Any]:
            return {"model": STUB_MODEL}

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompt = str(messages[-1].content)
            CALLS.append(prompt)
            reply = stub_reply(prompt)
            usage = {
                "input_tokens": len(prompt.split()),
                "output_tokens": len(reply.split()),
                "total_tokens": len(prompt.split()) + len(reply.split()),
            }
            message = AIMessage(content=reply, usage_metadata=usage)
            return ChatResult(generations=[ChatGeneration(message=message)])

    class OfflineStubProvider(SimpleLLMProvider):
        """A real provider with the network call replaced by the stub."""

        def _build_chat_model(self, model_settings, temperature=0.0, max_tokens=None):
            return OfflineStubChatModel(temperature=temperature)

    set_llm_provider(OfflineStubProvider("openai"))


def _words(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _share(part: Any, whole: Any) -> float:
    words, pool = _words(part), set(_words(whole))
    return sum(w in pool for w in words) / len(words) if words else 0.0


def _install_stub_judge() -> None:
    """The RAGAS judge scores each requested metric by word overlap, with no LLM."""
    from lasagna.knowledge_evaluation_sdk import experiment, ragas_evaluator

    class OfflineStubJudge:
        def __init__(self, llm=None, embeddings=None):
            pass

        def evaluate_detailed(
            self, questions, answers, contexts, ground_truths=None, metrics=None, **kwargs
        ):
            truths = ground_truths or [""] * len(questions)
            rows = []
            for answer, passages, truth in zip(answers, contexts, truths):
                joined = " ".join(passages or [])
                score = {
                    "faithfulness": _share(answer, joined),
                    "context_precision": _share(truth, joined),
                    "context_recall": _share(truth, joined),
                    "factual_correctness": _share(truth, answer),
                }
                rows.append({m: score.get(m, _share(answer, truth)) for m in metrics or []})
            aggregate = {m: sum(r[m] for r in rows) / len(rows) for m in metrics or [] if rows}
            return ragas_evaluator.DetailedEvalResult(aggregate=aggregate, per_sample=rows)

    ragas_evaluator.RAGASEvaluator = OfflineStubJudge  # type: ignore[misc]
    experiment._build_judge_llm = lambda *args, **kwargs: None


# ---------------------------------------------------------------------------
# A LangChain document compressor, duck-typed
# ---------------------------------------------------------------------------


class KeywordOverlapCompressor:
    """Orders passages by the words they share with the query.

    It has LangChain's ``compress_documents`` and nothing else (it does not
    subclass ``BaseDocumentCompressor``), and reports its score as
    ``relevance_score``, the way CohereRerank and CrossEncoderReranker do.
    """

    def __init__(self) -> None:
        self.calls = 0

    def compress_documents(self, documents, query, callbacks=None):
        self.calls += 1
        words = set(_words(query))
        scored = []
        for doc in documents:
            overlap = float(len(words & set(_words(doc.page_content))))
            metadata = {**doc.metadata, "relevance_score": overlap}
            scored.append(type(doc)(page_content=doc.page_content, metadata=metadata))
        return sorted(scored, key=lambda d: -d.metadata["relevance_score"])


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------


class Ran:
    def __init__(self, args: List[str], code: int, output: str, seconds: float):
        self.args, self.code, self.output, self.seconds = args, code, output, seconds
        #: The output with whitespace collapsed, so a wrapped line still matches.
        self.text = " ".join(output.split())

    def expect(self, code: int = 0) -> "Ran":
        check(
            self.code == code,
            f"`{self.command}` exited {self.code}, expected {code}: {self.last_line()}",
        )
        return self

    def says(self, *phrases: str) -> "Ran":
        for phrase in phrases:
            check(
                " ".join(phrase.split()) in self.text,
                f"`{self.command}` did not print {phrase!r}",
            )
        return self

    @property
    def command(self) -> str:
        return "lasagna " + shlex.join(self.args)

    def last_line(self) -> str:
        """The last line with words in it, without the borders of a Rich box."""
        for line in reversed(self.output.splitlines()):
            text = line.strip(" │╭╮╰╯─┃━")
            if re.search(r"\w", text):
                return " ".join(text.split())
        return "(no output)"


class Demo:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.log = open(workdir / "demo.log", "a", encoding="utf-8")
        self.steps: List[Dict[str, Any]] = []
        self.compressor = KeywordOverlapCompressor()
        self.reranker = ""

    def _quiet_logs(self) -> None:
        """Send the package's log lines to demo.log, not the terminal.

        Each `lasagna` command points loguru at the stream CliRunner captured,
        which is closed once the command returns.
        """
        from loguru import logger

        logger.remove()
        logger.add(self.log, level="INFO")

    def lasagna(self, *args: str) -> Ran:
        """Run one `lasagna` command in this process, as typed in the working folder."""
        from typer.testing import CliRunner

        from lasagna.cli import app

        argv = [str(a) for a in args]
        self.log.write(f"\n$ lasagna {shlex.join(argv)}\n")
        start = time.monotonic()
        result = CliRunner().invoke(app, argv)
        ran = Ran(argv, result.exit_code, result.output, time.monotonic() - start)
        self.log.write(result.output)
        if result.exception and not isinstance(result.exception, SystemExit):
            self.log.write("".join(traceback.format_exception(result.exception)))
        self.log.flush()
        self._quiet_logs()
        return ran

    def run(self, key: str, title: str, fn: Callable[[], str]) -> None:
        number = len(self.steps) + 1
        start = time.monotonic()
        record: Dict[str, Any] = {"key": key, "title": title}
        try:
            detail = fn()
            record.update(ok=True, detail=detail)
        except DemoFailure as exc:
            record.update(ok=False, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - reported as the step's failure
            self.log.write(traceback.format_exc())
            record.update(ok=False, detail=f"{type(exc).__name__}: {exc}")
        record["seconds"] = round(time.monotonic() - start, 1)
        self.steps.append(record)
        status = "ok" if record["ok"] else "FAILED"
        print(
            f"[{number}/{len(STEPS)}] {title}: {status} ({record['seconds']:.1f} s). "
            f"{record['detail']}",
            flush=True,
        )
        if not record["ok"]:
            raise DemoFailure(record["detail"])

    # -- 1. synthesize -----------------------------------------------------

    def synthesize(self) -> str:
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        docs = self.workdir / "docs"
        docs.mkdir()
        corpus = benchmark_registry.get("smoke").load_corpus()
        for doc_id, text in corpus.items():
            (docs / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")

        out = Path("my_bench.jsonl")
        args = ["benchmark", "synthesize", "docs", "--n", str(N_QUESTIONS), "--out", str(out)]
        dry = self.lasagna(*args, "--dry-run").expect(0)
        dry.says(
            f"Documents: {len(corpus)}",
            f"Questions: {N_QUESTIONS} (",
            "Dry run: no LLM was called and nothing was written.",
        )
        check(not out.exists(), f"the dry run wrote {out}")
        check(not CALLS, f"the dry run called the LLM {len(CALLS)} times")
        planned = re.search(r"LLM calls: (\d+)", dry.text)
        check(planned, "the dry run did not say how many LLM calls the run makes")

        ran = self.lasagna(*args).expect(0).says(f"Wrote {N_QUESTIONS} questions")
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        check(len(rows) == N_QUESTIONS, f"{out} has {len(rows)} rows, expected {N_QUESTIONS}")
        check(
            len(CALLS) == int(planned.group(1)),
            f"the run made {len(CALLS)} LLM calls; the dry run said {planned.group(1)}",
        )
        types = {}
        for number, row in enumerate(rows, 1):
            meta = row.get("metadata") or {}
            check(meta.get("synthetic") is True, f"row {number} is not marked synthetic: true")
            check(row.get("question") and row.get("contexts"), f"row {number} lacks a question")
            kind = meta.get("question_type")
            check(
                ("answer" in row) is (kind != "unanswerable"),
                f"row {number} ({kind}) has the wrong answer field",
            )
            types[kind] = types.get(kind, 0) + 1
        mix = ", ".join(f"{k} {v}" for k, v in sorted(types.items()))
        return (
            f"dry run: no LLM call, no file; then {len(rows)} rows ({mix}), "
            f"all synthetic: true, from {len(CALLS)} LLM calls in {ran.seconds:.1f} s"
        )

    # -- 2. validate-data --------------------------------------------------

    def validate(self) -> str:
        self.lasagna("benchmark", "validate-data", "my_bench.jsonl").expect(0).says(
            "valid jsonl data",
            f"Questions: {N_QUESTIONS}",
            "Retrieval: yes",
            "Generation: yes",
        )
        rows = Path("my_bench.jsonl").read_text().splitlines()
        number = next(i for i, line in enumerate(rows, 1) if "answer" in json.loads(line))
        broken = json.loads(rows[number - 1])
        broken["answer"] = [broken["answer"]]
        rows[number - 1] = json.dumps(broken)
        Path("broken.jsonl").write_text("\n".join(rows) + "\n")
        bad = self.lasagna("benchmark", "validate-data", "broken.jsonl").expect(1)
        bad.says(f'row {number}: field "answer" must be a string, got list')
        return f"my_bench.jsonl is valid for both tracks; broken.jsonl fails on row {number}"

    # -- 3. from_langchain -------------------------------------------------

    def wrap(self) -> str:
        import lasagna
        from lasagna.core.base_components import BaseReranker, ScoredDocument
        from lasagna.core.registry import registry

        self.reranker = lasagna.from_langchain(self.compressor)
        expected = "langchain.KeywordOverlapCompressor"
        check(self.reranker == expected, f"from_langchain returned {self.reranker!r}")
        self.lasagna("component", "list", "reranker").expect(0).says(
            expected, f"wrapped: {expected}"
        )
        component = registry.get("reranker", self.reranker)
        check(isinstance(component, BaseReranker), f"{expected} is not a BaseReranker")
        docs = [
            ScoredDocument(content="glass flows slowly", score=0.9, document_id="a", chunk_id="a"),
            ScoredDocument(content="tardigrades survive", score=0.1, document_id="b", chunk_id="b"),
        ]
        before = copy.deepcopy(docs)
        out = component.rerank("how do tardigrades survive", docs, top_k=2)
        check([d.document_id for d in out] == ["b", "a"], "the wrapped reranker kept input order")
        check(
            [(d.content, d.score) for d in docs] == [(d.content, d.score) for d in before],
            "the wrapped reranker changed the documents it was given",
        )
        return f"{expected} is a reranker, listed as wrapped, and ranks by overlap"

    # -- 4. experiment --dry-run -------------------------------------------

    def write_sweep(self) -> Path:
        import yaml

        sweep = {
            "name": "acceptance-demo",
            "description": "Every 0.1.0 baseline arm, plus a wrapped LangChain reranker",
            "benchmarks": [{"path": "my_bench.jsonl"}],
            "tracks": ["retrieval", "generation"],
            "mode": "live",
            "n_runs": 1,
            "max_concurrency": 1,
            "output_dir": str(self.workdir / "results" / "sweep"),
            "pipeline": {
                "base_profile": "speed",
                "config_overrides": [
                    {"name": "dense", "retriever": "dense"},
                    {"name": "reranked", "retriever": "dense", "reranker": self.reranker},
                    {"name": "closed_book", "retriever": "dense", "generator": "closed_book"},
                    {"name": "long_context", "retriever": "dense", "generator": "long_context"},
                    {"name": "no_rag", "no_rag": True},
                ],
            },
        }
        path = Path("demo.yaml")
        path.write_text(yaml.safe_dump(sweep, sort_keys=False))
        return path

    def dry_run(self) -> str:
        sweep = self.write_sweep()
        calls = len(CALLS)
        ran = self.lasagna("benchmark", "experiment", str(sweep), "--dry-run").expect(0)
        ran.says(
            f"Arms (5): {', '.join(ARMS)}",
            "Tasks (5): 1 benchmark x 5 arms x 1 run",
            "Models:",
            "all-MiniLM-L6-v2 cached",
            f"component {self.reranker}",
            f"generator {STUB_MODEL}",
            f"judge {STUB_JUDGE}",
        )
        results = self.workdir / "results"
        check(not results.exists(), f"the dry run wrote under {results}")
        research = Path.home() / ".lasagna" / "research" / "results"
        written = sorted(p.name for p in research.glob("*")) if research.exists() else []
        check(not written, f"the dry run wrote {written} under {research}")
        probe = re.search(r"Probe of \S+ \(estimate\)", ran.text)
        check(probe, "the dry run printed no probe estimate")
        return (
            f"5 arms, 5 tasks, MiniLM cached, the wrapped reranker named; "
            f"a probe of {len(CALLS) - calls} LLM calls; no results written"
        )

    # -- 5. the sweep ------------------------------------------------------

    def _results(self) -> Dict[str, Any]:
        path = self.workdir / "results" / "sweep" / "results.json"
        check(path.exists(), f"the sweep wrote no {path}")
        return json.loads(path.read_text())

    def sweep(self) -> str:
        compressor_calls = self.compressor.calls
        ran = self.lasagna("benchmark", "experiment", "demo.yaml").expect(0)
        ran.says("Experiment Complete", "Failed: 0")
        for arm in NO_CONTEXT_ARMS:
            ran.says(
                f"{arm}: no context: faithfulness and context metrics not reported; "
                f"compare exact_match, token_f1, factual_correctness"
            )
        noted = sorted(re.findall(r"^\s*(\S+): no context:", ran.output, re.M))
        check(noted == NO_CONTEXT_ARMS, f"the leaderboard footnote names {noted}")
        saved = self._results()
        tasks = {t["config_name"]: t for t in saved["results"]}
        check(sorted(tasks) == sorted(ARMS), f"results.json has arms {sorted(tasks)}")
        for arm, task in tasks.items():
            check(not task.get("error"), f"arm {arm} failed: {task.get('error')}")
            metrics = task["metrics"]
            judged = "faithfulness" in metrics
            check(
                judged is (arm not in NO_CONTEXT_ARMS),
                f"arm {arm} {'has' if judged else 'lacks'} faithfulness",
            )
            check("exact_match" in metrics, f"arm {arm} has no exact_match")
            retrieved = "ndcg@10" in metrics
            check(retrieved is (arm != "no_rag"), f"arm {arm} retrieval track: {retrieved}")
        check(
            self.compressor.calls > compressor_calls,
            "the reranked arm never called the wrapped compressor",
        )
        stages = tasks["reranked"].get("system", {}).get("stages", {})
        check(
            (stages.get("reranker") or {}).get("component") == self.reranker,
            f"the reranked arm records reranker {stages.get('reranker')}",
        )
        shutil.copy(self.workdir / "results" / "sweep" / "results.json", "first_run.json")
        board = {arm: f"{tasks[arm]['metrics']['exact_match']:.2f}" for arm in ARMS if arm in tasks}
        return (
            f"5 arms on both tracks in {ran.seconds:.0f} s; no faithfulness for "
            f"{' and '.join(NO_CONTEXT_ARMS)}, and the footnote says why; exact_match {board}"
        )

    # -- 6. the same sweep, replayed ----------------------------------------

    def replay(self) -> str:
        calls = len(CALLS)
        compressor_calls = self.compressor.calls
        self.lasagna("benchmark", "experiment", "demo.yaml").expect(0).says("Failed: 0")
        check(len(CALLS) == calls, f"the second run called the LLM {len(CALLS) - calls} times")
        second = self._results()
        first = json.loads(Path("first_run.json").read_text())
        cache = second["experiment_config"].get("llm_cache") or {}
        check(cache.get("hit_rate") == 1.0, f"the second run's LLM cache record is {cache}")
        check(_numbers(second) == _numbers(first), _difference(first, second))
        return (
            f"hit rate 1.0 ({cache.get('hits')} replayed), no LLM call, identical metrics; "
            f"the reranker ran again ({self.compressor.calls - compressor_calls} calls)"
        )

    # -- 7. results ---------------------------------------------------------

    def report(self) -> str:
        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults, verdict_lines

        results_json = str(self.workdir / "results" / "sweep" / "results.json")
        Path("bundles").mkdir()
        bundled = self.lasagna("results", "bundle", results_json, "-o", "bundles/run.json")
        bundled.expect(0).says("Wrote bundles/run.json", "run_id:")
        bundle = json.loads(Path("bundles/run.json").read_text())
        hit_rate = (bundle.get("experiment") or {}).get("llm_cache", {}).get("hit_rate")
        check(hit_rate == 1.0, f"the bundle's LLM cache hit rate is {hit_rate}")

        self.lasagna("results", "view", results_json, "-o", "view.html").expect(0).says(
            "Wrote view.html"
        )
        page = Path("view.html").read_text(encoding="utf-8")
        visible = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())
        for arm in ARMS:
            check(re.search(rf"\b{arm}\b", visible), f"view.html does not name arm {arm}")
        results = BenchmarkResults.load(results_json)
        (benchmark,) = results.benchmarks
        rows = results.pairwise_significance("ndcg@10", benchmark)
        verdict = verdict_lines(rows, "ndcg@10", benchmark, 0.05).lines
        check(verdict, "there is no significance verdict to look for")
        for line in verdict:
            check(" ".join(line.text.split()) in visible, f"view.html lacks {line.text!r}")

        latex = self.lasagna("results", "leaderboard", "bundles", "--format", "latex").expect(0)
        tables = latex.output.count("\\begin{tabular}")
        check(tables == 1, f"the LaTeX leaderboard has {tables} tables for 1 benchmark")
        for arm in ARMS:
            if arm != "no_rag":
                check(arm.replace("_", "\\_") in latex.output, f"the LaTeX table lacks {arm}")
        return (
            f"bundle {bundle['run_id']} (hit rate 1.0); view.html names all 5 arms and the "
            f"verdict ({verdict[0].text.strip().rstrip('.')}); 1 LaTeX table"
        )

    # -- 8. MCP --------------------------------------------------------------

    def mcp(self) -> str:
        import asyncio

        command = _mcp_command()
        start = time.monotonic()
        help_run = subprocess.run(
            [*command, "--help"], capture_output=True, text=True, timeout=60, env=dict(os.environ)
        )
        seconds = time.monotonic() - start
        check(help_run.returncode == 0, f"lasagna-mcp --help exited {help_run.returncode}")
        check("--transport" in help_run.stdout, "lasagna-mcp --help printed no options")
        check(
            seconds < MCP_HELP_BUDGET_S,
            f"lasagna-mcp --help took {seconds:.2f} s (budget {MCP_HELP_BUDGET_S:.0f} s)",
        )
        start = time.monotonic()
        tools = asyncio.run(_list_stdio_tools(command, self.log))
        listed = time.monotonic() - start
        check(tools, "the MCP server listed no tools")
        deleting = sorted(t for t in tools if t in DELETION_TOOLS or DELETION_PATTERN.match(t))
        check(not deleting, f"the default MCP tool set lists deletion tools {deleting}")
        check("search" in tools, "the default MCP tool set has no search tool")
        return (
            f"--help in {seconds:.2f} s; over stdio the server listed {len(tools)} tools, "
            f"none that deletes, {listed:.1f} s after it was started"
        )


def _mcp_command() -> List[str]:
    """``lasagna-mcp`` beside this interpreter, else the module it runs."""
    script = Path(sys.executable).with_name("lasagna-mcp")
    return [str(script)] if script.exists() else [sys.executable, "-m", "lasagna.mcp"]


async def _list_stdio_tools(command: List[str], log) -> List[str]:
    """Start the MCP server the way a client does (stdio, no flags) and list its tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command[0], args=command[1:], env=dict(os.environ))
    async with stdio_client(params, errlog=log) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return sorted(t.name for t in (await session.list_tools()).tools)


_VOLATILE = ("latency", "_ms", "seconds", "llm_cache_")


def _numbers(saved: Dict[str, Any]) -> Dict[str, Any]:
    """Every arm's metrics and per-query scores, minus timings and the cache's
    own counters."""

    def keep(key: str) -> bool:
        return not any(v in key for v in _VOLATILE)

    out = {}
    for task in saved["results"]:
        metrics = {k: v for k, v in task["metrics"].items() if keep(k)}
        # A replayed response counts the tokens it cost cold.
        metrics.update(
            {k: v for k, v in task.items() if k.startswith(("system_", "eval_")) and keep(k)}
        )
        rows = [{k: v for k, v in s.items() if keep(k)} for s in task["per_sample_scores"]]
        out[task["config_name"]] = (metrics, rows)
    return out


def _difference(first: Dict[str, Any], second: Dict[str, Any]) -> str:
    a, b = _numbers(first), _numbers(second)
    for arm in sorted(set(a) | set(b)):
        ma, mb = a.get(arm, ({}, []))[0], b.get(arm, ({}, []))[0]
        for key in sorted(set(ma) | set(mb)):
            if ma.get(key) != mb.get(key):
                return f"the replay changed {arm} {key}: {ma.get(key)} -> {mb.get(key)}"
        if a.get(arm) != b.get(arm):
            return f"the replay changed {arm}'s per-query scores"
    return "the replay changed the numbers"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _download_models() -> int:
    """Fetch the models into the Hugging Face cache; the demo itself stays offline."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)  # before huggingface_hub reads them at import
    from huggingface_hub import snapshot_download

    for model in MODELS:
        print(f"Downloading {model} into {_hf_home()}", flush=True)
        # The PyTorch weights only, not the ONNX, OpenVINO or TensorFlow copies.
        snapshot_download(model, ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.ot", "*.bin"])
    return 0


#: ``(key, title, Demo method)`` in the order they run. The keys and titles go
#: into ``--report``, so the release gate lists a step that was never reached.
STEPS = [
    ("synthesize", "synthesize 12 questions, after a dry run", "synthesize"),
    ("validate_data", "validate the file, and a broken copy", "validate"),
    ("from_langchain", "wrap a LangChain compressor as a reranker", "wrap"),
    ("experiment_dry_run", "dry-run the sweep", "dry_run"),
    ("sweep", "sweep five arms on both tracks", "sweep"),
    ("replay", "rerun the sweep from the LLM cache", "replay"),
    ("results", "bundle, view and LaTeX leaderboard", "report"),
    ("mcp", "MCP help and the default tool set", "mcp"),
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workdir", type=Path, help="run in this folder")
    parser.add_argument("--keep", action="store_true", help="keep the working folder")
    parser.add_argument("--report", type=Path, help="also write the step results as JSON")
    parser.add_argument(
        "--download-models", action="store_true", help="download the models, then exit"
    )
    args = parser.parse_args(argv)
    if args.download_models:
        return _download_models()

    report = args.report.resolve() if args.report else None
    workdir = (args.workdir or Path(tempfile.mkdtemp(prefix="lasagna-acceptance-"))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Registered before anything starts Milvus Lite: its server stops (and
    # writes the store out) in an atexit hook, and hooks run last-registered
    # first, so the folder goes after that.
    remove = []
    atexit.register(lambda: remove and shutil.rmtree(workdir, ignore_errors=True))
    _prepare_environment(workdir / "home")
    start = time.monotonic()
    demo: Optional[Demo] = None
    outcome, reason = "passed", ""
    try:
        _require_extras()
        _guard_network()
        os.chdir(workdir)
        demo = Demo(workdir)
        demo._quiet_logs()
        _install_stub_llm()
        _install_stub_judge()
        print(f"Lasagna acceptance demo in {workdir}", flush=True)
        for key, title, method in STEPS:
            demo.run(key, title, getattr(demo, method))
    except CannotRun as exc:
        outcome, reason = "cannot_run", str(exc)
        print(f"Cannot run the acceptance demo: {reason}", flush=True)
    except DemoFailure as exc:
        outcome, reason = "failed", str(exc)
    total = time.monotonic() - start
    if outcome == "passed":
        refused = f"; {len(set(REFUSED))} host(s) refused" if REFUSED else ""
        print(
            f"All {len(STEPS)} steps passed in {total:.0f} s, offline{refused}, with "
            f"{len(CALLS)} stub LLM calls.",
            flush=True,
        )
    if report:
        record = {
            "outcome": outcome,
            "reason": reason,
            "seconds": round(total, 1),
            "planned": [[key, title] for key, title, _ in STEPS],
            "steps": demo.steps if demo else [],
            "refused_hosts": sorted(set(REFUSED)),
            "llm_calls": len(CALLS),
        }
        report.write_text(json.dumps(record, indent=2))
    if outcome != "passed" or args.keep:
        print(f"Working folder kept: {workdir} (demo.log has every command's output)")
    else:
        os.chdir(workdir.parent)
        remove.append(True)
    return {"passed": 0, "failed": 1, "cannot_run": 2}[outcome]


if __name__ == "__main__":
    sys.exit(main())
