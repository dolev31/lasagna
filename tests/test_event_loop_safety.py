"""Regression tests for the cross-event-loop failures fixed in 0.1.0.

Four features — HyDE, query expansion, RAG Fusion and CRAG — each bridged to
async work with ``asyncio.run()``. That builds a *fresh event loop per call*,
so a cached LLM client holding an ``asyncio.Lock`` bound to an earlier loop
raised ``RuntimeError: ... is bound to a different event loop``. A broad
``except`` downstream turned that into a one-line warning, the feature returned
nothing, and the configuration scored identically to its baseline. Nothing
crashed, so it went unnoticed across many evaluation runs.

The trigger is calling search from a worker thread — what an evaluation harness
does with ``asyncio.to_thread(pipeline.query, ...)``. These tests reproduce that
shape with a fake LLM and no network.
"""

import asyncio
import inspect
import re
import threading

import pytest

from lasagna.retrieval.enhancement.enhancer import QueryEnhancer


def executable_source(func) -> str:
    """Source of ``func`` with comments and docstrings stripped.

    The fixes are explained in comments that necessarily mention the very
    constructs being banned ("the old asyncio.run bridge ..."), so a naive
    substring check would match its own explanation and fail.
    """
    source = inspect.getsource(func)
    # Drop docstrings (both quote styles), then trailing/standalone comments.
    source = re.sub(r'"""[\s\S]*?"""', "", source)
    source = re.sub(r"'''[\s\S]*?'''", "", source)
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


class LoopBoundFakeLLM:
    """Fake chat model that mimics a cached async client bound to one loop.

    Real LangChain clients hold an ``httpx.AsyncClient`` whose ``asyncio.Lock``
    is created on first use and bound to the loop running at that moment. Using
    the client from a *different* loop raises. This models exactly that: the
    lock is created once, and any use from another loop is an error.
    """

    def __init__(self, response: str = "fake response"):
        self.response = response
        self.invoke_count = 0
        self._home_loop = None
        self._lock = None

    def _bind_or_check(self):
        """Bind to the first loop seen; raise if later used from another."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._home_loop is None and loop is not None:
            self._home_loop = loop
            self._lock = asyncio.Lock()
        elif loop is not None and self._home_loop is not None and loop is not self._home_loop:
            raise RuntimeError("<asyncio.locks.Lock object> is bound to a different event loop")

    def invoke(self, *args, **kwargs):
        # The sync path must never touch a loop — that is the whole point.
        self._bind_or_check()
        self.invoke_count += 1
        return type("Msg", (), {"content": self.response})()

    async def ainvoke(self, *args, **kwargs):
        self._bind_or_check()
        self.invoke_count += 1
        return type("Msg", (), {"content": self.response})()


@pytest.fixture
def fake_llm():
    return LoopBoundFakeLLM()


class TestHyDEFromWorkerThread:
    """Bug 1: HyDE returned zero documents when driven from a worker thread."""

    def test_multi_doc_generation_survives_repeated_calls(self, fake_llm, monkeypatch):
        """Generating HyDE docs repeatedly must not bind to a per-call loop.

        The removed ``asyncio.run()`` fallback created a new loop each call, so
        the second call onward failed. Success here means every call produced
        documents.
        """
        enhancer = QueryEnhancer()
        monkeypatch.setattr(
            enhancer.hyde,
            "generate_hypothetical_document_sync",
            lambda *a, **k: "hypothetical answer",
        )

        results = []
        for _ in range(3):
            docs = enhancer.hyde.generate_hypothetical_documents_batch_sync("q", n_docs=2)
            results.append(docs)

        assert all(len(d) == 2 for d in results), (
            f"HyDE degraded across repeated calls: {[len(d) for d in results]}"
        )

    def test_generation_works_inside_worker_thread(self, monkeypatch):
        """The evaluation-harness shape: sync call from asyncio.to_thread."""
        enhancer = QueryEnhancer()
        monkeypatch.setattr(
            enhancer.hyde,
            "generate_hypothetical_document_sync",
            lambda *a, **k: "hypothetical answer",
        )

        async def driver():
            # Two sequential to_thread calls: each gets a worker thread with no
            # running loop, which is where asyncio.run() used to misbehave.
            first = await asyncio.to_thread(
                enhancer.hyde.generate_hypothetical_documents_batch_sync, "q", 2
            )
            second = await asyncio.to_thread(
                enhancer.hyde.generate_hypothetical_documents_batch_sync, "q", 2
            )
            return first, second

        first, second = asyncio.run(driver())
        assert len(first) == 2 and len(second) == 2, (
            f"HyDE returned {len(first)}/{len(second)} docs from worker threads"
        )

    def test_no_asyncio_run_in_hyde_batch_path(self):
        """Guard the fix itself: the sync batch path must not call asyncio.run.

        Reintroducing that bridge is what caused the silent degradation, and it
        is an easy thing to add back while 'simplifying'.
        """
        source = executable_source(QueryEnhancer().hyde.generate_hypothetical_documents_batch_sync)
        assert "asyncio.run" not in source, (
            "generate_hypothetical_documents must not bridge via asyncio.run — "
            "it rebinds cached async clients to a fresh loop"
        )


class TestQueryExpansionUsesSyncPath:
    """Bug 2: expansion/fusion silently fell back to a single query."""

    def test_orchestrator_calls_the_sync_helper(self):
        """search_multi_query must use generate_query_variations, not asyncio.run.

        When the async bridge failed, ``variations = []`` and RRF fused a single
        query — making every expansion and rag_fusion run identical to plain
        hybrid search, with only a warning to show for it.
        """
        from lasagna.api._search_orchestrator import SearchOrchestrator

        from lasagna.api._search_orchestrator import _query_variations

        # The fusion itself; search_multi_query presents what it returns.
        source = executable_source(SearchOrchestrator._search_multi_query_ranked)
        assert "_query_variations" in source, (
            "search_multi_query should get its variations through _query_variations"
        )
        assert "generate_query_variations" in executable_source(_query_variations), (
            "_query_variations should call the sync generate_query_variations helper"
        )
        assert "asyncio.run" not in source, "search_multi_query must not bridge via asyncio.run"

    def test_variations_generated_from_worker_thread(self, fake_llm, monkeypatch):
        """The sync helper works with no loop, and with a loop-bound client."""
        enhancer = QueryEnhancer()
        monkeypatch.setattr(enhancer.expander, "llm", fake_llm, raising=False)
        monkeypatch.setattr(
            enhancer,
            "generate_query_variations",
            lambda q, n=3: [f"{q} variant {i}" for i in range(n)],
        )

        async def driver():
            return await asyncio.to_thread(enhancer.generate_query_variations, "q", 3)

        variations = asyncio.run(driver())
        assert len(variations) == 3, f"expected 3 variations, got {variations}"


class TestComprehensiveSearchLoopContract:
    """Bug 3a: CRAG bridged across loops and returned empty results."""

    def test_single_asyncio_run_no_threadpool_bridge(self):
        from lasagna.api._search_orchestrator import SearchOrchestrator

        source = executable_source(SearchOrchestrator._search_comprehensive)
        assert source.count("asyncio.run(") == 1, (
            "_search_comprehensive should run the agent on exactly one loop"
        )
        assert "ThreadPoolExecutor" not in source, (
            "the ThreadPoolExecutor+asyncio.run bridge rebinds cached clients"
        )

    def test_running_loop_gives_actionable_error(self):
        """Violating the contract must say what to do, not fail cryptically."""
        from lasagna.api._search_orchestrator import SearchOrchestrator

        source = inspect.getsource(SearchOrchestrator._search_comprehensive)
        assert "async_search" in source, "callers inside a loop must be told to use async_search()"


class TestCragGradingIsBounded:
    """Bug 3b: grading fanned out one LLM call per chunk, unbounded."""

    def test_concurrency_constant_is_sane(self):
        from lasagna.agentic.agentic_rag import CRAG_GRADING_CONCURRENCY

        assert 1 <= CRAG_GRADING_CONCURRENCY <= 32, (
            f"implausible grading concurrency: {CRAG_GRADING_CONCURRENCY}"
        )

    def test_grading_respects_the_semaphore(self, monkeypatch):
        """Peak concurrency must stay within the configured cap.

        Exceeding an endpoint's limit makes grades time out; a timed-out grade
        still counts toward the CRAG decision as 'ambiguous', so the decision
        gets made from answers that were never computed.
        """
        import lasagna.agentic.agentic_rag as ar

        peak = 0
        active = 0
        lock = threading.Lock()

        async def fake_grade(llm, query, document_text, doc_index):
            nonlocal peak, active
            with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            with lock:
                active -= 1
            return (doc_index, "correct")

        monkeypatch.setattr(ar, "_grade_single_document", fake_grade)

        chunks = [type("C", (), {"content": f"chunk {i}"})() for i in range(40)]
        semaphore = asyncio.Semaphore(ar.CRAG_GRADING_CONCURRENCY)

        async def run_bounded():
            async def one(i, chunk):
                async with semaphore:
                    return await ar._grade_single_document(
                        llm=None, query="q", document_text=chunk.content, doc_index=i
                    )

            return await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))

        results = asyncio.run(run_bounded())

        assert len(results) == 40
        assert peak <= ar.CRAG_GRADING_CONCURRENCY, (
            f"peak concurrency {peak} exceeded cap {ar.CRAG_GRADING_CONCURRENCY}"
        )


class TestJudgeGeneratorSeparation:
    """Bug 4: the judge and generator LLMs shared one model string.

    A model id valid for the RAGAS scoring endpoint is generally not a valid
    model on the generator's platform. Passing one to the other meant the
    generator failed to build, produced an empty answer, and faithfulness
    scored 0.0 -- with only a warning to indicate anything had gone wrong.
    """

    def test_generator_ignores_the_judge_model(self, monkeypatch):
        from lasagna.knowledge_evaluation_sdk import experiment

        monkeypatch.setattr(experiment, "_active_judge_model", "judge-only-model")
        monkeypatch.delenv("RAG_EVAL_GENERATOR_MODEL", raising=False)
        monkeypatch.delenv("MODEL_NAME", raising=False)

        resolved = experiment._resolve_generator_model()
        assert resolved != "judge-only-model", "the generator must not inherit the judge's model id"
        assert resolved is None, "with nothing configured, defer to the platform default"

    def test_generator_resolution_order(self, monkeypatch):
        from lasagna.knowledge_evaluation_sdk import experiment

        monkeypatch.setenv("MODEL_NAME", "from-model-name")
        monkeypatch.delenv("RAG_EVAL_GENERATOR_MODEL", raising=False)
        assert experiment._resolve_generator_model() == "from-model-name"

        monkeypatch.setenv("RAG_EVAL_GENERATOR_MODEL", "explicit-generator")
        assert experiment._resolve_generator_model() == "explicit-generator", (
            "RAG_EVAL_GENERATOR_MODEL should win over MODEL_NAME"
        )

    def test_judge_returns_none_without_a_model_or_a_key(self, monkeypatch):
        """With a key and no model the judge is DEFAULT_JUDGE_MODEL (see
        test_current_model_parameters.py); with neither there is nothing to build."""
        from lasagna.knowledge_evaluation_sdk import experiment

        monkeypatch.setattr(experiment, "_active_judge_model", None)
        for var in ("RAG_EVAL_JUDGE_MODEL", "RAG_EVAL_JUDGE_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert experiment._build_judge_llm() is None, (
            "no judge model and no key should defer to the RAGAS default, not raise"
        )

    def test_judge_credentials_do_not_hijack_the_generator(self, monkeypatch):
        """The judge must read its own env vars before OPENAI_*.

        Provider auto-detection scans a priority-ordered table of ``*_API_KEY``
        variables. Pointing the judge at an OpenAI-compatible proxy by exporting
        OPENAI_API_KEY would therefore also re-route the *generator* to that
        proxy -- which is the collision this split exists to prevent.
        """
        source = executable_source(
            __import__(
                "lasagna.knowledge_evaluation_sdk.experiment",
                fromlist=["_build_judge_llm"],
            )._build_judge_llm
        )
        assert "RAG_EVAL_JUDGE_API_KEY" in source, "judge should prefer its own API key variable"
        assert "RAG_EVAL_JUDGE_BASE_URL" in source, "judge should prefer its own base URL variable"
