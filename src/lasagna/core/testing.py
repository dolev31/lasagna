"""ComponentTestHarness: validate custom components against their ABC contracts.

Researchers can use this to verify that their custom components meet the
expected interface before integrating them into a pipeline. The harness
runs a battery of contract checks using synthetic data and reports any
violations clearly.

Example:
    from lasagna.core.testing import ComponentTestHarness
    from lasagna.core.base_components import BaseReranker

    class MyReranker(BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            return sorted(documents, key=lambda d: d.score, reverse=True)[:top_k]

    harness = ComponentTestHarness()
    report = harness.validate(MyReranker())

    if report.passed:
        print("All checks passed!")
    else:
        for failure in report.failures:
            print(f"FAIL: {failure}")
"""

from __future__ import annotations

import copy
import asyncio
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type


from .base_components import (
    BaseChunker,
    BaseEmbedder,
    BaseReranker,
    BaseRetriever,
    BaseDocumentProcessor,
    BaseQueryEnhancer,
    BaseGenerator,
    BaseAgenticPipeline,
    AgenticPipelineInput,
    Chunk,
    ScoredDocument,
)


@dataclass
class ValidationReport:
    """Result of validating a component against its ABC contract.

    Attributes:
        component_class: Name of the component class.
        component_type: Detected component category (e.g., "reranker").
        checks: List of individual check results.
        passed: True if all checks passed.
        failures: List of failure messages (empty if all passed).
        latency_ms: Average latency across profiled calls.
    """

    component_class: str = ""
    component_type: str = ""
    checks: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    passed: bool = True
    latency_ms: Optional[float] = None

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else f"FAILED ({len(self.failures)} issues)"
        return (
            f"ValidationReport({self.component_class}, type={self.component_type}, status={status})"
        )

    def summary(self) -> str:
        """Human-readable summary of the validation."""
        lines = [
            f"Component: {self.component_class}",
            f"Type:      {self.component_type}",
            f"Status:    {'PASSED' if self.passed else 'FAILED'}",
            f"Checks:    {len(self.checks)} run, {len(self.failures)} failed",
        ]
        if self.latency_ms is not None:
            lines.append(f"Latency:   {self.latency_ms:.1f} ms (avg)")
        if self.failures:
            lines.append("\nFailures:")
            for f in self.failures:
                lines.append(f"  - {f}")
        return "\n".join(lines)


# Maps each ABC to its component type name
_ABC_MAP = {
    BaseReranker: "reranker",
    BaseChunker: "chunker",
    BaseEmbedder: "embedder",
    BaseRetriever: "retriever",
    BaseDocumentProcessor: "document_processor",
    BaseQueryEnhancer: "query_enhancer",
    BaseGenerator: "generator",
    BaseAgenticPipeline: "agentic_pipeline",
}


class ComponentTestHarness:
    """Validate custom components against their ABC contracts.

    Runs synthetic-data tests to verify return types, empty-input
    handling, and deterministic behaviour.
    """

    def validate(self, component: Any) -> ValidationReport:
        """Run all applicable contract checks on a component.

        Args:
            component: An instance of a component (must inherit from
                one of the Base* ABCs).

        Returns:
            ValidationReport with check results.
        """
        report = ValidationReport(component_class=component.__class__.__name__)

        # Detect component type
        detected_type = None
        for abc, type_name in _ABC_MAP.items():
            if isinstance(component, abc):
                detected_type = type_name
                break

        if detected_type is None:
            report.failures.append(
                f"{component.__class__.__name__} does not inherit from any "
                f"known base class ({', '.join(abc.__name__ for abc in _ABC_MAP)})"
            )
            report.passed = False
            return report

        report.component_type = detected_type

        # Run type-specific checks
        validator = _VALIDATORS.get(detected_type)
        if validator:
            validator(component, report)

        report.passed = len(report.failures) == 0
        return report


# ---------------------------------------------------------------------------
# Per-type validators
# ---------------------------------------------------------------------------


def _make_sample_documents(n: int = 5) -> List[ScoredDocument]:
    """Create synthetic ScoredDocument instances for testing."""
    return [
        ScoredDocument(
            content=f"Sample document {i} with some content for testing.",
            score=1.0 - i * 0.1,
            document_id=f"doc_{i}",
            chunk_id=f"chunk_{i}",
            metadata={"source": "test"},
        )
        for i in range(n)
    ]


def _validate_reranker(component: BaseReranker, report: ValidationReport) -> None:
    """Validate a BaseReranker implementation."""
    docs = _make_sample_documents(5)
    query = "test query for reranking"

    # Check: rerank returns a list
    _check(
        report,
        "rerank() returns a list",
        lambda: (_assert_type(component.rerank(query, docs, top_k=3), list)),
    )

    # Check: rerank respects top_k
    def _check_top_k():
        result = component.rerank(query, docs, top_k=2)
        _assert_true(len(result) <= 2, f"Expected <= 2 results but got {len(result)}")

    _check(report, "rerank() respects top_k", _check_top_k)

    # Check: result elements are ScoredDocument
    _check(
        report,
        "rerank() returns ScoredDocument instances",
        lambda: (_assert_all_type(component.rerank(query, docs, top_k=3), ScoredDocument)),
    )

    # Check: empty input handling
    _check(
        report,
        "rerank() handles empty document list",
        lambda: (_assert_type(component.rerank(query, [], top_k=3), list)),
    )

    # Check: the documents it was handed come back unchanged. A reranker that
    # writes scores or text onto its input changes what every later stage (and
    # the next arm, through a cache) sees, so its comparison is not attributable.
    def _check_no_mutation():
        handed = _make_sample_documents(5)
        before = [_state(d) for d in handed]
        component.rerank(query, handed, top_k=3)
        changed = [i for i, d in enumerate(handed) if _state(d) != before[i]]
        _assert_true(len(handed) == 5, f"the input list changed length to {len(handed)}")
        _assert_true(
            not changed,
            f"rerank() modified the documents it was handed (positions {changed}); "
            f"copy them before changing a score or the text",
        )

    _check(report, "rerank() does not modify its input", _check_no_mutation)

    # Check: deterministic output
    _check(
        report,
        "rerank() produces deterministic output",
        lambda: (
            _assert_deterministic(
                lambda: [d.content for d in component.rerank(query, docs, top_k=3)]
            )
        ),
    )

    # Profile latency
    _profile_latency(report, lambda: component.rerank(query, docs, top_k=3))


def _validate_chunker(component: BaseChunker, report: ValidationReport) -> None:
    """Validate a BaseChunker implementation."""
    text = (
        "This is a test document with multiple sentences. "
        "It should be chunked into smaller pieces. "
        "Each chunk should have meaningful content. "
        "The chunker should handle this gracefully."
    ) * 10  # Make it long enough to require chunking

    # Check: chunk returns a list
    _check(report, "chunk() returns a list", lambda: (_assert_type(component.chunk(text), list)))

    # Check: result elements are Chunk
    _check(
        report,
        "chunk() returns Chunk instances",
        lambda: (_assert_all_type(component.chunk(text), Chunk)),
    )

    # Check: chunks have non-empty content
    _check(
        report,
        "chunk() produces non-empty chunks",
        lambda: (
            _assert_true(
                all(c.content.strip() for c in component.chunk(text)),
                "Found chunk with empty content",
            )
        ),
    )

    # Check: empty input handling
    _check(
        report, "chunk() handles empty string", lambda: (_assert_type(component.chunk(""), list))
    )

    # Check: deterministic output
    _check(
        report,
        "chunk() produces deterministic output",
        lambda: (_assert_deterministic(lambda: [c.content for c in component.chunk(text)])),
    )

    _profile_latency(report, lambda: component.chunk(text))


def _validate_embedder(component: BaseEmbedder, report: ValidationReport) -> None:
    """Validate a BaseEmbedder implementation."""
    import numpy as np

    texts = ["sample text one", "sample text two", "sample text three"]

    # Check: encode returns ndarray
    _check(
        report,
        "encode() returns numpy array",
        lambda: (_assert_type(component.encode(texts), np.ndarray)),
    )

    def _check_no_mutation():
        handed = list(texts)
        component.encode(handed)
        _assert_true(handed == texts, "encode() modified the list of texts it was handed")

    _check(report, "encode() does not modify its input", _check_no_mutation)

    # Check: output shape
    _check(
        report,
        "encode() returns correct shape",
        lambda: (
            _assert_true(
                component.encode(texts).shape[0] == len(texts),
                f"Expected first dim = {len(texts)}, got {component.encode(texts).shape[0]}",
            )
        ),
    )

    # Check: encode_query returns 1D array
    _check(
        report,
        "encode_query() returns 1D array",
        lambda: (
            _assert_true(
                component.encode_query("test query").ndim == 1,
                f"Expected 1D array, got {component.encode_query('test query').ndim}D",
            )
        ),
    )

    # Check: dimension property
    _check(
        report,
        "dimension property returns positive int",
        lambda: (
            _assert_true(
                isinstance(component.dimension, int) and component.dimension > 0,
                f"dimension should be a positive int, got {component.dimension}",
            )
        ),
    )

    # Check: dimension consistency
    _check(
        report,
        "encode() output matches declared dimension",
        lambda: (
            _assert_true(
                component.encode(texts).shape[1] == component.dimension,
                f"encode() dim = {component.encode(texts).shape[1]} != "
                f"declared dimension = {component.dimension}",
            )
        ),
    )

    _profile_latency(report, lambda: component.encode(texts))


def _validate_retriever(component: BaseRetriever, report: ValidationReport) -> None:
    """Validate a BaseRetriever implementation (basic contract only)."""
    # Retriever validation is limited because it requires a real collection.
    # We only verify the interface exists and handles graceful failure.

    _check(
        report,
        "retrieve() is callable",
        lambda: (
            _assert_true(callable(getattr(component, "retrieve", None)), "Missing retrieve method")
        ),
    )

    _check(
        report,
        "name property exists",
        lambda: (_assert_true(isinstance(component.name, str), "name property should return str")),
    )


def _validate_document_processor(
    component: BaseDocumentProcessor, report: ValidationReport
) -> None:
    """Validate a BaseDocumentProcessor implementation."""
    _check(
        report,
        "extract() is callable",
        lambda: (
            _assert_true(callable(getattr(component, "extract", None)), "Missing extract method")
        ),
    )

    _check(
        report,
        "supported_extensions returns list",
        lambda: (_assert_type(component.supported_extensions, list)),
    )

    _check(
        report,
        "name property exists",
        lambda: (_assert_true(isinstance(component.name, str), "name property should return str")),
    )


def _validate_query_enhancer(component: BaseQueryEnhancer, report: ValidationReport) -> None:
    """Validate a BaseQueryEnhancer implementation."""
    query = "test query for enhancement"

    # Check: enhance returns a list
    _check(
        report, "enhance() returns a list", lambda: (_assert_type(component.enhance(query), list))
    )

    # Check: result elements are strings
    _check(
        report,
        "enhance() returns list of strings",
        lambda: (
            _assert_true(
                all(isinstance(q, str) for q in component.enhance(query)),
                "enhance() should return list of strings",
            )
        ),
    )

    # Check: non-empty results
    _check(
        report,
        "enhance() returns non-empty list",
        lambda: (_assert_true(len(component.enhance(query)) > 0, "enhance() returned empty list")),
    )

    # Check: empty input handling
    _check(
        report,
        "enhance() handles empty string",
        lambda: (_assert_type(component.enhance(""), list)),
    )

    _profile_latency(report, lambda: component.enhance(query))


class _StubLLM:
    """A deterministic stand-in model, so a generator's contract can be checked
    without a key. It answers with the last line of the prompt it was given."""

    def invoke(self, prompt):
        from types import SimpleNamespace

        return SimpleNamespace(content="Answer: stub answer")


def _validate_generator(component: Any, report: ValidationReport) -> None:
    """Validate a BaseGenerator implementation."""
    llm = _StubLLM()
    contexts = ["Paris is the capital of France.", "The Seine flows through Paris."]

    _check(
        report,
        "generate() returns a string",
        lambda: _assert_type(component.generate("capital of France?", contexts, llm=llm), str),
    )
    _check(
        report,
        "generate() handles no contexts",
        lambda: _assert_type(component.generate("capital of France?", [], llm=llm), str),
    )

    def _check_no_mutation():
        handed = list(contexts)
        component.generate("capital of France?", handed, llm=llm)
        _assert_true(handed == contexts, "generate() modified the contexts it was handed")

    _check(report, "generate() does not modify its input", _check_no_mutation)
    _profile_latency(report, lambda: component.generate("q", contexts, llm=llm))


def _agentic_stub_llm():
    """A local chat model that concludes without tools or external LLM calls."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class StubChatModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "component-validation-stub"

        def bind_tools(self, tools, **kwargs):
            return self

        # The prompt matches below serve Adaptive RAG, FLARE and Self-RAG,
        # respectively; the fallback finishes Self-Ask and IRCoT.
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompt = str(messages[-1].content)
            if "Class (A, B or C):" in prompt:
                answer = "A"
            elif "reply with exactly: NONE" in prompt:
                answer = "NONE"
            elif "[Yes]" in prompt and "[No]" in prompt:
                answer = "[No]"
            else:
                answer = "So the final answer is: stub answer"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    return StubChatModel()


def _validate_agentic_pipeline(component: BaseAgenticPipeline, report: ValidationReport) -> None:
    """Exercise the async stage with synthetic input and a local model.

    A pipeline's own retrieval dependencies still need to be available; as
    with other validators, infrastructure failures are allowed to surface.
    """
    from concurrent.futures import ThreadPoolExecutor

    from lasagna.agentic.agentic_rag import AgenticRAGResult
    from lasagna.core.config import RAGConfig

    def run(question):
        handed = AgenticPipelineInput(
            question=question,
            session_id="component-validation",
            document_filter=["synthetic-document"],
            # Keep real settings, but let the stub answer every model call.
            session_config=RAGConfig().with_query_overrides(
                {"agentic_models": {}, "adaptive_rag_classifier_model": ""}
            ),
            llm=_agentic_stub_llm(),
        )
        model = handed.llm
        config = handed.session_config
        # Models and configurations may update internal state (e.g. lazy
        # providers). Check their identity separately from the input data.
        before = _state(dataclasses.replace(handed, llm=None, session_config=None))
        started = time.perf_counter()
        result = asyncio.run(component.run(handed))
        report.latency_ms = (time.perf_counter() - started) * 1000
        _assert_type(result, AgenticRAGResult)
        _assert_true(
            handed.llm is model
            and handed.session_config is config
            and _state(dataclasses.replace(handed, llm=None, session_config=None)) == before,
            "run() modified the input it was handed; copy it before changing it",
        )

    def check(question):
        try:
            asyncio.get_running_loop()
        except RuntimeError:  # No running loop: a normal synchronous caller.
            run(question)
        else:
            # validate() is synchronous, but can also be called from a notebook
            # or async sweep. Do not nest asyncio.run in their running loop.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(run, question).result()

    _check(
        report,
        "run() returns AgenticRAGResult and does not modify its input",
        lambda: check("What is the capital of France?"),
    )
    _check(report, "run() handles an empty question", lambda: check(""))


_VALIDATORS = {
    "reranker": _validate_reranker,
    "chunker": _validate_chunker,
    "embedder": _validate_embedder,
    "retriever": _validate_retriever,
    "document_processor": _validate_document_processor,
    "query_enhancer": _validate_query_enhancer,
    "generator": _validate_generator,
    "agentic_pipeline": _validate_agentic_pipeline,
}


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _check(report: ValidationReport, name: str, fn) -> None:
    """Run a single check, recording pass/fail.

    Only catches validation-related exceptions (AssertionError, TypeError,
    ValueError, AttributeError). Infrastructure errors (ConnectionError,
    RuntimeError, etc.) are re-raised so they surface clearly.
    """
    try:
        fn()
        report.checks.append(f"PASS: {name}")
    except (AssertionError, TypeError, ValueError, AttributeError) as e:
        report.checks.append(f"FAIL: {name}")
        report.failures.append(f"{name}: {e}")


def _state(obj: Any) -> Any:
    """A deep snapshot of an object's fields, for before/after comparison."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return copy.deepcopy(dataclasses.asdict(obj))
    return copy.deepcopy(getattr(obj, "__dict__", obj))


def _assert_type(obj: Any, expected_type: Type) -> None:
    if not isinstance(obj, expected_type):
        raise AssertionError(f"Expected {expected_type.__name__}, got {type(obj).__name__}")


def _assert_all_type(lst: list, expected_type: Type) -> None:
    for i, item in enumerate(lst):
        if not isinstance(item, expected_type):
            raise AssertionError(
                f"Element {i} is {type(item).__name__}, expected {expected_type.__name__}"
            )


def _assert_true(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _results_equal(a: Any, b: Any) -> bool:
    """Compare two results, handling numpy arrays and nested structures."""
    try:
        import numpy as np

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return a == b


def _assert_deterministic(fn, n_runs: int = 3) -> None:
    """Check that a function produces the same output across multiple calls."""
    results = [fn() for _ in range(n_runs)]
    for i in range(1, n_runs):
        if not _results_equal(results[0], results[i]):
            raise AssertionError(f"Non-deterministic output: run 0 != run {i}")


def _profile_latency(report: ValidationReport, fn, n_runs: int = 5) -> None:
    """Profile the average latency of a function."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    report.latency_ms = sum(times) / len(times)
