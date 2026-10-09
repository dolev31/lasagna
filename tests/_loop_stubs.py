"""Stand-ins for the LLM and the pipeline's search, shared by the agentic-loop tests.

No LLM key exists in CI, so every loop test scripts the model's replies and
records what it was asked. The search stand-in replaces ``RAGBackend`` itself,
which is what the loops call, so a test that sees its calls proves the loop
searched through the pipeline rather than a private retriever.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage

from lasagna.core.base_components import AgenticPipelineInput

Reply = Union[str, AIMessage, Callable[[str], Union[str, AIMessage]]]


class ScriptedLLM:
    """Replies from a script, in order; records every prompt and stop list.

    A reply may be a function of the prompt, for a step whose wording the
    test does not want to pin. Running out of script is a test failure, so a
    loop that makes one call more than the method calls for is caught.
    """

    def __init__(self, replies: List[Reply]):
        self.replies = list(replies)
        self.prompts: List[str] = []
        self.stops: List[Optional[List[str]]] = []
        self.bound: List[Dict[str, Any]] = []

    async def ainvoke(self, messages, stop=None, **kwargs):
        prompt = messages[-1].content if isinstance(messages, list) else str(messages)
        self.prompts.append(prompt)
        self.stops.append(stop)
        if not self.replies:
            raise AssertionError(f"unscripted LLM call #{len(self.prompts)}:\n{prompt[-400:]}")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(prompt)
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)

    def bind(self, **kwargs):
        self.bound.append(kwargs)
        return self


def with_logprobs(text: str, probs: List[float]) -> AIMessage:
    """A reply carrying OpenAI-style token logprobs, one token per word."""
    import math

    words = text.split(" ")
    assert len(words) == len(probs), "one probability per word"
    content = [
        {"token": (w if i == 0 else " " + w), "logprob": math.log(p)}
        for i, (w, p) in enumerate(zip(words, probs))
    ]
    return AIMessage(content=text, response_metadata={"logprobs": {"content": content}})


def hit(chunk_id: str, content: str, document_id: str = "", filename: str = "") -> Any:
    """One search result in the shape ``RAGBackend.search`` returns."""
    return SimpleNamespace(
        chunk=SimpleNamespace(
            chunk_id=chunk_id,
            content=content,
            document_id=document_id or f"doc-{chunk_id}",
            chunk_index=0,
            metadata={},
            page_number=None,
        ),
        score=1.0,
        document=SimpleNamespace(filename=filename or f"{chunk_id}.txt"),
    )


class FakeBackend:
    """Replaces ``lasagna.api.rag_backend.RAGBackend``; answers from a corpus by query.

    ``corpus`` maps a query to the passages its search returns; a query not in
    it returns nothing. Every call is recorded with its keyword arguments.
    """

    calls: List[Dict[str, Any]] = []
    corpus: Dict[str, List[Any]] = {}
    default: List[Any] = []

    def search(self, query, session_id, top_k=None, **kwargs):
        FakeBackend.calls.append(
            {"query": query, "session_id": session_id, "top_k": top_k, **kwargs}
        )
        found = FakeBackend.corpus.get(query, FakeBackend.default)
        return list(found)[: top_k or None]


def install_backend(
    monkeypatch, corpus: Dict[str, List[Any]], default: Optional[List[Any]] = None
) -> type:
    """Replace ``RAGBackend``; ``default`` answers any query not in ``corpus``."""
    from lasagna.api import rag_backend

    FakeBackend.calls = []
    FakeBackend.corpus = dict(corpus)
    FakeBackend.default = list(default or [])
    monkeypatch.setattr(rag_backend, "RAGBackend", FakeBackend)
    return FakeBackend


class ArmConfig(SimpleNamespace):
    """A session config with an arm's per-query overrides, as the orchestrator passes."""

    def __init__(self, **settings):
        super().__init__(**settings)
        self._query_overrides = dict(settings)


def pipeline_input(question: str, llm: Any, config: Any = None, top_k: int = 3):
    return AgenticPipelineInput(
        question=question,
        session_id="arm-session",
        top_k=top_k,
        session_config=config if config is not None else ArmConfig(),
        llm=llm,
    )
