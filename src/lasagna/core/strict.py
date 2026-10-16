"""Strict stages: during an evaluation, a stage that cannot run fails the task.

When serving, a stage that fails (no LLM for query expansion, a HyDE call that
errors) falls back to plain search: an answer beats an error. When measuring,
the same fallback is a silent wrong number: an arm that enables query
expansion and cannot run it scores exactly the control, with nothing to say
so. The evaluation harness switches this on for its own tasks; serving never
sees it.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, List, Optional

_STRICT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "lasagna_strict_stages", default=False
)


def strict_stages() -> bool:
    """Whether a stage that cannot run must raise instead of falling back."""
    return _STRICT.get()


def enable_strict_stages() -> None:
    """Switch strict stages on for the current context (and threads it spawns)."""
    _STRICT.set(True)


@contextmanager
def strict() -> Iterator[None]:
    """Run a block with strict stages on."""
    token = _STRICT.set(True)
    try:
        yield
    finally:
        _STRICT.reset(token)


#: Set by the evaluation harness around one query; each reranker component
#: call appends whether it changed the order of the candidates it was given.
_RERANK_OBSERVATIONS: contextvars.ContextVar[Optional[List[bool]]] = contextvars.ContextVar(
    "lasagna_rerank_observations", default=None
)


def rerank_observations() -> Optional[List[bool]]:
    """The list a reranker call appends to, or None outside :func:`observe_reranking`."""
    return _RERANK_OBSERVATIONS.get()


@contextmanager
def observe_reranking() -> Iterator[List[bool]]:
    """Collect, for the calls made inside the block, whether a reranker moved anything.

    A reranker that hands its candidates back in the order it got them (a
    component missing its model, MMR with no embedder) scores the pipeline
    without a reranker under its own name. Compared only with another
    reranker, as in the two-arm sweep, its ranked lists differ from that arm's
    and nothing flagged it; the isolation check reads this instead.
    """
    seen: List[bool] = []
    token = _RERANK_OBSERVATIONS.set(seen)
    try:
        yield seen
    finally:
        _RERANK_OBSERVATIONS.reset(token)
