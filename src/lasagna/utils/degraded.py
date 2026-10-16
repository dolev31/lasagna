"""Which pipeline stages fell back during a search, reported to the caller.

HyDE, GraphRAG and hybrid search each catch their own failure and carry on
with less (the original query, the dense candidates, dense-only results). That
is the right behavior for a search, and the wrong thing to hide: a client got
a 200 and a result list that looked like a hybrid search's and was not. Each
fallback now calls `record`, which counts it in
``lasagna_degraded_fallbacks_total{stage=...}`` and, inside `collecting`,
adds the stage to the list a REST response reports as ``degraded_stages``.

A context variable, so the stages are collected per request, including from
the worker thread a search runs in (the async search copies its context).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Iterator, List, Optional

from loguru import logger

_stages: ContextVar[Optional[List[str]]] = ContextVar("lasagna_degraded_stages", default=None)


def record(stage: str, reason: object = "") -> None:
    """Note that ``stage`` fell back to a lesser result. Never raises."""
    try:
        from lasagna.utils.monitoring import METRICS

        METRICS.counter("lasagna_degraded_fallbacks", 1, tags={"stage": stage})
    except Exception as exc:  # metrics must never break a search
        logger.debug(f"Could not count the degraded {stage} stage: {exc}")
    stages = _stages.get()
    if stages is not None and stage not in stages:
        stages.append(stage)


def current() -> List[str]:
    """The stages collected so far by the active `collecting` block, or none."""
    return list(_stages.get() or [])


def report(stages: Iterable[str]) -> None:
    """Add stages another process fell back on (a server's ``degraded_stages``).

    Added to the active `collecting` block only: they were counted where they
    happened.
    """
    active = _stages.get()
    if active is None:
        return
    for stage in stages:
        if stage not in active:
            active.append(stage)


@contextmanager
def collecting() -> Iterator[List[str]]:
    """Collect the stages that fall back inside the ``with``, in order."""
    stages: List[str] = []
    token = _stages.set(stages)
    try:
        yield stages
    finally:
        _stages.reset(token)
