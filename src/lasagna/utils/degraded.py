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
from typing import Dict, Iterable, Iterator, List, Optional

from loguru import logger

_stages: ContextVar[Optional[List[str]]] = ContextVar("lasagna_degraded_stages", default=None)
_reasons: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "lasagna_degraded_reasons", default=None
)
#: ``{stage: {surface: fix}}``: the fix for a stage as one surface spells it.
_fixes: ContextVar[Optional[Dict[str, Dict[str, str]]]] = ContextVar(
    "lasagna_degraded_fixes", default=None
)


def record(stage: str, reason: object = "", fixes: Optional[Dict[str, str]] = None) -> None:
    """Note that ``stage`` fell back to a lesser result. Never raises.

    ``fixes``: what fixes it, as each surface spells it (``{"mcp": ...}``),
    for a fix that depends on the session the search ran in.
    """
    try:
        from lasagna.utils.monitoring import METRICS

        METRICS.counter("lasagna_degraded_fallbacks", 1, tags={"stage": stage})
    except Exception as exc:  # metrics must never break a search
        logger.debug(f"Could not count the degraded {stage} stage: {exc}")
    stages = _stages.get()
    if stages is not None and stage not in stages:
        stages.append(stage)
    reasons = _reasons.get()
    if reasons is not None and reason and stage not in reasons:
        reasons[stage] = str(reason)
    known = _fixes.get()
    if known is not None and fixes and stage not in known:
        known[stage] = dict(fixes)


def fix(stage: str, surface: str) -> Optional[str]:
    """The fix recorded for ``stage`` as ``surface`` spells it, or None."""
    return (_fixes.get() or {}).get(stage, {}).get(surface)


def current() -> List[str]:
    """The stages collected so far by the active `collecting` block, or none."""
    return list(_stages.get() or [])


def reasons() -> Dict[str, str]:
    """Why each collected stage fell back, as `record` was told: ``{stage: text}``.

    The MCP search tool shows them to the agent. A search that missed the
    session's documents (``documents_elsewhere``) replied "Try rephrasing your
    search", and the reason, with its fix, went to the server's log only.
    """
    return dict(_reasons.get() or {})


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
    reasons_token = _reasons.set({})
    fixes_token = _fixes.set({})
    try:
        yield stages
    finally:
        _fixes.reset(fixes_token)
        _reasons.reset(reasons_token)
        _stages.reset(token)
