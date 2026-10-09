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
from typing import Iterator

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
