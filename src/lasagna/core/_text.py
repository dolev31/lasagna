"""Text helpers shared by the CLI and the evaluation SDK."""

from __future__ import annotations

import difflib
from typing import Iterable


def did_you_mean(key: str, known: Iterable[str]) -> str:
    """``" (did you mean 'speed'?)"`` for the closest of ``known``, else ``""``.

    Written to follow a quoted name: ``f"Unknown profile {name!r}{did_you_mean(...)}"``.
    """
    close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""
