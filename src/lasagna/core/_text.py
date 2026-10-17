"""Text helpers shared by the CLI and the evaluation SDK."""

from __future__ import annotations

import difflib
from typing import Iterable


#: The log levels every surface accepts (``log_level``, ``lasagna --verbose``'s
#: sink, ``lasagna-mcp --log-level``): loguru's. The MCP server took four of the
#: seven the setting accepts.
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")


def did_you_mean(key: str, known: Iterable[str], quote: str = "") -> str:
    """``" (did you mean 'speed'?)"`` for the closest of ``known``, else ``""``.

    Written to follow a quoted name: ``f"Unknown profile {name!r}{did_you_mean(...)}"``.
    ``quote`` replaces the repr quoting, for a message that quotes its names
    another way (a data file's errors say ``unknown field "anwser"``).
    """
    close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)
    if not close:
        return ""
    shown = f"{quote}{close[0]}{quote}" if quote else repr(close[0])
    return f" (did you mean {shown}?)"
