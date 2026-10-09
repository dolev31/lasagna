"""Logging utilities for knowledge module.

This module provides logging configuration for the RAG knowledge base system.
"""

import os
import sys
from typing import Optional

from loguru import logger

_DEBUG_HANDLER_ID: Optional[int] = None


def configure_knowledge_logging(level: str) -> None:
    """Configure additional DEBUG logging for knowledge modules."""
    if os.getenv("LASAGNA_SKIP_LOGGING") == "1":
        return

    global _DEBUG_HANDLER_ID
    if _DEBUG_HANDLER_ID is not None:
        try:
            logger.remove(_DEBUG_HANDLER_ID)
        except Exception:
            pass
        _DEBUG_HANDLER_ID = None

    if str(level).upper() != "DEBUG":
        return

    def _filter(record) -> bool:
        name = record.get("name", "")
        return name.startswith("lasagna")

    _DEBUG_HANDLER_ID = logger.add(sys.stderr, level="DEBUG", filter=_filter)


__all__ = ["configure_knowledge_logging"]
