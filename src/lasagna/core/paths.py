"""Where Lasagna keeps its data, and the move from the old location.

Everything a user stores lives under one data directory: the metadata
database and the vector store in ``dbs/``, uploaded files in ``uploads/``. It is
``~/.lasagna`` unless ``LASAGNA_DATA_DIR`` (or ``RAG_DATA_DIR``) names another,
and each part can still be placed on its own with ``LASAGNA_DB_DIR`` and
``LASAGNA_UPLOAD_DIR`` (or the ``RAG_`` names they always had).

Before 0.1.0 the default was ``~/.rag_knowledge``. The first time the default
is used and only that directory exists, it is renamed to ``~/.lasagna``, with
one line in the log. Two existing directories are never merged and nothing is
deleted: when both exist and only the old one holds a database, a warning says
how to move it or keep using it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

DATA_DIR_NAME = ".lasagna"
LEGACY_DATA_DIR_NAME = ".rag_knowledge"

_checked: set = set()
_lock = threading.Lock()


def env_get(name: str, default: Optional[str] = None) -> Optional[str]:
    """``LASAGNA_<name>``, else ``RAG_<name>``, else ``default``.

    ``LASAGNA_`` wins when both are set. An empty value counts as unset, which
    is what ``${VAR:-}`` leaves in a compose file.
    """
    for prefix in ("LASAGNA_", "RAG_"):
        value = os.environ.get(prefix + name)
        if value:
            return value
    return default


def data_dir() -> Path:
    """The directory Lasagna stores data under (see the module docstring)."""
    explicit = env_get("DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    home = Path.home()
    target = home / DATA_DIR_NAME
    _move_legacy(home / LEGACY_DATA_DIR_NAME, target)
    return target


def db_dir() -> Path:
    """Where the metadata database and the Milvus Lite store live."""
    explicit = env_get("DB_DIR")
    return Path(explicit).expanduser() if explicit else data_dir() / "dbs"


def upload_dir() -> Path:
    """Where uploaded files are kept."""
    explicit = env_get("UPLOAD_DIR")
    return Path(explicit).expanduser() if explicit else data_dir() / "uploads"


def _move_legacy(legacy: Path, target: Path) -> None:
    """Rename ``legacy`` to ``target`` once, when only ``legacy`` exists."""
    key = (str(legacy), str(target))
    if key in _checked:
        return
    with _lock:
        if key in _checked:
            return
        _checked.add(key)
        if not legacy.is_dir():
            return
        if not target.exists():
            try:
                legacy.rename(target)
            except OSError as exc:
                if target.exists() and not legacy.exists():
                    return  # another process moved it first
                logger.warning(
                    f"Could not move {legacy} to {target}: {exc}. Lasagna now looks in "
                    f"{target}; to keep the data where it is, set LASAGNA_DATA_DIR={legacy}."
                )
                return
            logger.info(f"Moved Lasagna's data from {legacy} to {target}, its new location.")
            return
        if (legacy / "dbs").is_dir() and not (target / "dbs").exists():
            parts = " ".join(
                str(legacy / name)
                for name in ("dbs", "uploads")
                if (legacy / name).exists() and not (target / name).exists()
            )
            logger.warning(
                f"Documents from an earlier version are in {legacy}, which Lasagna no "
                f"longer reads; {target} already exists, so they were not moved. Move "
                f"them with: mv {parts} {target}/ "
                f"(or set LASAGNA_DATA_DIR={legacy} to keep using that directory)."
            )
