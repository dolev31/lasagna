"""The local Milvus Lite store's lock, and the ``lasagna serve`` that may hold it.

Milvus Lite admits one process per store. While ``lasagna serve`` held it,
``lasagna status``, ``search`` and ``upload`` loaded their models for about 7 s
and then failed with "Another process is using the local store", and so did a
``lasagna-mcp`` an agent started. ``serve`` now records its address in
``<data dir>/serve.json`` while it runs. A command that finds the store held
checks that file before loading anything: with a live server named there it
goes through the server's REST API, and otherwise it stops at once, naming the
fix.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


def _server_file() -> Path:
    from lasagna.core import paths

    return paths.data_dir() / "serve.json"


def announce(url: str) -> None:
    """Record that this process serves the local store at ``url``."""
    path = _server_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": os.getpid(), "url": url}))
    except OSError:
        pass


def withdraw() -> None:
    """Remove this process's record, if it is still this process's."""
    path = _server_file()
    try:
        if json.loads(path.read_text()).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def running_server() -> Optional[str]:
    """The URL of the ``lasagna serve`` that recorded itself, if it still runs."""
    try:
        record = json.loads(_server_file().read_text())
        pid, url = int(record["pid"]), str(record["url"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return url if pid != os.getpid() and _alive(pid) else None


def lite_store_locked(db_path: str) -> bool:
    """Whether another process holds the Milvus Lite store's lock.

    Milvus Lite admits one process per store, by an advisory lock on
    ``<store>/LOCK``. Probing it is the only way to tell that case apart from
    any other failure to open, since pymilvus drops the cause.
    """
    lock_path = os.path.join(db_path, "LOCK")
    if not os.path.exists(lock_path):
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return False
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def local_store_path() -> Optional[str]:
    """The Milvus Lite store this process would open; None for any other store."""
    try:
        from lasagna.core.config import rag_config

        if getattr(rag_config, "vector_backend", "milvus") != "milvus":
            return None
        cfg = rag_config.vector_backend_config or {}
        if cfg.get("uri") or os.environ.get("MILVUS_URI"):
            return None
        if cfg.get("db_path"):
            return str(cfg["db_path"])
        from lasagna.providers import get_config_provider

        return os.path.join(str(get_config_provider().db_dir), "rag_milvus.db")
    except Exception:
        return None


def _served_here(db_path: str) -> bool:
    """Whether this process's own Milvus Lite server holds the store.

    milvus_lite runs its server as a child of the process that opened the
    store, and the child holds the lock: that is this process using its store,
    not another one holding it. Read from ``sys.modules``, never imported.
    """
    manager = sys.modules.get("milvus_lite.server_manager")
    servers = getattr(getattr(manager, "server_manager_instance", None), "_servers", None)
    return isinstance(servers, dict) and os.path.abspath(db_path) in servers


def local_store_held() -> bool:
    """Whether another process holds the local store this one would open."""
    path = local_store_path()
    return bool(path) and lite_store_locked(path) and not _served_here(path)


def held_store_message(command: str) -> str:
    """One sentence for a command that found the store held and no server to use."""
    path = local_store_path() or ""
    home = os.path.expanduser("~")
    shown = "~" + path[len(home) :] if path.startswith(home + os.sep) else path
    return (
        f"Another process is using the local store at {shown}, so {command} cannot "
        f"open it: an MCP server an agent started, a benchmark, or a `lasagna serve` "
        f"from an older version. Stop it and run this again, or run `lasagna serve "
        f"--mcp` once so the CLI and agents share it (agents register "
        f"http://127.0.0.1:8000/mcp/sse by URL)."
    )
