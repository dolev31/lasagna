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

import ipaddress
import json
import os
import stat
import sys
from pathlib import Path
from typing import Optional


class StoreHeld(RuntimeError):
    """Another process holds the local store; the message says which and the fix."""


def _server_file() -> Path:
    from lasagna.core import paths

    return paths.data_dir() / "serve.json"


def announce(url: str, session: Optional[str] = None) -> None:
    """Record that this process serves the local store at ``url``.

    ``session`` is the session the server answers for when a request names
    none (``serve -s``); other commands name it when they used another.

    Written afresh with mode 0600: `running_server` follows only a record its
    own user owns and nobody else can write, since other commands send
    documents and queries wherever it points.
    """
    path = _server_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o600), "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "url": url, "session": session}))
    except OSError:
        pass


def _read_record() -> Optional[dict]:
    """The record, if this user wrote it and only this user can change it.

    Not a symlink, owned by this user, and not writable by group or others: a
    record someone else could write would let them choose where `status`,
    `search` and `upload` send documents and queries.
    """
    path = _server_file()
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            return None
        if hasattr(os, "getuid") and (
            info.st_uid != os.getuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return None
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def withdraw() -> None:
    """Remove this process's record, if it is still this process's."""
    record = _read_record()
    if record is not None and record.get("pid") == os.getpid():
        try:
            _server_file().unlink()
        except OSError:
            pass


def _alive(pid: int) -> bool:
    """Whether ``pid`` is a running process of this user.

    A pid this user may not signal belongs to someone else: the server that
    wrote the record has exited and the pid was reused, so it is not ours.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _loopback_http_url(url) -> Optional[str]:
    """``url`` if it is ``http://<loopback address>:<port>`` and nothing more."""
    if not isinstance(url, str):
        return None
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "http" or port is None or parts.username or parts.password:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    host = parts.hostname or ""
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return None
        except ValueError:
            return None
    return url


def running_server() -> Optional[str]:
    """The URL of the ``lasagna serve`` that recorded itself, if it still runs."""
    record = _read_record() or {}
    pid, url = record.get("pid"), _loopback_http_url(record.get("url"))
    if type(pid) is not int or pid <= 0 or url is None:
        return None
    return url if pid != os.getpid() and _alive(pid) else None


def running_session() -> Optional[str]:
    """The session the running ``lasagna serve`` answers for by default, if it recorded one."""
    if running_server() is None:
        return None
    session = (_read_record() or {}).get("session")
    return session if isinstance(session, str) and session else None


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
