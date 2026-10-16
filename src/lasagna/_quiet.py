"""Keep native libraries from writing to the terminal.

:func:`quiet_third_party_output` sets environment defaults, for the ``lasagna``
and ``lasagna-mcp`` commands only: a library must not change its host's process
environment, so importing ``lasagna`` never calls it. The other two helpers
are what the library path uses instead, and touch no environment variable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

#: Each is a default: a value already in the environment wins.
QUIET_DEFAULTS = {
    # gRPC, under Milvus Lite, logs fork() notices through absl at INFO on
    # every command that opens the store.
    "GRPC_VERBOSITY": "ERROR",
    "GLOG_minloglevel": "2",
    # tokenizers warns about fork-after-parallelism on the first upload.
    "TOKENIZERS_PARALLELISM": "false",
    # transformers draws a "Loading weights" bar for every model it loads,
    # which on a cached model lasts a few milliseconds. A delay hides bars for
    # anything shorter and keeps them for what needs one: the multi-gigabyte
    # first download (HF_HUB_DISABLE_PROGRESS_BARS would hide that too, and a
    # silent download reads as a hang) and a long ingest.
    "TQDM_DELAY": "2",
}


def quiet_third_party_output() -> None:
    """Apply :data:`QUIET_DEFAULTS` before any of those libraries is imported."""
    for key, value in QUIET_DEFAULTS.items():
        os.environ.setdefault(key, value)
    tqdm_without_process_lock()


def tqdm_without_process_lock() -> None:
    """Keep tqdm from creating its cross-process lock, unless it already has.

    The first tqdm bar in a process (even a disabled one) creates a
    multiprocessing ``RLock``, a POSIX semaphore, and registering it starts
    multiprocessing's resource tracker with fork and exec. Under Milvus Lite
    gRPC is running by then, and it writes "Other threads are currently calling
    into gRPC, skipping fork() handlers" and one "FD from fork parent still in
    poll list" line per descriptor to stderr; at exit the tracker reports the
    semaphore as leaked. The lock only orders bars drawn from several
    processes, which Lasagna never does; without it tqdm uses its thread lock,
    as it does where multiprocessing is unavailable.
    """
    try:
        from tqdm.std import TqdmDefaultWriteLock
    except ImportError:
        return
    if not hasattr(TqdmDefaultWriteLock, "mp_lock"):
        TqdmDefaultWriteLock.mp_lock = None


@contextmanager
def loading_bar_hidden(model: str) -> Iterator[None]:
    """Hide transformers' "Loading weights" bar while a cached ``model`` loads.

    On a cached model the bar lasts milliseconds, and from Python it was most
    of what the README's quickstart printed. A model that has to be downloaded
    keeps its bars, since a silent multi-gigabyte download reads as a hang.
    """
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        hf_logging = None
    from lasagna.core.model_downloads import is_cached

    if hf_logging is None or not hf_logging.is_progress_bar_enabled() or not is_cached(model):
        yield
        return
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.enable_progress_bar()
