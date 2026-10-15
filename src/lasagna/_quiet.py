"""Environment defaults that stop native libraries writing to the terminal.

For the ``lasagna`` and ``lasagna-mcp`` commands only. A library must not change
its host's process environment, so importing ``lasagna`` never calls this.
"""

from __future__ import annotations

import os

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
