"""One lock for Apple's MPS device, shared by every model in the process.

Metal aborts the whole process (``failed assertion _status <
MTLCommandBufferStatusCommitted``, or a bare segfault) when two threads encode
on MPS at once. Each model kept its own lock, but a sweep runs several arms
concurrently, each with its own embedder and reranker, so their calls overlapped
and every multi-arm sweep on an Apple Silicon laptop crashed. CUDA and CPU
need no such guard; there the lock is never taken.

Take it innermost, around one load or one batch of inference, so it never
waits on a lock that a holder of it might need.
"""

from __future__ import annotations

import contextlib
import threading
from typing import ContextManager, Optional

_MPS = threading.RLock()


def device_guard(device: Optional[str]) -> ContextManager:
    """Serialize work on ``device`` when it is MPS; a no-op otherwise."""
    if device is not None and str(device).startswith("mps"):
        return _MPS
    return contextlib.nullcontext()
