"""Serve a Celery worker's Prometheus metrics from a process of their own.

    python -m lasagna.infrastructure.tasks.worker_metrics <port> <worker pid>

Started by ``celery_app.serve_worker_metrics`` when a worker starts. The
server is not a thread in the worker: the worker's main process forks its pool
after that, and a fork while another thread runs can deadlock the child (Python
3.12 warns that it may). This process shares no lock with the worker. It reads
``PROMETHEUS_MULTIPROC_DIR``, where every pool process writes its counts, and
exits once the worker that started it is gone.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    port, worker = (int(v) for v in (argv if argv is not None else sys.argv[1:])[:2])
    from prometheus_client import CollectorRegistry, multiprocess, start_http_server

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(port, registry=registry)
    while os.getppid() == worker:  # re-parented once the worker exits
        time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
