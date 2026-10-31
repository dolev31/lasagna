"""The compose stack gives Qdrant enough file descriptors for client bursts.

Eight concurrent clients creating collections and payload indexes exhausted
the container's default 1024 FDs ("Too many open files" in the Qdrant log)
during stress testing of the collection-creation race fix. The Qdrant service
therefore raises `nofile` to 524288, the value that run validated; this test
pins that requirement to the compose file so a regression fails here instead
of in production.
"""

from __future__ import annotations

import yaml

from tests._paths import DEPLOY_DIR

MIN_NOFILE = 524288


def _soft_nofile(service: dict) -> int:
    ulimit = (service.get("ulimits") or {}).get("nofile")
    if isinstance(ulimit, dict):
        return int(ulimit.get("soft", 0))
    return int(ulimit or 0)


def test_qdrant_service_raises_nofile_ulimit():
    services = yaml.safe_load((DEPLOY_DIR / "backends/qdrant.yml").read_text())["services"]
    assert _soft_nofile(services["qdrant"]) >= MIN_NOFILE
