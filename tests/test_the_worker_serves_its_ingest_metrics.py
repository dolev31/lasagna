"""The Celery worker serves the metrics of the documents it processes.

``lasagna_ingests_total`` is counted where a document is processed, and with
the celery profile that is the worker container, which served no
``/metrics``. Nothing scraped it: the api's counter showed only the documents
the api processed itself, which with a worker running is none.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import lasagna.core.config as config_mod
from lasagna.infrastructure.tasks import celery_app

DEPLOY = Path(__file__).resolve().parents[1] / "src" / "lasagna" / "deploy"


@pytest.fixture
def served(monkeypatch, tmp_path):
    started = []

    class _Process:
        def __init__(self, argv):
            started.append(argv)

        def terminate(self):
            started.append("terminated")

    monkeypatch.setattr(celery_app.subprocess, "Popen", _Process)
    monkeypatch.setattr(celery_app, "_metrics_server", None)
    monkeypatch.setattr(config_mod, "rag_config", SimpleNamespace(enable_prometheus=True))
    monkeypatch.setenv("RAG_WORKER_METRICS_PORT", "9100")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    return started


def test_a_started_worker_serves_metrics_on_its_port(served, tmp_path):
    (tmp_path / "counter_123.db").write_bytes(b"left by the last run")
    celery_app.serve_worker_metrics(sender=None)
    (argv,) = served
    # A process of its own, not a thread the pool's fork would copy.
    assert argv[1:] == [
        "-m",
        "lasagna.infrastructure.tasks.worker_metrics",
        "9100",
        str(__import__("os").getpid()),
    ]
    assert not list(tmp_path.glob("*.db"))
    celery_app.stop_worker_metrics(sender=None)
    assert served[-1] == "terminated"


def test_without_the_port_the_worker_serves_nothing(served, monkeypatch):
    monkeypatch.delenv("RAG_WORKER_METRICS_PORT")
    celery_app.serve_worker_metrics(sender=None)
    assert served == []


def test_without_a_shared_directory_it_says_why(served, monkeypatch):
    from loguru import logger

    said = []
    sink = logger.add(lambda m: said.append(str(m)), level="WARNING", format="{message}")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR")
    try:
        celery_app.serve_worker_metrics(sender=None)
    finally:
        logger.remove(sink)
    assert served == [] and "PROMETHEUS_MULTIPROC_DIR" in said[0]


def test_the_metrics_process_serves_the_shared_directory(monkeypatch, tmp_path):
    import prometheus_client

    from lasagna.infrastructure.tasks import worker_metrics

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    started = []
    monkeypatch.setattr(
        prometheus_client, "start_http_server", lambda port, registry: started.append(port)
    )
    # Its worker is already gone (its parent is another process), so it returns
    # at once. Not pid 1: in a container the test's parent can be pid 1, and the
    # loop then waited forever (it hung the suite in a Linux CI container).
    gone = os.getppid() + 1
    assert worker_metrics.main(["9100", str(gone)]) == 0
    assert started == [9100]


def test_the_worker_start_serves_them(monkeypatch):
    class _Signal:
        def __init__(self):
            self.receivers = []

        def connect(self, fn, weak=True):
            self.receivers.append(fn)

    signals = ModuleType("celery.signals")
    signals.worker_init, signals.worker_process_init = _Signal(), _Signal()
    signals.worker_shutdown = _Signal()
    monkeypatch.setitem(sys.modules, "celery.signals", signals)
    monkeypatch.setattr(
        celery_app,
        "Celery",
        lambda *a, **k: SimpleNamespace(
            conf=SimpleNamespace(update=lambda **kw: None), set_default=lambda: None
        ),
    )
    import lasagna.core.model_downloads as model_downloads

    monkeypatch.setattr(model_downloads, "_refusing", 0)
    celery_app.create_celery_app()
    assert celery_app.serve_worker_metrics in signals.worker_init.receivers
    assert celery_app.stop_worker_metrics in signals.worker_shutdown.receivers


def test_compose_and_kubernetes_scrape_the_worker():
    compose = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())["services"]
    worker = compose["celery-worker"]
    assert worker["environment"]["RAG_WORKER_METRICS_PORT"] == "9100"
    assert "9100" in [str(p) for p in worker["expose"]]
    assert "RAG_WORKER_METRICS_PORT" not in (compose["api"].get("environment") or {})

    pod = yaml.safe_load((DEPLOY / "k8s" / "base" / "deployment-celery.yaml").read_text())
    template = pod["spec"]["template"]
    assert template["metadata"]["annotations"]["prometheus.io/port"] == "9100"
    container = template["spec"]["containers"][0]
    assert {"name": "RAG_WORKER_METRICS_PORT", "value": "9100"} in container["env"]
    assert container["ports"][0]["containerPort"] == 9100
