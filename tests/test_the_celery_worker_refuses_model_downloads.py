"""The Celery worker refuses model downloads as the API does.

``lasagna serve`` and the MCP server refuse to download a model inside a
request unless ``RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS`` allows it. The worker
processes the same uploads and had no such refusal: a session on a model the
image did not carry made the worker download it inside the task.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from lasagna.core import model_downloads
from lasagna.infrastructure.tasks import celery_app


class _Signal:
    def __init__(self):
        self.receivers = []

    def connect(self, fn, weak=True):
        self.receivers.append(fn)


@pytest.fixture
def worker(monkeypatch):
    signals = ModuleType("celery.signals")
    signals.worker_init, signals.worker_process_init = _Signal(), _Signal()
    monkeypatch.setitem(sys.modules, "celery.signals", signals)

    class _Celery:
        def __init__(self, *a, **k):
            self.conf = SimpleNamespace(update=lambda **kw: None)

        def set_default(self):
            pass

    monkeypatch.setattr(celery_app, "Celery", _Celery)
    monkeypatch.setattr(model_downloads, "_refusing", 0)
    monkeypatch.setattr(model_downloads, "_process_refusal", False, raising=False)
    celery_app.create_celery_app()
    return signals


def test_every_worker_process_refuses_downloads(worker):
    for signal in (worker.worker_init, worker.worker_process_init):
        assert celery_app.refuse_model_downloads_in_worker in signal.receivers


def test_a_started_worker_refuses_an_uncached_model(worker, monkeypatch):
    monkeypatch.setattr(model_downloads, "is_cached", lambda *a, **k: False)
    monkeypatch.setattr(model_downloads, "offline", lambda: False)
    model_downloads.check("acme/not-downloaded")  # an API process before start: no refusal
    worker.worker_process_init.receivers[0](sender=None)
    with pytest.raises(model_downloads.ModelNotDownloaded, match="acme/not-downloaded"):
        model_downloads.check("acme/not-downloaded")


def test_the_download_setting_lets_a_worker_download(worker, monkeypatch):
    monkeypatch.setattr(model_downloads, "is_cached", lambda *a, **k: False)
    monkeypatch.setattr(model_downloads, "offline", lambda: False)
    monkeypatch.setattr(model_downloads, "_allowed", lambda: True)
    worker.worker_init.receivers[0](sender=None)
    model_downloads.check("acme/not-downloaded")
