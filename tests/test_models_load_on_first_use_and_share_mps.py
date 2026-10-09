"""Models load when first used, and never run on MPS from two threads at once.

A backend is built from the settings-file default (bge-m3, 2.3 GB) before any
profile applies. Loading it then downloaded that model on the `speed` path the
docs sell as keyless and small, and failed every task offline. And Metal aborts
the process when two threads encode on MPS at once, which every multi-arm sweep
on an Apple Silicon laptop did.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from lasagna.core.accelerator import device_guard
from lasagna.core.config import rag_config
from lasagna.core.strict import strict
from lasagna.infrastructure.embedding import manager as manager_mod


class _FakeST:
    loads: list = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, path, device=None):
        if "missing" in str(path):
            raise OSError(f"{path} is not in the cache")
        _FakeST.loads.append(str(path))
        self.device = device

    def encode(self, texts, **kwargs):
        with _FakeST.lock:
            _FakeST.active += 1
            _FakeST.peak = max(_FakeST.peak, _FakeST.active)
        time.sleep(0.01)
        with _FakeST.lock:
            _FakeST.active -= 1
        if isinstance(texts, str):
            return np.ones(4, dtype=np.float32)
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch):
    _FakeST.loads, _FakeST.active, _FakeST.peak = [], 0, 0
    monkeypatch.setattr(manager_mod, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(manager_mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(manager_mod.EmbeddingManager, "_resolve_local_snapshot", lambda s, m: m)
    return _FakeST


def _manager(model="t-small-model", fallback="t-fallback-model"):
    config = rag_config.with_query_overrides(
        {
            "embedding_model": model,
            "embedding_model_fallback": fallback,
            "embedding_dim": 4,
            "enable_embedding_cache": False,
            "use_gpu": False,
        }
    )
    return manager_mod.EmbeddingManager(config)


def test_building_a_manager_loads_nothing(fake_st):
    m = _manager()
    assert fake_st.loads == [] and not m.is_loaded
    assert m.model_name == "t-small-model" and m.dimension == 4
    m.encode_query("hello")
    assert fake_st.loads == ["t-small-model"] and m.is_loaded


def test_a_model_that_cannot_load_fails_once_not_per_query(fake_st):
    m = _manager(model="t-missing-model", fallback="t-missing-fallback")
    with pytest.raises(RuntimeError, match="Could not load any embedding model"):
        m.encode_query("q")
    calls = len(fake_st.loads)
    with pytest.raises(RuntimeError, match="Could not load any embedding model"):
        m.encode_query("q")
    assert len(fake_st.loads) == calls, "the load was retried"


def test_an_evaluation_never_falls_back_to_another_embedder(fake_st):
    m = _manager(model="t-missing-model")
    with strict(), pytest.raises(RuntimeError, match="does not fall back"):
        m.encode_query("q")
    assert "t-fallback-model" not in fake_st.loads

    serving = _manager(model="t-missing-model")
    serving.encode_query("q")  # serving keeps the fallback
    assert serving.model_name == "t-fallback-model"


def test_two_managers_never_encode_on_mps_at_once(fake_st):
    a, b = _manager(), _manager()
    a.encode_query("warm")
    b.encode_query("warm")
    a._device = b._device = "mps"
    fake_st.peak = 0
    threads = [
        threading.Thread(
            target=lambda m=m: [m.encode_query(f"q{i}", use_cache=False) for i in range(5)]
        )
        for m in (a, b, a, b)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_st.peak == 1


def test_cpu_work_is_not_serialized(fake_st):
    a, b = _manager(), _manager()
    a.encode_query("warm")
    b.encode_query("warm")
    a._device = b._device = "cpu"
    fake_st.peak = 0
    threads = [
        threading.Thread(
            target=lambda m=m: [m.encode_query(f"q{i}", use_cache=False) for i in range(5)]
        )
        for m in (a, b)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_st.peak == 2


def test_the_guard_is_only_taken_for_mps():
    import contextlib

    assert isinstance(device_guard("cpu"), contextlib.nullcontext)
    assert isinstance(device_guard("cuda:0"), contextlib.nullcontext)
    assert not isinstance(device_guard("mps"), contextlib.nullcontext)


def test_the_first_encode_on_a_fresh_manager_is_guarded_too(fake_st, monkeypatch):
    # The guard read the device before the lazy load set it, so the first
    # encode on each manager (load plus first batch) ran on MPS unlocked.
    def load_on_mps(self):
        # What the real load does on Apple Silicon: under the device guard.
        with manager_mod.device_guard("mps"):
            self._device = "mps"
            self.model = _FakeST(self.model_name, device="mps")
            self.model.encode("test")

    monkeypatch.setattr(manager_mod.EmbeddingManager, "_load_sentence_transformer", load_on_mps)
    managers = [_manager(), _manager(), _manager()]
    fake_st.peak = 0
    threads = [
        threading.Thread(
            target=lambda m=m: m.encode_documents_batch(["a", "b"], show_progress=False)
        )
        for m in managers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_st.peak == 1
