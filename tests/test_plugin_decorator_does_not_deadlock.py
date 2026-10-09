"""A plugin that registers itself with the decorator must not deadlock.

docs/contributing-components.md shows both `@registry.register(...)` on the
class and shipping it through a `lasagna_components` entry point. Discovery
loads entry points while holding the registry's lock, and the decorator asked
for that lock again on the same thread: every process that touched the
registry (plugins list, doctor, experiments) hung forever.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import importlib.metadata as md

import importlib
from lasagna.core.base_components import BaseReranker


def test_a_decorator_inside_an_entry_point_registers_without_hanging(monkeypatch):
    # lasagna.core re-exports the registry object under the module's name.
    registry_module = importlib.import_module("lasagna.core.registry")
    reg = registry_module.ComponentRegistry()

    def load():
        @reg.register("reranker", "deco_demo")
        class DecoReranker(BaseReranker):
            def rerank(self, query, documents, top_k=5, **kwargs):
                return documents[:top_k]

        return DecoReranker

    fake_ep = SimpleNamespace(name="reranker.deco_demo", value="demo:DecoReranker", load=load)
    monkeypatch.setattr(md, "entry_points", lambda **kw: [fake_ep])

    outcome = {}
    worker = threading.Thread(
        target=lambda: outcome.setdefault("names", reg.list("reranker")), daemon=True
    )
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "registry deadlocked while loading a decorator-using plugin"
    assert "deco_demo" in [getattr(n, "name", n) for n in outcome["names"]] or "deco_demo" in str(
        outcome["names"]
    )
