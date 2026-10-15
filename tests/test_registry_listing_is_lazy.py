"""Listing components imports none of them.

Touching the registry resolved every built-in's factory, so `registry.list_*()`,
`lasagna plugins list` and `lasagna doctor` imported torch, transformers and
pymilvus: 6.6 s before the first line of output, and pymilvus's import read the
nearest .env. Built-ins now stay deferred and declare `implements_contract`;
listing reads the declaration, and only `get` imports a component.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from lasagna.core.registry import registry
from tests._paths import PACKAGE_DIR

HEAVY = ("torch", "transformers", "sentence_transformers", "pymilvus")
#: One module per built-in family; listing must leave each unimported.
BUILTIN_MODULES = (
    "lasagna.retrieval.search.ranker",
    "lasagna.retrieval.search.retrievers",
    "lasagna.processing.chunker.semantic",
    "lasagna.infrastructure.embedding.embedders",
    "lasagna.generation.strategies",
    "lasagna.agentic.graph_builder",
)


def _imported_after(code: str, tmp_path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update({"HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent)})
    probe = (
        code
        + "\nimport json, sys\n"
        + f"print('MODULES', json.dumps({{m: m in sys.modules for m in {HEAVY + BUILTIN_MODULES}}}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=300
    )
    line = next((x for x in out.stdout.splitlines() if x.startswith("MODULES ")), None)
    assert line, out.stdout + out.stderr
    return json.loads(line[len("MODULES ") :])


LISTINGS = (
    "from lasagna.core.registry import registry\n"
    "for listing in (registry.list_rerankers, registry.list_embedders, registry.list_chunkers,\n"
    "                registry.list_enhancers, registry.list_retrievers,\n"
    "                registry.list_document_processors, registry.list_pipelines,\n"
    "                registry.list_all):\n"
    "    assert listing()\n"
)


def _run_cli(*args: str) -> str:
    return (
        "import sys\n"
        f"sys.argv = ['lasagna', *{list(args)!r}]\n"
        "from lasagna.cli import app\n"
        "try:\n"
        "    app()\n"
        "except SystemExit:\n"
        "    pass\n"
    )


def test_listing_every_stage_imports_no_component(tmp_path):
    loaded = _imported_after(LISTINGS, tmp_path)
    assert not [m for m, present in loaded.items() if present], loaded


def test_component_list_imports_no_component(tmp_path):
    loaded = _imported_after(_run_cli("component", "list"), tmp_path)
    assert not [m for m, present in loaded.items() if present], loaded


def test_plugins_list_imports_no_component(tmp_path):
    loaded = _imported_after(_run_cli("plugins", "list"), tmp_path)
    assert not [m for m, present in loaded.items() if present], loaded


def test_doctor_imports_no_component(tmp_path):
    """`doctor` reports the torch version and GPU, so torch itself is allowed,
    and it asks the semantic chunker whether sentence splitting has fallen back
    to a regex."""
    loaded = _imported_after(_run_cli("doctor", "--json"), tmp_path)
    assert not loaded["pymilvus"]
    reported = {"lasagna.processing.chunker.semantic"}
    assert not [m for m in BUILTIN_MODULES if loaded[m] and m not in reported], loaded


def _builtins_by_stage():
    from lasagna.core.registry import _get_expected_abc

    listings = {
        "reranker": registry.list_rerankers,
        "embedder": registry.list_embedders,
        "chunker": registry.list_chunkers,
        "query_enhancer": registry.list_enhancers,
        "retriever": registry.list_retrievers,
        "document_processor": registry.list_document_processors,
        "agentic_pipeline": registry.list_pipelines,
    }
    for stage, listing in listings.items():
        for entry in listing():
            if registry.is_builtin(stage, entry["name"]):
                yield stage, entry, _get_expected_abc(stage)


@pytest.mark.parametrize(
    "stage,name", sorted((stage, entry["name"]) for stage, entry, _ in _builtins_by_stage())
)
def test_a_declared_contract_is_the_real_one(stage, name):
    """The declaration replaces `issubclass`, so it is checked against it here."""
    entry, abc = next((e, a) for s, e, a in _builtins_by_stage() if (s, e["name"]) == (stage, name))
    assert entry["implements_contract"] is True
    assert issubclass(registry.get_class(stage, name), abc)
