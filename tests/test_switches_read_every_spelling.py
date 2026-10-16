"""Every on/off environment variable accepts the same spellings.

``core/config.TRUTHY`` (1, true, yes, on) was one of six sets. OpenSearch
compared with ``== "true"``, so ``OPENSEARCH_VERIFY_CERTS=yes`` or ``=1``
left certificate checks off; ``on`` was ignored by the lifecycle settings,
``RAG_EMBED_MCP``, ``RAG_CONFIGURATOR``, ``RAG_ALLOW_HEADER_IDENTITY`` and
``RAG_LANGFUSE_TRACING``. All of them read ``env_flag`` now.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import lasagna
from lasagna.core.config import FALSY, TRUTHY, env_flag

ON = ["1", "true", "TRUE", "yes", "on", " On "]
OFF = ["0", "false", "no", "off", "OFF"]


@pytest.mark.parametrize("value", ON)
def test_every_truthy_spelling_is_on(monkeypatch, value):
    monkeypatch.setenv("LASAGNA_TEST_SWITCH", value)
    assert env_flag("LASAGNA_TEST_SWITCH") is True


@pytest.mark.parametrize("value", OFF)
def test_every_falsy_spelling_is_off(monkeypatch, value):
    monkeypatch.setenv("LASAGNA_TEST_SWITCH", value)
    assert env_flag("LASAGNA_TEST_SWITCH", default=True) is False


@pytest.mark.parametrize("value", ["", "maybe"])
def test_unset_or_unknown_is_the_default(monkeypatch, value):
    monkeypatch.setenv("LASAGNA_TEST_SWITCH", value)
    assert env_flag("LASAGNA_TEST_SWITCH", default=True) is True
    assert env_flag("LASAGNA_TEST_SWITCH") is False


@pytest.mark.parametrize("value", ["yes", "1", "on"])
def test_opensearch_ssl_and_certificate_checks_turn_on(monkeypatch, value):
    from lasagna.vector_backends import opensearch_backend

    for name in (
        "OPENSEARCH_USE_SSL",
        "OPENSEARCH_VERIFY_CERTS",
        "OPENSEARCH_ENABLE_NEURAL_SPARSE",
    ):
        monkeypatch.setenv(name, value)
    config = opensearch_backend._get_default_opensearch_config()
    assert config["use_ssl"] is True and config["verify_certs"] is True
    assert opensearch_backend._neural_sparse_enabled() is True


def test_a_lifecycle_switch_reads_on_and_off(monkeypatch):
    from lasagna.lifecycle.policy import LifecyclePolicy

    monkeypatch.setenv("RAG_GC_RECONCILE_QUOTA", "off")
    assert LifecyclePolicy.load().reconcile_quota_on_startup is False
    monkeypatch.setenv("RAG_GC_RECONCILE_QUOTA", "on")
    assert LifecyclePolicy.load().reconcile_quota_on_startup is True


def test_langfuse_tracing_reads_on(monkeypatch):
    from lasagna.utils.model_utils import is_langfuse_tracing_enabled

    monkeypatch.setenv("RAG_LANGFUSE_TRACING", "on")
    assert is_langfuse_tracing_enabled() is True


def _own_sets(tree: ast.AST):
    """Comparisons of a string with a literal holding "true" or "false"."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.In)):
                items = right.elts if isinstance(right, (ast.Tuple, ast.Set, ast.List)) else [right]
                words = {i.value for i in items if isinstance(i, ast.Constant)}
                if words & {"true", "false"}:
                    yield node.lineno


def test_no_module_parses_a_switch_with_its_own_set():
    package = Path(lasagna.__file__).parent
    allowed = {
        # `git rev-parse --is-inside-work-tree` prints "true"; not a setting.
        "core/provenance.py",
        # Names, not switches: an agentic HyDE strategy ("off", "always", ...)
        # and a similarity model of "none".
        "agentic/agentic_rag.py",
        "knowledge_evaluation_sdk/experiment.py",
        # LASAGNA_ENABLE_DOCS_API: its set is TRUTHY, spelled out.
        "mcp/server.py",
    }
    found = [
        f"{path.relative_to(package)}:{line}"
        for path in sorted(package.rglob("*.py"))
        if str(path.relative_to(package)) not in allowed
        for line in _own_sets(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert TRUTHY.isdisjoint(FALSY)
    assert not found, "use core.config.env_flag (or TRUTHY/FALSY): " + ", ".join(found)
