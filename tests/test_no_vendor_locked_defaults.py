"""No default may name a specific vendor's model route.

A default of ``"<vendor>/<model>"`` works for exactly one deployment: the one it
was developed against. Everyone else gets a 400 from their provider — and
because these particular settings feed optional features (LLM listwise
reranking, graph entity extraction), the failure surfaces as a degraded result
rather than a crash. The user sees worse retrieval and no explanation.

Five such defaults shipped in this package. This test is what stops the sixth,
because the mistake is invisible to anyone whose environment happens to serve
that route.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lasagna"

#: Prefixes that route to one provider. A bare model name (``gpt-4o-mini``) is
#: fine as a fallback: most gateways accept it, and `get_default_model_name`
#: already uses it. A routed name is not.
VENDOR_ROUTED = re.compile(
    r'["\'](?:Azure|GCP|AWS|Bedrock|Vertex|watsonx|IBM)/[\w.\-]+["\']',
    re.IGNORECASE,
)


def _python_files():
    return [p for p in SRC.rglob("*.py") if "test" not in p.name]


def _prose_nodes(tree: ast.AST) -> set:
    """Every string node that is documentation rather than a value.

    A string that stands alone as a statement is prose: module, class and
    function docstrings, and the attribute docstrings that follow a dataclass
    field. It is never used as a default, so naming a vendor route inside one
    is exactly how you explain the format to a reader.

    This is why the check parses instead of grepping — a regex over source text
    cannot tell an example from a default.
    """
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


class TestNoVendorRoutedDefaults:
    def test_no_vendor_route_is_used_as_a_value(self):
        """Documentation may name one; executable code may not default to one."""
        offenders = []
        for path in _python_files():
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:  # pragma: no cover - would fail elsewhere first
                continue
            docstrings = _prose_nodes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in docstrings:
                    continue
                if VENDOR_ROUTED.search(f'"{node.value}"'):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {node.value!r}")
        assert not offenders, (
            "vendor-routed model names used as values — these fail for every "
            "user not on that provider:\n  " + "\n  ".join(offenders)
        )


class TestTheDefaultsResolve:
    """The replacement must actually produce a usable name, not an empty string."""

    @pytest.fixture
    def config(self):
        from lasagna.core.config import rag_config

        return rag_config

    def test_llm_reranker_model_resolves(self, config):
        assert config.llm_reranker_model, "empty model name would fail at call time"

    def test_graph_extraction_model_resolves(self, config):
        assert config.graph_extraction_model

    def test_both_follow_the_configured_default(self, config, monkeypatch):
        """Setting MODEL_NAME must reach these, or they silently diverge."""
        monkeypatch.setenv("MODEL_NAME", "my-org/my-model")
        assert config.llm_reranker_model == "my-org/my-model"
        assert config.graph_extraction_model == "my-org/my-model"

    def test_an_explicit_setting_still_wins(self, config, monkeypatch):
        """The fallback must not override a user who configured a model."""
        monkeypatch.setenv("MODEL_NAME", "fallback-model")
        config._config["llm_reranker_model"] = "explicit-choice"
        try:
            assert config.llm_reranker_model == "explicit-choice"
        finally:
            config._config["llm_reranker_model"] = ""
