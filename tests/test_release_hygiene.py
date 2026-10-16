"""A first release ships no deprecated aliases, no leaked names, and plain punctuation.

The MCP server registered two deprecated aliases of ``search`` in a first
release; ``lasagna.TYPE_CHECKING`` was importable from the top-level package;
and runtime messages (the model license warning, the power verdict) used
" -- " where a colon or a comma belongs.
"""

from __future__ import annotations

import ast

import lasagna
from tests._paths import PACKAGE_DIR


def test_the_top_level_namespace_does_not_leak_typing():
    assert "TYPE_CHECKING" not in vars(lasagna)
    assert "TYPE_CHECKING" not in dir(lasagna)


def _runtime_strings():
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                yield path.relative_to(PACKAGE_DIR), node.lineno, node.value


def test_no_runtime_message_uses_a_double_dash():
    found = [
        f"{path}:{line}"
        for path, line, text in _runtime_strings()
        if " -- " in text and "redis.call" not in text  # a Lua script's comments
    ]
    assert not found, found


def test_the_mcp_server_has_no_deprecated_aliases():
    from lasagna.mcp import server

    names = {fn.__name__ for fn in server._TOOLS}
    assert not {"search_knowledge", "search_with_context"} & names
