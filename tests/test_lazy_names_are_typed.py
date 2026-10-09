"""Names resolved lazily are still visible to a type checker.

``lasagna`` and ``lasagna.sdk`` load most public names on first access through
a module ``__getattr__``, so a user's type checker saw ``lasagna.KnowledgeClient``
and ``from lasagna import reranker`` as ``Any``. Each module now repeats its
lazy names in an ``if TYPE_CHECKING:`` block. These tests keep that block in
step with ``_LAZY_IMPORTS``, so adding a lazy name without its typed import
fails here.
"""

from __future__ import annotations

import ast
import importlib

import pytest

from tests._paths import PACKAGE_DIR


def _typed_imports(path) -> dict:
    tree = ast.parse(path.read_text())
    blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    ]
    assert len(blocks) == 1, f"{path} needs exactly one `if TYPE_CHECKING:` block"
    found = {}
    for node in blocks[0].body:
        assert isinstance(node, ast.ImportFrom), ast.dump(node)
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            found[alias.asname or alias.name] = (module, alias.name)
    return found


@pytest.mark.parametrize("package", ["lasagna", "lasagna.sdk"])
def test_the_typed_block_mirrors_the_lazy_imports(package):
    module = importlib.import_module(package)
    path = PACKAGE_DIR.joinpath(*package.split(".")[1:], "__init__.py")
    assert _typed_imports(path) == dict(module._LAZY_IMPORTS)


def test_every_top_level_lazy_name_resolves():
    import lasagna

    missing = [name for name in lasagna._LAZY_IMPORTS if getattr(lasagna, name, None) is None]
    assert not missing
