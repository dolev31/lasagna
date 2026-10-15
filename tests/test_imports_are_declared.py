"""Every module must import on a clean install, and say so when it cannot.

`infrastructure/llm_manager.py` did `from langchain_ibm import ChatWatsonx` at
module level. `langchain-ibm` is declared in no dependency group — pyproject's
own comment lists it among the packages "deliberately absent" — so importing
that module on a clean install raised ImportError, and the 12 tests exercising
`LLMManager` passed only on machines that happened to have the distribution.

Two structural reasons nothing caught it:

- `tests/test_onboarding_prerequisites.py` checks imports against declared
  dependencies **for `cli.py` only**. Its own docstring says it was written after
  `questionary` broke `lasagna init`. The identical defect one directory over was
  invisible because the check never left that file.
- Nothing walks the package. `e2e/phases/p1_install.py`'s only outside-in probe
  is `import lasagna`, and `lasagna/__init__.py` is deliberately lazy — so it
  touches almost nothing. Meanwhile `p2`/`p3` run against the source tree with
  `PYTHONPATH=src` and the developer's site-packages, which is exactly the
  configuration in which an undeclared dependency is invisible.

This walks every module. It cannot prove a dependency is *declared* — that needs
a clean venv, which `e2e` p1 is the place for — but it does prove that an
optional import is guarded rather than bare.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import lasagna

PACKAGE_ROOT = pathlib.Path(lasagna.__file__).parent

#: Distributions the package deliberately does not depend on. A module-level
#: import of any of these is a broken install, not a missing extra.
#:
#: Kept in step with pyproject's own note about the extraction from the host
#: project: playwright, browsergym, e2b, tavily, langfuse, langchain-ibm.
UNDECLARED_BY_DESIGN = {
    "langchain_ibm",
    "langfuse",
    "playwright",
    "browsergym",
    "e2b",
    "tavily",
    # The agent framework Lasagna was extracted from. Only
    # integrations/cuga_agent.py uses it, importing cuga.sdk inside each
    # function so the evaluator can drive a whole CUGA agent; no extra installs
    # it (the adapter says `pip install cuga`), so a module-level import of it
    # anywhere would break a clean install.
    "cuga",
}


def _module_files():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "frontend" in path.parts or path.name == "__main__.py":
            continue
        yield path


def _toplevel_imports(tree: ast.AST) -> set:
    """Names imported at module scope — not inside a function or a try block.

    A `try: import x / except ImportError:` is the guarded form and is fine;
    those sit inside a `Try` node, which this deliberately does not descend into.
    """
    found: set = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


class TestNoModuleImportsAnUndeclaredDistributionAtTopLevel:
    def test_every_module(self):
        offenders = []
        for path in _module_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover
                continue
            for name in _toplevel_imports(tree) & UNDECLARED_BY_DESIGN:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {name}")

        assert not offenders, (
            "these modules import a distribution the package does not depend on, "
            "at module scope, so they raise ImportError on a clean install:\n  "
            + "\n  ".join(offenders)
            + "\nGuard them with try/except ImportError and raise an actionable "
            "error at the point of use."
        )

    def test_the_check_can_actually_fire(self):
        """Guards the guard: a scanner that never matches passes vacuously."""
        tree = ast.parse("from langchain_ibm import ChatWatsonx\n")
        assert _toplevel_imports(tree) & UNDECLARED_BY_DESIGN == {"langchain_ibm"}

    def test_a_guarded_import_is_not_flagged(self):
        """The whole point of the distinction."""
        tree = ast.parse(
            "try:\n    from langchain_ibm import ChatWatsonx\n"
            "except ImportError:\n    ChatWatsonx = None\n"
        )
        assert not (_toplevel_imports(tree) & UNDECLARED_BY_DESIGN)


class TestTheOptionalProviderStillFailsUsefully:
    """Deferring an import to the call site only helps if the failure explains
    itself. Otherwise `ImportError` at import becomes `'NoneType' object is not
    callable` several frames deeper, which is worse.
    """

    def test_watsonx_names_the_package_to_install(self, monkeypatch):
        from lasagna.infrastructure import llm_manager

        monkeypatch.setattr(llm_manager, "ChatWatsonx", None)
        monkeypatch.setenv("WATSONX_PROJECT_ID", "p")

        manager = llm_manager.LLMManager()
        with pytest.raises(ImportError, match="langchain-ibm"):
            manager._create_llm_instance(
                {"platform": "watsonx", "model_name": "m", "max_tokens": 16}
            )

    def test_the_module_imports_without_the_optional_provider(self):
        """The property that was broken: importing this module must not require
        a vendor SDK the package does not depend on."""
        import importlib

        module = importlib.import_module("lasagna.infrastructure.llm_manager")
        assert hasattr(module, "LLMManager")


class TestTheExtrasDeclareWhatTheirBackendsImport:
    """`pgvector = ["psycopg[binary]>=3.1"]` declared one of the three
    distributions `pgvector_backend.py` imports. The imports are guarded, so
    `pip install "lasagna-rag[pgvector]"` succeeded and left the backend
    silently unavailable — the user met it as "backend not available" with no
    indication that the extra was incomplete.
    """

    def test_the_pgvector_extra_covers_its_imports(self):
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib

        pyproject = tomllib.loads(
            (PACKAGE_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = " ".join(pyproject["project"]["optional-dependencies"]["pgvector"])

        source = (PACKAGE_ROOT / "vector_backends" / "pgvector_backend.py").read_text()
        if "import pgvector" in source or "from pgvector" in source:
            assert "pgvector" in declared
        if "psycopg_pool" in source:
            assert "pool" in declared, (
                "the backend imports psycopg_pool; psycopg[binary] does not "
                "provide it, psycopg[pool] does"
            )
