"""Things that must work before a stranger can use this at all.

Each of these failed for every fresh install while passing every test, because
they live in the gap between the source tree and what a user receives:

* `lasagna init` — the first command in the README's shell quickstart — imported
  an undeclared dependency and exited 1.
* `from lasagna.sdk import run_benchmark` bound `None` without the `eval` extra,
  so the failure surfaced later as `'NoneType' object is not callable`, naming
  neither the missing package nor the extra that provides it.
* The wheel shipped no example configs and no evaluation docs, while the README
  told people to run a config file that only exists in a git checkout.
"""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; the eval extra installs tomli there
    import tomli as tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


class TestFirstCommandDependencies:
    def test_every_module_the_cli_imports_is_declared(self):
        """`lasagna init` crashed on an undeclared `questionary`.

        Checked against the dependency list rather than by importing, so it
        fails in CI on a machine that happens to have the package.
        """
        import ast

        cli = (REPO / "src" / "lasagna" / "cli.py").read_text()
        tree = ast.parse(cli)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

        import sys

        # Authoritative rather than hand-maintained: a hand-written list is how
        # this check starts producing false positives and gets deleted.
        third_party = imported - set(sys.stdlib_module_names) - {"lasagna", "__future__"}

        declared_blob = " ".join(_pyproject()["project"]["dependencies"]).lower()
        missing = sorted(
            pkg for pkg in third_party if pkg.replace("_", "-").lower() not in declared_blob
        )
        assert not missing, (
            f"cli.py imports {missing}, which pyproject does not declare. Every "
            f"one of these crashes a command on a fresh install."
        )

    def test_questionary_specifically_is_declared(self):
        """Named because it is the one that broke, and the regression is cheap
        to reintroduce by moving the import."""
        blob = " ".join(_pyproject()["project"]["dependencies"])
        assert "questionary" in blob


class TestMissingOptionalDependenciesExplainThemselves:
    def test_sentinel_names_the_package_and_the_extra(self):
        from lasagna.sdk import _MissingDependency

        sentinel = _MissingDependency(
            "run_benchmark",
            "lasagna.knowledge_evaluation_sdk.experiment",
            ImportError("No module named 'ragas'"),
        )
        with pytest.raises(ImportError) as exc:
            sentinel()
        message = str(exc.value)
        assert "ragas" in message, "the error must name the missing package"
        assert "lasagna-rag[eval]" in message, "and the extra that provides it"

    def test_attribute_access_fails_the_same_way(self):
        from lasagna.sdk import _MissingDependency

        sentinel = _MissingDependency("X", "lasagna.x", ImportError("No module named 'ragas'"))
        with pytest.raises(ImportError):
            sentinel.some_attribute

    def test_sentinel_stays_falsy_like_the_none_it_replaced(self):
        """Callers written against the old behaviour keep working."""
        from lasagna.sdk import _MissingDependency

        assert not _MissingDependency("X", "lasagna.x", ImportError("boom"))

    def test_importing_a_name_never_raises(self):
        """Documentation and IDEs re-export these; the import must succeed even
        when the dependency is absent, or reading a signature breaks."""
        import lasagna.sdk as sdk

        for name in ("run_benchmark", "Experiment", "BenchmarkResults"):
            assert hasattr(sdk, name), f"{name} must be importable regardless of extras"


class TestTheWheelCarriesWhatTheDocsPromise:
    def test_example_configs_are_declared_as_package_data(self):
        patterns = _pyproject()["tool"]["setuptools"]["package-data"]["lasagna"]
        blob = " ".join(patterns)
        assert "research" in blob, (
            "the example experiment configs the README tells users to run must ship in the wheel"
        )

    def test_research_is_a_package_so_its_data_can_ship(self):
        """package-data only matches inside a discovered package; without an
        __init__.py the directory is skipped entirely and the patterns above
        silently match nothing."""
        assert (
            REPO / "src" / "lasagna" / "knowledge_evaluation_sdk" / "research" / "__init__.py"
        ).is_file()

    def test_the_quickstart_config_exists_and_is_runnable_as_written(self):
        import yaml

        path = REPO / "src/lasagna/knowledge_evaluation_sdk/research/experiments/quickstart.yaml"
        cfg = yaml.safe_load(path.read_text())
        names = [b["name"] if isinstance(b, dict) else b for b in cfg["benchmarks"]]
        assert names == ["smoke"], (
            "the quickstart must use the hermetic benchmark; it previously paired "
            "beir/nfcorpus with a mode that cannot score retrieval, so the config "
            "named 'quickstart' failed every time it was run"
        )
        assert cfg.get("mode") in (None, "auto", "live")
