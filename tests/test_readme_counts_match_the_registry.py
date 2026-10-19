"""The counts the README states are the counts the package has.

The stage table said 9 rerankers and 5 query enhancers, and the comparison
table said 21 components, while the registry listed 6, 4 and 17. A number a
reader can check in one command, and finds wrong, makes them doubt the ones
they cannot check.
"""

from __future__ import annotations

import re

import pytest

from lasagna.core.registry import registry
from tests._paths import PACKAGE_DIR

README = PACKAGE_DIR.parent.parent / "README.md"


def _builtins() -> dict:
    """Listed built-ins per stage. Plugins and components other tests register
    in this process are not what the README counts."""
    return {
        stage: [name for name in names if registry.is_builtin(stage, name)]
        for stage, names in registry.list_all().items()
    }


@pytest.fixture(scope="module")
def readme() -> str:
    if not README.is_file():
        pytest.skip("needs a source checkout")
    return README.read_text()


def test_the_stage_table_matches_the_registry(readme):
    stated = {
        m.group(1): int(m.group(2)) for m in re.finditer(r"^\| `(\w+)` \| (\d+) \|", readme, re.M)
    }
    actual = {stage: len(names) for stage, names in _builtins().items()}
    assert stated == actual


def test_the_component_total_matches_the_registry(readme):
    # "30 of 55 components cite the paper they implement": the total is the
    # second number; tests/test_the_notebook_and_the_readme_say_what_a_run_shows.py
    # checks the first.
    match = re.search(r"\| Published methods \| \d+ of (\d+) components ", readme)
    assert match, "README no longer states the component total"
    assert int(match.group(1)) == sum(len(names) for names in _builtins().values())


def test_the_benchmark_count_matches_the_public_catalogue(readme):
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

    public = [
        name
        for name in benchmark_registry.list_benchmarks()
        if name != "smoke"
        and benchmark_registry.is_builtin(name)
        and not name.startswith("private_eval/")
    ]
    stated = {int(n) for n in re.findall(r"\b(\d+) benchmarks\b", readme)}
    assert stated == {len(public)}, f"README states {stated}; the catalogue has {len(public)}"


def test_the_changelog_does_not_call_the_entries_datasets():
    """The changelog said "31 benchmark datasets" while the README said 31
    entries over about 22 datasets."""
    changelog = (README.parent / "CHANGELOG.md").read_text()
    assert not re.search(r"\b\d+ benchmark datasets\b", changelog)
