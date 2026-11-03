"""Every count the documentation states is the count the code has.

The README's counts were checked (``test_readme_counts_match_the_registry.py``),
and nothing else was. When ``2wikimultihopqa`` landed (#24), the SDK's benchmark
catalog (``src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md``) still said
"31 public benchmarks", its master table had no row for it, and its multi-hop
heading still said 5. A pull request that adds a benchmark, a component or a
published loop now fails here once for every page that states the old number,
so the count is updated everywhere in the same pull request.

What a page states as a total is read with narrow patterns ("32 benchmarks",
"55 components", "five published loops", a list of the published loops). A
number that is not a total, such as a category heading, is checked against its
own table, or listed in ``NOT_A_TOTAL`` with the reason.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tests._paths import PACKAGE_DIR

REPO_ROOT = PACKAGE_DIR.parent.parent

#: History and research notes, not statements about the package as it is.
_SKIPPED = ("CHANGELOG.md", "DATASETS.md")

#: (file, the text around the number) -> why that number is not a total.
NOT_A_TOTAL = {
    ("src/lasagna/core/docs/architecture.md", "ABCs: 8 components"): (
        "the eight stage interfaces, one per stage, not the component count"
    ),
}

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_BENCHMARK_TOTAL = re.compile(r"(?<![(\w])(\d+) (?:public )?benchmark(?:s| entries)\b")
_COMPONENT_TOTAL = re.compile(r"(?<!\w)(\d+) components\b")
_LOOP_COUNT = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r") published loops\b", re.IGNORECASE
)
_LOOP_LIST = re.compile(r"published loops \(([^)]*)\)")
_CATEGORY_HEADING = re.compile(r"^###+ .*\((\d+) (?:[a-z]+ )?benchmarks\)\s*$")
_TABLE_ROW = re.compile(r"^\| `([^`]+)` \|")


def _pages():
    """The tracked pages that describe the package, and the catalog generator."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md", "scripts/gen_catalog_doc.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    for name in sorted(filter(None, out.split("\0"))):
        path = REPO_ROOT / name
        if path.name in _SKIPPED or name.startswith((".", "tests/")):
            continue
        yield name, path.read_text(encoding="utf-8")


def _not_a_total(name: str, line: str) -> bool:
    return any(file == name and text in line for file, text in NOT_A_TOTAL)


def _stated(pattern: re.Pattern):
    """(file:line, number) for every match of ``pattern`` that is a total."""
    for name, text in _pages():
        for n, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                if not _not_a_total(name, line):
                    yield f"{name}:{n}", match.group(1)


@pytest.fixture(scope="module")
def counts():
    from lasagna.core.registry import registry
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

    builtins = {
        stage: [name for name in names if registry.is_builtin(stage, name)]
        for stage, names in registry.list_all().items()
    }
    loops = [
        name
        for name in builtins.get("agentic_pipeline", [])
        if registry.get_class("agentic_pipeline", name).__module__.startswith(
            "lasagna.agentic.loops."
        )
    ]
    benchmarks = {
        name: info
        for name, info in benchmark_registry.list_benchmarks().items()
        if name != "smoke"
        and benchmark_registry.is_builtin(name)
        and not name.startswith("private_eval")
    }
    return {
        "components": sum(len(names) for names in builtins.values()),
        "loops": loops,
        "benchmarks": benchmarks,
    }


def test_every_benchmark_total_is_the_catalogue_size(counts):
    expected = len(counts["benchmarks"])
    wrong = [(where, n) for where, n in _stated(_BENCHMARK_TOTAL) if int(n) != expected]
    assert not wrong, (
        f"the catalogue has {expected} public benchmarks; these say otherwise: {wrong}"
    )


def test_every_component_total_is_the_registry_size(counts):
    expected = counts["components"]
    wrong = [(where, n) for where, n in _stated(_COMPONENT_TOTAL) if int(n) != expected]
    assert not wrong, (
        f"the registry has {expected} built-in components; these say otherwise: {wrong}"
    )


def test_every_count_of_the_published_loops_is_right(counts):
    expected = len(counts["loops"])
    wrong = []
    for where, word in _stated(_LOOP_COUNT):
        number = int(word) if word.isdigit() else _NUMBER_WORDS.get(word.lower())
        if number != expected:
            wrong.append((where, word))
    assert not wrong, f"there are {expected} published loops ({counts['loops']}): {wrong}"


def test_every_list_of_the_published_loops_names_all_of_them(counts):
    expected = len(counts["loops"])
    wrong = []
    for where, items in _stated(_LOOP_LIST):
        named = [item for item in re.split(r",\s*|\s+and\s+", items) if item.strip()]
        if len(named) != expected:
            wrong.append((where, items))
    assert not wrong, f"there are {expected} published loops ({counts['loops']}): {wrong}"


def _sdk_catalog():
    return (REPO_ROOT / "src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md").read_text()


def test_the_sdk_master_table_lists_the_catalogue(counts):
    text = _sdk_catalog()
    table = text.split("## Master Table", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in table.splitlines():
        match = _TABLE_ROW.match(line)
        if match:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows[match.group(1)] = (cells[1], cells[2])
    expected = {
        name: (info.category.value, info.eval_mode.value)
        for name, info in counts["benchmarks"].items()
    }
    assert rows == expected


def test_each_category_heading_counts_its_own_table():
    lines = _sdk_catalog().splitlines()
    wrong = []
    for i, line in enumerate(lines):
        heading = _CATEGORY_HEADING.match(line)
        if not heading:
            continue
        rows = 0
        for following in lines[i + 1 :]:
            if following.startswith("#"):
                break
            rows += bool(_TABLE_ROW.match(following))
        if rows != int(heading.group(1)):
            wrong.append((line, rows))
    assert not wrong, f"headings whose count is not their table's row count: {wrong}"
