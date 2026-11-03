"""Every count the documentation states is the count the code has.

The README's counts were checked (``test_readme_counts_match_the_registry.py``),
and nothing else was. When ``2wikimultihopqa`` landed (#24), the SDK's benchmark
catalog (``src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md``) still said
"31 public benchmarks" and its master table had no row for it, and the example
notebook still told readers about "the 31 benchmarks". A pull request that adds
a benchmark, a component or a published loop now fails here once for every page
that states the old number, so the count is updated everywhere in the same pull
request.

What a page states as a total is read with narrow patterns ("32 benchmarks",
"55 components", "five published loops", a list of the published loops). A
number that is not a total, such as a category heading, is checked against its
own table, or listed in ``NOT_A_TOTAL`` with the reason. A docstring or comment
that only illustrates should not carry a count at all.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tests._paths import PACKAGE_DIR

REPO_ROOT = PACKAGE_DIR.parent.parent

#: History, not a statement about the package as it is. (The only DATASETS.md is
#: the research notes under knowledge_evaluation_sdk/research/.)
_SKIPPED = ("CHANGELOG.md", "DATASETS.md")

#: (file, text on the line) -> why the number on that line is not a total.
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

# "32 benchmarks", "32 public benchmarks", "32 benchmark entries". A number in
# parentheses is a category heading, checked against its table below.
_BENCHMARK_TOTAL = re.compile(
    r"(?<![(\w])(\d+) (?:[A-Za-z-]+ )?benchmark(?:s| entries| datasets)\b"
)
# Not "N built-in components": docs/catalog/stages.md states one per stage, and
# test_catalog_doc_current.py keeps that generated page in step.
_COMPONENT_TOTAL = re.compile(r"(?<!\w)(\d+) components\b")
_LOOP_COUNT = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r") published loops\b", re.IGNORECASE
)
_LOOP_LIST = re.compile(r"published loops \(([^)]*)\)")
_CATEGORY_HEADING = re.compile(r"^###+ .*\((\d+) (?:[a-z]+ )?benchmarks\)\s*$")
_TABLE_ROW = re.compile(r"^\| `([^`]+)` \|")


def _pages():
    """The tracked pages that describe the package, the notebooks and the catalog generator."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("needs a source checkout: the pages are the tracked files")
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md", "*.ipynb", "scripts/gen_catalog_doc.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    for name in sorted(filter(None, out.split("\0"))):
        path = REPO_ROOT / name
        if path.name in _SKIPPED or name.startswith((".", "tests/")) or not path.is_file():
            continue
        yield name, path.read_text(encoding="utf-8")


def _lines():
    """(file:line, file, line) for every line that may state a total."""
    for name, text in _pages():
        for n, line in enumerate(text.splitlines(), 1):
            if not any(file == name and part in line for file, part in NOT_A_TOTAL):
                yield f"{name}:{n}", name, line


def _stated(pattern: re.Pattern):
    """(file:line, captured text) for every match of ``pattern`` that is a total."""
    for where, _name, line in _lines():
        for match in pattern.finditer(line):
            yield where, match.group(1)


def _report(expected, wrong) -> str:
    return f"expected {expected}:\n" + "\n".join(f"  {where}: says {what}" for where, what in wrong)


@pytest.fixture(scope="module")
def counts():
    from lasagna.core.registry import registry
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

    builtins = {
        stage: [name for name in names if registry.is_builtin(stage, name)]
        for stage, names in registry.list_all().items()
    }
    # The published loops are the built-in agentic pipelines in agentic/loops/,
    # one file each; the default graph and legacy ReAct are not among them.
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
    assert not wrong, _report(f"{expected} public benchmarks", wrong)


def test_every_component_total_is_the_registry_size(counts):
    expected = counts["components"]
    wrong = [(where, n) for where, n in _stated(_COMPONENT_TOTAL) if int(n) != expected]
    assert not wrong, _report(f"{expected} built-in components", wrong)


def test_every_count_of_the_published_loops_is_right(counts):
    expected = len(counts["loops"])
    wrong = []
    for where, word in _stated(_LOOP_COUNT):
        number = int(word) if word.isdigit() else _NUMBER_WORDS[word.lower()]
        if number != expected:
            wrong.append((where, word))
    assert not wrong, _report(f"{expected} published loops ({', '.join(counts['loops'])})", wrong)


def test_every_list_of_the_published_loops_names_all_of_them(counts):
    """A list written out, by display name or by registered name."""
    loops = counts["loops"]
    wrong = []
    for where, items in _stated(_LOOP_LIST):
        named = [item for item in re.split(r",\s*|\s+and\s+", items) if item.strip()]
        if len(named) != len(loops):
            wrong.append((where, items))
    # A line about the published loops that names some of them by registered
    # name (the settings rows name two on purpose, but never say "published").
    slug = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, loops)) + r")(?![\w-])")
    for where, _name, line in _lines():
        if "published loop" not in line:
            continue
        named = set(slug.findall(line))
        if len(named) >= 2 and named != set(loops):
            wrong.append((where, f"{sorted(named)}, missing {sorted(set(loops) - named)}"))
    assert not wrong, _report(f"all {len(loops)} published loops", wrong)


def test_every_exemption_still_names_a_line_that_exists():
    """A NOT_A_TOTAL entry whose line has gone would hide nothing, silently."""
    for (file, part), _why in NOT_A_TOTAL.items():
        assert part in (REPO_ROOT / file).read_text(encoding="utf-8"), (file, part)


def _sdk_catalog():
    return (REPO_ROOT / "src/lasagna/knowledge_evaluation_sdk/docs/benchmarks.md").read_text()


def test_the_sdk_master_table_is_what_its_snippet_prints(counts):
    """The page's own regeneration snippet, compared row by row and in its order."""
    table = _sdk_catalog().split("## Master Table", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in table.splitlines():
        if _TABLE_ROW.match(line):
            rows.append(tuple(cell.strip() for cell in line.strip().strip("|").split("|")))
    expected = [
        (
            f"`{name}`",
            info.category.value,
            info.eval_mode.value,
            info.description.replace(" \u2014 ", ": "),
        )
        for name, info in sorted(counts["benchmarks"].items())
    ]
    assert rows == expected


def test_each_category_heading_counts_its_own_table():
    lines = _sdk_catalog().splitlines()
    headings = 0
    wrong = []
    for i, line in enumerate(lines):
        heading = _CATEGORY_HEADING.match(line)
        if not heading:
            continue
        headings += 1
        rows = 0
        for following in lines[i + 1 :]:
            if following.startswith("#"):
                break
            rows += bool(_TABLE_ROW.match(following))
        if rows != int(heading.group(1)):
            wrong.append((line, f"{rows} rows"))
    assert headings, "no category heading matched: the pattern no longer fits the page"
    assert not wrong, _report("each heading to count its table", wrong)
