"""The 0.1.0 entry is release notes a new user reads, and it states what changed numbers.

It was 1,580 lines, about 970 of them under "Fixed", plus a "Deprecated"
section, in what is the first public release (0.1.0a1 was never on PyPI). It
is now what 0.1.0 is, what changed for users of 0.1.0a1 (every metric
definition step, every change to the comparability key, the old names that
still work), known issues and the plan; the fixes are in the git history.
"""

from __future__ import annotations

import re

import lasagna
from lasagna.knowledge_evaluation_sdk import bundle
from tests._paths import PACKAGE_DIR

REPO = PACKAGE_DIR.parents[1]
CHANGELOG = (REPO / "CHANGELOG.md").read_text()


def _section(version: str) -> str:
    start = CHANGELOG.index(f"## [{version}]")
    rest = CHANGELOG[start + 4 :]
    end = rest.find("\n## [")
    return CHANGELOG[start : start + 4 + (end if end >= 0 else len(rest))]


RELEASE = _section("0.1.0")


def test_the_release_entry_is_short_and_has_the_sections_a_reader_needs():
    assert len(RELEASE.splitlines()) < 300
    for heading in (
        "### Added",
        "### Changed for users of 0.1.0a1",
        "### Known issues",
        "### Planned for 0.2.0",
    ):
        assert heading in RELEASE, heading
    assert "### Deprecated" not in RELEASE and "### Fixed" not in RELEASE


def test_every_metric_definition_step_is_stated():
    current = bundle.METRICS_VERSION
    major, minor = (int(p) for p in current.split("."))
    assert f"`METRICS_VERSION` {current}" in RELEASE
    for step in range(2, minor + 1):
        assert f"- **{major}.{step}:**" in RELEASE, f"{major}.{step} is not described"


def test_every_change_to_the_comparability_key_is_stated():
    fields = set(bundle._KEY_FIELDS_SINCE) | set(bundle._KEY_FIELDS_WHEN_SET)
    fields |= set(bundle.KEY_DEFAULTS)
    for field in fields:
        assert f"`{field}`" in RELEASE, field
    assert "each benchmark of a multi-benchmark run has its own key" in " ".join(RELEASE.split())


def test_every_old_name_that_still_works_is_listed():
    table = RELEASE[RELEASE.index("#### Renamed: the old name works through 0.1.x") :]
    for old in (
        "`lasagna benchmark compare`",
        "`lasagna benchmark leaderboard`",
        "`lasagna results interaction --bootstrap`",
    ):
        assert old in table, old
    assert f"the {len(lasagna._DEPRECATED_IMPORTS)} names dropped from `lasagna.__all__`" in table


def test_the_pre_release_entry_is_kept():
    assert "## [0.1.0a1] - 2026-08-18" in CHANGELOG
    assert "### Changed: metric definitions (`METRICS_VERSION` 1.0 to 1.1)" in CHANGELOG


def test_its_links_resolve():
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", RELEASE):
        if target.startswith("http"):
            continue
        assert (REPO / target).exists(), target
    slugs = {
        re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
        for h in re.findall(r"^#+ (.+)$", CHANGELOG, re.M)
    }
    for anchor in re.findall(r"\]\(#([^)]+)\)", RELEASE):
        assert anchor in slugs, anchor
