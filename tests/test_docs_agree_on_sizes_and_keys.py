"""The docs state one first-run download size per profile, and the same install and key facts.

The default download was 4.5 GB in the README and the guide, 3.1 GB in
docs/configuration.md and `lasagna init`, and 5.4 GB in code comments, and the
release gate measured 4.6 GB. None was what the code downloads: 3.1 GB was the
two models' weights alone, and the gate's 4.6 GB was BGE-M3 twice (transformers
fetched a second copy, no longer) without the reranker (the gate's query was
never reranked). Every figure now comes from `core/setup_text.py`, which
records what each model's first load downloads: 3.2 GB for the default
configuration and the profiles built on BGE-M3, 92 MB for `speed`.
CONTRIBUTING cited a CHANGELOG "Known issues" entry about pip backtracking
that does not exist, while the guide measured pip at about two minutes. The
SDK's getting-started page made an LLM key a prerequisite, which it is only
for generation, LLM components and the agentic loop.
"""

from __future__ import annotations

import re
from pathlib import Path

from lasagna.core.setup_text import (
    FLAG_EMBEDDING_EXTRA_BYTES,
    human_size,
    profile_download_size,
)

REPO = Path(__file__).resolve().parents[1]

#: Every shipped page that says how much the first run downloads.
PAGES = [
    "README.md",
    "AGENTS.md",
    "docs/guide.md",
    "docs/configuration.md",
    "examples/lasagna-plugin-template/README.md",
    "src/lasagna/skills/lasagna-rag/SKILL.md",
    "src/lasagna/skills/lasagna-eval/SKILL.md",
    "src/lasagna/skills/lasagna-support/SKILL.md",
]

#: Figures that were stated and were not what the code downloads.
STALE = ("3.1 GB", "4.5 GB", "4.6 GB", "5.4 GB", "91 MB", "870 MB")


def _text(path: str) -> str:
    return " ".join((REPO / path).read_text().split())


def test_every_page_states_the_measured_default_download():
    size = profile_download_size("standard")
    assert size == "3.2 GB" == profile_download_size(None)
    for page in PAGES:
        text = _text(page)
        assert size in text, f"{page} does not state {size}"
        for stale in STALE:
            assert stale not in text, f"{page} still says {stale}"


def test_pages_that_size_the_speed_profile_state_the_measured_figure():
    size = profile_download_size("speed")
    assert size == "92 MB"
    for page in ("README.md", "AGENTS.md", "docs/guide.md", "docs/configuration.md"):
        assert size in _text(page), f"{page} does not state {size}"
    for page in (
        "src/lasagna/skills/lasagna-rag/SKILL.md",
        "src/lasagna/skills/lasagna-eval/SKILL.md",
    ):
        assert size in _text(page), f"{page} does not state {size}"


def test_the_readme_states_the_computed_size_where_the_gate_checks_it():
    """tests/test_pypi_readme.py reads `N GB downloaded` and compares it with the gate."""
    match = re.search(r"([\d.]+) GB downloaded", _text("README.md"))
    assert match, "README no longer states `N GB downloaded`"
    assert f"{match.group(1)} GB" == profile_download_size(None)


def test_the_flag_embedding_extra_is_the_rest_of_the_repository():
    extra = human_size(FLAG_EMBEDDING_EXTRA_BYTES)
    for page in ("README.md", "docs/guide.md"):
        text = _text(page)
        assert f"{extra} more with the `flag-embedding` extra" in text, page
        assert "whole repository" in text, page


def test_the_benchmark_run_help_states_the_computed_sizes():
    """Help text is built when cli.py is imported, so it is written out, not computed."""
    cli = _text("src/lasagna/cli.py")
    speed, standard = profile_download_size("speed"), profile_download_size("standard")
    assert f"speed needs no API key and downloads {speed}" in cli
    assert f'"standard downloads {standard} and sends' in cli


def test_no_comment_in_the_package_cites_another_size():
    found = [
        f"{p.relative_to(REPO)}:{i}"
        for pattern in ("*.py", "*.toml", "*.yaml")
        for p in (REPO / "src" / "lasagna").rglob(pattern)
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"\b(3\.1|4\.5|5\.4) GB\b|\b(91|870) MB\b", line)
    ]
    assert not found, found


def test_contributing_cites_only_what_the_changelog_and_guide_say():
    contributing = (REPO / "CONTRIBUTING.md").read_text()
    changelog = (REPO / "CHANGELOG.md").read_text()
    known = changelog[changelog.index("### Known issues") :].split("\n### ", 1)[0]
    if '"Known issues"' in contributing:
        assert "backtrack" in known
    guide = " ".join((REPO / "docs" / "guide.md").read_text().split())
    assert "about two minutes" in guide and "about two minutes" in contributing


def test_getting_started_does_not_require_a_key_to_search():
    page = (REPO / "src" / "lasagna" / "sdk" / "docs" / "getting_started.md").read_text()
    assert "**Prerequisites:** Set one LLM API key" not in page
    assert "An LLM API key is optional" in page
