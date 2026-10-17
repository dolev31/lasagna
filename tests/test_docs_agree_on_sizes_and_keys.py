"""The docs state one first-run download size, and the same install and key facts.

The default download was 4.5 GB in the README and the guide, 3.1 GB in
docs/configuration.md and `lasagna init`, and 5.4 GB in code comments.
CONTRIBUTING cited a CHANGELOG "Known issues" entry about pip backtracking
that does not exist, while the guide measured pip at about two minutes. The
SDK's getting-started page made an LLM key a prerequisite, which it is only
for generation, LLM components and the agentic loop.
"""

from __future__ import annotations

import re
from pathlib import Path

from lasagna.core.setup_text import (
    BGE_M3_ONNX_BYTES,
    MODEL_WEIGHT_BYTES,
    human_size,
    profile_models,
)

REPO = Path(__file__).resolve().parents[1]

#: Every shipped page that says how much the first run downloads.
PAGES = [
    "README.md",
    "AGENTS.md",
    "docs/guide.md",
    "docs/configuration.md",
    "src/lasagna/skills/lasagna-rag/SKILL.md",
    "src/lasagna/skills/lasagna-eval/SKILL.md",
    "src/lasagna/skills/lasagna-support/SKILL.md",
]


def _default_download() -> str:
    return human_size(sum(MODEL_WEIGHT_BYTES[m] for m in profile_models("standard")))


def test_every_page_states_the_measured_default_download():
    size = _default_download()
    assert size == "3.1 GB"
    for page in PAGES:
        text = " ".join((REPO / page).read_text().split())
        assert size in text, f"{page} does not state {size}"
        for stale in ("4.5 GB", "5.4 GB"):
            assert stale not in text, f"{page} still says {stale}"


def test_the_flag_embedding_extra_is_the_onnx_copy():
    extra = human_size(BGE_M3_ONNX_BYTES)
    for page in ("README.md", "docs/guide.md"):
        text = " ".join((REPO / page).read_text().split())
        assert f"{extra} more with the `flag-embedding` extra" in text, page


def test_no_comment_in_the_package_cites_another_size():
    found = [
        f"{p.relative_to(REPO)}:{i}"
        for p in (REPO / "src" / "lasagna").rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"\b(4\.5|5\.4) GB\b", line)
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
