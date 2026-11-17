"""The contributor docs ask for a star, and never trade one for anything.

CONTRIBUTING.md opens with "Star the repository and watch its releases" as the
first step. GitHub's Acceptable Use Policies (section 4) forbid engagement
incentivized by rewards and automated starring, so no sentence may tie a star to
a merge, a review or credit, and an AI agent must never click it for its person.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAGES = (
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "docs/project/contributing.md",
    "docs/project/wanted.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
)
_STAR = re.compile(r"\bstar(s|red|ring)?\b", re.IGNORECASE)
_TRADE = re.compile(r"\b(merg\w*|review\w*|credit\w*|join\w*|condition\w*|requir\w*)\b", re.I)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()))


@pytest.mark.parametrize("page", PAGES)
def test_no_sentence_ties_a_star_to_a_merge_review_or_credit(page):
    traded = [
        s
        for s in _sentences((REPO / page).read_text(encoding="utf-8"))
        if _STAR.search(s) and _TRADE.search(s)
    ]
    assert not traded, traded


def test_the_star_is_the_first_step_and_a_person_clicks_it():
    contributing = " ".join((REPO / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
    agents = " ".join((REPO / "AGENTS.md").read_text(encoding="utf-8").split())
    assert "1. **Star the repository and watch its releases**" in contributing
    assert "must not star or watch through `gh` or the API" in contributing
    assert "never star or watch through `gh` or the API" in agents
