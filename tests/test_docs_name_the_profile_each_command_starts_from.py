"""The docs name the profile `benchmark run` and a sweep start from, as the code does.

`docs/configuration.md` and the evaluation skill said `benchmark run` starts
from `standard` (a 3.1 GB download) unless told otherwise, and the plugin
template said "Without `--profile speed`, step 3 runs the `standard` profile";
the command has defaulted to `speed` since the retrieval-only default landed.
A sweep file without `base_profile:` does start from `standard`. Each shipped
page that names either default is checked against the code.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from lasagna import cli
from lasagna.knowledge_evaluation_sdk.experiment import PipelineSweepSpec

REPO = Path(__file__).resolve().parents[1]

#: Every shipped page but the changelog, which quotes what the docs used to say.
PAGES = sorted(
    {
        *REPO.glob("*.md"),
        *REPO.glob("docs/**/*.md"),
        *REPO.glob("examples/**/README.md"),
        *REPO.glob("src/lasagna/**/*.md"),
    }
    - {REPO / "CHANGELOG.md"}
)


def _words(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _run_default() -> str:
    return inspect.signature(cli.benchmark_run).parameters["profile"].default.default


def test_the_defaults_are_what_this_test_assumes():
    assert _run_default() == "speed"
    assert PipelineSweepSpec().base_profile == "standard"


def test_every_page_names_the_profile_benchmark_run_starts_from():
    pattern = re.compile(r"`(?:lasagna )?benchmark run` starts from (?:the )?`(\w+)`")
    stated = {
        page.relative_to(REPO).as_posix(): match.group(1)
        for page in PAGES
        for match in pattern.finditer(_words(page))
    }
    assert stated, "no page names benchmark run's default profile any more"
    wrong = {page: name for page, name in stated.items() if name != _run_default()}
    assert not wrong, wrong


def test_no_page_implies_speed_must_be_asked_for():
    stale = [
        page.relative_to(REPO).as_posix()
        for page in PAGES
        if "Without `--profile speed`" in _words(page)
    ]
    assert not stale, stale


def test_a_page_naming_the_sweep_default_names_standard():
    pattern = re.compile(r"without `base_profile:`[^.]*? from `(\w+)`")
    for page in PAGES:
        for match in pattern.finditer(_words(page)):
            assert match.group(1) == PipelineSweepSpec().base_profile, page
