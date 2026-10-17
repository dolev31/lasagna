"""The README is the PyPI project page, and PyPI has no repo to resolve against.

`readme = "README.md"` means the file ships verbatim as the long_description.
Every relative link in it — `[Guide](docs/guide.md)` — becomes
`pypi.org/project/lasagna-rag/docs/guide.md`, which is a 404. Thirty of them
were, including the primary navigation row and the whole documentation table.

That page is where anyone who found the package by name rather than by repo
lands, so it is the first impression for exactly the users who have not already
decided to trust the project.
"""

from __future__ import annotations

import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; the eval extra installs tomli there
    import tomli as tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def repo_url() -> str:
    urls = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["urls"]
    return urls["Repository"].rstrip("/")


def _link_targets(text: str) -> list[str]:
    """Both plain links and badge links, whose nested brackets defeat the naive regex."""
    targets = [m.group(1) for m in re.finditer(r"\[!\[[^\]]*\]\([^)]*\)\]\(([^)]+)\)", text)]
    targets += [m.group(2) for m in re.finditer(r"(?<!\!)\[([^\]]*)\]\(([^)]+)\)", text)]
    return targets


class TestNoRelativeLinksSurvive:
    def test_every_link_is_absolute_or_an_anchor(self, readme):
        relative = [
            t
            for t in _link_targets(readme)
            if not t.startswith(("http://", "https://", "#", "mailto:"))
        ]
        assert not relative, (
            f"{len(relative)} relative link(s) would 404 on PyPI: {relative}. "
            f"Use absolute {tomllib.loads((REPO / 'pyproject.toml').read_text())['project']['urls']['Repository']}/blob/main/... URLs."
        )

    def test_the_readme_is_the_long_description(self, readme):
        """If this stops being true, this whole file is checking nothing."""
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        assert project["readme"] == "README.md"


class TestSelfLinksPointAtRealFiles:
    """Absolute links can rot in a way relative ones cannot — nothing resolves them."""

    def test_every_repo_link_targets_an_existing_path(self, readme, repo_url):
        broken = []
        for target in _link_targets(readme):
            if not target.startswith(repo_url):
                continue
            path = target[len(repo_url) :]
            if path.startswith("/actions/"):
                continue  # the CI status badge and its run page, not repo files
            path = re.sub(r"^/(blob|tree)/main/", "", path).split("#")[0]
            if path and not (REPO / path).exists():
                broken.append(path)
        assert not broken, f"README links to repo paths that do not exist: {broken}"

    def test_directories_use_tree_and_files_use_blob(self, readme, repo_url):
        """GitHub 404s on /blob/ for a directory."""
        wrong = []
        for target in _link_targets(readme):
            if not target.startswith(repo_url):
                continue
            match = re.match(rf"{re.escape(repo_url)}/(blob|tree)/main/([^#]*)", target)
            if not match:
                continue
            kind, path = match.group(1), match.group(2).rstrip("/")
            if not path:
                continue
            is_dir = (REPO / path).is_dir()
            if is_dir and kind != "tree":
                wrong.append(f"{path} is a directory but uses /blob/")
            if not is_dir and kind != "blob":
                wrong.append(f"{path} is a file but uses /tree/")
        assert not wrong, wrong


class TestTheSummaryMatchesTheCode:
    """The PyPI Summary is the one line under the package name in search results."""

    def test_stage_count_in_the_summary_is_real(self):
        from lasagna.core.registry import COMPONENT_TYPES

        summary = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["description"]
        words = {"seven": 7, "eight": 8, "nine": 9, "ten": 10}
        for word, n in words.items():
            if word in summary.lower():
                assert n == len(COMPONENT_TYPES), (
                    f"PyPI Summary says '{word}' stages; the registry defines "
                    f"{len(COMPONENT_TYPES)}"
                )

    def test_readme_and_summary_do_not_contradict_each_other(self, readme):
        """They sat 40 lines apart claiming different numbers."""
        from lasagna.core.registry import COMPONENT_TYPES

        words = {"seven": 7, "eight": 8, "nine": 9, "ten": 10}
        for word, n in words.items():
            if re.search(rf"\b{word}\b[^.]*pipeline stages", readme, re.I):
                assert n == len(COMPONENT_TYPES), (
                    f"README claims '{word}' pipeline stages; the registry defines "
                    f"{len(COMPONENT_TYPES)}"
                )


class TestPackagingIsComplete:
    def test_py_typed_exists_because_it_is_declared(self):
        """PEP 561: without the marker, every annotation we ship is discarded.

        It was listed in package-data and absent from the tree, so the wheel
        claimed to be typed and was not — type checkers and IDE integrations
        silently treated the whole package as untyped.
        """
        assert (REPO / "src" / "lasagna" / "py.typed").is_file()

    def test_declared_package_data_files_exist(self):
        """A literal (non-glob) pattern matching nothing is a packaging lie."""
        data = tomllib.loads((REPO / "pyproject.toml").read_text())
        patterns = data["tool"]["setuptools"]["package-data"]["lasagna"]
        pkg = REPO / "src" / "lasagna"
        missing = [p for p in patterns if not any(c in p for c in "*?[") and not (pkg / p).exists()]
        assert not missing, f"package-data names files that do not exist: {missing}"


class TestMeasuredClaimsMatchTheReports:
    """The cold-start numbers are offered as evidence, so they must stay true.

    The README cites `e2e/reports/` and states a download size and a wall-clock
    range. Both drifted: it claimed "three runs" and "11-26 minutes" while the
    committed reports held eight runs spanning 7-35 minutes. A number presented
    as measured, which quietly stopped being measured, is worse than no number.

    The first version of this class also asserted the *run count*, which was a
    mistake: every gate run adds a report, so the assertion failed on each CI
    run for a reason that was never a defect. A check that cries wolf on every
    green build gets deleted, and takes the useful check with it. The range is
    the claim worth defending -- it stays true as runs accumulate, and fails
    exactly when a measurement lands outside what the README promises.
    """

    @pytest.fixture(scope="class")
    def measured(self):
        """Each metric's values, oldest run first (report folders are UTC timestamps)."""
        import json

        runs = {}
        for path in sorted((REPO / "e2e" / "reports").glob("*/results.json")):
            for phase in json.loads(path.read_text()).get("phases", []):
                for metric in phase.get("metrics", []):
                    runs.setdefault(metric["name"], []).append(float(metric["value"]))
        if not runs.get("time_to_first_search"):
            pytest.skip(
                "no cold-start measurements here: `python -m e2e.run` writes them to "
                "e2e/reports/, which is not committed"
            )
        return runs

    def test_the_stated_range_contains_every_measurement(self, readme, measured):
        match = re.search(r"(\d+)[–-](\d+) minutes", readme)
        assert match, "README no longer states a cold-start range"
        low, high = int(match.group(1)), int(match.group(2))

        minutes = [v / 60 for v in measured["time_to_first_search"]]
        outside = [round(m, 1) for m in minutes if not (low <= m <= high)]
        assert not outside, (
            f"README claims {low}-{high} minutes, but measured runs fall outside "
            f"it: {sorted(outside)} (all: {sorted(round(m, 1) for m in minutes)})"
        )

    def test_the_download_size_is_right(self, readme, measured):
        """Against the newest run, not the largest.

        What is downloaded is a property of the code, so an older report
        describes code that no longer exists: checked against the largest,
        4.6 GB (BGE-M3 twice and no reranker, measured before both were
        fixed) stayed the figure to meet. The time range above keeps every
        run, since time is a property of the connection and each run is one
        more sample of it. Units are decimal, as the reports (MB = 10^6
        bytes) and `lasagna init` state them; the old factor of 1024 was off
        by 2.4%.
        """
        sizes = measured.get("model_download")
        if not sizes:
            pytest.skip("no download-size metric recorded")
        match = re.search(r"([\d.]+) GB downloaded", readme)
        assert match, "README no longer states a download size"
        claimed_mb = float(match.group(1)) * 1000
        newest = sizes[-1]
        assert abs(claimed_mb - newest) / newest < 0.05, (
            f"README claims {match.group(1)} GB; the newest report measured {newest / 1000:.2f} GB"
        )
