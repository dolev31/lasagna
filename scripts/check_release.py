"""Refuse a release whose tag, version, changelog and citation disagree.

    python scripts/check_release.py v0.1.0

Run by `.github/workflows/release.yml` before anything is built, and by hand
before `gh release create`. A PyPI filename can never be
reused, so a release cut from a tree that still says another version, or whose
changelog date is the `YYYY-MM-DD` placeholder, has to be caught before the
upload rather than after.

Checks, against the tree it runs in:

- the tag is `v` + `[project] version` in pyproject.toml;
- CHANGELOG.md has a `## [<version>] - <date>` section with a real date;
- CITATION.cff has the same `version` and `date-released` as that date;
- `constraints/<version>.txt` exists, and README.md installs that version with it;
- the Colab notebook and its script install that same version, because the
  output the notebook documents belongs to one release.

`--allow-placeholder` accepts the placeholder date. The test suite uses it, so
the tree can carry the next release's notes before its date is known.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; tomli is a core dependency there
    import tomli as tomllib

PLACEHOLDER = "YYYY-MM-DD"
#: Files that install the package at the release they document.
NOTEBOOK_FILES = (
    "examples/notebooks/compare_your_reranker.ipynb",
    "examples/notebooks/compare_your_reranker.py",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _changelog_date(text: str, version: str) -> str | None:
    m = re.search(rf"^## \[{re.escape(version)}\] [-—] (\S+)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


def _cff_field(text: str, field: str) -> str | None:
    m = re.search(rf"^{field}:\s*\"?([^\"\n#]+?)\"?\s*(#.*)?$", text, re.MULTILINE)
    return m.group(1) if m else None


def problems(tag: str, root: Path = REPO_ROOT, allow_placeholder: bool = False) -> list[str]:
    """Every reason `tag` cannot be released from `root`; empty when it can."""
    found = []
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    if tag != f"v{version}":
        found.append(f"tag {tag!r} does not match pyproject.toml version {version!r}")

    date = _changelog_date((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    if date is None:
        found.append(f"CHANGELOG.md has no '## [{version}] - <date>' section")
    elif date == PLACEHOLDER:
        if not allow_placeholder:
            found.append(f"CHANGELOG.md still dates {version} {PLACEHOLDER}: stamp the date")
    else:
        try:
            dt.date.fromisoformat(date)
        except ValueError:
            found.append(f"CHANGELOG.md dates {version} {date!r}, not an ISO date")

    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    if _cff_field(cff, "version") != version:
        found.append(f"CITATION.cff version is {_cff_field(cff, 'version')!r}, not {version!r}")
    released = _cff_field(cff, "date-released")
    if date not in (None, PLACEHOLDER) and released != date:
        found.append(f"CITATION.cff date-released is {released!r}, CHANGELOG.md says {date!r}")

    constraints = f"constraints/{version}.txt"
    if not (root / constraints).is_file():
        found.append(f"{constraints} does not exist")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if f'"lasagna-rag[eval]=={version}"' not in readme or constraints not in readme:
        found.append(f"README.md does not install lasagna-rag[eval]=={version} with {constraints}")
    for name in NOTEBOOK_FILES:
        if f"lasagna-rag[eval]=={version}" not in (root / name).read_text(encoding="utf-8"):
            found.append(f"{name} does not install lasagna-rag[eval]=={version}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="the release tag, e.g. v0.1.0")
    parser.add_argument("--allow-placeholder", action="store_true")
    args = parser.parse_args(argv)
    found = problems(args.tag, allow_placeholder=args.allow_placeholder)
    for p in found:
        print(f"::error::{p}")
    if not found:
        print(f"{args.tag}: version, changelog, citation and constraints agree")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
