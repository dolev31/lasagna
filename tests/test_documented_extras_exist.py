"""Every extra the docs and the Dockerfile name exists in pyproject.toml.

pip only warns about an unknown extra and installs the core package, so a
misnamed extra fails quietly. The Dockerfile installed ``.[rag]``, which has
never existed, so the image got none of what that line was meant to add; the
SDK getting-started guide told readers to do the same.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from tests._paths import PACKAGE_DIR

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

REPO = PACKAGE_DIR.parent.parent

# `lasagna-rag[a,b]`, and `.[a,b]` as written in a pip install of the checkout.
_NAMED_EXTRAS = re.compile(r"lasagna-rag\[([a-z0-9,_ -]+)\]|[\"' ]\.\[([a-z0-9,_ -]+)\]")


def _documents() -> list[Path]:
    found = [REPO / "README.md", REPO / "CONTRIBUTING.md"]
    found += sorted((REPO / "docs").rglob("*.md"))
    found += sorted((REPO / "examples").rglob("*.md"))
    found += sorted(PACKAGE_DIR.rglob("*.md"))
    found += sorted(PACKAGE_DIR.glob("settings*.toml"))
    found += [PACKAGE_DIR / "deploy" / "Dockerfile"]
    return [p for p in found if p.is_file() and "node_modules" not in p.parts]


@pytest.fixture(scope="module")
def extras() -> set[str]:
    pyproject = REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("needs a source checkout")
    with pyproject.open("rb") as f:
        return set(tomllib.load(f)["project"]["optional-dependencies"])


def test_every_named_extra_exists(extras):
    unknown = []
    for path in _documents():
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for match in _NAMED_EXTRAS.finditer(line):
                for name in (match.group(1) or match.group(2)).split(","):
                    if name.strip() not in extras:
                        unknown.append(f"{path.relative_to(REPO)}:{lineno}: [{name.strip()}]")
    assert not unknown, "extras that pyproject.toml does not define:\n" + "\n".join(unknown)
