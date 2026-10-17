"""The release's constraints file installs with the ``docling`` extra too.

``constraints/0.1.0.txt`` pinned typer 0.27.3, and docling-core requires
typer<0.27, so ``pip install "lasagna-rag[eval,docling]" -c constraints/0.1.0.txt``
(and ``[all]``) could not resolve. typer 0.26.8 is what an unconstrained
``[eval,docling]`` resolve picks, and Lasagna takes ``typer>=0.15.3,<1``.

Checked online with::

    uv pip compile pyproject.toml --all-extras --python-version 3.12 \\
        -c constraints/0.1.0.txt

Here, offline: the pin is below docling-core's bound, and where the docling
packages are installed (CI's Docling step), every pin the constraints file
holds satisfies what each of them requires.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _pins() -> dict:
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)[1]
    text = (ROOT / "constraints" / f"{version}.txt").read_text()
    return {
        m[1].lower().replace("_", "-"): m[2]
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==(\S+)$", text, re.M)
    }


def test_typer_is_pinned_below_docling_cores_bound():
    # docling-core 2.x: typer>=0.12.5,<0.27.
    assert Version(_pins()["typer"]) < Version("0.27")


def test_the_cli_takes_the_pinned_typer():
    requires = [Requirement(r) for r in metadata.requires("lasagna-rag") or []]
    typer = next(r for r in requires if r.name == "typer" and not r.marker)
    assert typer.specifier.contains(_pins()["typer"])


def test_every_pin_satisfies_the_installed_docling_packages():
    docling = [
        d for d in metadata.distributions() if d.metadata["Name"].lower().startswith("docling")
    ]
    if not docling:
        pytest.skip("docling is not installed (CI's Docling step installs it)")
    pins = _pins()
    unmet = []
    for dist in docling:
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            pinned = pins.get(req.name.lower().replace("_", "-"))
            if pinned and not req.specifier.contains(pinned, prereleases=True):
                unmet.append(f"{dist.metadata['Name']} requires {req}, constraints pin {pinned}")
    assert not unmet, unmet
