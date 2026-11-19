"""The image's Qdrant client must stay compatible with the stack's Qdrant server.

The client warns on every connection when it is more than one minor version
away from the server ("Major versions should match and minor version
difference must not exceed 1"). ``backends/qdrant.yml`` pins the server while
``pyproject.toml`` only floored the client (``qdrant-client>=1.7.0``), so the
image silently installed the newest client (1.20) against server v1.13.2 --
seven minor versions of untested territory in the production stack.

This test reads the server tag from ``backends/qdrant.yml`` and the client
bound from the ``vector-backends`` extra in ``pyproject.toml``, and fails if
the two can drift apart again. It fails on ``main`` (the client has no upper
bound there) and passes once the client is pinned to the server's range.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "src" / "lasagna" / "deploy"

# The rule the client itself enforces, quoted from its warning text.
MAX_MINOR_DIFF = 1


def _server_major_minor() -> tuple[int, int]:
    """(major, minor) of the Qdrant server pinned in ``backends/qdrant.yml``."""
    fragment = yaml.safe_load((DEPLOY_DIR / "backends" / "qdrant.yml").read_text())
    images = [
        service["image"]
        for service in fragment["services"].values()
        if isinstance(service.get("image"), str) and "qdrant/qdrant" in service["image"]
    ]
    assert images, "backends/qdrant.yml pins no qdrant/qdrant image"
    match = re.search(r":v?(\d+)\.(\d+)", images[0])
    assert match, f"cannot parse a server version from {images[0]!r}"
    return int(match.group(1)), int(match.group(2))


def _client_bounds() -> tuple[tuple[int, int], tuple[int, int] | None]:
    """(lower, upper_exclusive|None) of qdrant-client in the vector-backends extra."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    requirements = project["project"]["optional-dependencies"]["vector-backends"]
    line = next(req for req in requirements if req.startswith("qdrant-client"))
    lower_match = re.search(r">=\s*(\d+)\.(\d+)", line)
    upper_match = re.search(r"<\s*(\d+)\.(\d+)", line)
    assert lower_match, f"qdrant-client has no lower bound in {line!r}"
    lower = (int(lower_match.group(1)), int(lower_match.group(2)))
    upper = (int(upper_match.group(1)), int(upper_match.group(2))) if upper_match else None
    return lower, upper


def test_client_bound_stays_within_one_minor_of_the_server():
    server_major, server_minor = _server_major_minor()
    (lower_major, lower_minor), upper = _client_bounds()

    # The bound must be finite: an open upper end lets the image drift to the
    # newest client while the server stays pinned, which is the bug.
    assert upper is not None, (
        "qdrant-client has no upper bound in the vector-backends extra; "
        "the image will keep drifting ahead of the pinned server"
    )
    upper_major, upper_minor = upper

    # Major versions must match on both ends of the allowed range.
    assert lower_major == server_major == upper_major, (
        f"client range starts at {lower_major} and ends before {upper_major}, "
        f"but the server is {server_major}.{server_minor}"
    )

    # Every installable client minor (lower_minor .. upper_minor - 1) must be
    # within one of the server minor. The upper end is exclusive, so it may
    # sit at most one minor above the newest installable client.
    assert lower_minor >= server_minor - MAX_MINOR_DIFF, (
        f"client lower bound minor {lower_minor} is more than one minor "
        f"behind server {server_major}.{server_minor}"
    )
    assert upper_minor <= server_minor + MAX_MINOR_DIFF + 1, (
        f"client upper bound (exclusive) {upper_major}.{upper_minor} lets the "
        f"image install a client more than one minor ahead of server "
        f"{server_major}.{server_minor}"
    )
