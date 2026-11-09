"""Every REST endpoint count the docs state is the API's, and every route is documented.

The counts ("41 HTTP endpoints", "32 under `/api/v1/`") were written by hand in
three pages, and the endpoint reference lists each route in a table. Nothing
checked either against the API: a pull request adding a route could leave every
count and the reference behind. The truth is the reviewed OpenAPI snapshot
(``tests/snapshots/openapi.json``), which ``test_openapi_snapshot.py`` keeps
equal to the app.
"""

from __future__ import annotations

import json
import re

from tests._paths import PACKAGE_DIR

REPO_ROOT = PACKAGE_DIR.parent.parent
SNAPSHOT = REPO_ROOT / "tests" / "snapshots" / "openapi.json"
ENDPOINTS = PACKAGE_DIR / "api" / "docs" / "endpoints.md"
_METHODS = ("get", "post", "put", "patch", "delete")

#: What a page states, by the kind of total, read from whitespace-normalized text.
_STATED = {
    "total": [
        r"(\d+) HTTP endpoints",
        r"REST API with (\d+) endpoints",
        r"(\d+) endpoints across \d+ route modules",
        r"HTTP server with (\d+) endpoints",
    ],
    "versioned": [
        r"(\d+) versioned endpoints",
        r"Versioned Routes \(`/api/v1/`\): (\d+) endpoints",
        r"(\d+) under `/api/v1/`",
    ],
    "unversioned": [
        r"(\d+) non-versioned endpoints",
        r"Non-Versioned Routes: (\d+) endpoints",
        r"and (\d+) that are not versioned",
    ],
}


def _operations():
    paths = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["paths"]
    return [(m.upper(), p) for p, item in paths.items() for m in item if m in _METHODS]


def _actual():
    ops = _operations()
    versioned = sum(p.startswith("/api/v1/") for _, p in ops)
    return {"total": len(ops), "versioned": versioned, "unversioned": len(ops) - versioned}


def _pages():
    roots = [REPO_ROOT / "docs", PACKAGE_DIR]
    pages = [REPO_ROOT / "README.md"]
    for root in roots:
        pages += sorted(root.rglob("*.md"))
    return [p for p in pages if p.name != "CHANGELOG.md"]


def test_every_stated_endpoint_count_is_the_apis():
    actual = _actual()
    found = []
    for page in _pages():
        text = " ".join(page.read_text(encoding="utf-8").split())
        for kind, patterns in _STATED.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    found.append((page.relative_to(REPO_ROOT), kind, int(match.group(1))))
    wrong = [
        f"{page}: says {n} {kind}, the API has {actual[kind]}"
        for page, kind, n in found
        if n != actual[kind]
    ]
    assert not wrong, "\n".join(wrong)
    # The patterns still read the pages that state counts: a reworded page
    # would otherwise pass by stating nothing.
    assert len(found) >= 9, found


def test_every_route_has_a_row_in_the_endpoint_reference():
    reference = ENDPOINTS.read_text(encoding="utf-8")
    missing = [
        f"{method} {path}"
        for method, path in _operations()
        if not re.search(rf"\| `{method}` \| `{re.escape(path)}` \|", reference)
    ]
    assert not missing, "endpoints.md has no row for: " + ", ".join(missing)
