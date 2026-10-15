"""The REST contract under ``/api/v1`` changes only on purpose.

Routes, parameters, response models and error schemas are what every client
is written against, and nothing showed when an edit changed them: a renamed
field or a dropped route shipped as an incidental diff. GOVERNANCE.md puts
``/api/v1`` under the deprecation policy; this snapshot makes a change to it
visible in review.

To accept a deliberate change::

    LASAGNA_UPDATE_SNAPSHOTS=1 pytest tests/test_openapi_snapshot.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SNAPSHOT = Path(__file__).parent / "snapshots" / "openapi.json"


def _spec(monkeypatch) -> dict:
    # The optional mounts add no OpenAPI paths, but keep the app as shipped.
    for name in ("RAG_EMBED_MCP", "RAG_CONFIGURATOR", "RAG_REQUIRE_AUTH"):
        monkeypatch.delenv(name, raising=False)
    from lasagna.api.rest.app import create_app

    spec = create_app().openapi()
    spec["info"].pop("version", None)  # the package version, not the contract
    return spec


def test_the_api_matches_the_snapshot(monkeypatch):
    current = _spec(monkeypatch)
    if os.environ.get("LASAGNA_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    recorded = json.loads(SNAPSHOT.read_text())

    added = sorted(set(current["paths"]) - set(recorded["paths"]))
    removed = sorted(set(recorded["paths"]) - set(current["paths"]))
    assert not removed, f"routes removed: {removed}. Removing a route needs a deprecation."
    assert not added, f"routes added: {added}. Rerun with LASAGNA_UPDATE_SNAPSHOTS=1."
    changed = sorted(p for p in current["paths"] if current["paths"][p] != recorded["paths"][p])
    assert not changed, f"routes changed: {changed}. Rerun with LASAGNA_UPDATE_SNAPSHOTS=1."
    assert current == recorded, (
        "the OpenAPI document changed outside the routes (schemas or info). If that is "
        "deliberate, rerun with LASAGNA_UPDATE_SNAPSHOTS=1 and note it in CHANGELOG.md."
    )
