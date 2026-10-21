"""docs/catalog/ is written from the component and benchmark registries.

``scripts/gen_catalog_doc.py`` lists every built-in component by stage and every
public benchmark. This fails when a component or benchmark is added, removed or
re-described without regenerating the pages, and when a new component has no
entry saying what it needs (the generator stops on that).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "docs" / "catalog"


def test_the_pages_match_the_registries(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_catalog_doc.py"), "--check"],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_every_built_in_component_is_on_the_page():
    import lasagna.components  # noqa: F401
    from lasagna.core.registry import registry

    page = (CATALOG / "components.md").read_text(encoding="utf-8")
    built_in = [c for c in registry.list_components() if c["builtin"]]
    assert built_in
    for c in built_in:
        assert f"| `{c['name']}` |" in page, (c["stage"], c["name"])


def test_the_benchmark_count_is_the_one_the_readme_gives():
    page = (CATALOG / "benchmarks.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([^`]+)` \| (?:Single-hop|Multi-hop|Fact|Passage)", page, re.M)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert f"{len(rows)} benchmarks" in readme, len(rows)
    assert "smoke" not in rows


def test_no_page_has_an_em_dash():
    for page in CATALOG.glob("*.md"):
        assert "—" not in page.read_text(encoding="utf-8"), page.name
