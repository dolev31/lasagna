"""The MCP docs list the tools the server registers, and `lasagna-mcp --version` works.

The MCP README said "All 31", headed "Search (4 tools)" over a table of two,
and said the two deprecated search aliases "still answer" (they return
"Unknown tool"; the server registers 29), and the SDK's integration pages
repeated 31 tools, "the six" left out of `safe`, the aliases, and that every
tool requires `session_id`. `lasagna-mcp --version` was a usage error, and the
README told pip users to run `python -m e2e.run`, which exists only in a
source checkout. The tables are checked here against the tool snapshot, which
`tests/test_mcp_tool_schema_snapshot.py` checks against the server.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tests._paths import PACKAGE_DIR

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((REPO / "tests" / "snapshots" / "mcp_tools.json").read_text())
MCP_README = PACKAGE_DIR / "mcp" / "README.md"
ALL = set(SNAPSHOT)
SAFE = {name for name, tool in SNAPSHOT.items() if "safe" in tool["listed_in"]}

_ROW = re.compile(r"^\| `(\w+)` \| (.*)\|\s*$")
_HEADING = re.compile(r"^### (\w[\w ]*?)(?: \((\d+) tools\))?$")


def _tables():
    """``{heading: [(tool, description)]}`` for the README's tool tables."""
    text = MCP_README.read_text().split("## Tools", 1)[1].split("\n## ", 1)[0]
    tables, current = {}, None
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = line[4:]
            tables[current] = []
            continue
        row = _ROW.match(line)
        if row and current:
            tables[current].append((row.group(1), row.group(2)))
    return tables


def test_the_tables_name_every_registered_tool_once():
    named = [tool for rows in _tables().values() for tool, _ in rows]
    assert sorted(named) == sorted(ALL)


def test_all_only_marks_exactly_the_tools_safe_leaves_out():
    marked = {tool for rows in _tables().values() for tool, text in rows if "*(all only)*" in text}
    assert marked == ALL - SAFE


def test_every_heading_counts_its_rows():
    for heading, rows in _tables().items():
        count = re.search(r"\((\d+) tools\)", heading)
        assert count, f"{heading!r} states no count"
        assert int(count.group(1)) == len(rows), heading


def _words(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_every_stated_count_is_the_registered_one():
    pages = [
        REPO / "README.md",
        REPO / "AGENTS.md",
        MCP_README,
        PACKAGE_DIR / "sdk" / "docs" / "integrations.md",
        PACKAGE_DIR / "sdk" / "docs" / "ui" / "integration.md",
        PACKAGE_DIR / "skills" / "lasagna-rag" / "SKILL.md",
    ]
    for page in pages:
        words = _words(page)
        # A group's own count, "Search (2 tools)", is checked against its rows above.
        stated = {int(n) for n in re.findall(r"(?<!\()\b(\d+) (?:MCP )?tools\b", words)}
        stated |= {int(n) for n in re.findall(r"\bAll (\d+)\b", words)}
        assert stated <= {len(ALL), len(SAFE)}, (page.name, stated)
        assert "deprecated search aliases" not in words, page.name
        assert "requires a `session_id`" not in words, page.name
        assert "leaves out the six" not in words, page.name


def test_lasagna_mcp_prints_its_version(tmp_path):
    from lasagna import __version__

    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(PACKAGE_DIR.parent))
    code = "import sys; sys.argv = ['lasagna-mcp', '--version']; from lasagna.mcp.server import main; main()"
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"lasagna-mcp {__version__}"


def test_the_readme_says_the_gate_needs_a_checkout():
    words = _words(REPO / "README.md")
    for match in re.finditer(r"python -m e2e\.run", words):
        before = words[max(0, match.start() - 60) : match.start()]
        assert "source checkout" in before, before
