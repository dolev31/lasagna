"""`lasagna benchmark list` counts the benchmarks the README counts.

It printed every registered name, "32 benchmarks available" (33 with the
plugin template installed), while the README said 31: the README leaves out
`smoke`, a wiring check, and any benchmark a plugin adds. The list now says
what it counts, and both numbers come from the registry.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from lasagna.cli import app
from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
from tests._paths import PACKAGE_DIR

README = PACKAGE_DIR.parent.parent / "README.md"


def test_the_count_line_and_the_readme_agree():
    out = CliRunner().invoke(app, ["benchmark", "list"], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    last = out.output.strip().splitlines()[-1]
    counted = [n for n in benchmark_registry.list_benchmarks() if benchmark_registry.is_builtin(n)]
    measured = len([n for n in counted if n != "smoke"])
    assert last.startswith(f"{measured} benchmarks, plus smoke (a wiring check"), last
    readme = " ".join(README.read_text().split())
    assert f"{measured} benchmark entries" in readme
    assert "also shows `smoke`" in readme
    others = [
        n for n in benchmark_registry.list_benchmarks() if not benchmark_registry.is_builtin(n)
    ]
    if others:
        assert re.search(rf"and {len(others)} from plugins", last), last
