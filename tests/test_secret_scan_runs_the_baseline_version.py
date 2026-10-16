"""The secret scan runs the detect-secrets version that wrote the baseline.

The compliance job and scripts/ci-local.sh ran `uvx --from detect-secrets`,
which takes whatever release is newest, while `.secrets.baseline` was written
by 1.5.0 and pre-commit pins v1.5.0. A new release with new plugins or filters
reports findings the baseline never saw, and CI failed, or passed differently,
on an unchanged tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def version():
    return json.loads((REPO_ROOT / ".secrets.baseline").read_text())["version"]


@pytest.mark.parametrize("path", [".github/workflows/tests.yml", "scripts/ci-local.sh"])
def test_every_scan_pins_the_baseline_version(path, version):
    lines = (REPO_ROOT / path).read_text().splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    runs = re.findall(r"--from (detect-secrets\S*)", code)
    assert runs, f"{path} runs no detect-secrets scan"
    # `==` would read as a hard-coded secret to detect-secrets' keyword plugin.
    assert runs == [f"detect-secrets@{version}"] * len(runs), runs


def test_pre_commit_pins_the_baseline_version(version):
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    rev = re.search(r"Yelp/detect-secrets\s+rev:\s*(\S+)", text)
    assert rev and rev[1] == f"v{version}"
