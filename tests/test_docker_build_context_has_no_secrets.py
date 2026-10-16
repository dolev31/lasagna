"""The image build never sends a local .env (or other secret) to Docker.

There was no .dockerignore, the build context is the repository root, and the
Dockerfile copies all of src/lasagna/. deploy/README.md tells operators to put
the JWT secret, the Postgres password and API keys in src/lasagna/deploy/.env,
so every image built from a configured checkout carried them in a layer.

The rules are checked with a port of Docker's own matcher
(github.com/moby/patternmatcher, used by BuildKit for .dockerignore): `*` and
`?` stop at `/`, `**` spans directories, a pattern that matches a parent
directory excludes everything under it, and the last matching line wins.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "src" / "lasagna" / "deploy" / "Dockerfile"


def _compile(pattern: str) -> re.Pattern[str]:
    regex, i = "^", 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i + 1 : i + 2] == "*":
                i += 1
                if pattern[i + 1 : i + 2] == "/":
                    i += 1
                regex += ".*" if i + 1 >= len(pattern) else "(.*/)?"
            else:
                regex += "[^/]*"
        elif ch == "?":
            regex += "[^/]"
        elif ch == "[":
            end = pattern.index("]", i)
            regex += pattern[i : end + 1]
            i = end
        else:
            regex += re.escape(ch)
        i += 1
    return re.compile(regex + "$")


def _rules() -> list[tuple[bool, re.Pattern[str]]]:
    rules = []
    for line in (REPO_ROOT / ".dockerignore").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exclusion = line.startswith("!")
        pattern = line[1:].strip() if exclusion else line
        pattern = pattern.lstrip("/").rstrip("/")
        rules.append((exclusion, _compile(pattern)))
    return rules


def _excluded(path: str, rules) -> bool:
    parts = path.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    excluded = False
    for exclusion, regex in rules:
        if any(regex.match(c) for c in candidates):
            excluded = not exclusion
    return excluded


@pytest.fixture(scope="module")
def rules():
    path = REPO_ROOT / ".dockerignore"
    if not path.exists():
        pytest.fail(
            "There is no .dockerignore at the repository root, so `docker build` sends "
            "src/lasagna/deploy/.env (and every other local file) into the image."
        )
    return _rules()


@pytest.mark.parametrize(
    "path",
    [
        "src/lasagna/deploy/.env",
        "src/lasagna/deploy/.env.production",
        ".env",
        "examples/demo/.env.local",
        "src/lasagna/deploy/k8s/tls.key",
        ".venv-ci/bin/python",
        ".venv/lib/site.py",
        ".git/config",
        "src/lasagna/configurator/frontend/node_modules/react/index.js",
        "dist/lasagna_rag-0.1.0-py3-none-any.whl",
        "build/lib/lasagna/__init__.py",
        "src/lasagna/__pycache__/cli.cpython-312.pyc",
        ".pytest_cache/v/cache/lastfailed",
        "e2e/reports/run.json",
        "src/lasagna/rag_uploads/a.pdf",
        "src/lasagna/knowledge.db",
        ".coverage",
    ],
)
def test_secrets_and_local_artifacts_stay_out_of_the_build(rules, path):
    assert _excluded(path, rules), f"{path} would be sent to the Docker build"


def _copied_sources() -> list[str]:
    """The context paths the Dockerfile's COPY lines read (not COPY --from).

    ``${NAME}`` is the ARG's default, as Docker reads it.
    """
    sources, args = [], {}
    for line in DOCKERFILE.read_text().splitlines():
        words = line.split()
        if words[:1] == ["ARG"] and "=" in line:
            name, _, default = line[4:].partition("=")
            args[name.strip()] = default.strip().strip('"')
        if words[:1] == ["COPY"] and not any(w.startswith("--from") for w in words):
            sources += [
                re.sub(r"\$\{(\w+)\}", lambda m: args.get(m[1], m[0]), w)
                for w in words[1:-1]
                if not w.startswith("--")
            ]
    return sources


def test_every_file_the_image_copies_is_still_in_the_build(rules):
    """The ignore rules must not drop anything the Dockerfile needs."""
    sources = _copied_sources()
    assert "src/lasagna/" in sources and "pyproject.toml" in sources, sources
    assert "constraints/0.1.0.txt" in sources, sources
    tracked = subprocess.run(
        ["git", "ls-files", "--", *sources],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if not tracked:
        pytest.skip("not a git checkout")
    dropped = [p for p in tracked if _excluded(p, rules)]
    assert not dropped, f"tracked files the image needs are ignored: {dropped[:10]}"
    assert not _excluded("src/lasagna/deploy/.env.example", rules)
    assert not _excluded("src/lasagna/configurator/frontend/dist/index.html", rules)
