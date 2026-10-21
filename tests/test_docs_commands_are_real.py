"""Every ``lasagna ...`` command line in the docs names a real command and real options.

A command in a guide that no longer exists is the kind of error a reader finds
by pasting it. This reads the fenced code blocks in ``docs/**/*.md``, takes
each line that starts a ``lasagna`` command (after an optional ``$``), follows
it through the Typer app, and checks that the subcommand exists and that every
``--option`` or ``-o`` on the line is one that command (or the root) accepts.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAGES = sorted((REPO / "docs").rglob("*.md"))
FENCE = re.compile(r"^(\s*)(```|~~~)\s*([\w-]*)")
SHELL_LANGS = {"bash", "sh", "shell", "console", "text", "zsh", ""}
ALWAYS = {"--help", "--version", "--verbose", "-v", "--install-completion", "--show-completion"}


def _command_lines(path: Path):
    """Yield (line number, text) for each lasagna command in a shell-like fence."""
    in_fence = False
    lang = ""
    pending = None
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = FENCE.match(raw)
        if m:
            in_fence = not in_fence
            lang = m.group(3).lower() if in_fence else ""
            pending = None
            continue
        if not in_fence or lang not in SHELL_LANGS:
            continue
        line = raw.strip()
        if pending is not None:
            start, text = pending
            text += " " + line
            if text.endswith("\\"):
                pending = (start, text[:-1].rstrip())
                continue
            pending = None
            yield start, text
            continue
        line = line[2:] if line.startswith("$ ") else line
        if not re.match(r"^(uv run |python -m )?lasagna(\s|$)", line):
            continue
        line = re.sub(r"^(uv run |python -m )", "", line)
        if line.endswith("\\"):
            pending = (number, line[:-1].rstrip())
            continue
        yield number, line


def _split(text: str):
    """Tokens of the command, cut at the first shell operator or comment."""
    text = re.split(r"\s#\s|\s\|\s|\s&&\s|\s;\s|\s>\s|\s>>\s|\s<\s", text)[0]
    try:
        return shlex.split(text, comments=True)
    except ValueError:
        return text.split()


def _names(param):
    return set(getattr(param, "opts", [])) | set(getattr(param, "secondary_opts", []))


def _is_option(param):
    return "Argument" not in type(param).__name__


@pytest.fixture(scope="module")
def root():
    import typer.main

    from lasagna import cli

    return typer.main.get_command(cli.app)


def _check(root, tokens):
    """Return an error string, or None when the line is a real command."""
    node = root
    allowed = set(ALWAYS)
    for p in getattr(node, "params", []):
        allowed |= _names(p)
    path = []
    rest = tokens[1:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if hasattr(node, "commands") and not tok.startswith("-"):
            if tok in node.commands:
                node = node.commands[tok]
                path.append(tok)
                for p in getattr(node, "params", []):
                    allowed |= _names(p)
                i += 1
                continue
            if not path:
                return f"no command named {tok!r}"
            return f"{' '.join(path)} has no subcommand {tok!r}"
        break
    if hasattr(node, "commands") and path:
        # a group followed by nothing: legal, it prints help
        pass
    for tok in rest[i:]:
        if tok == "--":
            break
        if tok.startswith("--") or (tok.startswith("-") and len(tok) == 2 and tok[1].isalpha()):
            name = tok.split("=", 1)[0]
            if name not in allowed:
                return f"{' '.join(['lasagna'] + path)} has no option {name}"
    return None


def _cases():
    for page in PAGES:
        for number, text in _command_lines(page):
            yield pytest.param(page, number, text, id=f"{page.relative_to(REPO)}:{number}")


@pytest.mark.parametrize("page,number,text", list(_cases()))
def test_the_command_is_real(root, page, number, text):
    tokens = _split(text)
    if any(t.startswith("[") or t in {"<command>", "COMMAND"} for t in tokens[1:2]):
        pytest.skip("placeholder")
    tokens = [t for t in tokens if not (t.startswith("[") and t.endswith("]"))]
    error = _check(root, tokens)
    assert error is None, f"{page.relative_to(REPO)}:{number}: {error}\n  {text}"


def test_the_pages_have_commands_to_check():
    assert sum(1 for _ in _cases()) >= 30
