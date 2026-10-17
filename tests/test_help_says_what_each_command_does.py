"""Every command's help says what the command does, and claims nothing it does not.

`results validate --help` said the check "lets a reviewer reject tampering",
while the README says it is an integrity check against mistakes and naive
edits, "not a signature": anyone who edits a number can recompute the id.
`upload --help` told the story of an old README ("the README looped
`lasagna upload` over a folder") instead of what the command does.
"""

from __future__ import annotations

import pytest
from typer.main import get_command

from lasagna.cli import app


def _commands(command, path=()):
    subcommands = getattr(command, "commands", None)
    if subcommands:
        for name, sub in subcommands.items():
            yield from _commands(sub, (*path, name))
    else:
        yield " ".join(path), command


COMMANDS = dict(_commands(get_command(app)))


def _help(command) -> str:
    texts = [command.help or ""] + [getattr(p, "help", None) or "" for p in command.params]
    return " ".join(" ".join(texts).split())


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_no_help_tells_the_history_of_the_docs_or_claims_tamper_proofing(name):
    text = _help(COMMANDS[name])
    assert "README" not in text, text
    assert "reject tampering" not in text, text


def test_validate_says_what_it_detects():
    text = _help(COMMANDS["results validate"])
    assert "not tampering" in text and "not a signature" in text


def test_upload_says_what_several_files_buy():
    text = _help(COMMANDS["upload"])
    assert "models load once" in text
