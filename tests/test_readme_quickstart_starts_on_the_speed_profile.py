"""The README Quickstart runs the profile whose numbers it quotes.

Its code ran the default profile: 3.2 GB to download, 3 to 40 minutes to the
first result on release-gate runs, and seconds per warm search on two CPU
threads, while the fast figures beside it (92 MB, a warm search in tens of
milliseconds) were the speed profile's. The Quickstart now starts on `speed`,
in Python and in the shell, and then presents the default profile as the
quality option with its own costs.
"""

from __future__ import annotations

import re
from pathlib import Path

from lasagna.core.setup_text import profile_download_size

README = Path(__file__).resolve().parents[1] / "README.md"


def _quickstart() -> str:
    text = README.read_text(encoding="utf-8")
    return text[text.index("## Quickstart") : text.index("## Give your agent RAG")]


def _blocks(language: str) -> list:
    return re.findall(rf"```{language}\n(.*?)```", _quickstart(), re.S)


def test_the_python_quickstart_uses_the_speed_profile():
    (code,) = _blocks("python")
    assert "config=config_for_speed()" in code
    assert "from lasagna.sdk import KnowledgeClient, config_for_speed" in code


def test_the_shell_quickstart_configures_speed_before_it_uploads():
    commands = [line.split("#")[0].strip() for line in _blocks("bash")[1].splitlines()]
    commands = [c for c in commands if c]
    assert commands[0] == "lasagna configure --profile speed"
    assert commands.index("lasagna upload paper.pdf") > 0


def test_the_speed_profile_comes_first_and_the_default_follows_with_its_costs():
    words = " ".join(_quickstart().split())
    speed, default = (
        words.index("Start with the `speed` profile"),
        words.index("The default profile is the quality option"),
    )
    assert speed < default
    assert f"a {profile_download_size('speed')} embedding model" in words[:default]
    costs = words[default:]
    assert f"about {profile_download_size(None)} downloaded" in costs
    assert re.search(r"\d+[–-]\d+ minutes on a cold cache", costs)
    assert re.search(r"Warm, a search took [\d.]+ ms", costs)
