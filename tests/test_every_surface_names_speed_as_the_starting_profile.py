"""Every surface names the same starting profile: speed.

The README, `lasagna init` and doctor said to start with `speed` (a 92 MB
model, no LLM), while `lasagna configure` listed "standard ... (recommended
start)", the settings file said "STANDARD (Recommended Starter)",
configuration.md called standard "The recommended starting point", the SDK's
docstrings called standard the "Recommended starter" and balanced the
"Recommended default", and the configurator's sidebar badged standard as
recommended. Now speed is where to start, and standard and the default are for
quality, with what they cost.
"""

from __future__ import annotations

import inspect
import re
import subprocess

from tests._paths import PACKAGE_DIR

REPO = PACKAGE_DIR.parents[1]

#: A built-in profile's name, also inside an identifier (`config_for_standard`).
_PROFILE = re.compile(r"(?<![A-Za-z])(speed|standard|balanced|max[_ ]quality)(?![A-Za-z])", re.I)
_TEXT = {".py", ".md", ".toml", ".tsx", ".ts", ".html", ".txt", ".yaml", ".yml", ".ipynb"}


def _shipped_text():
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    for name in listed:
        path = REPO / name
        if path.suffix in _TEXT and not name.startswith(("tests/", "e2e/")):
            yield name, path.read_text(encoding="utf-8", errors="ignore")


def test_recommended_sits_next_to_no_profile_but_speed():
    found = []
    for name, text in _shipped_text():
        for match in re.finditer(r"recommended", text, re.I):
            low = max(0, match.start() - 120)
            near = [
                (abs(low + m.start() - match.start()), m.group(1).lower())
                for m in _PROFILE.finditer(text[low : match.end() + 80])
            ]
            if near and min(near)[1] != "speed":
                line = text.count("\n", 0, match.start()) + 1
                found.append(f"{name}:{line}: {min(near)[1]}")
    assert not found, "recommended next to a profile that is not where to start:\n" + "\n".join(
        found
    )


def test_only_the_speed_preset_says_recommended():
    from lasagna.sdk import config as presets

    for name, fn in inspect.getmembers(presets, inspect.isfunction):
        if name.startswith("config_for_") and name != "config_for_speed":
            assert "ecommend" not in (fn.__doc__ or ""), name
    assert "config_for_balanced() # Recommended" not in inspect.getsource(presets)


def test_the_profile_list_says_where_to_start_and_what_quality_costs():
    from lasagna.core.session_config._profiles import BUILTIN_PROFILE_DESCRIPTIONS as described
    from lasagna.core.setup_text import profile_download_size

    assert "start here" in described["speed"]
    assert f"{profile_download_size('speed')} download, no LLM" in described["speed"]
    assert "for quality" in described["standard"]
    assert f"{profile_download_size('standard')} download" in described["standard"]
    assert "LLM" in described["standard"]
    assert sum("start here" in text for text in described.values()) == 1


def test_the_readme_init_and_settings_file_agree():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Start with the `speed` profile" in readme
    cli = (PACKAGE_DIR / "cli.py").read_text(encoding="utf-8")
    assert '("Speed: vector search only, no LLM, recommended to start", "speed")' in cli
    settings = (PACKAGE_DIR / "settings.lasagna.toml").read_text(encoding="utf-8")
    assert "# 1. SPEED (start here" in settings
    assert "# 2. STANDARD (for quality" in settings
    sidebar = (PACKAGE_DIR / "configurator/frontend/src/components/layout/Sidebar.tsx").read_text(
        encoding="utf-8"
    )
    assert 'const STARTING_PROFILE = "speed";' in sidebar
