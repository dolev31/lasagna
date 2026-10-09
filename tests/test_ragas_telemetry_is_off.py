"""Lasagna turns off the usage events its judge library sends home.

ragas posts a usage event (its version, the LLM provider and model name, and a
random id kept in the user's data folder) each time a judge is built, unless
RAGAS_DO_NOT_TRACK is "true". An evaluation framework people run on private
corpora should not do that without asking, so importing lasagna sets the
opt-out unless the user has already chosen.

Run in a subprocess: the suite's own conftest sets the variable too.
"""

from __future__ import annotations

import os
import subprocess
import sys

_PROBE = "import os, lasagna; print(os.environ.get('RAGAS_DO_NOT_TRACK'))"


def _value_after_import(**env_overrides: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "RAGAS_DO_NOT_TRACK"}
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


def test_importing_lasagna_turns_ragas_usage_events_off():
    assert _value_after_import() == "true"


def test_a_user_can_turn_ragas_usage_events_back_on():
    assert _value_after_import(RAGAS_DO_NOT_TRACK="false") == "false"
