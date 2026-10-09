"""The suite runs in a throwaway home directory with telemetry off.

A run used to leave ``nltk_data/`` (64 MB), ragas's user id and a matplotlib
cache in the developer's real home, and to post ragas usage events. These pin
the isolation ``tests/conftest.py`` sets up, so a refactor of it cannot
quietly undo it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def test_home_is_a_throwaway_directory():
    home = Path.home().resolve()
    assert home.is_relative_to(Path(tempfile.gettempdir()).resolve()), home
    assert Path(os.environ["RAG_DB_DIR"]).resolve().is_relative_to(home.parent)


def test_ragas_usage_events_are_off():
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_the_model_cache_is_not_moved_into_the_throwaway_home():
    assert not Path(os.environ["HF_HOME"]).resolve().is_relative_to(Path.home().resolve())
