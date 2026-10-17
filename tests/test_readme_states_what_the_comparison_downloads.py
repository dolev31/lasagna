"""The README says exactly what its component comparison downloads.

It said "The component comparison below runs as is, with no model, corpus or
key". With an empty Hugging Face cache the sweep failed ("all-MiniLM-L6-v2 is
not downloaded"): only `python my_ideas.py` downloads nothing, and the sweep
fetches the `speed` profile's embedder and the cross-encoder control. The
README now states that size, computed here from the sweep it shows.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from lasagna.core.session_config._profiles import ProfileMixin
from lasagna.core.setup_text import MODEL_DOWNLOAD_BYTES, human_size

README = Path(__file__).resolve().parents[1] / "README.md"


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _readme_sweep() -> dict:
    block = re.search(r"```yaml\n# sweep\.yaml, next to my_ideas\.py\n(.*?)```", _text(), re.S)
    assert block, "the README no longer shows its comparison sweep"
    return yaml.safe_load(block.group(1))


def _sweep_download() -> str:
    """The models the README sweep loads: its profile's embedder, and the
    cross-encoder arm's model (`smoke` ships inside the package)."""
    sweep = _readme_sweep()
    assert [b["name"] for b in sweep["benchmarks"]] == ["smoke"]
    spec = ProfileMixin._PROFILE_SPECS[sweep["pipeline"]["base_profile"]]
    models = {spec["embedding_model"]}
    if "cross_encoder" in sweep["pipeline"]["rerankers"]:
        models.add(spec["reranker_model"])
    return human_size(sum(MODEL_DOWNLOAD_BYTES[m] for m in models))


def test_the_readme_no_longer_says_the_sweep_needs_no_model():
    words = " ".join(_text().split())
    assert "no model, corpus or key" not in words
    assert "`python my_ideas.py` downloads nothing" in words


def test_the_stated_sweep_download_is_what_the_sweep_loads():
    size = _sweep_download()
    assert size == "183 MB"
    words = " ".join(_text().split())
    assert f"the sweep downloads {size} once" in words
    assert f"downloads {size} of models the first time" in words
