"""Hand-written bundles made valid, for tests of what renders them.

`results leaderboard` and `results view` check a bundle as `results validate`
does and leave out (or refuse) one that fails. A test of the rendering writes a
few fields by hand; :func:`sealed` fills in what validation requires and
recomputes what it recomputes, so the bundle passes as a real one would.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from lasagna.knowledge_evaluation_sdk.bundle import (
    METRICS_VERSION,
    SCHEMA_VERSION,
    _content_hash,
    comparability_key,
    validation_problems,
)


def sealed(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """``bundle`` with the blocks validation requires, its keys and run id recomputed.

    A hand-written ``comparability_key`` is a group label: it goes into
    ``held_fixed.corpus_fingerprint`` (when that is not set), so bundles with
    the same label still share a key and different labels still do not.
    """
    out = copy.deepcopy(bundle)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("metrics_version", METRICS_VERSION)
    experiment = out.setdefault("experiment", {})
    experiment.setdefault(
        "environment", {"lasagna_version": "0.1.0", "python": "3.12.0", "platform": "test"}
    )
    out.setdefault("integrity", {"warnings": [], "blocking": []})
    for cfg in out.get("configs", []):
        held = cfg.setdefault("held_fixed", {})
        held.setdefault("metrics_version", out["metrics_version"])
        held.setdefault("corpus_fingerprint", cfg.get("comparability_key"))
        if cfg.get("benchmark"):
            held.setdefault("benchmark", cfg["benchmark"])
        cfg["comparability_key"] = comparability_key(held)
    out["run_id"] = _content_hash(out)
    assert not validation_problems(out), validation_problems(out)
    return out
