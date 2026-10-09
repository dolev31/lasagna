"""Seed every random source a benchmark task can reach.

Nothing in the harness seeded anything except dataset sampling, so two runs of
an identical configuration could differ for reasons no one recorded, and the
difference was reported as run-to-run variance of the component under test.

Deliberately *not* bit-level determinism. `torch.use_deterministic_algorithms`
raises on operations sentence-transformers uses, GPU kernels reorder float
reductions, and an LLM judge behind an API is outside our control entirely. The
claim this module supports is `determinism: "seeded"` — the same inputs, the
same sampling decisions, the same starting state — never "reproducible to the
last bit". Results record the weaker claim, because it is the true one.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

from loguru import logger


def seed_everything(seed: int, *, quiet: bool = False) -> Dict[str, Any]:
    """Seed Python, NumPy and Torch. Returns what was actually seeded.

    The return value is for the provenance record: "we tried to seed torch" and
    "torch was installed and seeded" are different claims, and only the second
    is worth putting in a results file.
    """
    seeded: Dict[str, Any] = {"seed": seed, "python": False, "numpy": False, "torch": False}

    random.seed(seed)
    seeded["python"] = True

    # Affects hash-ordering of sets/dicts of str in *child* processes only;
    # setting it here cannot change the current interpreter, so it is recorded
    # rather than claimed.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
            seeded["cuda"] = True
        seeded["torch"] = True
    except ImportError:
        pass

    if not quiet:
        logger.debug(f"Seeded {sorted(k for k, v in seeded.items() if v is True)} with {seed}")
    return seeded
