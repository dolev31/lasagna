"""Checkpoint manager for resumable benchmark experiments.

Persists per-task results atomically so that interrupted experiments can
be resumed without re-running completed tasks.

Follows the same atomic-write pattern as :mod:`datasets` (temp file + rename).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger


class CheckpointManager:
    """Manages checkpoints for a single experiment run.

    Each checkpoint is a JSON file named by task key (a hash of benchmark
    name + pipeline config + run index).  The checkpoint directory is:

        ``{output_dir}/.checkpoints/``

    Example::

        cp = CheckpointManager("/tmp/my_experiment")
        if not cp.is_completed("beir/nfcorpus|config_a|run_0"):
            result = run_task(...)
            cp.save("beir/nfcorpus|config_a|run_0", result)

        # Resume later — completed tasks are skipped automatically
        completed = cp.completed_keys()
    """

    def __init__(self, output_dir: str) -> None:
        self._dir = Path(output_dir) / ".checkpoints"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, task_key: str) -> Path:
        """Deterministic file path for a task key."""
        h = hashlib.sha256(task_key.encode()).hexdigest()[:16]
        safe = task_key.replace("/", "_").replace("|", "_")[:80]
        return self._dir / f"{safe}_{h}.json"

    def is_completed(self, task_key: str) -> bool:
        """Check whether a task has been checkpointed."""
        return self._key_path(task_key).exists()

    def save(self, task_key: str, result: Dict[str, Any]) -> None:
        """Atomically persist a task result."""
        path = self._key_path(task_key)
        data = {
            "task_key": task_key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }
        # Atomic write: temp file + rename
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._dir,
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(data, tmp, default=str)
            tmp_path = Path(tmp.name)
        tmp_path.rename(path)
        logger.debug(f"Checkpoint saved: {task_key}")

    def load(self, task_key: str) -> Optional[Dict[str, Any]]:
        """Load a checkpointed result, or ``None`` if not found."""
        path = self._key_path(task_key)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get("result")

    def completed_keys(self) -> Set[str]:
        """Return all completed task keys."""
        keys: Set[str] = set()
        for path in self._dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                keys.add(data["task_key"])
            except (json.JSONDecodeError, KeyError):
                continue
        return keys

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all checkpointed results."""
        results: List[Dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                if "result" in data:
                    results.append(data["result"])
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def clear(self) -> int:
        """Delete all checkpoints. Returns count deleted."""
        count = 0
        for path in self._dir.glob("*.json"):
            path.unlink()
            count += 1
        logger.info(f"Cleared {count} checkpoints")
        return count

    @property
    def checkpoint_dir(self) -> Path:
        return self._dir
