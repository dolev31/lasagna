"""Checkpoint manager for resumable benchmark experiments.

Persists per-task results atomically so that interrupted experiments can
be resumed without re-running completed tasks.

Follows the same atomic-write pattern as :mod:`datasets` (temp file + rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger


def _fsync_dir(directory: Path) -> None:
    """Flush a directory entry to disk, where the platform allows it (POSIX)."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """Write ``data`` as JSON to ``path``: afterwards it holds the old file or the new one.

    A temporary file in the same directory is written, flushed and fsynced,
    then renamed over ``path``, and the directory is fsynced so the rename
    survives a power cut too. Written in place, a crash or a full disk left a
    truncated file under the final name; renamed before the data reached the
    disk, it could leave an empty one.
    """
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with tmp:
            json.dump(data, tmp, **dump_kwargs)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


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
        # Atomic: an empty or half-written checkpoint under the final name
        # would be read by --resume as a finished task.
        write_json_atomic(path, data, default=str)
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
