"""Suite runner — run multiple YAML experiments concurrently.

Groups experiments by (mode, benchmark) so that live-mode experiments
sharing the same benchmark corpus can all benefit from the shared corpus
cache (ingest once, search many).  Offline-mode experiments and experiments
on different benchmarks run fully in parallel.

Usage::

    # Run all experiments in the default experiments/ directory
    runner = SuiteRunner()
    results = await runner.run_all()

    # Run a custom list of YAML files
    runner = SuiteRunner(experiment_files=["exp_a.yaml", "exp_b.yaml"])
    results = await runner.run_all()

    # CLI (via __main__.py):
    python -m lasagna.knowledge_evaluation_sdk.private_eval run-suite
    python -m lasagna.knowledge_evaluation_sdk.private_eval run-suite --dir experiments/ --concurrency 4
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# The experiment files the wheel ships; the same folder the command line defaults to.
_EXPERIMENTS_DIR = Path(__file__).parent.parent / "research" / "experiments"


class SuiteRunner:
    """Runs multiple YAML experiments with maximum parallelism.

    Experiments that share the same (mode, benchmark) tuple benefit from
    the shared corpus cache in live mode — only ONE corpus ingestion is
    needed regardless of how many experiments run simultaneously.

    Args:
        experiment_files: Explicit list of YAML paths to run.  If None,
            all ``*.yaml`` files in *experiments_dir* are used.
        experiments_dir: Directory to scan for YAML files.  Ignored when
            *experiment_files* is provided.
        max_parallel_experiments: Maximum number of experiments to execute
            concurrently (default: all at once).  The corpus-sharing cache
            is process-global, so there is no benefit in limiting this unless
            you are resource-constrained.
        skip_existing: Skip experiments whose result ``results.json`` already
            exists in the output directory (checkpoint-compatible).
    """

    def __init__(
        self,
        experiment_files: Optional[List[str]] = None,
        experiments_dir: Optional[str] = None,
        max_parallel_experiments: int = 0,
        skip_existing: bool = True,
    ) -> None:
        self._experiments_dir = Path(experiments_dir or _EXPERIMENTS_DIR)
        self._experiment_files = experiment_files
        self._max_parallel = max_parallel_experiments
        self._skip_existing = skip_existing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_all(
        self,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run all experiments and return a summary dict.

        Args:
            progress_callback: Optional ``(exp_name, status, elapsed_s) -> None``
                called when each experiment starts and finishes.

        Returns:
            ``{experiment_name: {"status": "ok"|"failed", "results": BenchmarkResults|None,
               "error": str|None, "elapsed_s": float}}``
        """
        yaml_files = self._discover_files()
        if not yaml_files:
            logger.warning(f"No YAML experiment files found in {self._experiments_dir}")
            return {}

        logger.info(
            f"Suite runner: {len(yaml_files)} experiments discovered, "
            f"max_parallel={self._max_parallel or 'unlimited'}"
        )

        # Optionally skip already-completed experiments
        if self._skip_existing:
            yaml_files = [f for f in yaml_files if not self._is_done(f)]
            logger.info(f"{len(yaml_files)} experiments need to run (others already done)")

        if not yaml_files:
            logger.info("All experiments already complete.")
            return {}

        semaphore = asyncio.Semaphore(self._max_parallel) if self._max_parallel > 0 else None

        async def _run_one(yaml_path: str) -> Tuple[str, Dict[str, Any]]:
            if semaphore:
                async with semaphore:
                    return await self._run_experiment(yaml_path, progress_callback)
            return await self._run_experiment(yaml_path, progress_callback)

        tasks = [asyncio.create_task(_run_one(f)) for f in yaml_files]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        summary: Dict[str, Any] = {}
        for yaml_path, outcome in zip(yaml_files, outcomes):
            exp_name = Path(yaml_path).stem
            if isinstance(outcome, Exception):
                summary[exp_name] = {"status": "failed", "error": str(outcome), "results": None}
            else:
                name, result = outcome
                summary[name] = result

        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _discover_files(self) -> List[str]:
        if self._experiment_files:
            return list(self._experiment_files)
        if not self._experiments_dir.exists():
            return []
        files = sorted(self._experiments_dir.glob("*.yaml"))
        return [str(f) for f in files]

    def _is_done(self, yaml_path: str) -> bool:
        """Return True if this experiment already produced results.json.

        Supports two output-dir strategies:
        - **Explicit** (``output_dir`` set in YAML): check that exact path.
        - **Timestamped** (no ``output_dir``): scan ``~/.lasagna/research/results/``
          for any directory whose name starts with ``{experiment_name}``.
          This covers both ``{name}_YYYYMMDD_HHMMSS`` (old) and ``{name}/``
          (stable) patterns without requiring a fresh Experiment object whose
          ``effective_output_dir()`` would generate a *new* timestamp.

        Previously, ``Experiment.from_file`` was called just to get the
        output path.  For timestamped dirs that always generates a NEW path
        that has never existed, so ``_is_done`` always returned False and
        every ``run-suite`` re-ran all experiments.
        """
        try:
            try:
                import yaml as _yaml

                with open(yaml_path) as f:
                    doc = _yaml.safe_load(f) or {}
            except Exception:
                return False

            exp_name = doc.get("name") or Path(yaml_path).stem
            explicit_out = doc.get("output_dir", "")

            # Explicit output_dir → deterministic path, simple check.
            if explicit_out:
                result = Path(os.path.expanduser(explicit_out)) / "results.json"
                done = result.exists()
                if done:
                    logger.debug(
                        f"Skipping (results exist at {explicit_out}): {Path(yaml_path).name}"
                    )
                return done

            # Timestamped output_dir → scan results root for any matching dir.
            from lasagna.core.paths import results_dir

            results_root = results_dir()
            # Match {name}/ (stable) or {name}_* (timestamped)
            for candidate in results_root.glob(f"{exp_name}*/results.json"):
                logger.debug(f"Skipping (results exist at {candidate}): {Path(yaml_path).name}")
                return True
            return False
        except Exception:
            return False

    async def _run_experiment(
        self,
        yaml_path: str,
        progress_callback: Optional[Any],
    ) -> Tuple[str, Dict[str, Any]]:
        from ..experiment import Experiment

        exp_name = Path(yaml_path).stem
        start = time.monotonic()
        logger.info(f"[suite] Starting: {exp_name}")
        if progress_callback:
            progress_callback(exp_name, "started", 0.0)

        try:
            exp = Experiment.from_file(yaml_path)
            results = await exp.run()
            elapsed = time.monotonic() - start
            logger.info(f"[suite] Done: {exp_name} in {elapsed:.1f}s")
            if progress_callback:
                progress_callback(exp_name, "done", elapsed)
            return exp_name, {
                "status": "ok",
                "results": results,
                "error": None,
                "elapsed_s": elapsed,
                "output_dir": exp.config.effective_output_dir(),
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(f"[suite] Failed: {exp_name} after {elapsed:.1f}s: {exc}")
            if progress_callback:
                progress_callback(exp_name, "failed", elapsed)
            return exp_name, {
                "status": "failed",
                "results": None,
                "error": str(exc),
                "elapsed_s": elapsed,
                "output_dir": None,
            }

    @staticmethod
    def _print_summary(summary: Dict[str, Any]) -> None:
        ok = [n for n, r in summary.items() if r["status"] == "ok"]
        failed = [n for n, r in summary.items() if r["status"] == "failed"]
        total_s = sum(r["elapsed_s"] for r in summary.values())

        lines = [
            "",
            "=" * 60,
            f"  Suite complete: {len(ok)} ok, {len(failed)} failed",
            f"  Total wall-clock: {total_s:.0f}s",
            "=" * 60,
        ]
        for name, r in sorted(summary.items(), key=lambda x: x[1]["elapsed_s"]):
            status = "OK  " if r["status"] == "ok" else "FAIL"
            lines.append(f"  [{status}] {name:<45} {r['elapsed_s']:>7.1f}s")
        if failed:
            lines.append("")
            lines.append("  Failures:")
            for name in failed:
                lines.append(f"    {name}: {summary[name]['error']}")
        lines.append("=" * 60)
        logger.info("\n".join(lines))
