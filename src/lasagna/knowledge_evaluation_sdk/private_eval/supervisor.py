"""Real-time supervisor for long-running RAG grid-search experiments.

Scans the results directory and checkpoint directories to produce a
live status table showing per-experiment progress, quality deltas vs
the baseline, ETA, accumulated cost, and anomaly warnings.

Usage::

    # One-shot status snapshot
    python -m lasagna.knowledge_evaluation_sdk.private_eval supervise

    # Live watch mode (refresh every 30s until Ctrl-C)
    python -m lasagna.knowledge_evaluation_sdk.private_eval supervise --watch

    # Programmatic
    sup = ExperimentSupervisor()
    report = sup.status()
    sup.print_status(report)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExperimentStatus:
    """Status of one experiment in the grid search."""

    name: str
    yaml_path: str
    state: str  # "done" | "in_progress" | "pending" | "failed"
    done_tasks: int = 0
    total_tasks: int = 0  # 0 = unknown (YAML parse failed)
    elapsed_s: float = 0.0
    eta_s: Optional[float] = None
    # Best non-baseline quality metric achieved so far
    best_metric: Optional[float] = None
    best_config: str = ""
    baseline_delta: Optional[float] = None
    errors: int = 0
    total_tokens: float = 0.0
    anomalies: List[str] = field(default_factory=list)
    # Raw metrics for anomaly analysis
    _all_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)


@dataclass
class SupervisorReport:
    """Aggregated supervisor snapshot."""

    timestamp: datetime
    experiments: List[ExperimentStatus]
    total_done_tasks: int
    total_tasks: int
    total_elapsed_s: float
    global_eta_s: Optional[float]
    total_tokens: float
    anomaly_count: int
    baseline_quality: Optional[float]  # ndcg@10 of "standard" config


# ---------------------------------------------------------------------------
# Anomaly constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_QUALITY = 0.30  # below this → quality collapse
_DEFAULT_MAX_LATENCY_S = 60.0  # per-query avg latency → stuck / misrouted
_DEFAULT_MAX_ERROR_RATE = 0.50  # fraction of tasks with errors
#: Total accumulated token budget across a supervised sweep. Priced in tokens
#: rather than dollars: rates change and differ by provider, and the dollar
#: version of this budget was summed from a field nothing ever wrote, so it was
#: always $0.00 and never once fired.
_DEFAULT_MAX_TOKENS = 20_000_000.0


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class ExperimentSupervisor:
    """Monitors a running (or completed) RAG grid-search suite.

    The supervisor is **read-only** — it never modifies results or checkpoints.
    It can be safely run in a separate terminal alongside ``run-suite``.

    Args:
        results_dir: Root directory where experiment result folders live.
        experiments_dir: Directory containing the YAML experiment files.
        ir_metric: Primary retrieval metric used for quality comparisons.
        gen_metric: Primary generation metric (used for offline experiments).
        min_quality_threshold: Anomaly threshold — configs below this signal
            a likely configuration error (embedding mismatch, broken routing, etc.).
        max_latency_s: Per-query avg latency above this (in seconds) signals
            agentic-pipeline misrouting or hung tasks.
        max_error_rate: Fraction of failed tasks above this threshold is anomalous.
        max_cost_usd: Total accumulated LLM cost budget — triggers warning when exceeded.
    """

    def __init__(
        self,
        results_dir: Optional[str] = None,
        experiments_dir: Optional[str] = None,
        ir_metric: str = "ndcg@10",
        gen_metric: str = "faithfulness",
        min_quality_threshold: float = _DEFAULT_MIN_QUALITY,
        max_latency_s: float = _DEFAULT_MAX_LATENCY_S,
        max_error_rate: float = _DEFAULT_MAX_ERROR_RATE,
        max_tokens: float = _DEFAULT_MAX_TOKENS,
    ) -> None:
        if results_dir is None:
            from lasagna.core.paths import results_dir as default_results_dir

            results_dir = str(default_results_dir())
        self._results_dir = Path(results_dir)
        self._experiments_dir = Path(experiments_dir or _EXPERIMENTS_DIR)
        self._ir_metric = ir_metric
        self._gen_metric = gen_metric
        self._min_quality = min_quality_threshold
        self._max_latency_ms = max_latency_s * 1000
        self._max_error_rate = max_error_rate
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> SupervisorReport:
        """Produce a one-shot status snapshot."""
        yaml_specs = self._scan_yaml_specs()
        results_index = self._scan_results()
        checkpoint_index = self._scan_checkpoints()

        # Derive baseline quality from completed baseline experiment
        baseline_quality = self._find_baseline(results_index)

        experiment_statuses: List[ExperimentStatus] = []
        total_done = 0
        total_tasks = 0
        total_tokens = 0.0
        task_durations: List[float] = []  # for ETA estimation

        for spec in yaml_specs:
            es = self._build_status(
                spec, results_index, checkpoint_index, baseline_quality, task_durations
            )
            experiment_statuses.append(es)
            total_done += es.done_tasks
            total_tasks += es.total_tasks
            total_tokens += es.total_tokens

        # Global ETA
        avg_task_s = (sum(task_durations) / len(task_durations)) if task_durations else None
        remaining = total_tasks - total_done
        global_eta = (avg_task_s * remaining) if (avg_task_s and remaining > 0) else None

        total_elapsed = sum(es.elapsed_s for es in experiment_statuses)
        anomaly_count = sum(len(es.anomalies) for es in experiment_statuses)

        # Check global spend anomaly, in tokens. The dollar version of this
        # check summed a field nothing ever wrote, so it never fired.
        if total_tokens > self._max_tokens:
            for es in experiment_statuses:
                if es.state == "in_progress":
                    es.anomalies.append(
                        f"⚠ Global token budget exceeded: {total_tokens:,.0f} > "
                        f"{self._max_tokens:,.0f}"
                    )
                    break

        return SupervisorReport(
            timestamp=datetime.now(),
            experiments=experiment_statuses,
            total_done_tasks=total_done,
            total_tasks=total_tasks,
            total_elapsed_s=total_elapsed,
            global_eta_s=global_eta,
            total_tokens=total_tokens,
            anomaly_count=anomaly_count,
            baseline_quality=baseline_quality,
        )

    def watch(self, interval_s: int = 30, clear_screen: bool = True) -> None:
        """Refresh status table every ``interval_s`` seconds until Ctrl-C."""
        print(f"Watching experiments (refresh every {interval_s}s). Press Ctrl-C to stop.\n")
        try:
            while True:
                report = self.status()
                if clear_screen:
                    print("\033[2J\033[H", end="")  # ANSI clear screen
                self.print_status(report)
                done = report.total_done_tasks
                total = report.total_tasks
                if total > 0 and done == total:
                    print("\n✅ All experiments complete!")
                    break
                time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\nSupervisor stopped.")

    def print_status(self, report: Optional[SupervisorReport] = None) -> None:
        """Print a formatted status table to stdout."""
        if report is None:
            report = self.status()

        ts = report.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 90}")
        print(f"  RAG Grid Search Supervisor   {ts}")
        print(f"{'=' * 90}")

        # Per-experiment table
        col_name = 40
        col_state = 12
        col_prog = 9
        col_q = 8
        col_delta = 8
        col_time = 8
        col_eta = 8
        col_warn = 20

        header = (
            f"{'Experiment':<{col_name}}"
            f"{'State':<{col_state}}"
            f"{'Progress':>{col_prog}}"
            f"{'Best Q':>{col_q}}"
            f"{'Δ base':>{col_delta}}"
            f"{'Elapsed':>{col_time}}"
            f"{'ETA':>{col_eta}}"
            f"  Warnings"
        )
        print(header)
        print("-" * 90)

        for es in report.experiments:
            state_icon = {
                "done": "✅ done",
                "in_progress": "🔄 running",
                "pending": "⏳ pending",
                "failed": "❌ failed",
            }.get(es.state, es.state)

            progress = (
                f"{es.done_tasks}/{es.total_tasks}" if es.total_tasks > 0 else f"{es.done_tasks}/??"
            )
            best_q = f"{es.best_metric:.4f}" if es.best_metric is not None else "-"
            delta = f"{es.baseline_delta:+.3f}" if es.baseline_delta is not None else "-"
            elapsed = _fmt_duration(es.elapsed_s) if es.elapsed_s > 0 else "-"
            eta = _fmt_duration(es.eta_s) if es.eta_s is not None else "-"
            warn_str = "; ".join(es.anomalies) if es.anomalies else ""

            line = (
                f"{es.name[: col_name - 1]:<{col_name}}"
                f"{state_icon:<{col_state}}"
                f"{progress:>{col_prog}}"
                f"{best_q:>{col_q}}"
                f"{delta:>{col_delta}}"
                f"{elapsed:>{col_time}}"
                f"{eta:>{col_eta}}"
                f"  {warn_str[:col_warn]}"
            )
            print(line)

        # Global summary
        print("-" * 90)
        pct = (report.total_done_tasks / report.total_tasks * 100) if report.total_tasks else 0
        eta_str = f"  ETA: ~{_fmt_duration(report.global_eta_s)}" if report.global_eta_s else ""
        print(
            f"  Overall: {report.total_done_tasks}/{report.total_tasks} tasks"
            f" ({pct:.0f}%)"
            f"  Elapsed: {_fmt_duration(report.total_elapsed_s)}"
            f"{eta_str}"
        )
        print(
            f"  Tokens: {report.total_tokens:,.0f} accumulated  (budget: {self._max_tokens:,.0f})"
        )
        if report.baseline_quality is not None:
            print(f"  Baseline (standard): {self._ir_metric} = {report.baseline_quality:.4f}")
        if report.anomaly_count > 0:
            print(f"  ⚠  {report.anomaly_count} anomaly warning(s): see above")
        print("=" * 90)

        # Detailed anomaly block
        for es in report.experiments:
            if es.anomalies:
                print(f"\n  ⚠  [{es.name}]")
                for a in es.anomalies:
                    print(f"      {a}")

    # ------------------------------------------------------------------
    # Internal: YAML scanning
    # ------------------------------------------------------------------

    def _scan_yaml_specs(self) -> List[Dict[str, Any]]:
        """Parse all YAML files to get experiment names and expected task counts."""
        if not self._experiments_dir.exists():
            return []
        specs = []
        try:
            import yaml as _yaml
        except ImportError:
            logger.warning("PyYAML not available: cannot parse experiment specs")
            return []

        for yaml_path in sorted(self._experiments_dir.glob("*.yaml")):
            try:
                with open(yaml_path) as f:
                    doc = _yaml.safe_load(f) or {}
                name = doc.get("name") or yaml_path.stem
                n_runs = doc.get("n_runs", 1)
                benchmarks = doc.get("benchmarks", [])
                n_benchmarks = len(benchmarks) if isinstance(benchmarks, list) else 1

                pipeline = doc.get("pipeline", {})
                overrides = pipeline.get("config_overrides", [])
                n_configs = len(overrides) if overrides else 1

                # Also count sweep dimensions
                for dim in ("rerankers", "embedders", "chunkers", "query_enhancers"):
                    dim_list = pipeline.get(dim, [])
                    if dim_list:
                        n_configs = max(n_configs, len(dim_list))

                total_tasks = n_benchmarks * n_configs * n_runs
                specs.append(
                    {
                        "name": name,
                        "yaml_path": str(yaml_path),
                        "total_tasks": total_tasks,
                        "mode": doc.get("mode", "offline"),
                        "output_dir": doc.get("output_dir", ""),
                    }
                )
            except Exception as exc:
                logger.debug(f"Could not parse {yaml_path}: {exc}")
                specs.append(
                    {
                        "name": yaml_path.stem,
                        "yaml_path": str(yaml_path),
                        "total_tasks": 0,
                        "mode": "?",
                        "output_dir": "",
                    }
                )
        return specs

    # ------------------------------------------------------------------
    # Internal: Results scanning
    # ------------------------------------------------------------------

    def _scan_results(self) -> Dict[str, Dict[str, Any]]:
        """{exp_name: parsed results dict} for completed experiments.

        When multiple result directories exist for the same experiment name
        (e.g. from re-runs), the most recently modified results.json wins.
        """
        index: Dict[str, Dict[str, Any]] = {}
        if not self._results_dir.exists():
            return index
        for candidate in self._results_dir.rglob("results.json"):
            try:
                mtime = candidate.stat().st_mtime
                with open(candidate) as f:
                    data = json.load(f)
                exp_name = data.get("experiment_name") or candidate.parent.name
                # Keep the most recently modified results.json for each experiment
                existing = index.get(exp_name)
                if existing is None or mtime > existing["mtime"]:
                    index[exp_name] = {
                        "path": candidate,
                        "data": data,
                        "mtime": mtime,
                    }
            except Exception:
                pass
        return index

    def _scan_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """{exp_name: {done_tasks, latest_ts, output_dir}} for in-progress experiments."""
        index: Dict[str, Dict[str, Any]] = {}
        if not self._results_dir.exists():
            return index

        for cp_dir in self._results_dir.rglob(".checkpoints"):
            # In-progress: .checkpoints/ exists but no results.json in parent
            output_dir = cp_dir.parent
            if (output_dir / "results.json").exists():
                continue  # Already done

            # Derive experiment name from directory name (strip _YYYYMMDD_HHMMSS suffix)
            dir_name = output_dir.name
            # Pattern: {name}_{YYYYMMDD}_{HHMMSS}
            parts = dir_name.rsplit("_", 2)
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                exp_name = parts[0]
            else:
                exp_name = dir_name

            cp_files = list(cp_dir.glob("*.json"))
            done_tasks = len(cp_files)
            latest_ts = 0.0
            for cp_file in cp_files:
                mtime = cp_file.stat().st_mtime
                if mtime > latest_ts:
                    latest_ts = mtime

            if exp_name not in index or index[exp_name]["done_tasks"] < done_tasks:
                index[exp_name] = {
                    "done_tasks": done_tasks,
                    "latest_mtime": latest_ts,
                    "output_dir": str(output_dir),
                    "start_mtime": output_dir.stat().st_mtime,
                }
        return index

    # ------------------------------------------------------------------
    # Internal: Status computation
    # ------------------------------------------------------------------

    def _build_status(
        self,
        spec: Dict[str, Any],
        results_index: Dict[str, Dict[str, Any]],
        checkpoint_index: Dict[str, Dict[str, Any]],
        baseline_quality: Optional[float],
        task_durations: List[float],  # mutated in-place for ETA estimation
    ) -> ExperimentStatus:
        name = spec["name"]
        total_tasks = spec["total_tasks"]

        # --- DONE ---
        if name in results_index:
            data = results_index[name]["data"]
            task_results = data.get("results", [])
            errors = sum(1 for r in task_results if r.get("error"))
            done_tasks = len(task_results)

            # Collect durations for global ETA
            for r in task_results:
                d = r.get("duration_seconds", 0.0)
                if d > 0:
                    task_durations.append(d)

            # Aggregate metrics per config (average across runs)
            config_metrics: Dict[str, Dict[str, List[float]]] = {}
            total_tokens = 0.0
            for r in task_results:
                if r.get("error"):
                    continue
                cfg = r.get("config_name", "unknown")
                config_metrics.setdefault(cfg, {})
                m = r.get("metrics", {})
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        config_metrics[cfg].setdefault(k, []).append(float(v))
                total_tokens += float(
                    (r.get("system_input_tokens") or 0) + (r.get("system_output_tokens") or 0)
                )

            all_metrics = {
                cfg: {k: sum(vs) / len(vs) for k, vs in kv.items()}
                for cfg, kv in config_metrics.items()
            }

            # Best non-baseline metric
            _BASELINE_NAMES = {
                "standard",
                "standard_baseline",
                "speed",
                "max_quality",
                "balanced",
                "baseline",
            }
            best_q, best_cfg, best_delta = self._best_quality(all_metrics, baseline_quality)

            state = "failed" if (errors == done_tasks and done_tasks > 0) else "done"
            anomalies = self._check_anomalies(all_metrics, errors, done_tasks)

            # Elapsed: difference between first and last checkpoint mtime (approx)
            cp_info = checkpoint_index.get(name)
            elapsed = 0.0
            if cp_info:
                elapsed = time.time() - cp_info["start_mtime"]
            elif task_results:
                elapsed = sum(r.get("duration_seconds", 0) for r in task_results)

            return ExperimentStatus(
                name=name,
                yaml_path=spec["yaml_path"],
                state=state,
                done_tasks=done_tasks,
                total_tasks=max(total_tasks, done_tasks),
                elapsed_s=elapsed,
                best_metric=best_q,
                best_config=best_cfg,
                baseline_delta=best_delta,
                errors=errors,
                total_tokens=total_tokens,
                anomalies=anomalies,
                _all_metrics=all_metrics,
            )

        # --- IN PROGRESS ---
        if name in checkpoint_index:
            cp = checkpoint_index[name]
            done_tasks = cp["done_tasks"]
            elapsed = time.time() - cp["start_mtime"]
            latest = cp["latest_mtime"]
            hung_threshold = self._max_latency_ms * 3 / 1000  # 3× latency threshold
            anomalies = []
            if time.time() - latest > hung_threshold and done_tasks > 0:
                stuck_min = (time.time() - latest) / 60
                anomalies.append(f"⚠ No progress for {stuck_min:.0f}m (may be hung)")

            avg_task_s = sum(task_durations) / len(task_durations) if task_durations else None
            remaining = max(total_tasks - done_tasks, 0) if total_tasks > 0 else None
            eta = (avg_task_s * remaining) if (avg_task_s and remaining is not None) else None

            return ExperimentStatus(
                name=name,
                yaml_path=spec["yaml_path"],
                state="in_progress",
                done_tasks=done_tasks,
                total_tasks=total_tasks,
                elapsed_s=elapsed,
                eta_s=eta,
                anomalies=anomalies,
            )

        # --- PENDING ---
        return ExperimentStatus(
            name=name,
            yaml_path=spec["yaml_path"],
            state="pending",
            done_tasks=0,
            total_tasks=total_tasks,
        )

    def _find_baseline(self, results_index: Dict[str, Dict[str, Any]]) -> Optional[float]:
        """Average the standard/baseline config quality across all completed experiments.

        Only collects the explicitly requested ``ir_metric`` (e.g. ndcg@10) — does NOT
        fall back to gen_metric.  Mixing IR and generation metrics into one baseline
        number (e.g. ndcg@10 ≈ 0.58 averaged with faithfulness ≈ 0.94) produces a
        meaningless reference that makes all deltas look negative.

        To monitor generation experiments, pass ``--ir-metric faithfulness``.
        """
        qs: List[float] = []
        for entry in results_index.values():
            for task in entry["data"].get("results", []):
                if task.get("config_name") in ("standard", "standard_baseline", "baseline"):
                    m = task.get("metrics", {})
                    q = m.get(self._ir_metric)
                    if q is not None and isinstance(q, (int, float)):
                        qs.append(float(q))
        return sum(qs) / len(qs) if qs else None

    def _best_quality(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        baseline_quality: Optional[float],
    ) -> Tuple[Optional[float], str, Optional[float]]:
        """Return (best_quality, best_config_name, delta_vs_baseline) excluding baselines.

        Only uses ``ir_metric`` — no gen_metric fallback, to prevent cross-track
        metric contamination in the delta computation.
        """
        _BASELINE_NAMES = {
            "standard",
            "standard_baseline",
            "speed",
            "max_quality",
            "balanced",
            "baseline",
        }
        best_q: Optional[float] = None
        best_cfg = ""
        for cfg, metrics in all_metrics.items():
            if cfg in _BASELINE_NAMES:
                continue
            q = metrics.get(self._ir_metric)
            if q is not None and (best_q is None or q > best_q):
                best_q = q
                best_cfg = cfg
        delta = (
            (best_q - baseline_quality)
            if (best_q is not None and baseline_quality is not None)
            else None
        )
        return best_q, best_cfg, delta

    def _check_anomalies(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        n_errors: int,
        total_tasks: int,
    ) -> List[str]:
        """Return human-readable anomaly strings for this experiment."""
        anomalies: List[str] = []
        _BASELINE_NAMES = {
            "standard",
            "standard_baseline",
            "speed",
            "max_quality",
            "balanced",
            "baseline",
        }

        # Error rate
        if total_tasks > 0:
            err_rate = n_errors / total_tasks
            if err_rate > self._max_error_rate:
                anomalies.append(
                    f"⚠ High error rate: {err_rate:.0%} ({n_errors}/{total_tasks} tasks failed)"
                )

        # Quality collapse — check all non-baseline configs
        for cfg, metrics in all_metrics.items():
            if cfg in _BASELINE_NAMES:
                continue
            q = metrics.get(self._ir_metric) or metrics.get(self._gen_metric)
            if q is not None and q < self._min_quality:
                anomalies.append(
                    f"⚠ Quality collapse on '{cfg}': {self._ir_metric}={q:.3f} (< {self._min_quality})"
                )

        # All non-baseline configs identical → config injection likely broken
        non_base = {
            cfg: metrics.get(self._ir_metric)
            for cfg, metrics in all_metrics.items()
            if cfg not in _BASELINE_NAMES and metrics.get(self._ir_metric) is not None
        }
        if len(non_base) > 1:
            values = list(non_base.values())
            if max(values) - min(values) < 0.001:
                anomalies.append("⚠ All configs identical: config injection may not be working")

        # Latency anomaly: compute from per-task latency fields if available
        latencies = [
            m.get("avg_latency_ms", 0.0)
            for m in all_metrics.values()
            if m.get("avg_latency_ms", 0.0) > 0
        ]
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            if avg_lat > self._max_latency_ms:
                anomalies.append(
                    f"⚠ High avg latency: {avg_lat / 1000:.0f}s/query (threshold {self._max_latency_ms / 1000:.0f}s)"
                )

        return anomalies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h{m:02d}m"
