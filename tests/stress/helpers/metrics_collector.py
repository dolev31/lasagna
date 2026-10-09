"""Metrics collection, Prometheus scraping, rich dashboard, and reporting.

Provides:
- PrometheusScraper: parses /metrics for cache hit rates, latency histograms
- LiveDashboard: rich console dashboard updated during test runs
- TimingReport: JSON report generation with percentile computation
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------


def percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute p50, p95, p99, mean, min, max from a list of latencies."""
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0, "count": 0}
    return {
        "p50": round(percentile(latencies, 50), 4),
        "p95": round(percentile(latencies, 95), 4),
        "p99": round(percentile(latencies, 99), 4),
        "mean": round(sum(latencies) / len(latencies), 4),
        "min": round(min(latencies), 4),
        "max": round(max(latencies), 4),
        "count": len(latencies),
    }


# ---------------------------------------------------------------------------
# Prometheus Scraper
# ---------------------------------------------------------------------------


class PrometheusScraper:
    """Scrape /metrics for cache and latency Prometheus counters.

    Metric names in Prometheus use underscores (sanitized from dots):
    - rag_search_cache_hits_total    (from rag.search.cache_hits)
    - rag_search_cache_misses_total  (from rag.search.cache_misses)
    - rag_http_request_duration_ms   (histogram)
    - rag_http_rate_limited_total    (counter)
    """

    def __init__(self, base_url: str) -> None:
        self._url = f"{base_url.rstrip('/')}/metrics"

    async def scrape_raw(self) -> str:
        """Fetch raw Prometheus metrics text."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self._url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.text()
                elif resp.status == 404:
                    logger.warning("Prometheus /metrics returned 404 — not enabled?")
                    return ""
                else:
                    logger.warning(f"Prometheus /metrics returned {resp.status}")
                    return ""

    def _parse_counter(self, text: str, name: str) -> float:
        """Extract a counter total value from Prometheus text format."""
        # Match lines like: rag_search_cache_hits_total 42.0
        # or with labels: rag_search_cache_hits_total{} 42.0
        pattern = rf"^{re.escape(name)}_total(?:\{{[^}}]*\}})?\s+(\S+)"
        for line in text.split("\n"):
            m = re.match(pattern, line)
            if m:
                return float(m.group(1))
        # Also try without _total suffix
        pattern2 = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)"
        for line in text.split("\n"):
            m = re.match(pattern2, line)
            if m and not line.startswith("#"):
                return float(m.group(1))
        return 0.0

    async def get_cache_hit_rate(self) -> Optional[float]:
        """Get search cache hit rate from Prometheus counters.

        Parses rag_search_cache_hits_total and rag_search_cache_misses_total.
        Returns hits / (hits + misses), or None if metrics unavailable.
        """
        text = await self.scrape_raw()
        if not text:
            return None

        hits = self._parse_counter(text, "rag_search_cache_hits")
        misses = self._parse_counter(text, "rag_search_cache_misses")
        total = hits + misses
        if total == 0:
            return 0.0
        return round(hits / total, 4)

    async def get_cache_counters(self) -> Dict[str, float]:
        """Get raw cache hit/miss counters."""
        text = await self.scrape_raw()
        return {
            "hits": self._parse_counter(text, "rag_search_cache_hits"),
            "misses": self._parse_counter(text, "rag_search_cache_misses"),
        }

    async def get_rate_limited_count(self) -> float:
        """Get total rate-limited request count."""
        text = await self.scrape_raw()
        return self._parse_counter(text, "rag_http_rate_limited")

    async def is_available(self) -> bool:
        """Check if /metrics endpoint is serving."""
        try:
            text = await self.scrape_raw()
            return len(text) > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Live Dashboard (rich)
# ---------------------------------------------------------------------------


class LiveDashboard:
    """Rich console dashboard for real-time stress test monitoring.

    Updates in-place with upload/search/deletion progress, latency percentiles,
    cache hit rates, and 429 counts.
    """

    def __init__(self, backend: str, num_tenants: int) -> None:
        self._backend = backend
        self._num_tenants = num_tenants
        self._start_time = time.monotonic()
        self._live = None
        self._table = None

        try:
            from rich.live import Live  # noqa: F401
            from rich.table import Table  # noqa: F401
            from rich.console import Console

            self._console = Console()
            self._rich_available = True
        except ImportError:
            self._rich_available = False

    def _elapsed(self) -> str:
        s = int(time.monotonic() - self._start_time)
        return f"{s // 60:02d}:{s % 60:02d}"

    def start(self) -> None:
        if not self._rich_available:
            return
        from rich.live import Live

        self._live = Live(console=self._console, refresh_per_second=1)
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def update(self, stats: DashboardStats) -> None:
        """Update the dashboard with current stats."""
        if not self._rich_available or not self._live:
            # Fallback to simple logging
            logger.info(
                f"[{self._backend}] {self._elapsed()} | "
                f"Uploads: {stats.uploads_done}/{stats.uploads_total} | "
                f"Searches: {stats.searches_done} | "
                f"429s: {stats.rate_limit_429s}rl/{stats.quota_429s}q | "
                f"Errors: {stats.errors}"
            )
            return

        from rich.table import Table

        table = Table(
            title=f"STRESS TEST: {self._backend} | "
            f"Tenants: {self._num_tenants} | "
            f"Elapsed: {self._elapsed()}",
            show_header=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("p50", justify="right")
        table.add_column("p95", justify="right")
        table.add_column("Errors", justify="right", style="red")

        table.add_row(
            "Uploads",
            f"{stats.uploads_done}/{stats.uploads_total}",
            f"{stats.upload_p50:.2f}s",
            f"{stats.upload_p95:.2f}s",
            str(stats.upload_errors),
        )
        table.add_row(
            "Searches",
            str(stats.searches_done),
            f"{stats.search_p50:.2f}s",
            f"{stats.search_p95:.2f}s",
            str(stats.search_errors),
        )
        table.add_row(
            "Deletions",
            str(stats.deletions_done),
            f"{stats.delete_p50:.2f}s",
            f"{stats.delete_p95:.2f}s",
            str(stats.delete_errors),
        )
        table.add_section()
        table.add_row(
            "Cache Hit Rate",
            f"{stats.cache_hit_rate:.1%}" if stats.cache_hit_rate is not None else "N/A",
            "",
            "",
            f"RL-429: {stats.rate_limit_429s} | Q-429: {stats.quota_429s}",
        )

        self._live.update(table)


@dataclass
class DashboardStats:
    """Snapshot of current stress test progress."""

    uploads_done: int = 0
    uploads_total: int = 0
    upload_p50: float = 0.0
    upload_p95: float = 0.0
    upload_errors: int = 0

    searches_done: int = 0
    search_p50: float = 0.0
    search_p95: float = 0.0
    search_errors: int = 0

    deletions_done: int = 0
    delete_p50: float = 0.0
    delete_p95: float = 0.0
    delete_errors: int = 0

    cache_hit_rate: Optional[float] = None
    rate_limit_429s: int = 0
    quota_429s: int = 0
    errors: int = 0
    healthy: bool = True


# ---------------------------------------------------------------------------
# Timing Report
# ---------------------------------------------------------------------------


@dataclass
class BackendReport:
    """Complete timing report for a single backend run."""

    backend: str
    architecture: str
    run_mode: Dict[str, Any] = field(default_factory=dict)
    startup_time_s: float = 0.0

    upload: Dict[str, Any] = field(default_factory=dict)
    search: Dict[str, Any] = field(default_factory=dict)
    citations: Dict[str, Any] = field(default_factory=dict)
    delete: Dict[str, Any] = field(default_factory=dict)

    cache: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)
    tenant_isolation: Dict[str, Any] = field(default_factory=dict)
    rate_limiting: Dict[str, Any] = field(default_factory=dict)
    endurance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        backend: str,
        backend_info: Any,
        client_metrics: List[Any],
    ) -> BackendReport:
        """Build report from collected client metrics."""
        import platform

        # Aggregate latencies across all clients
        all_upload = []
        all_search = []
        all_citations = []
        all_delete = []
        total_rl_429 = 0
        total_q_429 = 0
        total_errors = 0

        for m in client_metrics:
            all_upload.extend(m.latencies.get("upload", []))
            all_search.extend(m.latencies.get("search", []))
            all_citations.extend(m.latencies.get("citations", []))
            all_delete.extend(m.latencies.get("delete", []))
            total_rl_429 += m.rate_limit_429s
            total_q_429 += m.quota_429s
            total_errors += m.errors

        report = cls(
            backend=backend,
            architecture=platform.machine(),
            run_mode={
                "celery": backend_info.celery_enabled,
                "worker_mode": backend_info.worker_mode,
            },
            startup_time_s=backend_info.startup_time_s,
            upload=compute_latency_stats(all_upload),
            search=compute_latency_stats(all_search),
            citations=compute_latency_stats(all_citations),
            delete=compute_latency_stats(all_delete),
            rate_limiting={
                "rate_limit_429s": total_rl_429,
                "quota_429s": total_q_429,
                "total_errors": total_errors,
            },
        )
        return report

    def to_json(self) -> str:
        """Serialize to JSON."""
        from dataclasses import asdict

        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, output_dir: Path) -> Path:
        """Save report to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"stress_report_{self.backend}.json"
        path.write_text(self.to_json())
        logger.info(f"Report saved: {path}")
        return path


def print_comparison_table(reports: List[BackendReport]) -> None:
    """Print a cross-backend comparison table using rich."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="STRESS TEST COMPARISON REPORT")
        table.add_column("Backend", style="cyan")
        table.add_column("Startup", justify="right")
        table.add_column("Search p50", justify="right")
        table.add_column("Search p95", justify="right")
        table.add_column("Upload p50", justify="right")
        table.add_column("NDCG@5", justify="right")
        table.add_column("Notes", style="dim")

        for r in reports:
            notes = []
            if r.run_mode.get("worker_mode") == "uvicorn-single":
                notes.append("1 worker")
            if not r.run_mode.get("celery"):
                notes.append("no celery")
            note_str = ", ".join(notes) if notes else ""

            table.add_row(
                r.backend,
                f"{r.startup_time_s:.1f}s",
                f"{r.search.get('p50', 0):.2f}s",
                f"{r.search.get('p95', 0):.2f}s",
                f"{r.upload.get('p50', 0):.2f}s",
                f"{r.quality.get('ndcg@5', 0):.3f}" if r.quality else "N/A",
                note_str,
            )

        console.print(table)
    except ImportError:
        # Fallback without rich
        for r in reports:
            logger.info(
                f"{r.backend}: startup={r.startup_time_s:.1f}s "
                f"search_p50={r.search.get('p50', 0):.2f}s "
                f"search_p95={r.search.get('p95', 0):.2f}s"
            )
