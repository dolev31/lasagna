import json
import hashlib
import threading
from time import perf_counter
from typing import Any, Dict, Iterable, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Metrics backend protocol
# ---------------------------------------------------------------------------


class _MetricsBackend:
    """Base class for metrics backends."""

    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        return

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        return


class _PrometheusBackend(_MetricsBackend):
    """Prometheus metrics backend using prometheus-client.

    Lazily creates Counter and Histogram objects on first use.
    Metric names are sanitized (dots → underscores) to comply with
    Prometheus naming conventions.

    Thread-safe: a lock guards the lazy-creation dicts so that concurrent
    calls never attempt to register the same Prometheus metric twice
    (which would raise ``ValueError: Duplicated timeseries``).

    Latency histograms (names containing ``_ms``) use millisecond-tuned
    buckets; all others use the prometheus-client defaults.
    """

    # Buckets tuned for RAG pipeline latencies (milliseconds)
    _MS_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(self) -> None:
        from prometheus_client import Counter, Histogram  # noqa: F401 – import check

        self._counters: Dict[tuple, Any] = {}
        self._histograms: Dict[tuple, Any] = {}
        self._lock = threading.Lock()

    def _sanitize(self, name: str) -> str:
        return name.replace(".", "_").replace("-", "_")

    def _get_counter(self, name: str, tags: Optional[Dict[str, str]]) -> Any:
        label_names = tuple(sorted(tags.keys())) if tags else ()
        key = (name, label_names)
        existing = self._counters.get(key)
        if existing is not None:
            return existing
        with self._lock:
            # Double-check after acquiring lock
            existing = self._counters.get(key)
            if existing is not None:
                return existing
            from prometheus_client import Counter

            metric = Counter(
                self._sanitize(name),
                f"Counter: {name}",
                labelnames=label_names,
            )
            self._counters[key] = metric
            return metric

    def _get_histogram(self, name: str, tags: Optional[Dict[str, str]]) -> Any:
        label_names = tuple(sorted(tags.keys())) if tags else ()
        key = (name, label_names)
        existing = self._histograms.get(key)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._histograms.get(key)
            if existing is not None:
                return existing
            from prometheus_client import Histogram

            buckets = self._MS_BUCKETS if "_ms" in name else Histogram.DEFAULT_BUCKETS
            metric = Histogram(
                self._sanitize(name),
                f"Histogram: {name}",
                labelnames=label_names,
                buckets=buckets,
            )
            self._histograms[key] = metric
            return metric

    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        c = self._get_counter(name, tags)
        if tags:
            c.labels(**tags).inc(value)
        else:
            c.inc(value)

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        h = self._get_histogram(name, tags)
        if tags:
            h.labels(**tags).observe(value)
        else:
            h.observe(value)


# ---------------------------------------------------------------------------
# Public facade — object identity never changes, backend is swappable
# ---------------------------------------------------------------------------


class MetricsClient:
    """Metrics facade with pluggable backend.

    Default: no-op (zero overhead, single-user mode).
    Production: call ``configure_metrics(enable_prometheus=True)`` to activate
    Prometheus counters and histograms.  All existing ``METRICS.counter()`` /
    ``METRICS.histogram()`` call-sites light up immediately — no code changes
    needed anywhere else.

    The singleton ``METRICS`` object is imported by reference throughout the
    codebase.  The facade pattern ensures those references stay valid when the
    backend is swapped at runtime.

    Never raises: a backend error is logged once per metric name and the
    call returns, so recording a metric cannot change what a request does.
    """

    def __init__(self) -> None:
        self._backend: _MetricsBackend = _MetricsBackend()  # no-op by default
        self._failed: set = set()

    # -- public API (unchanged signature) ----------------------------------

    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        try:
            self._backend.counter(name, value, tags)
        except Exception as exc:
            self._failed_once(name, exc)

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        try:
            self._backend.histogram(name, value, tags)
        except Exception as exc:
            self._failed_once(name, exc)

    def _failed_once(self, name: str, exc: Exception) -> None:
        # Metrics are best-effort. A raise here used to reach the search's
        # routing `except`, which reran the standard pipeline and discarded
        # the agentic results it had just paid for.
        if name in self._failed:
            return
        self._failed.add(name)
        logger.warning(
            f"Could not record the metric {name} ({type(exc).__name__}: {exc}). "
            "The request it measures is unaffected; this is logged once per metric."
        )

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_prometheus_enabled(self) -> bool:
        return isinstance(self._backend, _PrometheusBackend)


def configure_metrics(enable_prometheus: bool = False) -> None:
    """Switch the global ``METRICS`` backend.

    Args:
        enable_prometheus: When *True*, activates Prometheus counters and
            histograms.  Requires ``prometheus-client`` to be installed;
            falls back to no-op with a warning if the package is missing.

    Idempotent: calling with ``enable_prometheus=True`` when Prometheus is
    already active is a no-op (avoids "Duplicated timeseries" errors from
    re-registering metrics in the global Prometheus CollectorRegistry).
    """
    if enable_prometheus:
        if METRICS.is_prometheus_enabled:
            return  # already active — nothing to do
        try:
            METRICS._backend = _PrometheusBackend()
            logger.info("Prometheus metrics enabled: all RAG instrumentation active")
        except ImportError:
            METRICS._backend = _MetricsBackend()
            logger.warning(
                "prometheus-client is not installed: metrics remain no-op.  "
                "Install with: pip install prometheus-client>=0.20.0"
            )
    else:
        METRICS._backend = _MetricsBackend()
        logger.debug("Metrics set to no-op mode")


METRICS = MetricsClient()


def trunc_text(value: Optional[str], max_len: int = 200) -> str:
    if value is None:
        return ""
    return value if len(value) <= max_len else f"{value[:max_len]}..."


def hash_text(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def summarize_document_ids(
    document_ids: Optional[Iterable[str]], max_list: int = 5
) -> Dict[str, Any]:
    if not document_ids:
        return {"document_ids_count": 0, "document_ids": []}
    doc_list = list(document_ids)
    payload = {"document_ids_count": len(doc_list)}
    if len(doc_list) <= max_list:
        payload["document_ids"] = doc_list
    return payload


def log_json(event: str, payload: Dict[str, Any]) -> None:
    body = {"event": event, **payload}
    logger.info(json.dumps(body, default=str))


def log_json_debug(event: str, payload: Dict[str, Any]) -> None:
    body = {"event": event, **payload}
    logger.debug(json.dumps(body, default=str))


class StageTimer:
    """Simple perf_counter-based timer."""

    def __init__(self) -> None:
        self._start = perf_counter()

    def elapsed_ms(self) -> float:
        return (perf_counter() - self._start) * 1000.0
