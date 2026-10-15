"""OpenTelemetry distributed tracing for the RAG search pipeline.

Provides dual-mode tracing:
    - **Default (no-op):** Zero overhead, no dependencies required.
    - **Production:** ``enable_tracing = true`` activates real OpenTelemetry spans.
      Requires ``opentelemetry-api`` and ``opentelemetry-sdk``.

Usage in pipeline code::

    from lasagna.utils.tracing import TRACER

    with TRACER.span("dense_search", attributes={"top_k": 75}) as span:
        results = vector_store.search(query_embedding, top_k=75)
        span.set_attribute("result_count", len(results))

The ``TRACER`` singleton follows the same facade pattern as ``METRICS``:
object identity never changes, backend is swapped via ``configure_tracing()``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse

from loguru import logger


# ---------------------------------------------------------------------------
# No-op span (default)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Span-like object that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        return

    def set_status(self, status: Any) -> None:
        return

    def record_exception(self, exception: BaseException) -> None:
        return


_NOOP_SPAN = _NoOpSpan()


# ---------------------------------------------------------------------------
# Tracing backends
# ---------------------------------------------------------------------------


class _TracingBackend:
    """Base no-op tracing backend."""

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NOOP_SPAN

    def shutdown(self) -> None:
        """No-op shutdown."""


class _OpenTelemetryBackend(_TracingBackend):
    """Real OpenTelemetry tracing backend.

    Creates spans using the OpenTelemetry API with proper nesting,
    attribute recording, and exception capture.
    """

    def __init__(self, service_name: str, exporter: str, endpoint: str) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        self._provider = TracerProvider(resource=resource)

        # Configure exporter
        span_exporter = self._create_exporter(exporter, endpoint)
        if span_exporter is not None:
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self._provider.add_span_processor(BatchSpanProcessor(span_exporter))

        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer("rag.knowledge")

    @staticmethod
    def _create_exporter(exporter: str, endpoint: str) -> Any:
        """Create the configured span exporter."""
        if exporter == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            return ConsoleSpanExporter()

        if exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                return OTLPSpanExporter(endpoint=endpoint)
            except ImportError:
                logger.warning(
                    "opentelemetry-exporter-otlp-proto-grpc not installed, "
                    "falling back to console exporter"
                )
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                return ConsoleSpanExporter()

        if exporter == "jaeger":
            logger.warning(
                "The 'jaeger' exporter is deprecated. Modern Jaeger accepts OTLP natively. "
                "Consider using exporter='otlp' with your Jaeger endpoint."
            )
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                parsed = urlparse(endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 6831
                return JaegerExporter(agent_host_name=host, agent_port=port)
            except ImportError:
                logger.warning("opentelemetry-exporter-jaeger not installed")
                return None

        if exporter == "zipkin":
            try:
                from opentelemetry.exporter.zipkin.json import ZipkinExporter

                return ZipkinExporter(endpoint=endpoint)
            except ImportError:
                logger.warning("opentelemetry-exporter-zipkin not installed")
                return None

        logger.warning(f"Unknown tracing exporter '{exporter}', using console")
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        with self._tracer.start_as_current_span(name, attributes=attributes or {}) as otel_span:
            try:
                yield otel_span
            except Exception as exc:
                otel_span.record_exception(exc)
                from opentelemetry.trace import StatusCode

                otel_span.set_status(StatusCode.ERROR, str(exc))
                raise

    def shutdown(self) -> None:
        """Flush pending spans and shut down the tracer provider."""
        try:
            self._provider.shutdown()
        except Exception as exc:
            logger.debug(f"TracerProvider shutdown error (expected on repeat): {exc}")


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class Tracer:
    """Tracing facade with pluggable backend.

    Default: no-op (zero overhead).
    Production: call ``configure_tracing(...)`` to activate OpenTelemetry.

    The ``TRACER`` singleton is imported by reference. The facade pattern
    keeps those references valid when the backend is swapped at runtime.
    """

    def __init__(self) -> None:
        self._backend: _TracingBackend = _TracingBackend()

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Create a trace span.

        Args:
            name: Span name (e.g. ``"dense_search"``, ``"rerank"``).
            attributes: Key-value pairs attached to the span.

        Yields:
            A span object with ``set_attribute()``, ``record_exception()``.
        """
        with self._backend.span(name, attributes) as s:
            yield s

    @property
    def is_tracing_enabled(self) -> bool:
        return isinstance(self._backend, _OpenTelemetryBackend)


def configure_tracing(
    enable: bool = False,
    service_name: str = "rag-knowledge",
    exporter: str = "otlp",
    endpoint: str = "http://localhost:4317",
) -> None:
    """Switch the global ``TRACER`` backend.

    Args:
        enable: When *True*, activates OpenTelemetry tracing.
        service_name: Service name reported in traces.
        exporter: Exporter type — ``'otlp'``, ``'jaeger'``, ``'zipkin'``, or ``'console'``.
        endpoint: Collector endpoint URL.
    """
    # Shut down previous backend to flush spans and release threads
    TRACER._backend.shutdown()

    if enable:
        try:
            TRACER._backend = _OpenTelemetryBackend(service_name, exporter, endpoint)
            logger.info(
                f"OpenTelemetry tracing enabled: service={service_name}, "
                f"exporter={exporter}, endpoint={endpoint}"
            )
        except ImportError:
            TRACER._backend = _TracingBackend()
            logger.warning(
                "opentelemetry-api/sdk not installed: tracing remains no-op.  "
                "Install with: pip install opentelemetry-api>=1.24.0 opentelemetry-sdk>=1.24.0"
            )
    else:
        TRACER._backend = _TracingBackend()
        logger.debug("Tracing set to no-op mode")


TRACER = Tracer()
