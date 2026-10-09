"""Async API client for stress tests with 429 differentiation.

Key design decisions:
- Does NOT blanket-retry on 429 — tracks rate-limit vs quota 429s separately
- Rate-limit 429: has Retry-After header → tracked, optionally waited
- Quota 429: no Retry-After header → tracked as hard failure
- All request timing recorded for percentile analysis
- Sessions are per-request to avoid event-loop mismatch with pytest-asyncio
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UploadResult:
    success: bool
    document_id: str = ""
    filename: str = ""
    status_code: int = 0
    latency_s: float = 0.0
    error: str = ""


@dataclass
class SearchResult:
    success: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    status_code: int = 0
    latency_s: float = 0.0
    error: str = ""
    routing: Optional[Dict[str, Any]] = None
    rate_limited: bool = False
    retry_after: Optional[float] = None


@dataclass
class CitationResult:
    success: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    total_results: int = 0
    status_code: int = 0
    latency_s: float = 0.0
    error: str = ""
    rate_limited: bool = False
    retry_after: Optional[float] = None


@dataclass
class DeleteResult:
    success: bool
    status_code: int = 0
    latency_s: float = 0.0
    error: str = ""


@dataclass
class HealthResult:
    healthy: bool
    status: str = ""
    checks: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 0


# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    """Per-client request metrics with 429 differentiation."""

    latencies: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "upload": [],
            "search": [],
            "citations": [],
            "delete": [],
            "health": [],
        }
    )
    status_codes: Counter = field(default_factory=Counter)
    rate_limit_429s: int = 0  # 429 with Retry-After header
    quota_429s: int = 0  # 429 without Retry-After header
    errors: int = 0  # 5xx, timeouts, connection errors
    total_requests: int = 0

    def record(
        self, endpoint: str, latency: float, status: int, has_retry_after: bool = False
    ) -> None:
        self.total_requests += 1
        if endpoint in self.latencies:
            self.latencies[endpoint].append(latency)
        self.status_codes[status] += 1

        if status == 429:
            if has_retry_after:
                self.rate_limit_429s += 1
            else:
                self.quota_429s += 1
        elif status >= 500:
            self.errors += 1


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


class StressAPIClient:
    """Async aiohttp client for a single tenant session.

    Each client has a fixed X-Session-ID and tracks its own request metrics.
    Sessions are created per-request to avoid event-loop mismatch with
    pytest-asyncio scoped fixtures.
    """

    def __init__(self, base_url: str, session_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.metrics = RequestMetrics()

    def _make_session(self, timeout: float = 60) -> aiohttp.ClientSession:
        """Create a new session bound to the current event loop."""
        return aiohttp.ClientSession(
            headers={"X-Session-ID": self.session_id},
            timeout=aiohttp.ClientTimeout(total=timeout),
        )

    async def close(self) -> None:
        """No-op — sessions are per-request."""
        pass

    def _parse_retry_after(self, resp: aiohttp.ClientResponse) -> Optional[float]:
        """Extract Retry-After header value (seconds)."""
        val = resp.headers.get("Retry-After")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    # -- Upload ----------------------------------------------------------------

    async def upload_document(self, file_path: Path, filename: str) -> UploadResult:
        """POST /api/v1/documents — upload a document.

        Uses a 180s timeout (uploads are synchronous and include embedding
        computation which can take 10-60s per document under load).
        """
        url = f"{self.base_url}/api/v1/documents"

        t0 = time.monotonic()
        fh = None
        try:
            fh = open(file_path, "rb")
            data = aiohttp.FormData()
            data.add_field(
                "file",
                fh,
                filename=filename,
                content_type="application/octet-stream",
            )
            async with self._make_session(timeout=180) as session:
                async with session.post(url, data=data) as resp:
                    latency = time.monotonic() - t0
                    has_retry = self._parse_retry_after(resp) is not None
                    self.metrics.record("upload", latency, resp.status, has_retry)

                    if resp.status in (200, 201):
                        body = await resp.json()
                        return UploadResult(
                            success=True,
                            document_id=body.get("document_id", ""),
                            filename=body.get("filename", filename),
                            status_code=resp.status,
                            latency_s=latency,
                        )
                    elif resp.status == 429:
                        retry_after = self._parse_retry_after(resp)
                        kind = "rate-limit" if retry_after else "quota"
                        return UploadResult(
                            success=False,
                            status_code=429,
                            latency_s=latency,
                            error=f"429 ({kind}), retry_after={retry_after}",
                        )
                    else:
                        body = await resp.text()
                        return UploadResult(
                            success=False,
                            status_code=resp.status,
                            latency_s=latency,
                            error=f"HTTP {resp.status}: {body[:200]}",
                        )
        except Exception as exc:
            latency = time.monotonic() - t0
            self.metrics.errors += 1
            self.metrics.total_requests += 1
            error_msg = str(exc) or f"{type(exc).__name__} after {latency:.1f}s"
            return UploadResult(
                success=False,
                latency_s=latency,
                error=error_msg,
            )
        finally:
            if fh is not None:
                fh.close()

    async def wait_for_ready(
        self,
        document_id: str,
        timeout: float = 60,
        poll_interval: float = 1.0,
    ) -> float:
        """Poll document status until 'ready'. Returns processing time in seconds."""
        url = f"{self.base_url}/api/v1/documents/{document_id}/status"
        t0 = time.monotonic()
        deadline = t0 + timeout

        while time.monotonic() < deadline:
            try:
                async with self._make_session() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            body = await resp.json()
                            status = body.get("status", "")
                            if status == "ready":
                                return time.monotonic() - t0
                            if status == "error":
                                raise RuntimeError(
                                    f"Document {document_id} failed: "
                                    f"{body.get('error_message', 'unknown')}"
                                )
            except (aiohttp.ClientError, RuntimeError):
                raise
            except Exception:
                pass

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Document {document_id} not ready after {timeout}s")

    # -- Search ----------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_reranking: bool = True,
        routing_tier: Optional[str] = None,
    ) -> SearchResult:
        """POST /api/v1/search — with feature flags."""
        url = f"{self.base_url}/api/v1/search"

        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "use_reranking": use_reranking,
        }
        if routing_tier:
            payload["routing_tier"] = routing_tier

        t0 = time.monotonic()
        try:
            async with self._make_session() as session:
                async with session.post(url, json=payload) as resp:
                    latency = time.monotonic() - t0
                    has_retry = self._parse_retry_after(resp) is not None
                    self.metrics.record("search", latency, resp.status, has_retry)

                    if resp.status == 200:
                        body = await resp.json()
                        return SearchResult(
                            success=True,
                            results=body.get("results", []),
                            total_results=body.get("total_results", 0),
                            status_code=200,
                            latency_s=latency,
                            routing=body.get("routing"),
                        )
                    elif resp.status == 429:
                        retry_after = self._parse_retry_after(resp)
                        is_rate_limit = retry_after is not None
                        kind = "rate-limit" if is_rate_limit else "quota"
                        return SearchResult(
                            success=False,
                            status_code=429,
                            latency_s=latency,
                            rate_limited=is_rate_limit,
                            retry_after=retry_after,
                            error=f"429 ({kind}), retry_after={retry_after}",
                        )
                    else:
                        return SearchResult(
                            success=False,
                            status_code=resp.status,
                            latency_s=latency,
                            error=f"HTTP {resp.status}",
                        )
        except Exception as exc:
            latency = time.monotonic() - t0
            self.metrics.errors += 1
            self.metrics.total_requests += 1
            return SearchResult(success=False, latency_s=latency, error=str(exc))

    async def search_with_citations(
        self,
        query: str,
        top_k: int = 5,
        routing_tier: Optional[str] = None,
    ) -> CitationResult:
        """POST /api/v1/search/citations.

        NOTE: This endpoint does NOT pass use_hyde or use_reranking flags.
        See search.py — only query, session_id, top_k, document_ids, routing_tier
        are forwarded to the backend.
        """
        url = f"{self.base_url}/api/v1/search/citations"

        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if routing_tier:
            payload["routing_tier"] = routing_tier

        t0 = time.monotonic()
        try:
            async with self._make_session() as session:
                async with session.post(url, json=payload) as resp:
                    latency = time.monotonic() - t0
                    has_retry = self._parse_retry_after(resp) is not None
                    self.metrics.record("citations", latency, resp.status, has_retry)

                    if resp.status == 200:
                        body = await resp.json()
                        return CitationResult(
                            success=True,
                            results=body.get("results", []),
                            citations=body.get("citations", []),
                            context=body.get("context", ""),
                            total_results=body.get("total_results", 0),
                            status_code=200,
                            latency_s=latency,
                        )
                    elif resp.status == 429:
                        retry_after = self._parse_retry_after(resp)
                        is_rate_limit = retry_after is not None
                        return CitationResult(
                            success=False,
                            status_code=429,
                            latency_s=latency,
                            rate_limited=is_rate_limit,
                            retry_after=retry_after,
                        )
                    else:
                        return CitationResult(
                            success=False,
                            status_code=resp.status,
                            latency_s=latency,
                            error=f"HTTP {resp.status}",
                        )
        except Exception as exc:
            latency = time.monotonic() - t0
            self.metrics.errors += 1
            self.metrics.total_requests += 1
            return CitationResult(success=False, latency_s=latency, error=str(exc))

    # -- Documents -------------------------------------------------------------

    async def delete_document(self, document_id: str) -> DeleteResult:
        """DELETE /api/v1/documents/{id}."""
        url = f"{self.base_url}/api/v1/documents/{document_id}"

        t0 = time.monotonic()
        try:
            async with self._make_session() as session:
                async with session.delete(url) as resp:
                    latency = time.monotonic() - t0
                    self.metrics.record("delete", latency, resp.status)
                    return DeleteResult(
                        success=resp.status == 200,
                        status_code=resp.status,
                        latency_s=latency,
                    )
        except Exception as exc:
            latency = time.monotonic() - t0
            self.metrics.errors += 1
            self.metrics.total_requests += 1
            return DeleteResult(success=False, latency_s=latency, error=str(exc))

    async def list_documents(self) -> List[Dict[str, Any]]:
        """GET /api/v1/documents — list all documents for this session."""
        url = f"{self.base_url}/api/v1/documents"
        try:
            async with self._make_session() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        return body.get("documents", [])
                    return []
        except Exception:
            return []

    # -- Health ----------------------------------------------------------------

    async def health_ready(self) -> HealthResult:
        """GET /api/v1/health/ready — readiness check."""
        url = f"{self.base_url}/api/v1/health/ready"

        try:
            async with self._make_session() as session:
                async with session.get(url) as resp:
                    body = await resp.json()
                    return HealthResult(
                        healthy=resp.status == 200,
                        status=body.get("status", ""),
                        checks=body.get("checks", {}),
                        status_code=resp.status,
                    )
        except Exception as exc:
            return HealthResult(healthy=False, status=str(exc))

    # -- Stats -----------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        """GET /api/v1/stats — system statistics.

        Note: vector_backend and cache fields are NOT populated by get_stats().
        Only features, embedding_model, embedding_dim are reliable.
        """
        url = f"{self.base_url}/api/v1/stats"
        try:
            async with self._make_session() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {}
        except Exception:
            return {}
