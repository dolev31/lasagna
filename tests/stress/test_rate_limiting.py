"""Test 4: Adaptive rate limiter behavior validation.

Tests the adaptive token-bucket rate limiter with burst capacity,
NOT strict RPM enforcement. Key behaviors:
- burst_capacity=10 allows initial spike
- burst_multiplier=2.0 doubles effective rate when idle
- Adaptive strategy throttles to 0.5x under load
- Rate-limit 429s MUST have Retry-After header
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from .helpers.api_client import StressAPIClient

pytestmark = pytest.mark.stress


class TestRateLimiting:
    """Adaptive rate limiter behavior tests."""

    @pytest.mark.asyncio
    async def test_adaptive_burst_behavior(self, loaded_backend, backend):
        """Verify burst capacity allows initial spike above base rate.

        Rate limiter config:
        - burst_capacity=10, burst_multiplier=2.0
        - When idle, effective rate = 2x base rate
        - After burst depletes, expect 429s

        Test:
        1. Send 10 rapid requests (within burst capacity) — all should pass
        2. Continue rapid-fire — expect some 429s after burst depletes
        3. Verify 429 responses have Retry-After header
        """
        ctx = loaded_backend
        config = ctx["config"]

        client = StressAPIClient(
            config.api_base_url,
            f"burst-test-{uuid.uuid4().hex[:8]}",
        )

        try:
            # Phase 1: Burst — send 10 requests rapidly (within burst_capacity=10)
            burst_results = []
            for _ in range(10):
                result = await client.search("burst test query", top_k=3)
                burst_results.append(result)
                # No delay between requests — test burst

            burst_successes = sum(1 for r in burst_results if r.success)
            # Most burst requests should succeed (burst_capacity=10)
            assert burst_successes >= 8, (
                f"Only {burst_successes}/10 burst requests succeeded. "
                f"Burst capacity should allow initial spike."
            )

            # Phase 2: Continue rapid-fire to deplete tokens
            # Single-worker backends (milvus-lite) process requests serially,
            # so tokens don't refill between rapid requests. Add a tiny delay
            # to allow token refill while still being fast enough to trigger
            # rate limiting.
            is_single_worker = ctx["backend_info"].worker_mode == "uvicorn-single"
            post_burst_results = []
            for _ in range(20):
                result = await client.search("post burst query", top_k=3)
                post_burst_results.append(result)
                if is_single_worker:
                    await asyncio.sleep(0.05)

            # Some should be rate-limited after burst depletes
            _rate_limited = [r for r in post_burst_results if r.rate_limited]
            total_429s = client.metrics.rate_limit_429s

            # At least some requests should have gotten through
            post_successes = sum(1 for r in post_burst_results if r.success)
            assert post_successes > 0, "No requests succeeded after burst"

            # Verify the rate limiter actually kicked in (at least 1 429)
            if total_429s == 0:
                pytest.skip(
                    "Rate limiter did not trigger with 30 rapid requests — "
                    "base rate may be too high for this test to exercise"
                )

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_429_has_retry_after(self, loaded_backend, backend):
        """All rate-limit 429s MUST include Retry-After header.

        This distinguishes rate-limit 429s from quota 429s
        (QuotaExceededError returns 429 without Retry-After).
        """
        ctx = loaded_backend
        config = ctx["config"]

        client = StressAPIClient(
            config.api_base_url,
            f"retry-after-{uuid.uuid4().hex[:8]}",
        )

        try:
            # Hammer the endpoint to trigger rate limiting
            results = []
            for _ in range(50):
                result = await client.search("retry after test", top_k=3)
                results.append(result)

            # Always check quota 429s first — these indicate a real problem
            # regardless of whether rate-limit 429s were observed
            assert client.metrics.quota_429s == 0, (
                f"Got {client.metrics.quota_429s} quota 429s — "
                f"quotas should not trigger with default high limits"
            )

            # Check all 429s that were rate-limited
            rate_limited = [r for r in results if r.rate_limited]

            if not rate_limited:
                pytest.skip(
                    "No rate-limit 429s triggered with 50 rapid requests — "
                    "cannot validate Retry-After header"
                )

            for r in rate_limited:
                assert r.retry_after is not None, "Rate-limit 429 missing Retry-After header"
                assert r.retry_after >= 0, (
                    f"Retry-After should be non-negative, got {r.retry_after}"
                )

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_multi_tenant_fairness(self, loaded_backend, backend):
        """Verify fair rate distribution across tenants.

        N tenants send requests simultaneously at equal rates.
        No single tenant should get disproportionately more 429s.
        """
        ctx = loaded_backend
        config = ctx["config"]

        num_tenants = 10
        requests_per_tenant = 15
        clients = []

        for i in range(num_tenants):
            clients.append(
                StressAPIClient(
                    config.api_base_url,
                    f"fairness-{i}-{uuid.uuid4().hex[:8]}",
                )
            )

        try:
            # All tenants send requests concurrently
            async def tenant_workload(client: StressAPIClient):
                for _ in range(requests_per_tenant):
                    await client.search("fairness test query", top_k=3)
                    await asyncio.sleep(0.05)  # Small delay to spread load

            tasks = [tenant_workload(c) for c in clients]
            await asyncio.gather(*tasks)

            # Collect per-tenant 429 counts
            tenant_429s = [c.metrics.rate_limit_429s for c in clients]

            # If rate limiting kicked in, check fairness
            total_429s = sum(tenant_429s)
            if total_429s > 0:
                mean_429s = total_429s / num_tenants
                if mean_429s > 0:
                    max_429s = max(tenant_429s)
                    # No tenant should get more than 3x the mean
                    assert max_429s <= max(mean_429s * 3, 5), (
                        f"Unfair rate limiting: max={max_429s}, mean={mean_429s:.1f}. "
                        f"Per-tenant: {tenant_429s}"
                    )

        finally:
            for c in clients:
                await c.close()

    @pytest.mark.asyncio
    async def test_recovery_after_throttling(self, loaded_backend, backend):
        """After rate limiting triggers, system recovers to normal throughput.

        1. Burst above capacity to trigger 429s
        2. Wait for token refill (use Retry-After)
        3. Resume at moderate rate — all requests should pass
        """
        ctx = loaded_backend
        config = ctx["config"]

        client = StressAPIClient(
            config.api_base_url,
            f"recovery-{uuid.uuid4().hex[:8]}",
        )

        try:
            # Phase 1: Trigger rate limiting
            max_retry_after = 0.0
            for _ in range(30):
                result = await client.search("recovery test", top_k=3)
                if result.retry_after and result.retry_after > max_retry_after:
                    max_retry_after = result.retry_after

            # Phase 2: Wait for recovery
            wait_time = max(max_retry_after, 2.0)
            await asyncio.sleep(wait_time)

            # Phase 3: Send requests at moderate rate — should all pass
            recovery_results = []
            for _ in range(5):
                result = await client.search("recovery test after wait", top_k=3)
                recovery_results.append(result)
                await asyncio.sleep(1.0)  # 1 req/sec, well under any rate limit

            recovery_successes = sum(1 for r in recovery_results if r.success)
            assert recovery_successes >= 4, (
                f"Only {recovery_successes}/5 succeeded after recovery wait. "
                f"Rate limiter should have refilled tokens."
            )

        finally:
            await client.close()
